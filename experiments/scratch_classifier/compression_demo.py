"""Lossless compression demo: causal 2D RoPE byte model + arithmetic coding.

The model's next-pixel predictions provide probability distributions for
arithmetic coding, turning our autoregressive model into a lossless compressor.
"""

import argparse
import gzip
import io
import json
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from train_rope_byte_2d import RoPE2DByteLM, MNISTByteDataset


# ============================================================
# Arithmetic coding (integer-based, 32-bit precision)
# ============================================================

PRECISION = 32
WHOLE = 1 << PRECISION
HALF = WHOLE >> 1
QUARTER = WHOLE >> 2


class ArithmeticEncoder:
    """Encodes a sequence of symbols given per-symbol CDFs."""

    def __init__(self):
        self.low = 0
        self.high = WHOLE
        self.pending = 0
        self.bits = []

    def _output_bit(self, bit):
        self.bits.append(bit)
        for _ in range(self.pending):
            self.bits.append(1 - bit)
        self.pending = 0

    def encode_symbol(self, cdf, symbol):
        """Encode one symbol. cdf is length 257 (cumulative for 256 symbols)."""
        rng = self.high - self.low
        total = cdf[256]
        self.high = self.low + (rng * cdf[symbol + 1]) // total
        self.low = self.low + (rng * cdf[symbol]) // total

        while True:
            if self.high <= HALF:
                self._output_bit(0)
            elif self.low >= HALF:
                self._output_bit(1)
                self.low -= HALF
                self.high -= HALF
            elif self.low >= QUARTER and self.high <= 3 * QUARTER:
                self.pending += 1
                self.low -= QUARTER
                self.high -= QUARTER
            else:
                break
            self.low <<= 1
            self.high <<= 1

    def finish(self):
        self.pending += 1
        if self.low < QUARTER:
            self._output_bit(0)
        else:
            self._output_bit(1)
        return self.bits

    def num_bits(self):
        return len(self.bits)


class ArithmeticDecoder:
    """Decodes a sequence given per-symbol CDFs (for verification)."""

    def __init__(self, bits):
        self.bits = bits
        self.pos = 0
        self.low = 0
        self.high = WHOLE
        self.value = 0
        for i in range(PRECISION):
            self.value = (self.value << 1) | self._read_bit()

    def _read_bit(self):
        if self.pos < len(self.bits):
            b = self.bits[self.pos]
            self.pos += 1
            return b
        return 0

    def decode_symbol(self, cdf):
        rng = self.high - self.low
        total = cdf[256]
        scaled = ((self.value - self.low + 1) * total - 1) // rng

        # Binary search for symbol
        lo, hi = 0, 256
        while lo < hi:
            mid = (lo + hi) // 2
            if cdf[mid + 1] <= scaled:
                lo = mid + 1
            else:
                hi = mid
        symbol = lo

        self.high = self.low + (rng * cdf[symbol + 1]) // total
        self.low = self.low + (rng * cdf[symbol]) // total

        while True:
            if self.high <= HALF:
                pass
            elif self.low >= HALF:
                self.low -= HALF
                self.high -= HALF
                self.value -= HALF
            elif self.low >= QUARTER and self.high <= 3 * QUARTER:
                self.low -= QUARTER
                self.high -= QUARTER
                self.value -= QUARTER
            else:
                break
            self.low <<= 1
            self.high <<= 1
            self.value = (self.value << 1) | self._read_bit()

        return symbol


def probs_to_cdf(probs, scale=1 << 16):
    """Convert probability array to integer CDF for arithmetic coding."""
    # Quantize probabilities to integers summing to scale
    counts = np.maximum((probs * scale).astype(np.int64), 1)  # min 1 to avoid zero-prob
    # Adjust so they sum to scale
    diff = scale - counts.sum()
    counts[counts.argmax()] += diff

    cdf = np.zeros(257, dtype=np.int64)
    for i in range(256):
        cdf[i + 1] = cdf[i] + counts[i]
    return cdf


# ============================================================
# Compression with model
# ============================================================

@torch.no_grad()
def compress_image(model, pixels, device):
    """Compress a single 784-pixel image using the model.

    Returns: number of bits in the compressed representation.
    """
    img_tensor = torch.tensor(pixels, dtype=torch.long).unsqueeze(0).to(device)
    logits = model(img_tensor)  # (1, 784, 256)
    probs_all = F.softmax(logits[0].float(), dim=-1).cpu().numpy()  # (784, 256)

    encoder = ArithmeticEncoder()

    # First pixel: uniform distribution (no context)
    uniform_cdf = np.arange(257, dtype=np.int64) * 256  # 0, 256, 512, ..., 65536
    encoder.encode_symbol(uniform_cdf, int(pixels[0]))

    # Remaining pixels: model prediction at position i-1 predicts pixel i
    for i in range(1, 784):
        cdf = probs_to_cdf(probs_all[i - 1])
        encoder.encode_symbol(cdf, int(pixels[i]))

    bits = encoder.finish()
    return len(bits)


@torch.no_grad()
def compress_batch_theoretical(model, images, device):
    """Compute theoretical compressed size (cross-entropy) for a batch.
    Faster than actual arithmetic coding — gives a tight lower bound.
    """
    images = images.to(device)
    logits = model(images)
    log_probs = F.log_softmax(logits.float(), dim=-1)

    # Shifted: logits[i] predicts pixel[i+1]
    total_bits = 0.0
    n_images = images.shape[0]

    # First pixel: uniform = 8 bits each
    total_bits += n_images * 8.0

    # Remaining pixels: -log2(p(pixel[i+1] | context))
    for i in range(783):
        targets = images[:, i + 1]  # (B,)
        lp = log_probs[:, i, :]  # (B, 256)
        bits_per_pixel = -lp[torch.arange(n_images), targets] / math.log(2)
        total_bits += bits_per_pixel.sum().item()

    return total_bits


def png_size(pixels):
    """Compress 28x28 grayscale image with PNG, return size in bits."""
    from PIL import Image
    img = Image.fromarray(pixels.reshape(28, 28).astype(np.uint8), mode='L')
    buf = io.BytesIO()
    img.save(buf, format='PNG', optimize=True)
    return len(buf.getvalue()) * 8


def gzip_size(pixels):
    """Compress raw bytes with gzip, return size in bits."""
    return len(gzip.compress(pixels.tobytes(), compresslevel=9)) * 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps')
    parser.add_argument('--n-images', type=int, default=1000)
    parser.add_argument('--verify', type=int, default=5,
                        help='Number of images to verify with actual arithmetic coding')
    args = parser.parse_args()

    device = args.device

    # Load model
    ckpt_path = 'pile_runs/rope_byte/mnist_2d_d64_causal_tp1800/best_model.pt'
    print(f'Loading causal model from {ckpt_path}...')
    model = RoPE2DByteLM(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                         height=28, width=28, bidirectional=False).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Load test data
    mnist_test = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True)
    test_labels = np.array([l for _, l in mnist_test])
    val_data = np.fromfile('pile_runs/custom_vocab/byte_mnist/val.bytes', dtype=np.uint8)
    n_images = min(args.n_images, len(val_data) // 784)

    print(f'\nCompressing {n_images} test images...\n')

    # ============================================================
    # Theoretical compression (cross-entropy bound)
    # ============================================================
    print('Computing model-based compression (theoretical)...')
    batch_size = 128
    total_model_bits = 0.0
    t0 = time.time()
    for start in range(0, n_images, batch_size):
        end = min(start + batch_size, n_images)
        batch_pixels = []
        for i in range(start, end):
            pixels = val_data[i * 784:(i + 1) * 784].astype(np.int64)
            batch_pixels.append(pixels)
        batch_tensor = torch.tensor(np.stack(batch_pixels), dtype=torch.long)
        total_model_bits += compress_batch_theoretical(model, batch_tensor, device)
    model_time = time.time() - t0

    # ============================================================
    # Baseline compression
    # ============================================================
    print('Computing baselines (PNG, gzip)...')
    total_png_bits = 0
    total_gzip_bits = 0
    total_raw_bits = n_images * 784 * 8

    try:
        from PIL import Image
        has_pil = True
    except ImportError:
        has_pil = False
        print('  (PIL not available, skipping PNG)')

    for i in range(n_images):
        pixels = val_data[i * 784:(i + 1) * 784]
        if has_pil:
            total_png_bits += png_size(pixels)
        total_gzip_bits += gzip_size(pixels)

    # ============================================================
    # Actual arithmetic coding verification (a few images)
    # ============================================================
    print(f'\nVerifying actual arithmetic coding on {args.verify} images...')
    verify_results = []
    for i in range(min(args.verify, n_images)):
        pixels = val_data[i * 784:(i + 1) * 784].astype(np.int64)
        t0v = time.time()
        actual_bits = compress_image(model, pixels, device)
        tv = time.time() - t0v

        # Compute theoretical bits for this image
        with torch.no_grad():
            img_tensor = torch.tensor(pixels, dtype=torch.long).unsqueeze(0).to(device)
            logits = model(img_tensor)
            probs_all = F.softmax(logits[0].float(), dim=-1).cpu().numpy()

        # Theoretical bits for this image
        theory_bits = 8.0  # first pixel uniform
        for j in range(1, 784):
            p = probs_all[j - 1, pixels[j]]
            if p > 0:
                theory_bits += -math.log2(p)
            else:
                theory_bits += 16  # penalty for zero-prob

        if has_pil:
            img_png = png_size(pixels)
        else:
            img_png = 0
        img_gzip = gzip_size(pixels)

        verify_results.append({
            'label': int(test_labels[i]),
            'actual_bits': actual_bits,
            'theoretical_bits': theory_bits,
            'png_bits': img_png,
            'gzip_bits': img_gzip,
            'raw_bits': 784 * 8,
            'time_s': tv,
        })

        print(f'  Image {i} (digit {test_labels[i]}): '
              f'model={actual_bits} bits ({actual_bits/784:.2f} bpp), '
              f'theory={theory_bits:.0f}, '
              f'PNG={img_png}, gzip={img_gzip}, '
              f'raw={784*8}')

    # ============================================================
    # Summary
    # ============================================================
    print('\n' + '=' * 60)
    print('  COMPRESSION RESULTS')
    print('=' * 60)

    avg_model_bpp = total_model_bits / (n_images * 784)
    avg_gzip_bpp = total_gzip_bits / (n_images * 784)
    avg_raw_bpp = 8.0

    print(f'\n  {"Method":<30s} {"Avg bits/pixel":>14s} {"Avg bits/image":>14s} {"Ratio":>8s}')
    print(f'  {"-"*30} {"-"*14} {"-"*14} {"-"*8}')
    print(f'  {"Raw (uncompressed)":<30s} {avg_raw_bpp:>14.3f} {avg_raw_bpp*784:>14.0f} {"1.00x":>8s}')
    if has_pil:
        avg_png_bpp = total_png_bits / (n_images * 784)
        print(f'  {"PNG":<30s} {avg_png_bpp:>14.3f} {avg_png_bpp*784:>14.0f} {avg_raw_bpp/avg_png_bpp:>7.1f}x')
    print(f'  {"gzip":<30s} {avg_gzip_bpp:>14.3f} {avg_gzip_bpp*784:>14.0f} {avg_raw_bpp/avg_gzip_bpp:>7.1f}x')
    print(f'  {"Model (arithmetic coding)":<30s} {avg_model_bpp:>14.3f} {avg_model_bpp*784:>14.0f} {avg_raw_bpp/avg_model_bpp:>7.1f}x')

    print(f'\n  Model compression time: {model_time:.1f}s for {n_images} images '
          f'({n_images/model_time:.0f} images/s)')

    # Per-digit breakdown (vectorized)
    print(f'\n  Per-digit compression (model, bits/pixel):')
    all_image_bits = []
    all_image_labels = []
    for start in range(0, n_images, batch_size):
        end = min(start + batch_size, n_images)
        batch_pixels = []
        batch_labels = []
        for i in range(start, end):
            pixels = val_data[i * 784:(i + 1) * 784].astype(np.int64)
            batch_pixels.append(pixels)
            batch_labels.append(test_labels[i])
        batch_tensor = torch.tensor(np.stack(batch_pixels), dtype=torch.long).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            log_probs = F.log_softmax(logits.float(), dim=-1)  # (B, 784, 256)
            targets = batch_tensor[:, 1:]  # (B, 783)
            lp_selected = log_probs[:, :-1, :].gather(2, targets.unsqueeze(-1)).squeeze(-1)  # (B, 783)
            bits_per_image = (8.0 + (-lp_selected / math.log(2)).sum(dim=1)).cpu().numpy()  # (B,)
        all_image_bits.extend(bits_per_image.tolist())
        all_image_labels.extend(batch_labels)

    all_image_bits = np.array(all_image_bits)
    all_image_labels = np.array(all_image_labels)
    for d in range(10):
        mask = all_image_labels == d
        if mask.any():
            vals = all_image_bits[mask] / 784
            print(f'    Digit {d}: {np.mean(vals):.3f} bpp '
                  f'(min={np.min(vals):.3f}, max={np.max(vals):.3f}, n={mask.sum()})')

    # Save results
    results = {
        'n_images': n_images,
        'avg_model_bpp': avg_model_bpp,
        'avg_gzip_bpp': avg_gzip_bpp,
        'avg_raw_bpp': avg_raw_bpp,
        'model_compression_ratio': avg_raw_bpp / avg_model_bpp,
        'gzip_compression_ratio': avg_raw_bpp / avg_gzip_bpp,
        'verify': verify_results,
    }
    if has_pil:
        results['avg_png_bpp'] = avg_png_bpp
        results['png_compression_ratio'] = avg_raw_bpp / avg_png_bpp

    out_path = Path('pile_runs/rope_byte/compression_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to {out_path}')


if __name__ == '__main__':
    main()
