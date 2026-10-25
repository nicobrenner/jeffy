"""Compress a real photo with the MNIST byte model to show domain specificity.

The model was trained on 28x28 grayscale MNIST digits. When applied to natural
photos, it performs WORSE than PNG/gzip because it doesn't know natural image
statistics. This is the flip side of compression = prediction: a model that
doesn't understand the data can't compress it.
"""

import argparse
import gzip
import io
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_rope_byte_2d import RoPE2DByteLM


@torch.no_grad()
def compress_patches(model, patches, device, batch_size=64):
    """Compute theoretical compression (cross-entropy) for 28x28 patches."""
    total_bits = 0.0
    n_patches = len(patches)

    for start in range(0, n_patches, batch_size):
        end = min(start + batch_size, n_patches)
        batch = torch.tensor(
            np.stack(patches[start:end]).astype(np.int64), dtype=torch.long
        ).to(device)
        logits = model(batch)
        log_probs = F.log_softmax(logits.float(), dim=-1)

        total_bits += batch.shape[0] * 8.0

        targets = batch[:, 1:]
        lp = log_probs[:, :-1, :]
        bits = -lp.gather(2, targets.unsqueeze(-1)).squeeze(-1) / math.log(2)
        total_bits += bits.sum().item()

    return total_bits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('image', help='Path to image file')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--patch-size', type=int, default=28)
    parser.add_argument('--max-patches', type=int, default=500,
                        help='Max patches to process (for speed)')
    args = parser.parse_args()

    img = Image.open(args.image)
    print(f"Image: {args.image}")
    print(f"  Original: {img.size[0]}x{img.size[1]}, {img.mode}")

    gray = img.convert('L')
    w, h = gray.size
    ps = args.patch_size

    n_cols = w // ps
    n_rows = h // ps
    total_patches = n_cols * n_rows
    print(f"  Grayscale: {w}x{h}")
    print(f"  Patches ({ps}x{ps}): {n_cols} x {n_rows} = {total_patches}")

    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            patch = np.array(gray.crop((c*ps, r*ps, (c+1)*ps, (r+1)*ps)))
            patches.append(patch.flatten())
    patches = np.array(patches)

    if len(patches) > args.max_patches:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(patches), args.max_patches, replace=False)
        idx.sort()
        patches = patches[idx]
        print(f"  Subsampled to {len(patches)} patches")

    n = len(patches)
    pixels_per_patch = ps * ps
    total_pixels = n * pixels_per_patch

    # Load model
    ckpt_path = 'pile_runs/rope_byte/mnist_2d_d64_causal_tp1800/best_model.pt'
    print(f"\nLoading model from {ckpt_path}...")
    model = RoPE2DByteLM(
        vocab_size=256, d_model=64, n_layers=4, n_heads=4,
        height=28, width=28, bidirectional=False
    ).to(args.device)
    ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()

    # Model compression
    print("Computing model compression...")
    t0 = time.time()
    model_bits = compress_patches(model, patches, args.device)
    model_time = time.time() - t0
    model_bpp = model_bits / total_pixels

    # Baselines
    print("Computing baselines...")
    raw_bits = total_pixels * 8
    raw_bpp = 8.0

    png_bits = 0
    gz_bits = 0
    for p in patches:
        patch_img = Image.fromarray(p.reshape(ps, ps).astype(np.uint8), mode='L')
        buf = io.BytesIO()
        patch_img.save(buf, format='PNG', optimize=True)
        png_bits += len(buf.getvalue()) * 8
        gz_bits += len(gzip.compress(p.tobytes(), compresslevel=9)) * 8

    png_bpp = png_bits / total_pixels
    gz_bpp = gz_bits / total_pixels

    # Full-image baselines
    print("Computing full-image baselines...")
    buf_full_png = io.BytesIO()
    gray.save(buf_full_png, format='PNG', optimize=True)
    full_png_bits = len(buf_full_png.getvalue()) * 8
    full_png_bpp = full_png_bits / (w * h)

    full_gz_bits = len(gzip.compress(np.array(gray).tobytes(), compresslevel=9)) * 8
    full_gz_bpp = full_gz_bits / (w * h)

    # JPEG (original file)
    with open(args.image, 'rb') as f:
        jpeg_bits = len(f.read()) * 8
    jpeg_bpp = jpeg_bits / (img.size[0] * img.size[1] * (3 if img.mode == 'RGB' else 1))

    print(f"\n{'='*60}")
    print(f"  COMPRESSION RESULTS")
    print(f"{'='*60}")
    print(f"\n  Patch-level ({n} patches of {ps}x{ps}):")
    print(f"  {'Method':<25s} {'bpp':>8s} {'Ratio':>8s}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    print(f"  {'Raw':<25s} {raw_bpp:>8.3f} {'1.0x':>8s}")
    print(f"  {'PNG (per patch)':<25s} {png_bpp:>8.3f} {raw_bpp/png_bpp:>7.1f}x")
    print(f"  {'gzip (per patch)':<25s} {gz_bpp:>8.3f} {raw_bpp/gz_bpp:>7.1f}x")
    print(f"  {'MNIST model':<25s} {model_bpp:>8.3f} {raw_bpp/model_bpp:>7.1f}x")

    print(f"\n  Full-image (grayscale {w}x{h}):")
    print(f"  {'PNG':<25s} {full_png_bpp:>8.3f} {8.0/full_png_bpp:>7.1f}x")
    print(f"  {'gzip':<25s} {full_gz_bpp:>8.3f} {8.0/full_gz_bpp:>7.1f}x")
    print(f"  {'Original JPEG (color)':<25s} {jpeg_bpp:>8.3f}")

    print(f"\n  Key insight: the MNIST model uses {model_bpp:.1f} bpp on natural photos")
    print(f"  vs {0.92:.2f} bpp on MNIST digits. It's {model_bpp/0.92:.1f}x worse on")
    print(f"  out-of-distribution data because it can't predict natural textures.")
    print(f"  This IS the OOD detection signal — high bits = unfamiliar data.")

    print(f"\n  Model time: {model_time:.1f}s ({n/model_time:.0f} patches/s)")


if __name__ == '__main__':
    main()
