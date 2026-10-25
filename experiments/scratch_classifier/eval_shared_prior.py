"""Does a tiny compression prior transfer across images?

Loads the frozen shared model D (d32, 2L, trained on one 700x700 photo) and:
  1. Arithmetic-codes the training photo for real (not the cross-entropy estimate),
     decode-verifying a sample of patches to confirm losslessness.
  2. Codes unseen images with the same frozen weights (no retraining).

Reports, per image: PNG bytes, CE-estimated bytes, actual coded bytes,
total if the model must be shipped (+model bytes), total if the prior is shared.
"""

import io
import math
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from compression_demo import ArithmeticDecoder, ArithmeticEncoder
from train_rope_byte_2d import RoPE2DByteLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
PHOTO_DIR = "pile_runs/rope_byte/photo_compressor"
CKPT = f"{PHOTO_DIR}/color_700_shared_D.pt"
P = 28
VERIFY_PATCHES = 5

IMAGES = {
    "train_photo_700": f"{PHOTO_DIR}/color_700_original.png",
    "pluto_nasa_700": "/Users/nico/Documents/Projects/Blursort/pluto_natural_color_20150714_detail.jpg",
    "sample_rgb_448x364": "/Users/nico/Documents/Projects/ImageDepth/assets/sample-rgb.jpg",
}


def center_crop_to_grid(img):
    w, h = img.size
    gw, gh = (w // P) * P, (h // P) * P
    gw, gh = min(gw, 700), min(gh, 700)
    left, top = (w - gw) // 2, (h - gh) // 2
    return img.crop((left, top, left + gw, top + gh))


def png_bytes(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return len(buf.getvalue())


def probs_to_cdf(probs, scale=1 << 16):
    counts = np.maximum((probs * scale).astype(np.int64), 1)
    counts[counts.argmax()] += scale - counts.sum()
    cdf = np.zeros(257, dtype=np.int64)
    cdf[1:] = np.cumsum(counts)
    return cdf


UNIFORM_CDF = np.arange(257, dtype=np.int64) * 256


@torch.no_grad()
def code_channel(model, chan, verify_idx):
    """Return (ce_bits, actual_bits, n_verified_ok) for one HxW uint8 channel."""
    H, W = chan.shape
    ce_bits = 0.0
    actual_bits = 0
    ok = 0
    k = 0
    for r in range(H // P):
        for c in range(W // P):
            px = chan[r * P:(r + 1) * P, c * P:(c + 1) * P].flatten()
            ids = torch.tensor(px, dtype=torch.long).unsqueeze(0).to(DEVICE)
            logits = model(ids)[0].float()
            logp = F.log_softmax(logits, dim=-1).cpu()
            tgt = torch.tensor(px[1:], dtype=torch.long)
            ce_bits += 8.0 + (-logp[:-1][torch.arange(P * P - 1), tgt] / math.log(2)).sum().item()

            probs = logp.exp().numpy()
            enc = ArithmeticEncoder()
            cdfs = [UNIFORM_CDF] + [probs_to_cdf(probs[i - 1]) for i in range(1, P * P)]
            for i in range(P * P):
                enc.encode_symbol(cdfs[i], int(px[i]))
            bits = enc.finish()
            actual_bits += len(bits)

            if k in verify_idx:
                dec = ArithmeticDecoder(bits)
                decoded = [dec.decode_symbol(cdfs[i]) for i in range(P * P)]
                ok += int(np.array_equal(np.array(decoded, dtype=np.uint8), px))
            k += 1
    return ce_bits, actual_bits, ok


def main():
    t0 = time.time()
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = RoPE2DByteLM(vocab_size=256, d_model=32, n_heads=2, n_layers=2,
                         height=P, width=P, bidirectional=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    model_bytes = n_params * 2
    print(f"Model D: {n_params:,} params = {model_bytes / 1024:.1f} KB f16, trained {ckpt['epoch']} epochs on train_photo_700", flush=True)
    print(f"Device: {DEVICE}\n", flush=True)

    rows = []
    for name, path in IMAGES.items():
        img = center_crop_to_grid(Image.open(path).convert("RGB"))
        arr = np.array(img)
        H, W, _ = arr.shape
        if name != "train_photo_700":
            img.save(f"{PHOTO_DIR}/shared_prior_{name}.png")
        n_patches = (H // P) * (W // P)
        verify_idx = set(np.linspace(0, n_patches - 1, VERIFY_PATCHES, dtype=int).tolist())

        ce_total = 0.0
        ac_total = 0
        ok_total = 0
        for ch in range(3):
            ce, ac, ok = code_channel(model, arr[:, :, ch], verify_idx)
            ce_total += ce
            ac_total += ac
            ok_total += ok
        png = png_bytes(arr)
        ce_b, ac_b = ce_total / 8, ac_total / 8
        rows.append((name, H, W, png, ce_b, ac_b, ok_total, 3 * VERIFY_PATCHES))
        print(f"{name}: {H}x{W}  PNG={png / 1024:.1f}KB  CE-est={ce_b / 1024:.1f}KB  "
              f"AC-actual={ac_b / 1024:.1f}KB  decode-verified {ok_total}/{3 * VERIFY_PATCHES} patches  "
              f"({time.time() - t0:.0f}s)", flush=True)

    print(f"\n{'image':<20}{'PNG':>9}{'AC actual':>11}{'+model':>9}{'vs PNG (shipped)':>18}{'vs PNG (shared)':>17}")
    for name, H, W, png, ce_b, ac_b, ok, n in rows:
        shipped = ac_b + model_bytes
        print(f"{name:<20}{png / 1024:>8.0f}K{ac_b / 1024:>10.0f}K{shipped / 1024:>8.0f}K"
              f"{shipped / png:>17.2f}x{ac_b / png:>16.2f}x")
    print("\n'shipped' = model weights sent with every image (COIN-style). 'shared' = receiver already has the prior.")
    print(f"AC overhead vs CE estimate on train photo: {(rows[0][5] - rows[0][4]) / 1024:+.2f} KB")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
