"""Pre-cut training windows (local CPU job — never on a rented GPU).

Train: N random 29x29x3 windows from a directory of images. Row 0 / col 0 are the pixel row above
and column left of the 28x28 patch (pad 128 at image borders). Saved as uint8 npy (N, 29, 29, 3).
Eval : Kodak-24 tiled fully at stride 28 over the 756x504 crop -> (24, 486, 29, 29, 3), plus PNG bpp per image.

Usage:
  python3 prep_shared_prior_data.py train <image_dir> <N> <out.npy>
  python3 prep_shared_prior_data.py kodak data/kodak <out.npz>
"""

import io
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

P, PAD = 28, 128
HALO = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 1  # trailing integer arg = halo width in px
EXTS = {".png", ".jpg", ".jpeg"}


def padded(arr):
    """Add a HALO-pixel PAD border on top and left."""
    H, W, C = arr.shape
    out = np.full((H + HALO, W + HALO, C), PAD, dtype=np.uint8)
    out[HALO:, HALO:] = arr
    return out


def cut_train(image_dir, n, out):
    files = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in EXTS)
    per = -(-n // len(files))
    rng = np.random.default_rng(0)
    windows = np.empty((per * len(files), P + HALO, P + HALO, 3), dtype=np.uint8)
    k, t0 = 0, time.time()
    for i, f in enumerate(files):
        arr = padded(np.array(Image.open(f).convert("RGB")))
        H, W, _ = arr.shape
        for _ in range(per):
            r = rng.integers(0, H - P - HALO + 1)
            c = rng.integers(0, W - P - HALO + 1)
            windows[k] = arr[r:r + P + HALO, c:c + P + HALO]
            k += 1
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(files)} images, {k:,} windows ({time.time() - t0:.0f}s)", flush=True)
    windows = windows[:k]
    rng.shuffle(windows)
    np.save(out, windows)
    print(f"Saved {out}: {windows.shape} uint8 = {windows.nbytes / 1048576:.0f} MB from {len(files)} images", flush=True)


def cut_kodak(image_dir, out):
    files = sorted(Path(image_dir).glob("kodim*.png"))
    tiles, png_bpp, names = [], [], []
    for f in files:
        arr = np.array(Image.open(f).convert("RGB"))
        H, W = (arr.shape[0] // P) * P, (arr.shape[1] // P) * P
        arr = arr[:H, :W]
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG", optimize=True)
        png_bpp.append(len(buf.getvalue()) * 8 / (H * W))
        pa = padded(arr)
        t = [pa[r:r + P + HALO, c:c + P + HALO] for r in range(0, H, P) for c in range(0, W, P)]
        tiles.append(np.stack(t))
        names.append(f.name)
    tiles = np.stack(tiles)  # (24, 486, 29, 29, 3) — all Kodak images are 768x512 or 512x768 -> same tile count
    np.savez(out, tiles=tiles, png_bpp=np.array(png_bpp), names=np.array(names), hw=np.array([H, W]))
    print(f"Saved {out}: tiles {tiles.shape}, mean PNG bpp {np.mean(png_bpp):.3f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1] == "train":
        cut_train(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        cut_kodak(sys.argv[2], sys.argv[3])
