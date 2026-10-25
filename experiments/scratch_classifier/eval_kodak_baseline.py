"""C0: Kodak-24 lossless benchmark for a frozen shared prior (no training).

Images are cropped to multiples of 28 (768x512 -> 756x504, 486 patches/channel).
Reports per-image and mean bits-per-pixel (bpp = bits / (H*W), counting all 3 channels)
for: model cross-entropy (AC-verified elsewhere to within 0.15%), optimized PNG, gzip.
Model bytes are reported separately (shared prior: sent once, not per image).

Usage: python3 eval_kodak_baseline.py [ckpt] [d_model] [n_layers] [n_heads]
"""

import gzip
import io
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_rope_byte_2d import RoPE2DByteLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
P = 28
KODAK = sorted(Path("data/kodak").glob("kodim*.png"))
CKPT = sys.argv[1] if len(sys.argv) > 1 else "pile_runs/rope_byte/photo_compressor/color_700_shared_D.pt"
D_MODEL, N_LAYERS, N_HEADS = (int(x) for x in (sys.argv[2:5] or (32, 2, 2)))
OUT = Path("pile_runs/rope_byte/kodak_results.json")


def crop(arr):
    H, W = arr.shape[:2]
    return arr[: (H // P) * P, : (W // P) * P]


@torch.no_grad()
def model_bits(model, arr):
    H, W, _ = arr.shape
    patches = []
    for ch in range(3):
        for r in range(H // P):
            for c in range(W // P):
                patches.append(arr[r * P:(r + 1) * P, c * P:(c + 1) * P, ch].flatten())
    x = torch.tensor(np.stack(patches), dtype=torch.long, device=DEVICE)
    total = 0.0
    for i in range(0, len(x), 64):
        b = x[i:i + 64]
        logp = F.log_softmax(model(b).float(), dim=-1)
        tgt = b[:, 1:]
        total += 8.0 * len(b) + (-logp[:, :-1].gather(-1, tgt.unsqueeze(-1)).squeeze(-1) / math.log(2)).sum().item()
    return total


def png_bits(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", optimize=True)
    return len(buf.getvalue()) * 8


def main():
    t0 = time.time()
    ckpt = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model = RoPE2DByteLM(vocab_size=256, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
                         height=P, width=P, bidirectional=False).to(DEVICE)
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Prior: {CKPT} ({n_params:,} params = {n_params * 2 / 1024:.1f} KB f16) on {len(KODAK)} Kodak images, device {DEVICE}\n", flush=True)

    rows = []
    for p in KODAK:
        arr = crop(np.array(Image.open(p).convert("RGB")))
        H, W, _ = arr.shape
        npx = H * W
        mb, pb, gb = model_bits(model, arr), png_bits(arr), len(gzip.compress(arr.tobytes(), 9)) * 8
        rows.append(dict(image=p.name, h=H, w=W, model_bpp=mb / npx, png_bpp=pb / npx, gzip_bpp=gb / npx))
        print(f"{p.name}: model {mb / npx:5.2f} bpp | PNG {pb / npx:5.2f} | gzip {gb / npx:5.2f} | model/PNG {mb / pb:.3f}", flush=True)

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("model_bpp", "png_bpp", "gzip_bpp")}
    wins = sum(r["model_bpp"] < r["png_bpp"] for r in rows)
    print(f"\nMEAN  model {mean['model_bpp']:.3f} bpp | PNG {mean['png_bpp']:.3f} | gzip {mean['gzip_bpp']:.3f} | "
          f"model/PNG {mean['model_bpp'] / mean['png_bpp']:.3f} | beats PNG on {wins}/{len(rows)} images | raw 24 bpp")
    print(f"Shared prior cost (sent once): {n_params * 2 / 1024:.1f} KB f16. Time {time.time() - t0:.0f}s")

    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev[Path(CKPT).stem] = dict(ckpt=CKPT, params=n_params, mean=mean, wins=wins, per_image=rows)
    OUT.write_text(json.dumps(prev, indent=1))
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
