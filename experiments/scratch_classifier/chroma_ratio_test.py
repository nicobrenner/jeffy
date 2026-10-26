"""Nico's "fractional RGB" idea, tested as a colour representation for coding (2026-10-25): intensity + ratio chromaticity
  I = (R+G+B)/3 at full resolution,  r = R/(R+G+B), g = G/(R+G+B) (b implied) at HALF resolution, ratios quantised to 8 bits,
against the standard JPEG-style YCbCr 4:2:0 (Y full, Cb/Cr half) and against plain RGB lossless. Each plane is coded losslessly
with JPEG XL (grayscale), so the only loss in the two subsampled pipelines is the chroma subsampling + ratio quantisation; we
report bytes and PSNR of the reconstruction vs the original RGB. Kodak 512² crops.

  python3 -u chroma_ratio_test.py --images 8
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from blursort_progressive import psnr


def jxl_gray(a, tmp):
    Image.fromarray(a.astype(np.uint8), "L").save(f"{tmp}/g.png")
    subprocess.run(["cjxl", f"{tmp}/g.png", f"{tmp}/g.jxl", "-d", "0", "-e", "7", "--quiet"], check=True, capture_output=True)
    return os.path.getsize(f"{tmp}/g.jxl")


def jxl_rgb(a, tmp):
    Image.fromarray(a).save(f"{tmp}/c.png")
    subprocess.run(["cjxl", f"{tmp}/c.png", f"{tmp}/c.jxl", "-d", "0", "-e", "7", "--quiet"], check=True, capture_output=True)
    return os.path.getsize(f"{tmp}/c.jxl")


def down2(p):  # box filter
    return ((p[0::2, 0::2].astype(np.float64) + p[1::2, 0::2] + p[0::2, 1::2] + p[1::2, 1::2]) / 4)


def up2(p, H, W):
    return np.array(Image.fromarray(p.astype(np.float32), "F").resize((W, H), Image.BILINEAR))


def ycbcr_420(arr, tmp):
    x = arr.astype(np.float64); R, G, B = x[..., 0], x[..., 1], x[..., 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B; Cb = 128 - 0.168736 * R - 0.331264 * G + 0.5 * B; Cr = 128 + 0.5 * R - 0.418688 * G - 0.081312 * B
    Yq = np.round(Y).clip(0, 255); Cbq = np.round(down2(Cb)).clip(0, 255); Crq = np.round(down2(Cr)).clip(0, 255)
    b = jxl_gray(Yq, tmp) + jxl_gray(Cbq, tmp) + jxl_gray(Crq, tmp)
    H, W = Y.shape; cb = up2(Cbq, H, W) - 128; cr = up2(Crq, H, W) - 128
    rec = np.stack([Yq + 1.402 * cr, Yq - 0.344136 * cb - 0.714136 * cr, Yq + 1.772 * cb], -1)
    return b, np.clip(np.round(rec), 0, 255).astype(np.uint8)


def ratio_420(arr, tmp):
    x = arr.astype(np.float64); S = x.sum(-1); I = S / 3
    r = np.where(S > 0, x[..., 0] / np.maximum(S, 1e-9), 1 / 3); g = np.where(S > 0, x[..., 1] / np.maximum(S, 1e-9), 1 / 3)
    Iq = np.round(I).clip(0, 255); rq = np.round(down2(r) * 255).clip(0, 255); gq = np.round(down2(g) * 255).clip(0, 255)
    b = jxl_gray(Iq, tmp) + jxl_gray(rq, tmp) + jxl_gray(gq, tmp)
    H, W = I.shape; ru = up2(rq, H, W) / 255; gu = up2(gq, H, W) / 255; bu = np.clip(1 - ru - gu, 0, 1)
    S3 = Iq * 3; rec = np.stack([S3 * ru, S3 * gu, S3 * bu], -1)
    return b, np.clip(np.round(rec), 0, 255).astype(np.uint8)


def ratio_full(arr, tmp):
    """ratios at FULL resolution (no subsampling): isolates the quantisation cost of the representation itself."""
    x = arr.astype(np.float64); S = x.sum(-1); I = S / 3
    r = np.where(S > 0, x[..., 0] / np.maximum(S, 1e-9), 1 / 3); g = np.where(S > 0, x[..., 1] / np.maximum(S, 1e-9), 1 / 3)
    Iq = np.round(I).clip(0, 255); rq = np.round(r * 255).clip(0, 255); gq = np.round(g * 255).clip(0, 255)
    b = jxl_gray(Iq, tmp) + jxl_gray(rq, tmp) + jxl_gray(gq, tmp)
    ru, gu = rq / 255, gq / 255; bu = np.clip(1 - ru - gu, 0, 1); S3 = Iq * 3
    return b, np.clip(np.round(np.stack([S3 * ru, S3 * gu, S3 * bu], -1)), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--images", type=int, default=8); ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args(); tmp = tempfile.mkdtemp(); allres = []
    for f in sorted(Path("data/kodak").glob("kodim*.png"))[:a.images]:
        im = np.array(Image.open(f).convert("RGB")); H, W, _ = im.shape
        arr = im[(H - 512) // 2:(H - 512) // 2 + 512, (W - 512) // 2:(W - 512) // 2 + 512].copy(); npx = 512 * 512
        r = {"image": f.name, "rgb_lossless_bpp": 8 * jxl_rgb(arr, tmp) / npx}
        for name, fn in (("ycbcr_420", ycbcr_420), ("ratio_420", ratio_420), ("ratio_full", ratio_full)):
            b, rec = fn(arr, tmp); r[name] = {"bpp": 8 * b / npx, "psnr": psnr(arr, rec)}
        allres.append(r)
        print(f"{f.name}: RGB lossless {r['rgb_lossless_bpp']:.2f} bpp | YCbCr 4:2:0 {r['ycbcr_420']['bpp']:.2f} bpp @ {r['ycbcr_420']['psnr']:.1f} dB | "
              f"intensity+ratios 4:2:0 {r['ratio_420']['bpp']:.2f} @ {r['ratio_420']['psnr']:.1f} | ratios full-res {r['ratio_full']['bpp']:.2f} @ {r['ratio_full']['psnr']:.1f}")
    print(f"\nMEANS over {len(allres)}: RGB lossless {np.mean([r['rgb_lossless_bpp'] for r in allres]):.2f} bpp | " + " | ".join(
        f"{n} {np.mean([r[n]['bpp'] for r in allres]):.2f} bpp @ {np.mean([r[n]['psnr'] for r in allres]):.1f} dB" for n in ("ycbcr_420", "ratio_420", "ratio_full")))
    Path(a.out).mkdir(parents=True, exist_ok=True); json.dump(allres, open(Path(a.out) / f"chroma_ratio_kodak_{a.images}.json", "w"), indent=1)
    print("saved", Path(a.out) / f"chroma_ratio_kodak_{a.images}.json")


if __name__ == "__main__":
    main()
