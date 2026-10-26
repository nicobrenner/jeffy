"""BlurSort as a progressive format, measured (Nico, 2026-10-25: "fractality ... progressive image loading").

The dyadic polyphase order is a progressive transmission order: phase (0,0) is a 1/s² image, each further phase refines it.
This script measures, on Kodak crops, with NO new training:
  1. scale self-similarity: bpp of the frozen 215K byte-LM prior on the full image vs on each polyphase phase coded as an
     independent image (s = 2: 4 phases, s = 4: 16 phases). If natural images were scale-invariant and aliasing free, the
     phases would cost the same bpp as the full image; the gap is the price of independent (unconditioned) phase coding.
  2. progressive curve: bytes received (phases coded independently with the prior, dyadic order) vs PSNR of the preview
     reconstructed from the phases received so far (nearest-decoded-pixel fill), against JPEG XL's own progressive lossless
     mode (cjxl -p, file truncated at fractions, djxl --allow_partial_files) on the same crop, and PNG as the
     all-or-nothing reference.

  python3 -u blursort_progressive.py --images 8 --device cpu
"""
import argparse
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

import compress_with_prior as C

P = 28


def dyadic_order(s):
    """bit-reversed (coarse-to-fine) order over the s x s phase grid, e.g. s=2: (0,0),(1,1),(0,1),(1,0)."""
    if s == 2:
        return [(0, 0), (1, 1), (0, 1), (1, 0)]
    coarse = [(2 * dx, 2 * dy) for dx, dy in dyadic_order(s // 2)]
    out = list(coarse)
    for dx, dy in coarse:
        out += [(dx + 1, dy + 1), (dx, dy + 1), (dx + 1, dy)]
    return out


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def prior_bits(m, arr, bs=32):
    _, tiles = C.tile_image(arr)
    ce, *_ = C.model_code(m, tiles, bs, False)
    return ce


def preview(arr, received, s):
    """fill pixels not yet received with the nearest received pixel (phase grid)."""
    H, W, _ = arr.shape
    mask = np.zeros((H, W), bool)
    for dx, dy in received:
        mask[dy::s, dx::s] = True
    idx = ndimage.distance_transform_edt(~mask, return_distances=False, return_indices=True)
    return arr[idx[0], idx[1]]


def jxl_progressive(arr, tmp, fracs):
    src, dst, part, dec = [os.path.join(tmp, n) for n in ("p.png", "p.jxl", "part.jxl", "dec.png")]
    Image.fromarray(arr).save(src)
    subprocess.run(["cjxl", src, dst, "-d", "0", "-e", "7", "-p", "--quiet"], check=True, capture_output=True)
    b = open(dst, "rb").read()
    curve = []
    for f in fracs:
        open(part, "wb").write(b[:int(len(b) * f)])
        if os.path.exists(dec):
            os.remove(dec)
        r = subprocess.run(["djxl", part, dec, "--allow_partial_files", "--quiet"], capture_output=True)
        if r.returncode or not os.path.exists(dec):
            continue
        curve.append((int(len(b) * f), psnr(arr, np.array(Image.open(dec).convert("RGB")))))
    return len(b), curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=int, default=8); ap.add_argument("--device", default="cpu")
    ap.add_argument("--ckpt", default="checkpoints/prior_full_d64_20ep_best.pt"); ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args()
    C.DEVICE = a.device
    m, nbytes, _ = C.load_prior(a.ckpt, int8=True)
    tmp = tempfile.mkdtemp()
    files = sorted(Path("data/kodak").glob("kodim*.png"))[:a.images]
    Path(a.out).mkdir(parents=True, exist_ok=True)
    allres = []
    for f in files:
        im = np.array(Image.open(f).convert("RGB"))
        if im.shape[0] > im.shape[1]:
            im = np.rot90(im).copy()
        H, W = 448, 672  # 16 x 24 tiles; s=2 phases 224x336 (8x12 tiles); s=4 phases 112x168 (4x6 tiles)
        t, l = (im.shape[0] - H) // 2, (im.shape[1] - W) // 2
        arr = im[t:t + H, l:l + W].copy()
        npx = H * W
        res = {"image": f.name, "crop": [H, W]}
        full_bits = prior_bits(m, arr)
        res["prior_full_bpp"] = full_bits / npx
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, "PNG", optimize=True); res["png_bpp"] = 8 * buf.tell() / npx
        jxl_bytes, jcurve = jxl_progressive(arr, tmp, [0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        res["jxl_bpp"] = 8 * jxl_bytes / npx
        res["jxl_curve"] = [(8 * b / npx, p) for b, p in jcurve]
        print(f"\n{f.name} {W}x{H}: prior full {res['prior_full_bpp']:.3f} bpp | PNG {res['png_bpp']:.3f} | JXL {res['jxl_bpp']:.3f}; JXL progressive: " +
              ", ".join(f"{bpp:.2f}bpp→{p:.1f}dB" for bpp, p in res["jxl_curve"]))
        for s in (2, 4):
            order = dyadic_order(s)
            phases = {(dx, dy): arr[dy::s, dx::s].copy() for dx, dy in order}
            bits = {ph: prior_bits(m, phases[ph]) for ph in order}
            phase_bpp = {f"{dx},{dy}": bits[(dx, dy)] / phases[(dx, dy)].size * 3 for dx, dy in order}
            total = sum(bits.values())
            curve, got, cum = [], [], 0
            for ph in order:
                got.append(ph); cum += bits[ph]
                pv = preview(arr, got, s)
                curve.append((cum / npx, psnr(arr, pv)))
            res[f"s{s}"] = {"phase_bpp": phase_bpp, "mean_phase_bpp": float(np.mean(list(phase_bpp.values()))),
                            "total_bpp": total / npx, "curve": curve}
            jb = np.array([c[0] for c in res["jxl_curve"]]); jp = np.array([c[1] for c in res["jxl_curve"]])
            cmp = []
            for bpp, p in curve[:-1]:
                jx = float(np.interp(bpp, jb, jp)) if jb.min() <= bpp <= jb.max() else float("nan")
                cmp.append((bpp, p, jx))
            print(f"  s={s}: phases coded independently total {total/npx:.3f} bpp (= {total/full_bits:.3f}x the full-image prior), mean phase bpp {res[f's{s}']['mean_phase_bpp']:.3f}")
            print("        progressive (bpp received → preview PSNR | JXL progressive at same bpp): " +
                  ", ".join(f"{b:.2f}→{p:.1f}dB|{j:.1f}" for b, p, j in cmp[:6]))
        allres.append(res)
        json.dump(allres, open(Path(a.out) / f"progressive_kodak_{a.images}.json", "w"), indent=1)
    # summary
    print("\nSUMMARY over", len(allres), "images (means):")
    print(f"  prior full {np.mean([r['prior_full_bpp'] for r in allres]):.3f} bpp | PNG {np.mean([r['png_bpp'] for r in allres]):.3f} | JXL {np.mean([r['jxl_bpp'] for r in allres]):.3f}")
    for s in (2, 4):
        print(f"  s={s}: independent phases {np.mean([r[f's{s}']['total_bpp'] for r in allres]):.3f} bpp ({np.mean([r[f's{s}']['total_bpp']/r['prior_full_bpp'] for r in allres]):.3f}x full); "
              f"first phase alone: {np.mean([r[f's{s}']['curve'][0][0] for r in allres]):.3f} bpp → {np.mean([r[f's{s}']['curve'][0][1] for r in allres]):.1f} dB preview")
    print("saved", Path(a.out) / f"progressive_kodak_{a.images}.json")


if __name__ == "__main__":
    main()
