"""Nico's observation (2026-10-25): a BlurSorted image is much harder to compress than the original. Two questions:
 (1) how much harder, lossless and lossy (JPEG at matched quality / matched bytes)?  -> "anti-compression"
 (2) is there a flip side: because every block of the BlurSorted canvas is a *phase* (a sparse full-field sample of the
     image), a localised loss on the canvas becomes a scattered, interpolable loss on the image. Test: erase a block /
     a set of 8x8 blocks, un-BlurSort, inpaint by nearest known pixel, compare PSNR with the same erasure on the original.
BlurSort for a 2^k square = bit-reversal permutation of row and column indices (blursort([0..7]) = [0,4,2,6,1,5,3,7]).

  python3 -u blursort_anticompression.py --images 8
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


def bitrev_perm(n):
    k = n.bit_length() - 1; assert 1 << k == n
    return np.array([int(f"{i:0{k}b}"[::-1], 2) for i in range(n)])


def blursort2d(img):
    p = bitrev_perm(img.shape[0]); q = bitrev_perm(img.shape[1])
    return img[p][:, q]


def unblursort2d(canvas):
    p = bitrev_perm(canvas.shape[0]); q = bitrev_perm(canvas.shape[1])
    inv_p = np.argsort(p); inv_q = np.argsort(q)
    return canvas[inv_p][:, inv_q]


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def jpeg(arr, q):
    b = io.BytesIO(); Image.fromarray(arr).save(b, "JPEG", quality=q, subsampling=0); size = b.tell(); b.seek(0)
    return size, np.array(Image.open(b).convert("RGB"))


def lossless_sizes(arr, tmp):
    out = {}
    b = io.BytesIO(); Image.fromarray(arr).save(b, "PNG", optimize=True); out["png"] = b.tell()
    b = io.BytesIO(); Image.fromarray(arr).save(b, "WEBP", lossless=True, quality=100, method=6); out["webp"] = b.tell()
    Image.fromarray(arr).save(f"{tmp}/a.png"); subprocess.run(["cjxl", f"{tmp}/a.png", f"{tmp}/a.jxl", "-d", "0", "-e", "7", "--quiet"], check=True, capture_output=True)
    out["jxl"] = os.path.getsize(f"{tmp}/a.jxl")
    return out


def inpaint_nearest(arr, mask):
    idx = ndimage.distance_transform_edt(mask, return_distances=False, return_indices=True)
    return arr[idx[0], idx[1]]


def erase_tests(arr, rng):
    H, W, _ = arr.shape
    res = {}
    masks = {"block_64": np.zeros((H, W), bool), "blocks_8x8_10pct": np.zeros((H, W), bool), "rows_32": np.zeros((H, W), bool)}
    masks["block_64"][H // 2 - 32:H // 2 + 32, W // 2 - 32:W // 2 + 32] = True
    for r in range(0, H, 8):
        for c in range(0, W, 8):
            if rng.random() < 0.10:
                masks["blocks_8x8_10pct"][r:r + 8, c:c + 8] = True
    masks["rows_32"][H // 2 - 16:H // 2 + 16, :] = True
    for name, m in masks.items():
        # erase on the original, inpaint
        rec_o = inpaint_nearest(arr, m)
        # erase the same canvas region on the BlurSorted image, un-BlurSort, inpaint the scattered holes
        canvas = blursort2d(arr); m_canvas = m
        img_back = unblursort2d(canvas); m_back = unblursort2d(m_canvas[:, :, None])[:, :, 0]
        rec_b = inpaint_nearest(img_back, m_back)
        res[name] = {"erased_pct": float(m.mean() * 100), "psnr_original": psnr(arr, rec_o), "psnr_blursorted": psnr(arr, rec_b)}
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--images", type=int, default=8); ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args(); tmp = tempfile.mkdtemp(); rng = np.random.default_rng(0)
    files = sorted(Path("data/kodak").glob("kodim*.png"))[:a.images]; allres = []
    for f in files:
        im = np.array(Image.open(f).convert("RGB")); H, W, _ = im.shape
        arr = im[(H - 512) // 2:(H - 512) // 2 + 512, (W - 512) // 2:(W - 512) // 2 + 512].copy()
        bs = blursort2d(arr); assert np.array_equal(unblursort2d(bs), arr)
        npx = 512 * 512
        r = {"image": f.name, "lossless_original": lossless_sizes(arr, tmp), "lossless_blursorted": lossless_sizes(bs, tmp)}
        jo, jb = [], []
        for q in (30, 50, 75, 90, 95):
            b1, d1 = jpeg(arr, q); jo.append({"q": q, "bpp": 8 * b1 / npx, "psnr": psnr(arr, d1)})
            b2, d2 = jpeg(bs, q); jb.append({"q": q, "bpp": 8 * b2 / npx, "psnr": psnr(arr, unblursort2d(d2))})
        r["jpeg_original"], r["jpeg_blursorted"] = jo, jb
        r["erase"] = erase_tests(arr, rng)
        allres.append(r)
        print(f"{f.name}: lossless PNG/WebP/JXL original {8*r['lossless_original']['png']/npx:.2f}/{8*r['lossless_original']['webp']/npx:.2f}/{8*r['lossless_original']['jxl']/npx:.2f} bpp | "
              f"blursorted {8*r['lossless_blursorted']['png']/npx:.2f}/{8*r['lossless_blursorted']['webp']/npx:.2f}/{8*r['lossless_blursorted']['jxl']/npx:.2f}")
        print("   JPEG q75: original %.2f bpp @ %.1f dB | blursorted %.2f bpp @ %.1f dB" % (jo[2]["bpp"], jo[2]["psnr"], jb[2]["bpp"], jb[2]["psnr"]))
        print("   erase+inpaint PSNR (original vs blursorted): " + ", ".join(f"{k} {v['psnr_original']:.1f} vs {v['psnr_blursorted']:.1f} dB" for k, v in r["erase"].items()))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(allres, open(Path(a.out) / f"anticompression_kodak_{a.images}.json", "w"), indent=1)
    npx = 512 * 512
    print("\nMEANS over", len(allres), "images")
    for c in ("png", "webp", "jxl"):
        o = np.mean([8 * r["lossless_original"][c] / npx for r in allres]); b = np.mean([8 * r["lossless_blursorted"][c] / npx for r in allres])
        print(f"  lossless {c}: original {o:.2f} bpp, blursorted {b:.2f} bpp ({b/o:.2f}x; raw = 24)")
    for i, q in enumerate((30, 50, 75, 90, 95)):
        print(f"  JPEG q{q}: original {np.mean([r['jpeg_original'][i]['bpp'] for r in allres]):.2f} bpp @ {np.mean([r['jpeg_original'][i]['psnr'] for r in allres]):.1f} dB | "
              f"blursorted {np.mean([r['jpeg_blursorted'][i]['bpp'] for r in allres]):.2f} bpp @ {np.mean([r['jpeg_blursorted'][i]['psnr'] for r in allres]):.1f} dB")
    for k in allres[0]["erase"]:
        print(f"  erase {k} ({allres[0]['erase'][k]['erased_pct']:.0f}% of pixels): recovered PSNR original {np.mean([r['erase'][k]['psnr_original'] for r in allres]):.1f} dB vs blursorted {np.mean([r['erase'][k]['psnr_blursorted'] for r in allres]):.1f} dB")
    print("saved", Path(a.out) / f"anticompression_kodak_{a.images}.json")


if __name__ == "__main__":
    main()
