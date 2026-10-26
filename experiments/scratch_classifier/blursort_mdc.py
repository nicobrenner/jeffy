"""Multiple-description transmission with BlurSort phases (Nico: "code each phase as its own unit and accept losing some").
Goal: quality as a function of lost packets, at a stated byte cost, against two conventional ways of packetising an image.

  phases   : the image's s×s polyphase phases, each coded as its own JPEG XL file (one packet per phase). Any received subset
             gives a full-field image; missing pixels are filled by bilinear interpolation from the received phases.
  jxl_prog : one progressive JPEG XL file cut into s² equal byte packets; decodable only up to the first lost packet.
  jpeg_rows: JPEG q90 cut into s² horizontal stripes (one packet per stripe); lost stripes filled by interpolation from the rows above/below.
Lossless image coders throughout for the phase/JXL arms (so quality loss comes only from the missing packets); JPEG is lossy by nature,
so it is reported at its own PSNR with zero loss. k packets lost at random, 5 trials per k.

  python3 -u blursort_mdc.py --images 6 --s 4
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
from scipy.interpolate import griddata

from blursort_progressive import dyadic_order, psnr


def jxl_bytes(arr, tmp, name, progressive=False):
    src, dst = f"{tmp}/{name}.png", f"{tmp}/{name}.jxl"
    Image.fromarray(arr).save(src)
    subprocess.run(["cjxl", src, dst, "-d", "0", "-e", "7", "--quiet"] + (["-p"] if progressive else []), check=True, capture_output=True)
    return open(dst, "rb").read()


def jxl_decode_partial(b, tmp):
    open(f"{tmp}/part.jxl", "wb").write(b)
    if os.path.exists(f"{tmp}/part.png"):
        os.remove(f"{tmp}/part.png")
    r = subprocess.run(["djxl", f"{tmp}/part.jxl", f"{tmp}/part.png", "--allow_partial_files", "--quiet"], capture_output=True)
    if r.returncode or not os.path.exists(f"{tmp}/part.png"):
        return None
    return np.array(Image.open(f"{tmp}/part.png").convert("RGB"))


def fill_bilinear(arr, known):
    """interpolate missing pixels (known=False) from known ones, per channel; falls back to nearest where griddata leaves NaN."""
    H, W, _ = arr.shape
    yy, xx = np.mgrid[0:H, 0:W]
    pts = np.stack([yy[known], xx[known]], 1)
    out = arr.astype(np.float64).copy()
    near = ndimage.distance_transform_edt(~known, return_distances=False, return_indices=True)
    for c in range(3):
        v = griddata(pts, arr[:, :, c][known].astype(np.float64), (yy, xx), method="linear")
        nan = np.isnan(v)
        v[nan] = arr[:, :, c][near[0], near[1]][nan]
        out[:, :, c] = v
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--images", type=int, default=6); ap.add_argument("--s", type=int, default=4)
    ap.add_argument("--trials", type=int, default=5); ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args(); s = a.s; n_pk = s * s; tmp = tempfile.mkdtemp(); rng = np.random.default_rng(0)
    files = sorted(Path("data/kodak").glob("kodim*.png"))[:a.images]
    ks = [0, 1, 2, 4, 8, 12] if n_pk == 16 else [0, 1, 2, 3]
    allres = []
    for f in files:
        im = np.array(Image.open(f).convert("RGB")); H, W, _ = im.shape
        arr = im[(H - 512) // 2:(H - 512) // 2 + 512, (W - 512) // 2:(W - 512) // 2 + 512].copy(); npx = 512 * 512
        order = dyadic_order(s)
        # phases as packets (JXL lossless each)
        pk_phase = {ph: jxl_bytes(arr[ph[1]::s, ph[0]::s], tmp, "ph") for ph in order}
        bytes_phase = sum(len(b) for b in pk_phase.values())
        # one progressive JXL cut into n_pk packets
        prog = jxl_bytes(arr, tmp, "prog", progressive=True); cut = len(prog) // n_pk
        pk_prog = [prog[i * cut:(i + 1) * cut] if i < n_pk - 1 else prog[i * cut:] for i in range(n_pk)]
        # JPEG q90 stripes
        buf = io.BytesIO(); Image.fromarray(arr).save(buf, "JPEG", quality=90, subsampling=0); jpeg_bytes = buf.tell(); buf.seek(0)
        jdec = np.array(Image.open(buf).convert("RGB")); rows = 512 // n_pk
        res = {"image": f.name, "bpp": {"phases": 8 * bytes_phase / npx, "jxl_prog": 8 * len(prog) / npx, "jpeg_q90": 8 * jpeg_bytes / npx}, "curve": {}}
        for k in ks:
            ph_ps, pr_ps, jp_ps = [], [], []
            for t in range(a.trials if k else 1):
                lost = set(rng.choice(n_pk, size=k, replace=False).tolist()) if k else set()
                # phases
                known = np.zeros((512, 512), bool)
                for i, ph in enumerate(order):
                    if i not in lost:
                        known[ph[1]::s, ph[0]::s] = True
                ph_ps.append(psnr(arr, fill_bilinear(arr, known)) if k else 99.0)
                # progressive JXL: usable prefix = packets before the first lost one
                first = min(lost) if lost else n_pk
                dec = jxl_decode_partial(b"".join(pk_prog[:first]), tmp) if first else None
                pr_ps.append(psnr(arr, dec) if dec is not None else 0.0)
                # JPEG stripes
                known = np.ones((512, 512), bool)
                for i in lost:
                    known[i * rows:(i + 1) * rows, :] = False
                jp_ps.append(psnr(arr, fill_bilinear(jdec, known)) if k else psnr(arr, jdec))
            res["curve"][k] = {"phases": float(np.mean(ph_ps)), "jxl_prog": float(np.mean(pr_ps)), "jpeg_rows": float(np.mean(jp_ps))}
        allres.append(res)
        print(f"{f.name}: bytes phases {res['bpp']['phases']:.2f} bpp | progressive JXL {res['bpp']['jxl_prog']:.2f} | JPEG q90 {res['bpp']['jpeg_q90']:.2f}")
        print("   lost k → PSNR phases / jxl-prog / jpeg-rows: " + ", ".join(f"{k}: {v['phases']:.1f}/{v['jxl_prog']:.1f}/{v['jpeg_rows']:.1f}" for k, v in res["curve"].items()))
    print(f"\nMEANS over {len(allres)} images, {n_pk} packets:")
    print("  bytes: " + ", ".join(f"{m} {np.mean([r['bpp'][m] for r in allres]):.2f} bpp" for m in ("phases", "jxl_prog", "jpeg_q90")))
    for k in ks:
        print(f"  {k:>2} of {n_pk} packets lost: phases {np.mean([r['curve'][k]['phases'] for r in allres]):.1f} dB | progressive JXL {np.mean([r['curve'][k]['jxl_prog'] for r in allres]):.1f} dB | JPEG stripes {np.mean([r['curve'][k]['jpeg_rows'] for r in allres]):.1f} dB")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    json.dump(allres, open(Path(a.out) / f"mdc_kodak_{a.images}_s{s}.json", "w"), indent=1, default=float)
    print("saved", Path(a.out) / f"mdc_kodak_{a.images}_s{s}.json")


if __name__ == "__main__":
    main()
