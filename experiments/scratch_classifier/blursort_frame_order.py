"""Does frame ORDER matter when BlurSort polyphase sub-images are coded as a video?

BlurSort on a 2^k square image = bit-reversal permutation = polyphase decomposition: at scale s the
4^s "frames" are img[dy::2^s, dx::2^s] for phases (dx, dy). We write the frames as a video under several
orders and compare file sizes with lossless codecs (exactness verified by decoding) plus one lossy point.

  python3 blursort_frame_order.py IMAGE [--size 512] [--scales 2 3] [--out pile_runs/blursort]

Orders:
  dyadic   native BlurSort block order (row = rev(dx), col = rev(dy) over the phase grid, row-major)
  raster   phases in (dy, dx) row-major order
  snake    boustrophedon over the phase grid (consecutive frames differ by one full-res pixel)
  gray     Gray-code order on each axis, snake over rows (also one-pixel steps)
  random   fixed-seed shuffle (control)
Codecs:
  ffv1 (lossless, intra-only => order should NOT matter: null control), libx264rgb -qp 0 (lossless, inter),
  libx264rgb -crf 18 (lossy, inter; PSNR reported). Baselines: PNG and WebP-lossless of the crop.
"""

import argparse
import io
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
from PIL import Image


def bitrev(i, k):
    return int(format(i, f"0{k}b")[::-1], 2) if k else 0


def phase_orders(s, seed=0):
    n = 2 ** s
    raster = [(dx, dy) for dy in range(n) for dx in range(n)]
    dyadic = [(bitrev(r, s), bitrev(c, s)) for r in range(n) for c in range(n)]  # block (r,c) holds phase (rev r, rev c)
    snake = [(dx if dy % 2 == 0 else n - 1 - dx, dy) for dy in range(n) for dx in range(n)]
    gray = [g ^ (g >> 1) for g in range(n)]
    grayo = [(gray[i] if j % 2 == 0 else gray[n - 1 - i], gray[j]) for j in range(n) for i in range(n)]
    rnd = list(raster)
    np.random.default_rng(seed).shuffle(rnd)
    return {"dyadic": dyadic, "raster": raster, "snake": snake, "gray": grayo, "random": rnd}


def frames(img, s, order):
    step = 2 ** s
    return [np.ascontiguousarray(img[dy::step, dx::step]) for dx, dy in order]


def write_video(frs, path, codec_args, fps=30):
    h, w, _ = frs[0].shape
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "pipe:0"] + codec_args + [path]
    p = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frs), capture_output=True)
    if p.returncode:
        raise RuntimeError(p.stderr.decode())
    return os.path.getsize(path)


def read_video(path, n, h, w):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    out = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(out, dtype=np.uint8).reshape(n, h, w, 3)


CODECS = {
    "ffv1_lossless": ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24"],
    "x264rgb_qp0": ["-c:v", "libx264rgb", "-qp", "0", "-preset", "medium", "-pix_fmt", "rgb24"],
    "x264rgb_crf18": ["-c:v", "libx264rgb", "-crf", "18", "-preset", "medium", "-pix_fmt", "rgb24"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--scales", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args()

    im = Image.open(a.image).convert("RGB")
    W, H = im.size
    assert min(W, H) >= a.size, f"image {W}x{H} smaller than {a.size}"
    l, t = (W - a.size) // 2, (H - a.size) // 2
    img = np.array(im.crop((l, t, l + a.size, t + a.size)))
    name = Path(a.image).stem
    npix = a.size * a.size
    res = {"image": a.image, "size": a.size, "baselines": {}, "scales": {}}

    buf = io.BytesIO(); Image.fromarray(img).save(buf, "PNG", optimize=True); res["baselines"]["png_bytes"] = buf.tell()
    buf = io.BytesIO(); Image.fromarray(img).save(buf, "WEBP", lossless=True, quality=100, method=6); res["baselines"]["webp_lossless_bytes"] = buf.tell()
    print(f"{name} {a.size}x{a.size}: PNG {res['baselines']['png_bytes']:,} B ({8*res['baselines']['png_bytes']/npix:.3f} bpp), "
          f"WebP-lossless {res['baselines']['webp_lossless_bytes']:,} B ({8*res['baselines']['webp_lossless_bytes']/npix:.3f} bpp)")

    tmp = tempfile.mkdtemp()
    for s in a.scales:
        orders = phase_orders(s)
        step = 2 ** s
        fh = fw = a.size // step
        res["scales"][s] = {}
        print(f"\nscale s={s}: {4**s} frames of {fw}x{fh}")
        print(f"{'codec':<16}{'order':<9}{'bytes':>10}{'bpp':>8}{'exact':>7}{'psnr':>7}{'sec':>6}")
        for cname, cargs in CODECS.items():
            res["scales"][s][cname] = {}
            for oname, order in orders.items():
                frs = frames(img, s, order)
                path = os.path.join(tmp, f"{name}_s{s}_{cname}_{oname}.mkv")
                t0 = time.time()
                nbytes = write_video(frs, path, cargs)
                dec = read_video(path, len(frs), fh, fw)
                ref = np.stack(frs)
                exact = bool(np.array_equal(dec, ref))
                mse = float(np.mean((dec.astype(np.float64) - ref.astype(np.float64)) ** 2))
                psnr = float("inf") if mse == 0 else 10 * np.log10(255 ** 2 / mse)
                res["scales"][s][cname][oname] = {"bytes": nbytes, "bpp": 8 * nbytes / npix, "exact": exact, "psnr": psnr}
                print(f"{cname:<16}{oname:<9}{nbytes:>10,}{8*nbytes/npix:>8.3f}{str(exact):>7}{psnr:>7.2f}{time.time()-t0:>6.1f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = Path(a.out) / f"frame_order_{name}_{a.size}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
