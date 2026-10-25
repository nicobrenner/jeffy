"""LOSSY video->image test in native YUV 4:2:0: tile+JXL (lossy, per plane) and per-frame JXL vs AV1 / x265 lossy, as
rate-distortion curves (bpp per luma pixel vs PSNR on Y and on YUV 6:1:1). Same crops and frames as blursort_video_yuv.py.

  python3 -u blursort_video_lossy.py data/video_xiph/foreman_cif.y4m --crop 256 --k 16
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from blursort_video_yuv import read_y4m, raw420, tile


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def yuv_psnr(frames, dec_frames):
    y = psnr(np.stack([f[0] for f in frames]), np.stack([f[0] for f in dec_frames]))
    u = psnr(np.stack([f[1] for f in frames]), np.stack([f[1] for f in dec_frames]))
    v = psnr(np.stack([f[2] for f in frames]), np.stack([f[2] for f in dec_frames]))
    return y, (6 * y + u + v) / 8


def split420(raw, k, crop):
    ysz, csz = crop * crop, (crop // 2) ** 2
    out = []
    for i in range(k):
        b = raw[i * (ysz + 2 * csz):(i + 1) * (ysz + 2 * csz)]
        out.append((np.frombuffer(b[:ysz], np.uint8).reshape(crop, crop), np.frombuffer(b[ysz:ysz + csz], np.uint8).reshape(crop // 2, crop // 2),
                    np.frombuffer(b[ysz + csz:], np.uint8).reshape(crop // 2, crop // 2)))
    return out


def enc_video(frames, args, tmp, name):
    h, w = frames[0][0].shape
    path = os.path.join(tmp, f"{name}.mkv")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "30",
           "-i", "pipe:0"] + args + [path]
    r = subprocess.run(cmd, input=raw420(frames), capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[-300:])
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"],
                         capture_output=True).stdout
    return os.path.getsize(path), split420(dec, len(frames), h)


def jxl_gray_lossy(a, tmp, dist):
    p = os.path.join(tmp, "g.png"); Image.fromarray(a, "L").save(p)
    q = os.path.join(tmp, "g.jxl")
    r = subprocess.run(["cjxl", p, q, "-d", str(dist), "-e", "7", "--quiet"], capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[-300:])
    subprocess.run(["djxl", q, os.path.join(tmp, "g_dec.png"), "--quiet"], capture_output=True)
    return os.path.getsize(q), np.array(Image.open(os.path.join(tmp, "g_dec.png")).convert("L"))


def untile(sheet, k, h, w):
    s = int(round(k ** 0.5))
    return [sheet[(i // s) * h:(i // s + 1) * h, (i % s) * w:(i % s + 1) * w] for i in range(k)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("y4m"); ap.add_argument("--crop", type=int, default=256); ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    frames = read_y4m(a.y4m, a.k, a.crop)
    nl = a.k * a.crop * a.crop
    res = {"av1": [], "x265": [], "tile_jxl": [], "per_frame_jxl": []}
    print(f"\n{a.y4m}: K={a.k} x {a.crop}² (4:2:0 native); bpp per luma pixel / Y-PSNR / YUV-PSNR")
    for crf in [20, 28, 36, 44, 52]:
        size, dec = enc_video(frames, ["-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0", "-cpu-used", "4", "-pix_fmt", "yuv420p"], tmp, f"av1_{crf}")
        y, yuv = yuv_psnr(frames, dec); res["av1"].append({"crf": crf, "bpp": 8 * size / nl, "y_psnr": y, "yuv_psnr": yuv})
        print(f"  av1 crf{crf:<3} {8*size/nl:6.3f} bpp  Y {y:5.2f}  YUV {yuv:5.2f}")
    for crf in [16, 22, 28, 34, 40]:
        size, dec = enc_video(frames, ["-c:v", "libx265", "-crf", str(crf), "-preset", "slower", "-pix_fmt", "yuv420p"], tmp, f"x265_{crf}")
        y, yuv = yuv_psnr(frames, dec); res["x265"].append({"crf": crf, "bpp": 8 * size / nl, "y_psnr": y, "yuv_psnr": yuv})
        print(f"  x265 crf{crf:<3} {8*size/nl:6.3f} bpp  Y {y:5.2f}  YUV {yuv:5.2f}")
    h = w = a.crop
    for dist in [0.5, 1.0, 2.0, 3.5, 6.0, 10.0]:
        tot = 0; dec = [[None, None, None] for _ in range(a.k)]
        for pi in range(3):
            planes = [fr[pi] for fr in frames]
            b, sheet = jxl_gray_lossy(tile(planes), tmp, dist); tot += b
            for i, p in enumerate(untile(sheet, a.k, planes[0].shape[0], planes[0].shape[1])):
                dec[i][pi] = p
        y, yuv = yuv_psnr(frames, [tuple(d) for d in dec]); res["tile_jxl"].append({"dist": dist, "bpp": 8 * tot / nl, "y_psnr": y, "yuv_psnr": yuv})
        print(f"  tile+jxl d{dist:<4} {8*tot/nl:6.3f} bpp  Y {y:5.2f}  YUV {yuv:5.2f}")
        tot = 0; dec = [[None, None, None] for _ in range(a.k)]
        for i, fr in enumerate(frames):
            for pi in range(3):
                b, p = jxl_gray_lossy(fr[pi], tmp, dist); tot += b; dec[i][pi] = p
        y, yuv = yuv_psnr(frames, [tuple(d) for d in dec]); res["per_frame_jxl"].append({"dist": dist, "bpp": 8 * tot / nl, "y_psnr": y, "yuv_psnr": yuv})
        print(f"  frame+jxl d{dist:<4} {8*tot/nl:6.3f} bpp  Y {y:5.2f}  YUV {yuv:5.2f}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = Path(a.out) / f"video_lossy_{Path(a.y4m).stem}_c{a.crop}_k{a.k}.json"
    json.dump({"y4m": a.y4m, "crop": a.crop, "k": a.k, "results": res}, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
