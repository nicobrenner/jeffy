"""Nico's experiment (2026-10-25): start from an EXISTING compressed video (what a phone or a download gives you), extract its
frames, compress them as images (per-frame / tile sheet / interleave, JPEG XL), and compare the total image bytes with the size of
the video file itself. The decoded frames are what the user actually has, so the image side is scored on reproducing them:
lossless (bit-exact) and lossy at a mild JXL distance (visually transparent, d=1) and a stronger one (d=2).

Source clips are raw y4m (never compressed); we first make the "existing video" with AV1 / HEVC at a typical quality (crf),
in the codec's native yuv420p, then decode to frames. All image coding is on the decoded Y/U/V planes (no RGB conversion).

  python3 -u blursort_existing_video.py data/video_xiph/akiyo_cif.y4m --k 256 --codec av1 --crf 30
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from blursort_video_yuv import read_y4m, raw420, tile, interleave
from blursort_video_to_image import snake_phases
from blursort_video_lossy import split420, jxl_gray_lossy, yuv_psnr


def make_video(frames, codec, crf, tmp):
    h, w = frames[0][0].shape
    path = os.path.join(tmp, f"src_{codec}_{crf}.mkv")
    args = {"av1": ["-c:v", "libaom-av1", "-crf", str(crf), "-b:v", "0", "-cpu-used", "4"],
            "hevc": ["-c:v", "libx265", "-crf", str(crf), "-preset", "medium"]}[codec]
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "30",
           "-i", "pipe:0"] + args + ["-pix_fmt", "yuv420p", path]
    r = subprocess.run(cmd, input=raw420(frames), capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[-300:])
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"],
                         capture_output=True).stdout
    return os.path.getsize(path), split420(dec, len(frames), h, w)


def jxl_lossless(a, tmp):
    b, back = jxl_gray_lossy(a, tmp, 0)
    assert np.array_equal(back, a)
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("y4m"); ap.add_argument("--k", type=int, default=256); ap.add_argument("--codec", default="av1"); ap.add_argument("--crf", type=int, default=30)
    ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    src = read_y4m(a.y4m, a.k, crop=None)
    vbytes, frames = make_video(src, a.codec, a.crf, tmp)
    h, w = frames[0][0].shape
    npix = a.k * h * w
    s = int(round(a.k ** 0.5)); assert s * s == a.k, "K must be square for the interleave"
    order = snake_phases(s)
    ypsnr, _ = yuv_psnr(src, frames)
    print(f"\n{a.y4m}: {a.k} frames {w}x{h}; existing video = {a.codec} crf{a.crf}: {vbytes:,} B = {8*vbytes/npix:.3f} bpp, Y-PSNR vs raw {ypsnr:.2f} dB")
    res = {"video_bytes": vbytes, "video_bpp": 8 * vbytes / npix, "video_psnr_vs_raw": ypsnr, "image": {}}
    for dist in [0, 1.0, 2.0]:
        for mode in ["per_frame", "tile", "interleave"]:
            tot = 0; dec = [[None] * 3 for _ in range(a.k)]
            for pi in range(3):
                planes = [f[pi] for f in frames]
                if mode == "per_frame":
                    for i, p in enumerate(planes):
                        b, d = jxl_gray_lossy(p, tmp, dist); tot += b; dec[i][pi] = d
                else:
                    sheet = tile(planes) if mode == "tile" else interleave(planes, order)
                    b, d = jxl_gray_lossy(sheet, tmp, dist); tot += b
                    ph, pw = planes[0].shape
                    for i in range(a.k):
                        if mode == "tile":
                            dec[i][pi] = d[(i // s) * ph:(i // s + 1) * ph, (i % s) * pw:(i % s + 1) * pw]
                        else:
                            dx, dy = order[i]; dec[i][pi] = d[dy::s, dx::s]
            y_dec, _ = yuv_psnr(frames, [tuple(x) for x in dec]); y_raw, _ = yuv_psnr(src, [tuple(x) for x in dec])
            exact = all(np.array_equal(dec[i][pi], frames[i][pi]) for i in range(a.k) for pi in range(3))
            res["image"][f"{mode}_jxl_d{dist}"] = {"bytes": tot, "bpp": 8 * tot / npix, "ratio_vs_video": tot / vbytes, "psnr_vs_decoded": y_dec, "psnr_vs_raw": y_raw, "exact": exact}
            print(f"  {mode:<11} JXL d={dist:<4} {tot:>12,} B  {8*tot/npix:6.3f} bpp  = {tot/vbytes:6.1f}x the video  PSNR vs decoded {y_dec:5.2f} / vs raw {y_raw:5.2f}  exact={exact}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = Path(a.out) / f"existing_video_{Path(a.y4m).stem}_{a.codec}{a.crf}_k{a.k}.json"
    json.dump({"y4m": a.y4m, "k": a.k, "codec": a.codec, "crf": a.crf, "results": res}, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
