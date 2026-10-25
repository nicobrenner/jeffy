"""Stronger lossless-video controls for the video->image test: x265 lossless, libaom AV1 lossless, VP9 lossless, plus the
image-side winners (tile+JXL, interleave+JXL) on the same frames. ffv1 / x264-qp0 are weak inter predictors in lossless
mode; a claim against "video codecs" needs these. Same frames, same crops as blursort_video_to_image.py.

  python3 -u blursort_video_codec_controls.py data/video_xiph/akiyo_frames --pattern 'frame*.png' --crop 256 --k 16
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from blursort_video_to_image import load_frames, merge_tile, merge_interleave, snake_phases, jxl_bytes, webp_bytes


def enc(frs, args, tmp, name, ext="mkv"):
    h, w, _ = frs[0].shape
    path = os.path.join(tmp, f"{name}.{ext}")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
           "-i", "pipe:0"] + args + [path]
    r = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frs), capture_output=True)
    if r.returncode:
        return None, r.stderr.decode()[-300:]
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
                         capture_output=True).stdout
    try:
        exact = np.array_equal(np.frombuffer(dec, np.uint8).reshape(len(frs), h, w, 3), np.stack(frs))
    except ValueError:
        exact = False
    return os.path.getsize(path), exact


CODECS = {
    # RGB-exact lossless paths (gbrp = planar RGB inside the codec; no chroma subsampling)
    "x264rgb_qp0": ["-c:v", "libx264rgb", "-qp", "0", "-preset", "veryslow", "-pix_fmt", "rgb24"],
    "x265_lossless": ["-c:v", "libx265", "-x265-params", "lossless=1", "-preset", "slower", "-pix_fmt", "gbrp"],
    "vp9_lossless": ["-c:v", "libvpx-vp9", "-lossless", "1", "-pix_fmt", "gbrp"],
    "av1_lossless": ["-c:v", "libaom-av1", "-crf", "0", "-aom-params", "lossless=1", "-cpu-used", "4", "-pix_fmt", "gbrp"],
    "ffv1": ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir")
    ap.add_argument("--pattern", default="frame*.png")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    frs = load_frames(a.frame_dir, a.pattern, 0, a.k * a.stride, a.crop)[::a.stride]
    npix = a.k * a.crop * a.crop
    res = {}
    print(f"\n{a.frame_dir}: K={a.k} x {a.crop}² (stride {a.stride})")
    for name, args in CODECS.items():
        size, exact = enc(frs, args, tmp, name)
        res[name] = {"bytes": size, "exact": exact}
        print(f"  {name:<16}{(size or 0):>10,} B  {8*(size or 0)/npix:6.3f} bpp  exact={exact}")
    tile = merge_tile(frs); inter = merge_interleave(frs, snake_phases(int(round(a.k ** 0.5))))
    for name, b in [("tile_jxl", jxl_bytes(tile, tmp)), ("tile_webp", webp_bytes(tile)),
                    ("interleave_jxl", jxl_bytes(inter, tmp)), ("per_frame_jxl", sum(jxl_bytes(f, tmp) for f in frs))]:
        res[name] = {"bytes": b, "exact": True}
        print(f"  {name:<16}{b:>10,} B  {8*b/npix:6.3f} bpp")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = Path(a.out) / f"video_codec_controls_{Path(a.frame_dir.rstrip('/')).name}_c{a.crop}_k{a.k}_s{a.stride}.json"
    json.dump({"frame_dir": a.frame_dir, "crop": a.crop, "k": a.k, "stride": a.stride, "results": res}, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
