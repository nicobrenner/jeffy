"""Reverse BlurSort: merge K video frames into ONE image by phase interleaving, then compress with image codecs.

Nico's question (2026-10-25): take a video, extract frames, merge them with "reverse BlurSort" (each frame becomes
one polyphase phase of a sqrt(K)x larger image), compress the merged image with WebP / JPEG XL — is that smaller than
the video / the frames? Controls: the same frames tiled side by side (sprite sheet), per-frame image coding, and
lossless video codecs (ffv1 intra-only, x264rgb qp0 inter).

  python3 blursort_video_to_image.py FRAME_DIR [--pattern 'frame0*.jpg'] [--crop 320] [--ks 4 16 64] [--start 0]
"""

import argparse
import glob
import io
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


def load_frames(d, pattern, start, k, crop):
    files = sorted(glob.glob(os.path.join(d, pattern)))[start:start + k]
    assert len(files) == k, f"need {k} frames, found {len(files)}"
    out = []
    for f in files:
        im = Image.open(f).convert("RGB")
        W, H = im.size
        l, t = (W - crop) // 2, (H - crop) // 2
        out.append(np.array(im.crop((l, t, l + crop, t + crop))))
    return out


def snake_phases(s):
    return [(dx if dy % 2 == 0 else s - 1 - dx, dy) for dy in range(s) for dx in range(s)]


def merge_interleave(frs, order):
    """frame i -> phase order[i]; merged[y*s+dy, x*s+dx] = frame_i[y, x]  (inverse of the polyphase split)."""
    s = int(round(len(frs) ** 0.5)); h, w, _ = frs[0].shape
    out = np.zeros((h * s, w * s, 3), dtype=np.uint8)
    for f, (dx, dy) in zip(frs, order):
        out[dy::s, dx::s] = f
    return out


def merge_tile(frs):
    s = int(round(len(frs) ** 0.5)); h, w, _ = frs[0].shape
    out = np.zeros((h * s, w * s, 3), dtype=np.uint8)
    for i, f in enumerate(frs):
        r, c = divmod(i, s)
        out[r * h:(r + 1) * h, c * w:(c + 1) * w] = f
    return out


def png_bytes(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, "PNG", optimize=True); return b.tell()


def webp_bytes(a):
    b = io.BytesIO(); Image.fromarray(a).save(b, "WEBP", lossless=True, quality=100, method=6); return b.tell()


def jxl_bytes(a, tmp):
    p = os.path.join(tmp, "x.png"); Image.fromarray(a).save(p)
    q = os.path.join(tmp, "x.jxl")
    r = subprocess.run(["cjxl", p, q, "-d", "0", "-e", "7", "--quiet"], capture_output=True)
    if r.returncode:
        return None
    return os.path.getsize(q)


def video_bytes(frs, codec_args, tmp, name):
    h, w, _ = frs[0].shape
    path = os.path.join(tmp, name + ".mkv")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "30",
           "-i", "pipe:0"] + codec_args + [path]
    r = subprocess.run(cmd, input=b"".join(f.tobytes() for f in frs), capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode())
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"], capture_output=True).stdout
    exact = np.array_equal(np.frombuffer(dec, np.uint8).reshape(len(frs), h, w, 3), np.stack(frs))
    return os.path.getsize(path), exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir")
    ap.add_argument("--pattern", default="frame0*.jpg")
    ap.add_argument("--crop", type=int, default=320)
    ap.add_argument("--ks", type=int, nargs="+", default=[4, 16, 64])
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1, help="take every Nth frame (larger = less similar frames)")
    ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    res = {"frame_dir": a.frame_dir, "crop": a.crop, "stride": a.stride, "ks": {}}
    for k in a.ks:
        files_needed = a.stride * k
        frs = load_frames(a.frame_dir, a.pattern, a.start, files_needed, a.crop)[::a.stride]
        s = int(round(k ** 0.5)); npix = k * a.crop * a.crop
        r = {}
        r["per_frame_webp"] = sum(webp_bytes(f) for f in frs)
        r["per_frame_png"] = sum(png_bytes(f) for f in frs)
        jx = [jxl_bytes(f, tmp) for f in frs]
        r["per_frame_jxl"] = sum(jx) if all(jx) else None
        tile = merge_tile(frs)
        r["tile_webp"], r["tile_png"], r["tile_jxl"] = webp_bytes(tile), png_bytes(tile), jxl_bytes(tile, tmp)
        for oname, order in {"snake": snake_phases(s), "raster": [(dx, dy) for dy in range(s) for dx in range(s)]}.items():
            m = merge_interleave(frs, order)
            r[f"interleave_{oname}_webp"], r[f"interleave_{oname}_png"], r[f"interleave_{oname}_jxl"] = webp_bytes(m), png_bytes(m), jxl_bytes(m, tmp)
            if oname == "snake":
                Image.fromarray(m).save(os.path.join(a.out, f"video_merge_k{k}_{oname}.png")) if Path(a.out).exists() else None
        r["video_ffv1"], ex1 = video_bytes(frs, ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "rgb24"], tmp, f"ffv1_{k}")
        r["video_x264_qp0"], ex2 = video_bytes(frs, ["-c:v", "libx264rgb", "-qp", "0", "-preset", "medium", "-pix_fmt", "rgb24"], tmp, f"x264_{k}")
        r["lossless_verified"] = bool(ex1 and ex2)
        res["ks"][k] = r
        print(f"\nK={k} frames of {a.crop}² (stride {a.stride}); bpp = bits per pixel over all K frames")
        for name, v in r.items():
            if isinstance(v, int):
                print(f"  {name:<24}{v:>10,} B  {8*v/npix:6.3f} bpp")
        print(f"  lossless video round-trip exact: {r['lossless_verified']}")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    tag = Path(a.frame_dir.rstrip('/')).name
    out = Path(a.out) / f"video_to_image_{tag}_c{a.crop}_s{a.stride}.json"
    json.dump(res, open(out, "w"), indent=1)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
