"""Video->image test in the codecs' NATIVE domain: YUV 4:2:0 planes straight from the y4m, no RGB upsampling.

Nico's question (2026-10-25): why does per-frame JXL beat AV1 lossless under motion — is that real or an RGB-mode artefact?
Here every method codes the identical Y (crop x crop), U, V (crop/2 x crop/2) planes of K consecutive frames:
  video side : AV1 / x265 / VP9 lossless and ffv1 (intra), yuv420p in, bit-exact decode checked
  image side : per-frame JXL (3 grayscale planes/frame), tile+JXL (one sheet per plane), interleave+JXL (snake order)
bpp is bits per *luma* pixel over the K frames (the standard video convention; 1.5 samples per luma pixel).

  python3 -u blursort_video_yuv.py data/video_xiph/akiyo_cif.y4m --crop 256 --k 16
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from blursort_video_to_image import snake_phases


def read_y4m(path, k, crop, start=0):
    with open(path, "rb") as f:
        hdr = f.readline().decode()
        W = int([t for t in hdr.split() if t[0] == "W"][0][1:]); H = int([t for t in hdr.split() if t[0] == "H"][0][1:])
        assert "C420" in hdr or " C" not in hdr, hdr  # y4m default is 420
        ysz, csz = W * H, (W // 2) * (H // 2)
        frames = []
        for i in range(start + k):
            tag = f.readline()
            assert tag.startswith(b"FRAME"), tag
            buf = f.read(ysz + 2 * csz)
            if i < start:
                continue
            y = np.frombuffer(buf[:ysz], np.uint8).reshape(H, W)
            u = np.frombuffer(buf[ysz:ysz + csz], np.uint8).reshape(H // 2, W // 2)
            v = np.frombuffer(buf[ysz + csz:], np.uint8).reshape(H // 2, W // 2)
            if crop is None:
                frames.append((y.copy(), u.copy(), v.copy())); continue
            l, t = (W - crop) // 2, (H - crop) // 2
            l -= l % 2; t -= t % 2
            frames.append((y[t:t + crop, l:l + crop].copy(), u[t // 2:t // 2 + crop // 2, l // 2:l // 2 + crop // 2].copy(),
                           v[t // 2:t // 2 + crop // 2, l // 2:l // 2 + crop // 2].copy()))
    return frames


def raw420(frames):
    return b"".join(y.tobytes() + u.tobytes() + v.tobytes() for y, u, v in frames)


def enc_video(frames, args, tmp, name):
    h, w = frames[0][0].shape
    path = os.path.join(tmp, f"{name}.mkv")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "30",
           "-i", "pipe:0"] + args + [path]
    raw = raw420(frames)
    r = subprocess.run(cmd, input=raw, capture_output=True)
    if r.returncode:
        return None, r.stderr.decode()[-300:]
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"],
                         capture_output=True).stdout
    return os.path.getsize(path), dec == raw


def jxl_gray(a, tmp):
    p = os.path.join(tmp, "g.png"); Image.fromarray(a, "L").save(p)
    q = os.path.join(tmp, "g.jxl")
    r = subprocess.run(["cjxl", p, q, "-d", "0", "-e", "7", "--quiet"], capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[-300:])
    dec = subprocess.run(["djxl", q, os.path.join(tmp, "g_dec.png"), "--quiet"], capture_output=True)
    back = np.array(Image.open(os.path.join(tmp, "g_dec.png")).convert("L"))
    assert np.array_equal(back, a), "jxl not lossless"
    return os.path.getsize(q)


def tile(planes):
    s = int(round(len(planes) ** 0.5)); h, w = planes[0].shape
    out = np.zeros((h * s, w * s), np.uint8)
    for i, p in enumerate(planes):
        r, c = divmod(i, s); out[r * h:(r + 1) * h, c * w:(c + 1) * w] = p
    return out


def interleave(planes, order):
    s = int(round(len(planes) ** 0.5)); h, w = planes[0].shape
    out = np.zeros((h * s, w * s), np.uint8)
    for p, (dx, dy) in zip(planes, order):
        out[dy::s, dx::s] = p
    return out


CODECS = {
    "x265_lossless": ["-c:v", "libx265", "-x265-params", "lossless=1", "-preset", "slower", "-pix_fmt", "yuv420p"],
    "vp9_lossless": ["-c:v", "libvpx-vp9", "-lossless", "1", "-pix_fmt", "yuv420p"],
    "av1_lossless": ["-c:v", "libaom-av1", "-crf", "0", "-aom-params", "lossless=1", "-cpu-used", "4", "-pix_fmt", "yuv420p"],
    "x264_qp0": ["-c:v", "libx264", "-qp", "0", "-preset", "veryslow", "-pix_fmt", "yuv420p"],
    "ffv1": ["-c:v", "ffv1", "-level", "3", "-pix_fmt", "yuv420p"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("y4m")
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    frames = read_y4m(a.y4m, a.k, a.crop, a.start)
    nluma = a.k * a.crop * a.crop
    res = {}
    print(f"\n{a.y4m}: K={a.k} x {a.crop}² luma (4:2:0 native); bpp per luma pixel")
    for name, args in CODECS.items():
        size, exact = enc_video(frames, args, tmp, name)
        res[name] = {"bytes": size, "exact": exact}
        print(f"  {name:<18}{(size or 0):>10,} B  {8*(size or 0)/nluma:6.3f} bpp  exact={exact}")
    s = int(round(a.k ** 0.5)); order = snake_phases(s)
    per_frame = tile_b = inter_b = 0
    for pi in range(3):
        planes = [fr[pi] for fr in frames]
        per_frame += sum(jxl_gray(p, tmp) for p in planes)
        tile_b += jxl_gray(tile(planes), tmp)
        inter_b += jxl_gray(interleave(planes, order), tmp)
    for name, b in [("per_frame_jxl", per_frame), ("tile_jxl", tile_b), ("interleave_jxl", inter_b)]:
        res[name] = {"bytes": b, "exact": True}
        print(f"  {name:<18}{b:>10,} B  {8*b/nluma:6.3f} bpp")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = Path(a.out) / f"video_yuv_{Path(a.y4m).stem}_c{a.crop}_k{a.k}.json"
    json.dump({"y4m": a.y4m, "crop": a.crop, "k": a.k, "start": a.start, "bpp_basis": "per luma pixel", "results": res}, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
