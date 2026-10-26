"""Frame REORDERING as a codec-agnostic lever, on a REAL edited clip (Nico, 2026-10-25: "why synthetic?").

Input: a y4m with real shot cuts (Sintel trailer, Blender Foundation CC-BY, from media.xiph.org). We take a window of N
frames, detect shot boundaries (luma MSE jump between consecutive frames), and build frame orders:
  timeline      : as shot
  shot_grouped  : frames of the same shot made contiguous; shots ordered by first appearance; a shot that RETURNS later
                  is appended to its earlier occurrence (this is the cheap "map" version: 1 byte/frame to undo)
  chain         : greedy nearest-neighbour walk over all frames by downsampled-luma MSE (no shot notion at all)
Each order is encoded with AV1 lossless, x265 lossless, and AV1 lossy (crf 30, same quantiser for every order, PSNR
reported so the orders are compared at equal distortion). Image side for completeness: grouped search (chunks of 16
along each order, tile vs interleave per chunk, JXL lossless per plane).

  python3 -u blursort_reorder_real.py data/video_xiph/sintel/sintel_trailer_2k_480p24.y4m --start 300 --n 240
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from blursort_video_yuv import read_y4m, raw420
from blursort_video_lossy import split420, yuv_psnr
from blursort_global_grouping import code_set, chain_order


def encode(frames, args, tmp, name, lossless):
    h, w = frames[0][0].shape
    path = os.path.join(tmp, f"{name}.mkv")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "24",
           "-i", "pipe:0"] + args + ["-pix_fmt", "yuv420p", path]
    raw = raw420(frames)
    r = subprocess.run(cmd, input=raw, capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[-300:])
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"], capture_output=True).stdout
    if lossless:
        assert dec == raw, name
        return os.path.getsize(path), None
    y, _ = yuv_psnr(frames, split420(dec, len(frames), h, w))
    return os.path.getsize(path), y


CODECS = {
    "av1_lossless": (["-c:v", "libaom-av1", "-crf", "0", "-aom-params", "lossless=1", "-cpu-used", "5"], True),
    "x265_lossless": (["-c:v", "libx265", "-x265-params", "lossless=1", "-preset", "medium"], True),
    "av1_crf30": (["-c:v", "libaom-av1", "-crf", "30", "-b:v", "0", "-cpu-used", "5"], False),
}


def shots(frames, factor=8.0):
    small = np.stack([f[0][::4, ::4].astype(np.float32) for f in frames])
    step = ((small[1:] - small[:-1]) ** 2).mean(axis=(1, 2))
    thr = np.median(step) * factor + 50.0
    cuts = [i + 1 for i, s in enumerate(step) if s > thr]
    bounds = [0] + cuts + [len(frames)]
    segs = [list(range(bounds[i], bounds[i + 1])) for i in range(len(bounds) - 1)]
    # merge a segment with an earlier one if their mean downsampled frames are close (a returning shot)
    means = [small[s].mean(0) for s in segs]
    group_of = list(range(len(segs)))
    for i in range(len(segs)):
        for j in range(i):
            if group_of[j] == j and ((means[i] - means[j]) ** 2).mean() < thr:
                group_of[i] = j; break
    return segs, group_of, step, thr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("y4m"); ap.add_argument("--start", type=int, default=0); ap.add_argument("--n", type=int, default=240)
    ap.add_argument("--scale", type=int, default=1, help="downsample factor applied to the frames before coding (1 = full)")
    ap.add_argument("--out", default="pile_runs/blursort")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    frames = read_y4m(a.y4m, a.n, crop=None, start=a.start)
    if a.scale > 1:
        s = a.scale
        frames = [(y[::s, ::s].copy(), u[::s, ::s].copy(), v[::s, ::s].copy()) for y, u, v in frames]
        frames = [(y[:y.shape[0] // 2 * 2, :y.shape[1] // 2 * 2], u[:y.shape[0] // 4 * 2, :y.shape[1] // 4 * 2], v[:y.shape[0] // 4 * 2, :y.shape[1] // 4 * 2]) for y, u, v in frames]
    h, w = frames[0][0].shape; nl = a.n * h * w
    segs, group_of, step, thr = shots(frames)
    returning = sum(1 for i, g in enumerate(group_of) if g != i)
    print(f"\n{a.y4m} frames {a.start}..{a.start+a.n-1} at {w}x{h}: {len(segs)} shots (cut threshold MSE {thr:.0f}), {returning} returning shot(s); shot lengths {[len(s) for s in segs]}")
    orders = {"timeline": list(range(a.n))}
    groups = {}
    for i, s in enumerate(segs):
        groups.setdefault(group_of[i], []).extend(s)
    orders["shot_grouped"] = [f for g in sorted(groups) for f in groups[g]]
    orders["chain"] = chain_order(frames)[0]
    res = {"shots": [len(s) for s in segs], "group_of": group_of, "returning": returning, "orders": {}}
    for oname, order in orders.items():
        fr = [frames[i] for i in order]
        r = {}
        for cname, (args, ll) in CODECS.items():
            b, psnr = encode(fr, args, tmp, f"{oname}_{cname}", ll)
            r[cname] = {"bytes": b, "bpp": 8 * b / nl, "y_psnr": psnr}
            print(f"  {oname:<13}{cname:<15}{b:>12,} B  {8*b/nl:6.3f} bpp" + (f"  Y-PSNR {psnr:.2f}" if psnr else ""))
        # image side: chunks of 16 along this order, best of tile / interleave per chunk
        tot = 0
        for i in range(0, a.n, 16):
            ch = fr[i:i + 16]
            bt = code_set(ch, "tile", tmp)
            bi = code_set(ch, "interleave", tmp) if len(ch) == 16 else None
            tot += min([bt] + ([bi] if bi else []))
        r["grouped_jxl"] = {"bytes": tot + a.n, "bpp": 8 * (tot + a.n) / nl}
        print(f"  {oname:<13}{'grouped_jxl':<15}{tot + a.n:>12,} B  {8*(tot + a.n)/nl:6.3f} bpp")
        res["orders"][oname] = r
    base = res["orders"]["timeline"]
    for oname in ["shot_grouped", "chain"]:
        print(f"  {oname} vs timeline: " + ", ".join(f"{c} {res['orders'][oname][c]['bytes']/base[c]['bytes']:.3f}x" for c in list(CODECS) + ["grouped_jxl"]))
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = Path(a.out) / f"reorder_real_{Path(a.y4m).stem}_s{a.start}_n{a.n}_x{a.scale}.json"
    json.dump({"y4m": a.y4m, "start": a.start, "n": a.n, "scale": a.scale, "w": w, "h": h, "results": res}, open(out, "w"), indent=1, default=int)
    print("saved", out)


if __name__ == "__main__":
    main()
