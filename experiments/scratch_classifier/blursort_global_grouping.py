"""Nico's grouping idea, tested end to end (2026-10-25): instead of coding the frames in timeline order, search over ALL frames
for groups of similar frames (not necessarily adjacent in time), and for each group pick interleave or tile by trying both.
Lossless, native YUV 4:2:0, JPEG XL per plane. The frame map (which frame went where) is counted (1 byte per frame).

Grouping: frames are compared on a 4x-downsampled luma (MSE). A nearest-neighbour chain (greedy MST walk, the DCC'06 idea)
orders all frames by similarity; the chain is cut into runs where the step MSE exceeds a threshold, and runs are split into
square-able chunks of G frames (G = 16 or 4; a leftover chunk is tiled, never interleaved). Each chunk: tile vs interleave
(snake order along the chain), keep the smaller.

Controls on the SAME frames: AV1 lossless in timeline order; AV1 lossless in the chain order (gives the video codec the same
reordering); tile-all; interleave-all in timeline order; per-frame JXL.

Clips: the three natural ones (first 64 frames), plus a synthetic "cutback" clip that alternates 8-frame segments of akiyo and
foreman (64 frames) -- the case where returning content is out of reach of a video codec's reference window.

  python3 -u blursort_global_grouping.py --clip akiyo --k 64
"""
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from blursort_video_yuv import read_y4m, raw420, tile, interleave
from blursort_video_to_image import snake_phases
from blursort_video_lossy import jxl_gray_lossy

Y4M = {s: f"data/video_xiph/{s}_cif.y4m" for s in ["akiyo", "foreman", "bus"]}


def load_clip(name, k, crop):
    if name == "cutback":
        a = read_y4m(Y4M["akiyo"], k // 2, crop); b = read_y4m(Y4M["foreman"], k // 2, crop)
        out = []
        for seg in range(k // 16):
            out += a[seg * 8:(seg + 1) * 8] + b[seg * 8:(seg + 1) * 8]
        return out
    return read_y4m(Y4M[name], k, crop)


def av1_lossless(frames, tmp, name):
    h, w = frames[0][0].shape
    path = os.path.join(tmp, f"{name}.mkv")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "yuv420p", "-s", f"{w}x{h}", "-r", "30", "-i", "pipe:0",
           "-c:v", "libaom-av1", "-crf", "0", "-aom-params", "lossless=1", "-cpu-used", "4", "-pix_fmt", "yuv420p", path]
    r = subprocess.run(cmd, input=raw420(frames), capture_output=True)
    if r.returncode:
        raise RuntimeError(r.stderr.decode()[-300:])
    dec = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "yuv420p", "pipe:1"], capture_output=True).stdout
    assert dec == raw420(frames)
    return os.path.getsize(path)


def jxl_ll(a, tmp):
    b, back = jxl_gray_lossy(a, tmp, 0); assert np.array_equal(back, a); return b


def code_set(frames, mode, tmp):
    """lossless JXL bytes for a set of frames as per-frame / tile / interleave (sum over the 3 planes)."""
    tot = 0
    s = int(round(len(frames) ** 0.5))
    for pi in range(3):
        planes = [f[pi] for f in frames]
        if mode == "per_frame":
            tot += sum(jxl_ll(p, tmp) for p in planes)
        elif mode == "tile":
            tot += jxl_ll(tile_any(planes), tmp)
        else:
            assert s * s == len(frames)
            tot += jxl_ll(interleave(planes, snake_phases(s)), tmp)
    return tot


def tile_any(planes):
    n = len(planes); h, w = planes[0].shape
    cols = int(np.ceil(np.sqrt(n))); rows = int(np.ceil(n / cols))
    out = np.zeros((rows * h, cols * w), np.uint8)
    for i, p in enumerate(planes):
        r, c = divmod(i, cols); out[r * h:(r + 1) * h, c * w:(c + 1) * w] = p
    return out


def chain_order(frames):
    small = np.stack([f[0][::4, ::4].astype(np.float32) for f in frames])
    n = len(frames)
    d = ((small[:, None] - small[None]) ** 2).mean(axis=(2, 3))
    # start from the frame with the smallest mean distance to the rest; greedy nearest-neighbour walk
    cur = int(np.argmin(d.sum(1))); order = [cur]; left = set(range(n)) - {cur}; steps = []
    while left:
        nxt = min(left, key=lambda j: d[cur, j]); steps.append(d[cur, nxt]); order.append(nxt); left.remove(nxt); cur = nxt
    return order, steps, d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", default="akiyo"); ap.add_argument("--k", type=int, default=64); ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--group", type=int, default=16); ap.add_argument("--out", default="pile_runs/blursort")
    ap.add_argument("--no-cut", action="store_true", help="do not cut the chain at MSE jumps; just chunk the similarity order into G")
    a = ap.parse_args()
    tmp = tempfile.mkdtemp()
    frames = load_clip(a.clip, a.k, a.crop)
    nl = a.k * a.crop * a.crop
    order, steps, d = chain_order(frames)
    adj = [d[i, i + 1] for i in range(a.k - 1)]
    print(f"\n{a.clip}: {a.k} frames {a.crop}²; timeline step MSE median {np.median(adj):.1f} max {np.max(adj):.1f}; chain step MSE median {np.median(steps):.1f} max {np.max(steps):.1f}")
    res = {}
    res["av1_lossless_timeline"] = av1_lossless(frames, tmp, "av1_t")
    res["av1_lossless_chain_order"] = av1_lossless([frames[i] for i in order], tmp, "av1_c")
    res["per_frame_jxl"] = code_set(frames, "per_frame", tmp)
    res["tile_all"] = code_set(frames, "tile", tmp)
    res["interleave_all_timeline"] = code_set(frames, "interleave", tmp)
    res["interleave_all_chain"] = code_set([frames[i] for i in order], "interleave", tmp)
    for k, v in res.items():
        print(f"  {k:<28}{v:>10,} B  {8*v/nl:6.3f} bpp")
    # grouped search: cut the chain where the step MSE jumps, chunk into G, choose per chunk
    thr = float("inf") if a.no_cut else float(np.median(steps) * 4 + 1.0)
    runs = [[order[0]]]
    for st, idx in zip(steps, order[1:]):
        if st > thr:
            runs.append([idx])
        else:
            runs[-1].append(idx)
    G = a.group
    chunks = []
    for r in runs:
        i = 0
        while i < len(r):
            for g in (G, 4, 1):
                if len(r) - i >= g:
                    chunks.append(r[i:i + g]); i += g; break
    tot = 0; choice = []
    for ch in chunks:
        fr = [frames[i] for i in ch]
        bt = code_set(fr, "tile", tmp) if len(ch) > 1 else code_set(fr, "per_frame", tmp)
        s = int(round(len(ch) ** 0.5))
        bi = code_set(fr, "interleave", tmp) if s * s == len(ch) and len(ch) > 1 else None
        best = min([bt] + ([bi] if bi else []))
        choice.append({"frames": [int(x) for x in ch], "tile": int(bt), "interleave": (int(bi) if bi else None), "pick": "interleave" if bi and bi < bt else "tile"})
        tot += best
    tot += a.k  # frame map, 1 byte per frame
    res["grouped_search"] = tot
    print(f"  {'grouped_search':<28}{tot:>10,} B  {8*tot/nl:6.3f} bpp   runs {len(runs)} chunks {len(chunks)} picks: "
          + ", ".join(f"{len(c['frames'])}:{c['pick'][0]}" for c in choice))
    best_ctrl = min(res["av1_lossless_timeline"], res["av1_lossless_chain_order"])
    print(f"  grouped_search vs best AV1: {tot/best_ctrl:.3f}x ; vs tile_all {tot/res['tile_all']:.3f}x")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    out = Path(a.out) / f"global_grouping_{a.clip}_k{a.k}_g{G}{'_nocut' if a.no_cut else ''}.json"
    json.dump({"clip": a.clip, "k": a.k, "crop": a.crop, "group": G, "order": [int(x) for x in order], "threshold": thr,
               "results": {k: int(v) for k, v in res.items()}, "chunks": choice}, open(out, "w"), indent=1)
    print("saved", out)


if __name__ == "__main__":
    main()
