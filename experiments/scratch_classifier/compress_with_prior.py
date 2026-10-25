"""Compress ANY image with the shared byte-LM prior and compare against PNG / WebP / JPEG XL.

  python3 compress_with_prior.py photo.jpg                 # fast: exact coded size from the model's probabilities (~1 min / MP on MPS)
  python3 compress_with_prior.py photo.jpg --ac            # also run the real arithmetic coder, write photo.bytelm, verify round-trip
  python3 compress_with_prior.py photo.jpg --heatmap       # also save a per-pixel bits heatmap next to the image
  python3 compress_with_prior.py photo.jpg --ckpt checkpoints/prior_full_d64_20ep_best.pt --int8

The image is cropped to a multiple of 28 px (4032x3024 phone photos need no crop). Nothing is uploaded anywhere.
"""

import argparse
import io
import json
import math
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from compression_demo import ArithmeticDecoder, ArithmeticEncoder
from shared_prior_model import CONFIGS, PriorLM
from train_shared_prior import as_ids

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
P, PAD = 28, 128
UNIFORM = np.arange(257, dtype=np.int64) * 256


def load_prior(path, int8):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = PriorLM(**ck["arch"], **CONFIGS[ck["config"]])
    state, names = ck["model_state"], {n for n, _ in m.named_parameters()}
    nbytes = 0
    if int8:
        q = {}
        for k, v in state.items():
            if k in names:
                s = (v.abs().amax(dim=tuple(range(1, v.dim())), keepdim=True) / 127.0).clamp(min=1e-8) if v.dim() > 1 else (v.abs().max() / 127.0).clamp(min=1e-8)
                q[k] = (torch.round(v / s).clamp(-127, 127) * s).to(v.dtype)
                nbytes += v.numel() + 2 * (v.shape[0] if v.dim() > 1 else 1)
            else:
                q[k] = v
        state = q
    else:
        nbytes = sum(v.numel() for k, v in state.items() if k in names) * 2
    m.load_state_dict(state)
    return m.to(DEVICE).eval(), nbytes, ck


def tile_image(arr):
    """Crop to a multiple of 28 and cut 29x29x3 windows with the 1-px decoded-neighbour context (PAD at borders)."""
    H, W = (arr.shape[0] // P) * P, (arr.shape[1] // P) * P
    arr = arr[:H, :W]
    pa = np.full((H + 1, W + 1, 3), PAD, np.uint8)
    pa[1:, 1:] = arr
    tiles = np.stack([pa[r:r + P + 1, c:c + P + 1] for r in range(0, H, P) for c in range(0, W, P)])
    return arr, tiles


def probs_to_cdf(p, scale=1 << 16):
    c = np.maximum((p * scale).astype(np.int64), 1)
    c[c.argmax()] += scale - c.sum()
    cdf = np.zeros(257, dtype=np.int64)
    cdf[1:] = np.cumsum(c)
    return cdf


@torch.no_grad()
def model_code(m, tiles, bs, do_ac, verify_every=200):
    """Returns (ce_bits, per_tile_bits array, ac_bits or None, bitstreams or None, verified_ok, verified_n)."""
    is_patch = m.is_patch.cpu().numpy()
    ce_total, per_tile, ac_total, streams, ok, nver = 0.0, [], 0, [], 0, 0
    for i in range(0, len(tiles), bs):
        ids = as_ids(m, tiles[i:i + bs], DEVICE)
        logits = m(ids).float()
        logp = F.log_softmax(logits, dim=-1)
        nll = (-logp[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) / math.log(2)) * torch.tensor(is_patch[1:], device=DEVICE, dtype=torch.float32)
        b = nll.sum(1) + (8.0 if is_patch[0] else 0.0)
        ce_total += b.sum().item()
        per_tile.extend(b.cpu().numpy().tolist())
        if do_ac:
            probs = logp.exp().cpu().numpy()
            ids_np = ids.cpu().numpy()
            for j in range(len(ids_np)):
                enc, cdfs = ArithmeticEncoder(), []
                for t in range(ids_np.shape[1]):
                    if not is_patch[t]:
                        continue
                    cdf = UNIFORM if t == 0 else probs_to_cdf(probs[j, t - 1])
                    cdfs.append((t, cdf))
                    enc.encode_symbol(cdf, int(ids_np[j, t]))
                bits = enc.finish()
                ac_total += len(bits)
                streams.append(bits)
                if (i + j) % verify_every == 0:
                    dec = ArithmeticDecoder(bits)
                    ok += all(dec.decode_symbol(cdf) == int(ids_np[j, t]) for t, cdf in cdfs)
                    nver += 1
    return ce_total, np.array(per_tile), (ac_total if do_ac else None), (streams if do_ac else None), ok, nver


def classical_sizes(arr):
    out = {}
    for name, fmt, kw in [("PNG", "PNG", dict(optimize=True)), ("WebP-lossless", "WEBP", dict(lossless=True, quality=100, method=6, exact=True))]:
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format=fmt, **kw)
        out[name] = len(buf.getvalue())
    try:
        with tempfile.TemporaryDirectory() as d:
            src, dst = Path(d) / "in.png", Path(d) / "out.jxl"
            Image.fromarray(arr).save(src, format="PNG")
            subprocess.run(["cjxl", str(src), str(dst), "-d", "0", "-e", "7", "--quiet"], check=True, capture_output=True)
            out["JPEG XL-lossless"] = dst.stat().st_size
    except Exception:
        pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--ckpt", default="checkpoints/prior_full_d64_20ep_best.pt")
    ap.add_argument("--int8", action="store_true", help="per-channel int8 weights (216 KB decoder)")
    ap.add_argument("--ac", action="store_true", help="run the real arithmetic coder and write <image>.bytelm")
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    t0 = time.time()

    m, dec_bytes, ck = load_prior(args.ckpt, args.int8)
    arr, tiles = tile_image(np.array(Image.open(args.image).convert("RGB")))
    H, W, _ = arr.shape
    npx, raw = H * W, H * W * 3
    print(f"{args.image}: {W}x{H} RGB = {raw / 1024:.0f} KB raw | {len(tiles):,} tiles | prior {Path(args.ckpt).name} ({ck['config']}, {m.n_params():,} params, decoder {dec_bytes / 1024:.0f} KB {'int8' if args.int8 else 'f16'}) | {DEVICE}", flush=True)

    ce_bits, per_tile, ac_bits, streams, ok, nver = model_code(m, tiles, args.bs, args.ac)
    ours = (ac_bits if args.ac else ce_bits) / 8
    sizes = classical_sizes(arr)
    print(f"\n{'codec':<20}{'bytes':>12}{'bpp':>8}{'vs PNG':>9}")
    for name, s in sizes.items():
        print(f"{name:<20}{s:>12,}{s * 8 / npx:>8.2f}{s / sizes['PNG']:>9.3f}x")
    print(f"{'byte-LM prior' + (' (AC)' if args.ac else ' (est)'):<20}{int(ours):>12,}{ours * 8 / npx:>8.2f}{ours / sizes['PNG']:>9.3f}x")
    if args.ac:
        print(f"arithmetic coder vs estimate: {(ac_bits / ce_bits - 1) * 100:+.2f}% | round-trip verified on {ok}/{nver} sampled tiles")
        out = Path(args.image).with_suffix(".bytelm")
        payload = bytearray(b"BLM1") + np.array([H, W], np.uint32).tobytes()
        for bits in streams:
            arrb = np.packbits(np.array(bits, np.uint8))
            payload += np.array([len(bits)], np.uint32).tobytes() + arrb.tobytes()
        out.write_bytes(payload)
        print(f"wrote {out} ({out.stat().st_size:,} bytes incl. {len(streams) * 4 + 12:,} bytes of framing)")
    if args.heatmap:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        gh, gw = H // P, W // P
        per_px = np.zeros((len(tiles), P * P), np.float32)
        # redistribute per-tile bits uniformly is uninformative; recompute per-pixel cost quickly
        with torch.no_grad():
            k = 0
            for i in range(0, len(tiles), args.bs):
                ids = as_ids(m, tiles[i:i + args.bs], DEVICE)
                logp = F.log_softmax(m(ids)[:, :-1].float(), dim=-1)
                nll = (-logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) / math.log(2)).cpu().numpy()
                nll = np.concatenate([np.full((len(ids), 1), 8.0, np.float32), nll], 1)
                g = m.grid
                nll = nll.reshape(len(ids), g * g, m.n_ch).sum(2).reshape(len(ids), g, g)
                nll = nll[:, 1:, 1:] if m.l_context else nll
                per_px[k:k + len(ids)] = nll.reshape(len(ids), -1)
                k += len(ids)
        heat = per_px.reshape(gh, gw, P, P).transpose(0, 2, 1, 3).reshape(H, W)
        fig, ax = plt.subplots(1, 2, figsize=(14, 6 * H / W + 1))
        ax[0].imshow(arr); ax[0].set_title(f"{Path(args.image).name}  {ours * 8 / npx:.2f} bpp (PNG {sizes['PNG'] * 8 / npx:.2f})"); ax[0].axis("off")
        im = ax[1].imshow(heat, cmap="magma", vmin=0, vmax=np.percentile(heat, 99)); ax[1].set_title("bits per pixel (R+G+B)"); ax[1].axis("off")
        plt.colorbar(im, ax=ax[1], fraction=0.03)
        hp = Path(args.image).with_name(Path(args.image).stem + "_bits.png")
        fig.tight_layout(); fig.savefig(hp, dpi=100)
        print(f"heatmap: {hp}")
    print(f"\nTotal {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
