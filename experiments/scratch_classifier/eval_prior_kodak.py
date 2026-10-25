"""Per-image Kodak-24 results for a trained shared prior: CE bpp (f16 and int8 weights), real arithmetic
coding on a subset of images (decode-verified), and the bytes-vs-PNG table.

Usage: python3 eval_prior_kodak.py checkpoints/prior_full_d64_best.pt [n_ac_images=3]
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from compression_demo import ArithmeticDecoder, ArithmeticEncoder
from shared_prior_model import CONFIGS, PriorLM
from train_shared_prior import as_ids

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/prior_full_d64_best.pt"
N_AC = int(sys.argv[2]) if len(sys.argv) > 2 else 3
OUT = Path("pile_runs/rope_byte/kodak_prior_results.json")
UNIFORM = np.arange(257, dtype=np.int64) * 256


def probs_to_cdf(p, scale=1 << 16):
    c = np.maximum((p * scale).astype(np.int64), 1)
    c[c.argmax()] += scale - c.sum()
    cdf = np.zeros(257, dtype=np.int64)
    cdf[1:] = np.cumsum(c)
    return cdf


def load(ckpt_path, int8=False):
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = PriorLM(**ck["arch"], **CONFIGS[ck["config"]])
    state = ck["model_state"]
    nbytes = 0
    if int8:
        q = {}
        for k, v in state.items():
            if k in {n for n, _ in m.named_parameters()}:
                # per-output-channel symmetric int8 (one fp16 scale per row); 1-D tensors keep a single scale
                s = (v.abs().amax(dim=tuple(range(1, v.dim())), keepdim=True) / 127.0).clamp(min=1e-8) if v.dim() > 1 else (v.abs().max() / 127.0).clamp(min=1e-8)
                q[k] = (torch.round(v / s).clamp(-127, 127) * s).to(v.dtype)
                nbytes += v.numel() + 2 * (v.shape[0] if v.dim() > 1 else 1)
            else:
                q[k] = v
        state = q
    else:
        nbytes = sum(v.numel() for k, v in state.items() if k in {n for n, _ in m.named_parameters()}) * 2
    m.load_state_dict(state)
    return m.to(DEVICE).eval(), ck, nbytes


@torch.no_grad()
def image_bits(m, tiles, bs):
    total = 0.0
    for i in range(0, len(tiles), bs):
        total += m.bits(as_ids(m, tiles[i:i + bs], DEVICE)).sum().item()
    return total


@torch.no_grad()
def image_ac_bits(m, tiles, bs, verify=2):
    """Real arithmetic coding of every patch token of one image; decode-verify `verify` tiles."""
    bits_total, ok = 0, 0
    is_patch = m.is_patch.cpu().numpy()
    for i in range(0, len(tiles), bs):
        ids = as_ids(m, tiles[i:i + bs], DEVICE)
        probs = F.softmax(m(ids).float(), dim=-1).cpu().numpy()
        ids_np = ids.cpu().numpy()
        for j in range(len(ids_np)):
            enc = ArithmeticEncoder()
            cdfs = []
            for t in range(ids_np.shape[1]):
                if not is_patch[t]:
                    continue
                cdf = UNIFORM if t == 0 else probs_to_cdf(probs[j, t - 1])
                cdfs.append((t, cdf))
                enc.encode_symbol(cdf, int(ids_np[j, t]))
            bits = enc.finish()
            bits_total += len(bits)
            if i == 0 and j < verify:
                dec = ArithmeticDecoder(bits)
                ok += all(dec.decode_symbol(cdf) == int(ids_np[j, t]) for t, cdf in cdfs)
    return bits_total, ok


def main():
    t0 = time.time()
    kodak = np.load("data/prior_kodak.npz")
    tiles, png_bpp, names, (H, W) = kodak["tiles"], kodak["png_bpp"], kodak["names"], kodak["hw"]
    npx = H * W
    m16, ck, b16 = load(CKPT)
    m8, _, b8 = load(CKPT, int8=True)
    bs = 16 if m16.seq_len > 1000 else 64
    print(f"{Path(CKPT).name}: config {ck['config']} {ck['arch']} | f16 {b16 / 1024:.1f} KB, int8 {b8 / 1024:.1f} KB | {len(tiles)} Kodak images\n", flush=True)

    rows = []
    for k in range(len(tiles)):
        ce16 = image_bits(m16, tiles[k], bs) / npx
        ce8 = image_bits(m8, tiles[k], bs) / npx
        row = dict(image=str(names[k]), png_bpp=float(png_bpp[k]), ce_f16_bpp=ce16, ce_int8_bpp=ce8)
        if k < N_AC:
            ac_bits, ok = image_ac_bits(m8, tiles[k], bs)
            row.update(ac_int8_bpp=ac_bits / npx, decode_ok=int(ok))
        rows.append(row)
        extra = f" | AC {row['ac_int8_bpp']:.3f} (verify {row['decode_ok']}/2)" if "ac_int8_bpp" in row else ""
        print(f"{row['image']}: PNG {row['png_bpp']:6.2f} | f16 {ce16:6.3f} | int8 {ce8:6.3f} ({ce8 / row['png_bpp']:.3f}x){extra}  ({time.time() - t0:.0f}s)", flush=True)

    mean = lambda key: float(np.mean([r[key] for r in rows if key in r]))
    wins = int(sum(r["ce_int8_bpp"] < r["png_bpp"] for r in rows))
    ac_rows = [r for r in rows if "ac_int8_bpp" in r]
    ac_vs_ce = float(np.mean([r["ac_int8_bpp"] / r["ce_int8_bpp"] - 1 for r in ac_rows])) if ac_rows else float("nan")
    print(f"\nMEAN  PNG {mean('png_bpp'):.3f} | f16 {mean('ce_f16_bpp'):.3f} ({mean('ce_f16_bpp') / mean('png_bpp'):.3f}x) | "
          f"int8 {mean('ce_int8_bpp'):.3f} ({mean('ce_int8_bpp') / mean('png_bpp'):.3f}x) | beats PNG on {wins}/{len(rows)} | "
          f"AC vs CE on {len(ac_rows)} images: {ac_vs_ce:+.2%} | decoder {b8 / 1024:.0f} KB int8")
    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev[Path(CKPT).stem] = dict(ckpt=CKPT, f16_bytes=b16, int8_bytes=b8, mean_png=mean("png_bpp"), mean_f16=mean("ce_f16_bpp"),
                                 mean_int8=mean("ce_int8_bpp"), wins=wins, ac_vs_ce=ac_vs_ce, per_image=rows)
    OUT.write_text(json.dumps(prev, indent=1))
    print(f"Saved {OUT}. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
