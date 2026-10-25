"""A real codec around the shared byte-LM prior: encode an image to a .bytelm file, decode it back, verify bit-exact.

  python3 bytelm_codec.py encode photo.png photo.bytelm [--int8] [--ckpt ...]
  python3 bytelm_codec.py decode photo.bytelm photo_decoded.png [--int8] [--ckpt ...]
  python3 bytelm_codec.py roundtrip photo.png [--int8]        # encode, decode, compare, report sizes

Encoder and decoder run the SAME incremental (KV-cached) model step, token by token, so their probability
tables are bit-identical on the same device. Tiles are processed in wavefront order (tile row + col = d), so
the one-pixel neighbour context of every tile has already been decoded. Image borders use PAD=128.

File format (BLM1): magic | H,W (uint32) | per tile: bit length (uint32) + packed arithmetic-coded bits.
Decode speed is bounded by sequential model steps: (#wavefronts) x 2523 steps, batched over the tiles of a wavefront.
"""

import argparse
import hashlib
import math
import struct
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from compress_with_prior import classical_sizes, load_prior
from compression_demo import ArithmeticDecoder, ArithmeticEncoder

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
P, PAD, MAGIC = 28, 128, b"BLM1"


def probs_to_cdf(p, scale=1 << 16):
    c = np.maximum((p * scale).astype(np.int64), 1)
    c[c.argmax()] += scale - c.sum()
    cdf = np.zeros(257, dtype=np.int64)
    cdf[1:] = np.cumsum(c)
    return cdf


class Incremental:
    """Token-by-token forward of PriorLM with per-layer KV caches. Float32 throughout."""

    def __init__(self, m):
        self.m = m.float()
        self.freqs = m.freqs  # (S, hd/2) complex
        self.grid, self.n_ch, self.S = m.grid, m.n_ch, m.seq_len
        self.is_patch = m.is_patch.cpu().numpy()

    def start(self, B):
        self.cache = [(None, None) for _ in self.m.layers]
        self.B = B

    def rope(self, x, t):  # x: (B, H, 1, hd)
        xc = torch.view_as_complex(x.reshape(x.shape[0], x.shape[1], 1, -1, 2))
        return torch.view_as_real(xc * self.freqs[t]).reshape(x.shape)

    @torch.no_grad()
    def step(self, tok, t):
        """tok: (B,) long at position t. Returns logits (B, 256) predicting position t+1."""
        m = self.m
        x = m.token_emb(tok).unsqueeze(1)  # (B,1,d)
        if m.interleave:
            x = x + m.channel_emb(m.chan_idx[t])
        new_cache = []
        for (K, V), blk in zip(self.cache, m.layers):
            h = blk.ln1(x)
            q, k, v = blk.qkv(h).reshape(x.shape[0], 1, 3, blk.n_heads, blk.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
            q, k = self.rope(q, t), self.rope(k, t)
            K = k if K is None else torch.cat([K, k], 2)
            V = v if V is None else torch.cat([V, v], 2)
            att = torch.softmax((q @ K.transpose(-2, -1)) * blk.head_dim ** -0.5, dim=-1)
            x = x + blk.out((att @ V).transpose(1, 2).reshape(x.shape[0], 1, -1))
            x = x + blk.ff(blk.ln2(x))
            new_cache.append((K, V))
        self.cache = new_cache
        return m.lm_head(m.norm(x))[:, 0]


def tile_sequence_known(img_pad, tr, tc, grid, n_ch):
    """The full token sequence of a tile from a (padded) image, as (grid*grid*n_ch,) — used by the encoder."""
    w = img_pad[tr * P:tr * P + grid, tc * P:tc * P + grid]  # (grid, grid, 3)
    if n_ch == 1:
        raise NotImplementedError("codec supports the interleaved-channel configs (chan/full)")
    return w.reshape(-1)


def wavefronts(gh, gw):
    for d in range(gh + gw - 1):
        yield [(tr, d - tr) for tr in range(max(0, d - gw + 1), min(gh, d + 1))]


def encode(arr, inc):
    H, W, _ = arr.shape
    gh, gw = H // P, W // P
    grid, S = inc.grid, inc.S
    pad = np.full((H + 1, W + 1, 3), PAD, np.uint8)
    pad[1:, 1:] = arr
    streams = {}
    t0 = time.time()
    for wf in wavefronts(gh, gw):
        seqs = np.stack([tile_sequence_known(pad, tr, tc, grid, inc.n_ch) for tr, tc in wf]).astype(np.int64)  # (B,S)
        B = len(wf)
        inc.start(B)
        encs = [ArithmeticEncoder() for _ in range(B)]
        toks = torch.tensor(seqs[:, 0], device=DEVICE)
        for t in range(S - 1):
            logits = inc.step(toks, t)
            nxt = seqs[:, t + 1]
            if inc.is_patch[t + 1]:
                probs = torch.softmax(logits, -1).cpu().numpy()
                for j in range(B):
                    encs[j].encode_symbol(probs_to_cdf(probs[j]), int(nxt[j]))
            toks = torch.tensor(nxt, device=DEVICE)
        for j, (tr, tc) in enumerate(wf):
            streams[(tr, tc)] = encs[j].finish()
    print(f"  encoded {gh * gw} tiles in {time.time() - t0:.0f}s", flush=True)
    return streams


def write_file(path, H, W, streams, gh, gw):
    out = bytearray(MAGIC + struct.pack("<II", H, W))
    for tr in range(gh):
        for tc in range(gw):
            bits = streams[(tr, tc)]
            out += struct.pack("<I", len(bits)) + np.packbits(np.array(bits, np.uint8)).tobytes()
    Path(path).write_bytes(out)
    return len(out)


def read_file(path):
    data = Path(path).read_bytes()
    assert data[:4] == MAGIC, "not a BLM1 file"
    H, W = struct.unpack("<II", data[4:12])
    gh, gw = H // P, W // P
    pos, streams = 12, {}
    for tr in range(gh):
        for tc in range(gw):
            n = struct.unpack("<I", data[pos:pos + 4])[0]
            nbytes = (n + 7) // 8
            bits = np.unpackbits(np.frombuffer(data[pos + 4:pos + 4 + nbytes], np.uint8))[:n].tolist()
            streams[(tr, tc)] = bits
            pos += 4 + nbytes
    return H, W, streams


def decode(H, W, streams, inc):
    gh, gw = H // P, W // P
    grid, S = inc.grid, inc.S
    pad = np.full((H + 1, W + 1, 3), PAD, np.uint8)
    t0 = time.time()
    for wf in wavefronts(gh, gw):
        B = len(wf)
        # known context tokens come from already-decoded neighbours (row 0 / col 0 of each tile window)
        win = np.stack([pad[tr * P:tr * P + grid, tc * P:tc * P + grid].copy() for tr, tc in wf])  # (B,grid,grid,3)
        seqs = win.reshape(B, -1).astype(np.int64)  # context positions correct; patch positions to be filled
        decs = [ArithmeticDecoder(streams[(tr, tc)]) for tr, tc in wf]
        inc.start(B)
        toks = torch.tensor(seqs[:, 0], device=DEVICE)
        for t in range(S - 1):
            logits = inc.step(toks, t)
            if inc.is_patch[t + 1]:
                probs = torch.softmax(logits, -1).cpu().numpy()
                for j in range(B):
                    seqs[j, t + 1] = decs[j].decode_symbol(probs_to_cdf(probs[j]))
            toks = torch.tensor(seqs[:, t + 1], device=DEVICE)
        win = seqs.reshape(B, grid, grid, 3).astype(np.uint8)
        for j, (tr, tc) in enumerate(wf):
            pad[tr * P + 1:tr * P + grid, tc * P + 1:tc * P + grid] = win[j, 1:, 1:]
    print(f"  decoded {gh * gw} tiles in {time.time() - t0:.0f}s", flush=True)
    return pad[1:, 1:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["encode", "decode", "roundtrip"])
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--ckpt", default="checkpoints/prior_full_d64_20ep_best.pt")
    ap.add_argument("--int8", action="store_true")
    args = ap.parse_args()
    m, dec_bytes, ck = load_prior(args.ckpt, args.int8)
    assert m.interleave, "codec needs an interleaved-channel prior (chan/full)"
    inc = Incremental(m)
    print(f"prior {Path(args.ckpt).name}: {m.n_params():,} params, decoder {dec_bytes / 1024:.0f} KB ({'int8' if args.int8 else 'f16'}), device {DEVICE}", flush=True)

    if args.cmd in ("encode", "roundtrip"):
        arr = np.array(Image.open(args.src).convert("RGB"))
        H, W = (arr.shape[0] // P) * P, (arr.shape[1] // P) * P
        arr = arr[:H, :W]
        out = args.dst or str(Path(args.src).with_suffix(".bytelm"))
        print(f"encode {args.src} ({W}x{H}) -> {out}", flush=True)
        streams = encode(arr, inc)
        nbytes = write_file(out, H, W, streams, H // P, W // P)
        sizes = classical_sizes(arr)
        print(f"  .bytelm {nbytes:,} bytes = {nbytes * 8 / (H * W):.2f} bpp | PNG {sizes['PNG']:,} ({nbytes / sizes['PNG']:.3f}x) | WebP {sizes['WebP-lossless']:,} ({nbytes / sizes['WebP-lossless']:.3f}x)"
              + (f" | JXL {sizes['JPEG XL-lossless']:,} ({nbytes / sizes['JPEG XL-lossless']:.3f}x)" if "JPEG XL-lossless" in sizes else ""), flush=True)
        if args.cmd == "encode":
            return
        src_arr, blm = arr, out
        dst = str(Path(args.src).with_name(Path(args.src).stem + "_decoded.png"))
    else:
        blm, dst = args.src, args.dst or str(Path(args.src).with_suffix(".decoded.png"))
        src_arr = None

    H, W, streams = read_file(blm)
    print(f"decode {blm} ({W}x{H}) -> {dst}", flush=True)
    rec = decode(H, W, streams, inc)
    Image.fromarray(rec).save(dst)
    if src_arr is not None:
        same = np.array_equal(rec, src_arr)
        print(f"  round-trip {'BIT-EXACT' if same else 'MISMATCH'} | sha256 src {hashlib.sha256(src_arr.tobytes()).hexdigest()[:16]} dec {hashlib.sha256(rec.tobytes()).hexdigest()[:16]}", flush=True)


if __name__ == "__main__":
    main()
