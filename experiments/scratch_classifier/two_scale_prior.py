"""Two-scale (coarse-to-fine) coding with the byte-LM prior — the MVE of notes/blursort_polyphase.md §5.

Phase (0,0) of an image (every other pixel) is coded with the raster prior as-is. This script trains and measures the
NEXT step: phase (1,1) (the diagonal odd pixels) coded CONDITIONED on the already-decoded phase (0,0). The model is the
same PriorLM; its sequence is [coarse 29x29 window of phase (0,0) at the same tile position] + [L-halo 29x29 window of
phase (1,1)], with a 2-way level embedding; causal attention lets every target token see the whole coarse prefix; loss and
bits are only on the target patch. Initialised from the frozen raster prior so it only has to learn to use the prefix.

Measured against: the same model with the coarse prefix replaced by PAD (= unconditioned, what blursort_progressive.py
measured as "independent phases"), on Kodak phase (1,1).

  python3 two_scale_prior.py cut                       # data: CLIC pairs -> data/two_scale_train.npy, Kodak -> data/two_scale_kodak.npz
  python3 two_scale_prior.py train --epochs 3 --windows-per-epoch 60000 --tag ts_d64 [--init checkpoints/prior_full_d64_20ep_best.pt]
  python3 two_scale_prior.py eval checkpoints/prior_ts_d64_best.pt
"""
import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from shared_prior_model import PriorLM, PAD, CONFIGS

P, H1 = 28, 1
DATA, CKPT_DIR = Path("data"), Path("checkpoints")


def phases2(arr):
    return arr[0::2, 0::2], arr[1::2, 1::2]  # coarse (0,0), target (1,1)


def padded(a):
    H, W, C = a.shape
    out = np.full((H + H1, W + H1, C), PAD, np.uint8); out[H1:, H1:] = a
    return out


def cut_train(image_dir, n, out):
    files = sorted(p for p in Path(image_dir).rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    per = -(-n // len(files)); rng = np.random.default_rng(0)
    pairs = np.empty((per * len(files), 2, P + H1, P + H1, 3), np.uint8); k = 0; t0 = time.time()
    for i, f in enumerate(files):
        arr = np.array(Image.open(f).convert("RGB"))
        c, t = phases2(arr); c, t = padded(c), padded(t)
        H, W, _ = t.shape
        for _ in range(per):
            r = rng.integers(0, H - P - H1 + 1); q = rng.integers(0, W - P - H1 + 1)
            pairs[k, 0] = c[r:r + P + H1, q:q + P + H1]; pairs[k, 1] = t[r:r + P + H1, q:q + P + H1]; k += 1
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)} images, {k:,} pairs ({time.time()-t0:.0f}s)", flush=True)
    pairs = pairs[:k]; rng.shuffle(pairs); np.save(out, pairs)
    print(f"saved {out}: {pairs.shape} = {pairs.nbytes/1048576:.0f} MB", flush=True)


def cut_kodak(image_dir, out):
    tiles, hw = [], None
    for f in sorted(Path(image_dir).glob("kodim*.png")):
        arr = np.array(Image.open(f).convert("RGB"))
        c, t = phases2(arr)
        H, W = (t.shape[0] // P) * P, (t.shape[1] // P) * P
        c, t = padded(c[:H, :W]), padded(t[:H, :W]); hw = (H, W)
        tiles.append(np.stack([np.stack([c[r:r + P + H1, q:q + P + H1], t[r:r + P + H1, q:q + P + H1]]) for r in range(0, H, P) for q in range(0, W, P)]))
    np.savez(out, tiles=np.stack(tiles), hw=np.array(hw))
    print(f"saved {out}: {len(tiles)} images x {tiles[0].shape} (phase (1,1) {hw[1]}x{hw[0]})", flush=True)


class TwoScalePrior(PriorLM):
    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.token_emb.embedding_dim
        self.level_emb = nn.Embedding(2, d); nn.init.normal_(self.level_emb.weight, std=0.02)
        n = self.seq_len
        self.register_buffer("freqs2", torch.cat([self.freqs, self.freqs], 0), persistent=False)
        self.register_buffer("chan2", torch.cat([self.chan_idx, self.chan_idx]), persistent=False)
        self.register_buffer("level2", torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)]), persistent=False)
        self.register_buffer("is_patch2", torch.cat([torch.zeros(n, dtype=torch.bool), self.is_patch]), persistent=False)

    def forward(self, ids):  # ids: (B, 2*seq_len) = coarse ++ target
        x = self.token_emb(ids) + self.channel_emb(self.chan2) + self.level_emb(self.level2)
        for blk in self.layers:
            x = blk(x, self.freqs2)
        return self.lm_head(self.norm(x))

    def bits(self, ids):
        logp = F.log_softmax(self(ids)[:, :-1].float(), dim=-1)
        nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) / math.log(2)
        return (nll * self.is_patch2[1:].to(nll.dtype)).sum(1)

    def loss(self, ids):
        logits = self(ids)[:, :-1].float(); tgt = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        m = self.is_patch2[1:].to(nll.dtype)
        return (nll * m).sum() / (m.sum() * tgt.shape[0])


def to_ids(pairs_np, device, blank_coarse=False):
    """(B, 2, 29, 29, 3) uint8 -> (B, 2*2523) long: coarse tokens then target tokens."""
    p = torch.from_numpy(pairs_np).long()
    if blank_coarse:
        p = p.clone(); p[:, 0] = PAD
    return p.reshape(p.shape[0], 2, -1).reshape(p.shape[0], -1).to(device)


def build(init, device, d_model=64, n_layers=4, n_heads=4):
    m = TwoScalePrior(d_model=d_model, n_layers=n_layers, n_heads=n_heads, **CONFIGS["full"])
    if init:
        ck = torch.load(init, map_location="cpu", weights_only=False)
        missing, unexpected = m.load_state_dict(ck["model_state"], strict=False)
        print(f"init from {init}: missing {missing}, unexpected {unexpected}", flush=True)
    return m.to(device)


@torch.no_grad()
def eval_frozen_unconditioned(init, kodak, device, bs):
    """the fair unconditioned baseline: the frozen raster prior coding the phase (1,1) windows on their own."""
    ck = torch.load(init, map_location="cpu", weights_only=False)
    f = PriorLM(d_model=64, n_layers=4, n_heads=4, **CONFIGS["full"]); f.load_state_dict(ck["model_state"]); f = f.to(device).eval()
    H, W = kodak["hw"]; out = []
    for img in kodak["tiles"]:
        bits = 0.0
        for i in range(0, len(img), bs):
            w = torch.from_numpy(np.ascontiguousarray(img[i:i + bs, 1])).long().reshape(min(bs, len(img) - i), -1).to(device)
            bits += f.bits(w).sum().item()
        out.append(bits / (H * W))
    return float(np.mean(out))


@torch.no_grad()
def eval_kodak(m, kodak, device, bs, blank=False):
    m.eval(); H, W = kodak["hw"]; out = []
    for img in kodak["tiles"]:
        bits = sum(m.bits(to_ids(img[i:i + bs], device, blank)).sum().item() for i in range(0, len(img), bs))
        out.append(bits / (H * W))
    m.train(); return float(np.mean(out)), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["cut", "train", "eval"]); ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--n", type=int, default=60000); ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--windows-per-epoch", type=int, default=60000); ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--tag", default="ts_d64")
    ap.add_argument("--init", default="checkpoints/prior_full_d64_20ep_best.pt"); ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--device", default=None); ap.add_argument("--eval-images", type=int, default=24)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    if a.cmd == "cut":
        cut_train(DATA / "clic2020", a.n, DATA / "two_scale_train.npy"); cut_kodak(DATA / "kodak", DATA / "two_scale_kodak.npz"); return
    kz = np.load(DATA / "two_scale_kodak.npz"); kodak = {"tiles": kz["tiles"][:a.eval_images], "hw": tuple(kz["hw"])}
    ebs = 8
    if a.cmd == "eval":
        m = build(None, device); ck = torch.load(a.ckpt, map_location=device, weights_only=False); m.load_state_dict(ck["model_state"])
        cond, per = eval_kodak(m, kodak, device, ebs); unc, _ = eval_kodak(m, kodak, device, ebs, blank=True)
        print(f"Kodak phase (1,1): conditioned {cond:.3f} bpp | coarse blanked {unc:.3f} | ratio {cond/unc:.3f}"); return
    m = build(a.init, device)
    unc_frozen = eval_frozen_unconditioned(a.init, kodak, device, ebs)
    print(f"Frozen raster prior on phase (1,1) alone (unconditioned baseline): {unc_frozen:.3f} bpp", flush=True)
    data = np.load(DATA / "two_scale_train.npy", mmap_mode="r")
    print(f"{m.n_params():,} params | seq {2*m.seq_len} | train pairs {len(data):,} | device {device}", flush=True)
    cond0, _ = eval_kodak(m, kodak, device, ebs); unc0, _ = eval_kodak(m, kodak, device, ebs, blank=True)
    print(f"Epoch 0: Kodak phase(1,1) conditioned {cond0:.3f} bpp | blanked {unc0:.3f}", flush=True)
    opt = torch.optim.AdamW(m.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * (a.windows_per_epoch // a.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(total, 1), pct_start=0.05)
    best = float("inf"); rng = np.random.default_rng(0); use_amp = device == "cuda"
    for ep in range(1, a.epochs + 1):
        idx = np.sort(rng.choice(len(data), size=min(a.windows_per_epoch, len(data)), replace=False))
        t0, tok, ls, steps = time.time(), 0, 0.0, 0
        for i in range(0, len(idx), a.batch_size):
            ids = to_ids(data[idx[i:i + a.batch_size]], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = m.loss(ids)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            if sched.last_epoch < total - 1:
                sched.step()
            tok += ids.numel(); ls += loss.item(); steps += 1
            if steps % 100 == 0:
                print(f"  ep {ep} step {steps} loss {ls/steps/math.log(2):.3f} bits/tok {tok/(time.time()-t0):,.0f} tok/s", flush=True)
            if a.max_steps and steps >= a.max_steps:
                break
        cond, per = eval_kodak(m, kodak, device, ebs); unc, _ = eval_kodak(m, kodak, device, ebs, blank=True)
        print(f"Epoch {ep}/{a.epochs}: train {ls/max(steps,1)/math.log(2):.3f} bits/tok | Kodak phase(1,1) conditioned {cond:.3f} bpp | vs frozen unconditioned {unc_frozen:.3f} ({cond/unc_frozen:.3f}x) | blanked {unc:.3f} | {tok/(time.time()-t0):,.0f} tok/s | {time.time()-t0:.0f}s{' *' if cond < best else ''}", flush=True)
        state = dict(model_state=m.state_dict(), epoch=ep, kodak_cond=cond, kodak_blank=unc, kodak_unc_frozen=unc_frozen, per_image=per, config="full", arch=dict(d_model=64, n_layers=4, n_heads=4))
        torch.save(state, CKPT_DIR / f"prior_{a.tag}_last.pt")
        if cond < best:
            best = cond; torch.save(state, CKPT_DIR / f"prior_{a.tag}_best.pt")


if __name__ == "__main__":
    main()
