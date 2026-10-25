"""Train a shared image prior on pre-cut windows; evaluate Kodak-24 bpp every epoch.

Checkpoints every epoch (last + best), resumes automatically, appends results JSON per epoch.
Device: cuda -> mps -> cpu. Run with python3 -u and tee to a log.

python3 -u train_shared_prior.py --config full --data data/prior_train.npy --kodak data/prior_kodak.npz \
    --epochs 6 --windows-per-epoch 100000 --batch-size 32 --tag full_d64
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from shared_prior_model import CONFIGS, PriorLM

CKPT_DIR = Path("checkpoints")
RESULTS = Path("pile_runs/rope_byte/shared_prior_results.json")


def pick_device(force_cpu=False):
    if force_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def as_ids(model, windows_np, device):
    """(B,29,29,3) uint8 -> (B or 3B, S) long on device, cropping to 28x28 if no l_context, splitting channels if not interleaved."""
    w = torch.from_numpy(windows_np)
    g = model.grid  # windows may carry a larger halo than the model uses: keep the bottom-right grid x grid
    w = w[:, -g:, -g:]
    if not model.interleave:
        w = w.permute(0, 3, 1, 2).reshape(-1, model.grid, model.grid, 1)
    return model.to_sequence(w).to(device)


@torch.no_grad()
def eval_kodak(model, kodak, device, bs):
    model.eval()
    tiles, png_bpp, (H, W) = kodak["tiles"], kodak["png_bpp"], kodak["hw"]
    bpp = []
    for img_tiles in tiles:
        bits = 0.0
        for i in range(0, len(img_tiles), bs):
            ids = as_ids(model, img_tiles[i:i + bs], device)
            bits += model.bits(ids).sum().item()
        bpp.append(bits / (H * W))
    model.train()
    return float(np.mean(bpp)), float(np.mean(png_bpp)), bpp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", choices=CONFIGS, required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--kodak", required=True)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--windows-per-epoch", type=int, default=100000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: stop after N steps")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    device = pick_device(args.cpu)
    print(f"Device: {device}", flush=True)
    torch.manual_seed(0)

    model = PriorLM(args.d_model, args.n_layers, args.n_heads, **CONFIGS[args.config]).to(device)
    print(f"Config {args.config} {CONFIGS[args.config]} | {model.n_params():,} params = {model.n_params() * 2 / 1024:.1f} KB f16 | seq_len {model.seq_len}", flush=True)

    data = np.load(args.data, mmap_mode="r")
    kodak = np.load(args.kodak)
    print(f"Train windows: {len(data):,} | Kodak tiles: {kodak['tiles'].shape}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = args.epochs * (args.windows_per_epoch // args.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(total_steps, 1), pct_start=0.05)

    CKPT_DIR.mkdir(exist_ok=True)
    last_path, best_path = CKPT_DIR / f"prior_{args.tag}_last.pt", CKPT_DIR / f"prior_{args.tag}_best.pt"
    start_epoch, best_bpp = 0, float("inf")
    if last_path.exists():
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        opt.load_state_dict(ck["opt_state"])
        sched.load_state_dict(ck["sched_state"])
        start_epoch, best_bpp = ck["epoch"], ck["best_bpp"]
        print(f"Resumed from epoch {start_epoch}, best Kodak bpp {best_bpp:.3f}", flush=True)

    eval_bs = max(4, (64 * 784) // model.seq_len) if device != "cuda" else 64
    kbpp, png_bpp, _ = eval_kodak(model, kodak, device, eval_bs)
    print(f"Epoch {start_epoch}: Kodak bpp {kbpp:.3f} (PNG {png_bpp:.3f}, ratio {kbpp / png_bpp:.3f})", flush=True)

    use_amp = device == "cuda"
    rng = np.random.default_rng(start_epoch)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        model.train()
        idx = rng.choice(len(data), size=min(args.windows_per_epoch, len(data)), replace=False)
        idx.sort()
        t0, tokens, loss_sum, steps = time.time(), 0, 0.0, 0
        for i in range(0, len(idx), args.batch_size):
            ids = as_ids(model, data[idx[i:i + args.batch_size]], device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model.loss(ids)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < total_steps - 1:
                sched.step()
            tokens += ids.numel()
            loss_sum += loss.item()
            steps += 1
            if steps % 200 == 0:
                print(f"  ep {epoch} step {steps} loss {loss_sum / steps:.4f} ({loss_sum / steps / math.log(2):.3f} bits/tok) {tokens / (time.time() - t0):,.0f} tok/s", flush=True)
            if args.max_steps and steps >= args.max_steps:
                break
        train_time = time.time() - t0
        kbpp, png_bpp, per_img = eval_kodak(model, kodak, device, eval_bs)
        improved = kbpp < best_bpp
        best_bpp = min(best_bpp, kbpp)
        print(f"Epoch {epoch}/{args.epochs}: train {loss_sum / max(steps, 1) / math.log(2):.3f} bits/tok | "
              f"Kodak {kbpp:.3f} bpp (PNG {png_bpp:.3f}, ratio {kbpp / png_bpp:.3f}) | {tokens / train_time:,.0f} tok/s | {train_time:.0f}s{' *' if improved else ''}", flush=True)

        state = dict(model_state=model.state_dict(), opt_state=opt.state_dict(), sched_state=sched.state_dict(),
                     epoch=epoch, best_bpp=best_bpp, config=args.config,
                     arch=dict(d_model=args.d_model, n_layers=args.n_layers, n_heads=args.n_heads), kodak_bpp=kbpp)
        torch.save(state, last_path)
        if improved:
            torch.save(state, best_path)

        res = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        res.setdefault(args.tag, dict(config=args.config, params=model.n_params(), epochs=[]))["epochs"].append(
            dict(epoch=epoch, kodak_bpp=kbpp, png_bpp=png_bpp, per_image=per_img, train_bits_per_tok=loss_sum / max(steps, 1) / math.log(2),
                 tok_per_s=tokens / train_time, seconds=train_time))
        res[args.tag]["best_kodak_bpp"] = best_bpp
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        RESULTS.write_text(json.dumps(res, indent=1))
        if args.max_steps:
            break

    print(f"Done. Best Kodak bpp {best_bpp:.3f} vs PNG {png_bpp:.3f}. Checkpoint: {best_path}", flush=True)


if __name__ == "__main__":
    main()
