"""Test model variants for color photo compression.

Goal: find the best model_size + coded_bits tradeoff to beat PNG.

Variants:
  A) Current: 3 × d64 4L models (215K params each, 1.29 MB total f16)
  B) Tiny: 3 × d32 2L models (~33K params each, ~198 KB total f16)
  C) Shared: 1 × d64 4L model with channel token (215K params, 430 KB f16)
  D) Shared tiny: 1 × d32 2L model with channel token (~33K, 66 KB f16)
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from train_rope_byte_2d import RoPE2DByteLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 150
PHOTO_DIR = "pile_runs/rope_byte/photo_compressor"


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def make_patches(arr, patch_h=28, patch_w=28):
    H, W = arr.shape
    pH, pW = H // patch_h, W // patch_w
    patches = []
    for r in range(pH):
        for c in range(pW):
            patch = arr[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
            patches.append(torch.tensor(patch.flatten(), dtype=torch.long))
    return patches


def train_model(model, patches, epochs, label="", save_path=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best_ce_sum = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for patch in patches:
            ids = patch.unsqueeze(0).to(DEVICE)
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, model.vocab_size),
                ids[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Eval on ALL patches (this is compression, not generalization)
        model.eval()
        ce_sum = 0.0
        with torch.no_grad():
            for patch in patches:
                ids = patch.unsqueeze(0).to(DEVICE)
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, model.vocab_size),
                    ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
                ce_sum += loss.item()

        if ce_sum < best_ce_sum:
            best_ce_sum = ce_sum
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if save_path:
                torch.save({
                    "model_state": best_state,
                    "epoch": epoch,
                    "ce_sum": best_ce_sum,
                    "n_params": sum(p.numel() for p in model.parameters()),
                }, save_path)

        if epoch % 25 == 0 or epoch == 1:
            coded_bits = ce_sum / math.log(2)
            avg_bpb = coded_bits / (len(patches) * 783)
            print(f"  {label} ep {epoch:3d}: coded={coded_bits / 8 / 1024:.1f}KB avg={avg_bpb:.3f} bits/byte", flush=True)

    return best_ce_sum, best_state


def eval_compression(model, patches, model_bits, label):
    model.eval()
    ce_sum = 0.0
    with torch.no_grad():
        for patch in patches:
            ids = patch.unsqueeze(0).to(DEVICE)
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, model.vocab_size),
                ids[:, 1:].reshape(-1),
                reduction="sum",
            )
            ce_sum += loss.item()

    coded_bits = ce_sum / math.log(2)
    total_bits = model_bits + coded_bits
    return coded_bits, total_bits


def main():
    print("=== COLOR COMPRESSION: MODEL SIZE VARIANTS ===", flush=True)
    t0 = time.time()

    img = Image.open(f"{PHOTO_DIR}/color_700_original.png")
    arr = np.array(img)
    H, W, C = arr.shape
    total_pixels = H * W

    png_bytes = os.path.getsize(f"{PHOTO_DIR}/color_700_original.png")
    png_bpp = (png_bytes * 8) / total_pixels
    print(f"Image: {H}x{W}x{C}, PNG: {png_bytes:,} bytes = {png_bpp:.2f} bpp", flush=True)
    print(f"Epochs: {EPOCHS}, Device: {DEVICE}", flush=True)

    channels = {}
    for ch, name in enumerate(["R", "G", "B"]):
        channels[name] = make_patches(arr[:, :, ch])
    n_patches = len(channels["R"])
    print(f"Patches per channel: {n_patches}", flush=True)

    results = {}

    # B already ran: hardcode results
    results["B_tiny_separate"] = {
        "model_bytes": 3 * 33408 * 2,  # 3 models f16
        "coded_bytes": (191 + 190 + 199) * 1024,  # from prior run
        "total_bytes": 774.3 * 1024,
        "bpp": (774.3 * 1024 * 8) / total_pixels,
    }
    print(f"\n--- B: TINY SEPARATE (already done) → 774 KB ---", flush=True)

    # For shared models: train each channel separately with the SAME model
    # (no prefix token — just train one model on all 3 channels' patches)
    # This works because the model learns byte prediction patterns common to all channels

    # --- Variant D: Tiny shared model (d32, 2L) ---
    print(f"\n--- D: TINY SHARED (d32, 2L) ---", flush=True)
    all_patches = []
    for ch_name, patches in channels.items():
        all_patches.extend(patches)

    model_d = RoPE2DByteLM(
        vocab_size=256, d_model=32, n_heads=2, n_layers=2,
        height=28, width=28, bidirectional=False,
    ).to(DEVICE)
    n_params_d = count_params(model_d)
    model_bits_d = n_params_d * 16
    print(f"  Model: {n_params_d:,} params = {model_bits_d // 8:,} bytes (f16)", flush=True)
    print(f"  Training on {len(all_patches)} patches (3 channels pooled)", flush=True)

    best_ce_d, best_state_d = train_model(model_d, all_patches, EPOCHS, label="D",
                                          save_path=f"{PHOTO_DIR}/color_700_shared_D.pt")
    coded_bits_d = best_ce_d / math.log(2)
    total_bits_d = model_bits_d + coded_bits_d
    results["D_tiny_shared"] = {
        "model_bytes": model_bits_d / 8,
        "coded_bytes": coded_bits_d / 8,
        "total_bytes": total_bits_d / 8,
        "bpp": total_bits_d / total_pixels,
    }
    elapsed_d = time.time() - t0
    print(f"  D total: {total_bits_d / 8 / 1024:.1f} KB, time: {elapsed_d:.0f}s", flush=True)

    # --- Variant C: Shared full model (d64, 4L) ---
    print(f"\n--- C: SHARED FULL (d64, 4L) ---", flush=True)
    model_c = RoPE2DByteLM(
        vocab_size=256, d_model=64, n_heads=4, n_layers=4,
        height=28, width=28, bidirectional=False,
    ).to(DEVICE)
    n_params_c = count_params(model_c)
    model_bits_c = n_params_c * 16
    print(f"  Model: {n_params_c:,} params = {model_bits_c // 8:,} bytes (f16)", flush=True)
    print(f"  Training on {len(all_patches)} patches (3 channels pooled)", flush=True)

    best_ce_c, best_state_c = train_model(model_c, all_patches, EPOCHS, label="C",
                                          save_path=f"{PHOTO_DIR}/color_700_shared_C.pt")
    coded_bits_c = best_ce_c / math.log(2)
    total_bits_c = model_bits_c + coded_bits_c
    results["C_shared_full"] = {
        "model_bytes": model_bits_c / 8,
        "coded_bytes": coded_bits_c / 8,
        "total_bytes": total_bits_c / 8,
        "bpp": total_bits_c / total_pixels,
    }
    elapsed_c = time.time() - t0
    print(f"  C total: {total_bits_c / 8 / 1024:.1f} KB, time: {elapsed_c:.0f}s", flush=True)

    # --- Summary ---
    print(f"\n{'='*70}", flush=True)
    print(f"RESULTS — 700×700 color photo compression", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"{'Variant':<22} {'Model':<12} {'Coded':<12} {'Total':<12} {'BPP':<8} {'vs PNG'}", flush=True)
    for name, r in sorted(results.items()):
        ratio = r["total_bytes"] / png_bytes
        marker = "WIN!" if r["total_bytes"] < png_bytes else f"{ratio:.1f}x bigger"
        print(
            f"{name:<22} {r['model_bytes']/1024:>8.1f} KB  "
            f"{r['coded_bytes']/1024:>8.1f} KB  "
            f"{r['total_bytes']/1024:>8.1f} KB  "
            f"{r['bpp']:>6.2f}  {marker}",
            flush=True,
        )
    print(f"{'PNG':<22} {'':>12} {'':>12} {png_bytes/1024:>8.1f} KB  {png_bpp:>6.2f}", flush=True)
    print(f"\nTotal time: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
