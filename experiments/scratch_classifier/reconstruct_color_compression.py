"""Train a tiny shared compression model and reconstruct the image.

Trains model D (d32, 2L shared across R/G/B) then decodes via
teacher-forced argmax: at each pixel position, the model's most
likely prediction. This shows what the model "learned" about the image.
"""

import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_rope_byte_2d import RoPE2DByteLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 150
PHOTO_DIR = "pile_runs/rope_byte/photo_compressor"


def make_patches(arr, patch_h=28, patch_w=28):
    H, W = arr.shape
    pH, pW = H // patch_h, W // patch_w
    patches = []
    for r in range(pH):
        for c in range(pW):
            patch = arr[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
            patches.append(torch.tensor(patch.flatten(), dtype=torch.long))
    return patches


def reconstruct_channel(model, patches, patch_h=28, patch_w=28, grid_h=25, grid_w=25):
    """Reconstruct a channel via teacher-forced argmax decoding."""
    model.eval()
    recon = np.zeros((grid_h * patch_h, grid_w * patch_w), dtype=np.uint8)
    idx = 0
    with torch.no_grad():
        for r in range(grid_h):
            for c in range(grid_w):
                ids = patches[idx].unsqueeze(0).to(DEVICE)
                logits = model(ids)
                preds = logits[:, :-1].argmax(dim=-1).squeeze(0).cpu().numpy()
                # First pixel has no prediction (no context), use original
                patch_pixels = np.concatenate([[ids[0, 0].item()], preds])
                recon[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w] = (
                    patch_pixels.reshape(patch_h, patch_w)
                )
                idx += 1
    return recon


def main():
    print("=== RECONSTRUCT COLOR COMPRESSION (Model D: d32, 2L shared) ===", flush=True)
    t0 = time.time()

    img = Image.open(f"{PHOTO_DIR}/color_700_original.png")
    arr = np.array(img)
    H, W, C = arr.shape
    print(f"Image: {H}x{W}x{C}, Device: {DEVICE}", flush=True)

    channels = {}
    for ch, name in enumerate(["R", "G", "B"]):
        channels[name] = make_patches(arr[:, :, ch])

    all_patches = []
    for ch_name in ["R", "G", "B"]:
        all_patches.extend(channels[ch_name])

    n_patches_per_ch = len(channels["R"])
    grid_h = H // 28
    grid_w = W // 28
    print(f"Patches: {n_patches_per_ch} per channel, {len(all_patches)} total", flush=True)
    print(f"Grid: {grid_h}x{grid_w}", flush=True)

    # Train model D
    model = RoPE2DByteLM(
        vocab_size=256, d_model=32, n_heads=2, n_layers=2,
        height=28, width=28, bidirectional=False,
    ).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params = {n_params * 2:,} bytes (f16)", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    best_ce = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for patch in all_patches:
            ids = patch.unsqueeze(0).to(DEVICE)
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 256),
                ids[:, 1:].reshape(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        ce_sum = 0.0
        with torch.no_grad():
            for patch in all_patches:
                ids = patch.unsqueeze(0).to(DEVICE)
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, 256),
                    ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
                ce_sum += loss.item()

        if ce_sum < best_ce:
            best_ce = ce_sum
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            ckpt_path = f"{PHOTO_DIR}/color_700_shared_D.pt"
            torch.save({
                "model_state": best_state,
                "epoch": epoch,
                "ce_sum": best_ce,
                "n_params": n_params,
            }, ckpt_path)

        if epoch % 25 == 0 or epoch == 1:
            coded_kb = ce_sum / math.log(2) / 8 / 1024
            elapsed = time.time() - t0
            print(f"  ep {epoch:3d}: coded={coded_kb:.1f}KB ({elapsed:.0f}s)", flush=True)

    # Load best weights
    model.load_state_dict(best_state)
    model.to(DEVICE)

    coded_bits = best_ce / math.log(2)
    model_bytes = n_params * 2
    total_bytes = model_bytes + coded_bits / 8
    print(f"\nBest: model={model_bytes / 1024:.1f}KB + coded={coded_bits / 8 / 1024:.1f}KB = {total_bytes / 1024:.1f}KB", flush=True)

    png_bytes = os.path.getsize(f"{PHOTO_DIR}/color_700_original.png")
    ratio = total_bytes / png_bytes
    print(f"PNG: {png_bytes / 1024:.1f}KB — model is {ratio:.2f}x {'smaller' if ratio < 1 else 'bigger'}", flush=True)

    # Reconstruct each channel
    print("\nReconstructing...", flush=True)
    recon_channels = []
    for ch, name in enumerate(["R", "G", "B"]):
        recon = reconstruct_channel(model, channels[name], grid_h=grid_h, grid_w=grid_w)
        recon_channels.append(recon)
        print(f"  {name} channel done", flush=True)

    # Assemble RGB image
    recon_rgb = np.stack(recon_channels, axis=-1)
    recon_img = Image.fromarray(recon_rgb)

    out_path = f"{PHOTO_DIR}/color_700_recon_D.png"
    recon_img.save(out_path)
    print(f"\nSaved reconstruction: {out_path}", flush=True)

    # Also save the original cropped to same size for comparison
    crop_h = grid_h * 28
    crop_w = grid_w * 28
    original_crop = arr[:crop_h, :crop_w]
    orig_path = f"{PHOTO_DIR}/color_700_original_cropped.png"
    Image.fromarray(original_crop).save(orig_path)
    print(f"Saved original (cropped): {orig_path}", flush=True)

    # Compute PSNR
    mse = np.mean((recon_rgb.astype(float) - original_crop.astype(float)) ** 2)
    if mse > 0:
        psnr = 10 * math.log10(255 ** 2 / mse)
    else:
        psnr = float("inf")
    print(f"PSNR: {psnr:.1f} dB", flush=True)

    print(f"\nTotal time: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
