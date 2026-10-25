"""BlurSort vs raster pixel ordering for byte-level image compression.

Hypothesis: BlurSort creates a coarse-to-fine ordering that's more
predictable for an autoregressive model than raster scan order.

This is experiment 1 from the BlurSort compression research log:
just reorder pixels, measure BPP change. No model changes needed.
"""

import math
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_rope_byte_2d import RoPE2DByteLM

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
EPOCHS = 50
MAX_PATCHES = 500  # subsample for speed; full image has 15K+
PHOTO_DIR = "pile_runs/rope_byte/photo_compressor"


def blursort(array, dimensions=2):
    """BlurSort: recursive dimensional interleaving."""
    if len(array) <= 1:
        return array
    buckets = [[] for _ in range(dimensions)]
    for i, x in enumerate(array):
        buckets[i % dimensions].append(x)
    result = []
    for bucket in buckets:
        result.extend(blursort(bucket, dimensions))
    return result


def make_patches(arr, patch_h=28, patch_w=28, ordering="raster"):
    H, W = arr.shape
    pH, pW = H // patch_h, W // patch_w
    patches = []
    for r in range(pH):
        for c in range(pW):
            patch = arr[r * patch_h : (r + 1) * patch_h, c * patch_w : (c + 1) * patch_w]
            pixels = patch.flatten().tolist()
            if ordering == "blursort":
                pixels = blursort(pixels, 2)
            patches.append(torch.tensor(pixels, dtype=torch.long))
    return patches


def train_and_eval(patches, label, t0):
    n_patches = len(patches)
    n_val = max(1, n_patches // 10)
    train_p = patches[:-n_val]
    val_p = patches[-n_val:]

    model = RoPE2DByteLM(
        vocab_size=256, d_model=64, n_heads=4, n_layers=4,
        height=28, width=28, bidirectional=False,
    ).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")

    best_val = float("inf")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for patch in train_p:
            ids = patch.unsqueeze(0).to(DEVICE)
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1)
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        val_sum = 0.0
        with torch.no_grad():
            for patch in val_p:
                ids = patch.unsqueeze(0).to(DEVICE)
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1)
                )
                val_sum += loss.item()
        val_loss = val_sum / len(val_p)

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "epoch": epoch,
                "val_loss": best_val,
            }, f"{PHOTO_DIR}/blursort_{label}.pt")

        if epoch % 10 == 0:
            bpp = val_loss / math.log(2)
            elapsed = time.time() - t0
            print(f"  ep {epoch:2d}: val_loss={val_loss:.4f} bpp={bpp:.2f} ({elapsed:.0f}s)")

    return best_val


def main():
    print("=== BLURSORT vs RASTER PIXEL ORDERING ===")
    t0 = time.time()

    img = Image.open(f"{PHOTO_DIR}/original_grayscale.png")
    arr = np.array(img)
    H, W = arr.shape
    pH, pW = H // 28, W // 28
    print(f"Image: {H}x{W}, {pH}x{pW} = {pH * pW} patches")

    results = {}
    for ordering in ["raster", "blursort"]:
        print(f"\n--- {ordering.upper()} ---")
        patches = make_patches(arr, ordering=ordering)
        if MAX_PATCHES and len(patches) > MAX_PATCHES:
            np.random.seed(42)
            idx = np.random.choice(len(patches), MAX_PATCHES, replace=False)
            patches = [patches[i] for i in sorted(idx)]
            print(f"  Subsampled to {len(patches)} patches")
        best_val = train_and_eval(patches, ordering, t0)
        bpp = best_val / math.log(2)
        results[ordering] = {"val_loss": best_val, "bpp": bpp}
        print(f"  Best: val_loss={best_val:.4f} bpp={bpp:.2f}")

    print(f"\n{'='*60}")
    print(f"COMPARISON")
    print(f"{'='*60}")
    r_bpp = results["raster"]["bpp"]
    b_bpp = results["blursort"]["bpp"]
    print(f"Raster:   {r_bpp:.4f} bpp")
    print(f"BlurSort: {b_bpp:.4f} bpp")
    diff = r_bpp - b_bpp
    if diff > 0:
        print(f"BlurSort wins by {diff:.4f} bpp ({diff / r_bpp * 100:.1f}% improvement)")
    else:
        print(f"Raster wins by {-diff:.4f} bpp ({-diff / r_bpp * 100:.1f}% better)")
    print(f"Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
