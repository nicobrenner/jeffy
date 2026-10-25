"""Train a byte model on patches from a personal photo, then measure compression.

Demonstrates that a model trained on YOUR image's statistics can compress it
far better than a generic model or PNG. This is domain-specific learned compression.

Usage:
    python3 -u train_photo_compressor.py /path/to/photo.jpg --epochs 50
    python3 -u train_photo_compressor.py /path/to/photo.jpg --epochs 50 --device cpu
"""

import argparse
import gzip
import io
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_rope_byte_2d import RoPE2DByteLM


class PhotoPatchDataset(Dataset):
    """28x28 grayscale patches from a photo."""
    def __init__(self, patches):
        self.patches = torch.tensor(patches.astype(np.int64), dtype=torch.long)

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return self.patches[idx]


def tile_image(image_path, patch_size=28):
    img = Image.open(image_path).convert("L")
    w, h = img.size
    arr = np.array(img)
    n_rows = h // patch_size
    n_cols = w // patch_size
    patches = []
    for r in range(n_rows):
        for c in range(n_cols):
            patch = arr[r*patch_size:(r+1)*patch_size, c*patch_size:(c+1)*patch_size]
            patches.append(patch.flatten())
    return np.array(patches), img, n_rows, n_cols


@torch.no_grad()
def compute_bpp(model, patches, device, batch_size=64):
    total_bits = 0.0
    n_pixels = len(patches) * 784
    for start in range(0, len(patches), batch_size):
        end = min(start + batch_size, len(patches))
        batch = torch.tensor(
            patches[start:end].astype(np.int64), dtype=torch.long
        ).to(device)
        logits = model(batch)
        log_probs = F.log_softmax(logits.float(), dim=-1)
        # First pixel: uniform (8 bits)
        total_bits += batch.shape[0] * 8.0
        targets = batch[:, 1:]
        lp = log_probs[:, :-1, :]
        bits = -lp.gather(2, targets.unsqueeze(-1)).squeeze(-1) / math.log(2)
        total_bits += bits.sum().item()
    return total_bits / n_pixels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=4)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str,
                        default="mps" if torch.backends.mps.is_available() else
                        ("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--patch-size", type=int, default=28)
    parser.add_argument("--output-dir", type=str, default="pile_runs/rope_byte/photo_compressor")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}", flush=True)
    print(f"  Photo Compressor Training", flush=True)
    print(f"{'='*60}", flush=True)

    patches, img, n_rows, n_cols = tile_image(args.image, args.patch_size)
    n_patches = len(patches)
    print(f"  Image: {args.image}", flush=True)
    print(f"  Size: {img.size[0]}x{img.size[1]} grayscale", flush=True)
    print(f"  Patches: {n_cols}x{n_rows} = {n_patches} ({args.patch_size}x{args.patch_size})", flush=True)

    # Train/val split: 90/10
    rng = np.random.default_rng(42)
    idx = rng.permutation(n_patches)
    n_val = max(n_patches // 10, 1)
    val_patches = patches[idx[:n_val]]
    train_patches = patches[idx[n_val:]]
    print(f"  Train: {len(train_patches)}, Val: {len(val_patches)}", flush=True)

    train_ds = PhotoPatchDataset(train_patches)
    val_ds = PhotoPatchDataset(val_patches)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = RoPE2DByteLM(
        vocab_size=256, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, height=args.patch_size, width=args.patch_size,
        bidirectional=False, tie_weights=True,
    ).to(args.device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params, device={args.device}", flush=True)
    print(f"  Epochs: {args.epochs}", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # Checkpointing
    ckpt_path = out_dir / "checkpoint.pt"
    best_path = out_dir / "best_model.pt"
    start_epoch = 0
    best_val_loss = float("inf")

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=args.device, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt["best_val_loss"]
        print(f"  Resumed from epoch {start_epoch}, best_loss={best_val_loss:.4f}", flush=True)

    # Baselines
    print(f"\n  Computing baselines...", flush=True)
    raw_bpp = 8.0
    png_bits = 0
    gz_bits = 0
    for p in patches:
        patch_img = Image.fromarray(p.reshape(args.patch_size, args.patch_size).astype(np.uint8), mode="L")
        buf = io.BytesIO()
        patch_img.save(buf, format="PNG", optimize=True)
        png_bits += len(buf.getvalue()) * 8
        gz_bits += len(gzip.compress(p.tobytes(), compresslevel=9)) * 8
    total_pixels = n_patches * args.patch_size * args.patch_size
    png_bpp = png_bits / total_pixels
    gz_bpp = gz_bits / total_pixels
    print(f"  Raw: {raw_bpp:.2f} bpp", flush=True)
    print(f"  PNG: {png_bpp:.2f} bpp ({raw_bpp/png_bpp:.1f}x)", flush=True)
    print(f"  gzip: {gz_bpp:.2f} bpp ({raw_bpp/gz_bpp:.1f}x)", flush=True)

    # MNIST model baseline
    mnist_ckpt_path = "pile_runs/rope_byte/mnist_2d_d64_causal_tp1800/best_model.pt"
    try:
        mnist_model = RoPE2DByteLM(
            vocab_size=256, d_model=64, n_layers=4, n_heads=4,
            height=28, width=28, bidirectional=False,
        ).to(args.device)
        mnist_ckpt = torch.load(mnist_ckpt_path, map_location=args.device, weights_only=False)
        mnist_model.load_state_dict(mnist_ckpt["model_state"])
        mnist_model.eval()
        mnist_bpp = compute_bpp(mnist_model, patches, args.device)
        print(f"  MNIST model on photo: {mnist_bpp:.2f} bpp ({raw_bpp/mnist_bpp:.1f}x)", flush=True)
        del mnist_model
    except Exception as e:
        print(f"  MNIST model baseline failed: {e}", flush=True)
        mnist_bpp = None

    print(f"\n  Training...", flush=True)
    print(f"{'='*60}", flush=True)

    t_total = time.time()
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss_sum = 0
        n_batches = 0
        for batch in train_loader:
            ids = batch.to(args.device)
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1].contiguous().view(-1, 256),
                ids[:, 1:].contiguous().view(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1

        train_loss = train_loss_sum / max(n_batches, 1)

        model.eval()
        val_sum = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch.to(args.device)
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1].contiguous().view(-1, 256),
                    ids[:, 1:].contiguous().view(-1),
                )
                val_sum += loss.item()
                val_n += 1
        val_loss = val_sum / max(val_n, 1)

        tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            tag = " *BEST*"
            torch.save({
                "model_state": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "val_loss": val_loss,
                "epoch": epoch + 1,
                "n_params": n_params,
            }, best_path)

        tp = (epoch + 1) * len(train_patches) * 784 / n_params

        if (epoch + 1) % 5 == 0 or epoch == 0 or tag:
            print(
                f"  ep {epoch+1:3d}: train={train_loss:.4f} val={val_loss:.4f} "
                f"ppl={math.exp(min(val_loss, 20)):.2f} tp={tp:.0f}{tag}",
                flush=True,
            )

        torch.save({
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
        }, ckpt_path)

    total_time = time.time() - t_total
    print(f"\n  Training done in {total_time:.1f}s", flush=True)

    # Final compression measurement on ALL patches
    print(f"\n  Final compression measurement (all {n_patches} patches):", flush=True)
    best_ckpt = torch.load(best_path, map_location=args.device, weights_only=False)
    model.load_state_dict(best_ckpt["model_state"])
    model.eval()
    photo_bpp = compute_bpp(model, patches, args.device)

    print(f"\n{'='*60}", flush=True)
    print(f"  RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Method':<30s} {'bpp':>8s} {'Ratio':>8s}", flush=True)
    print(f"  {'-'*30} {'-'*8} {'-'*8}", flush=True)
    print(f"  {'Raw':<30s} {raw_bpp:>8.2f} {'1.0x':>8s}", flush=True)
    print(f"  {'PNG (per patch)':<30s} {png_bpp:>8.2f} {raw_bpp/png_bpp:>7.1f}x", flush=True)
    print(f"  {'gzip (per patch)':<30s} {gz_bpp:>8.2f} {raw_bpp/gz_bpp:>7.1f}x", flush=True)
    if mnist_bpp:
        print(f"  {'MNIST model (wrong domain)':<30s} {mnist_bpp:>8.2f} {raw_bpp/mnist_bpp:>7.1f}x", flush=True)
    print(f"  {'Photo-trained model':<30s} {photo_bpp:>8.2f} {raw_bpp/photo_bpp:>7.1f}x", flush=True)
    print(f"\n  Model size: {n_params:,} params", flush=True)
    print(f"  Training time: {total_time:.0f}s", flush=True)


if __name__ == "__main__":
    main()
