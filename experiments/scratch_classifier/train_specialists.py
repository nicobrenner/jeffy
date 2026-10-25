"""Train 10 per-digit specialist models + classify by lowest loss.

Each specialist is a small 2D RoPE byte model trained ONLY on images
of its digit. Classification: run all 10 specialists on a test image, pick
the one with the lowest loss. This is the "overtraining" thesis in action —
each tiny model becomes an expert at its digit's pixel patterns.

Supports both causal and bidirectional specialists via --bidirectional flag.
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_rope_byte_2d import (
    RoPE2DByteLM,
    MNISTByteDataset,
    evaluate,
)


class SingleDigitDataset(Dataset):
    """MNIST images for a single digit class."""
    def __init__(self, bytes_path, labels, digit, seq_len=784):
        all_data = np.fromfile(bytes_path, dtype=np.uint8)
        n_images = len(all_data) // seq_len
        labels = labels[:n_images]
        mask = labels == digit
        indices = np.where(mask)[0]
        self.images = []
        for idx in indices:
            start = idx * seq_len
            self.images.append(all_data[start:start + seq_len].copy())
        self.images = np.array(self.images)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return torch.from_numpy(self.images[idx].astype(np.int64))


def train_specialist(digit, args, train_labels, val_labels, device, bidirectional=False):
    """Train one specialist model for a single digit."""
    seq_len = 784
    run_dir = Path(args.output_dir) / f'specialist_digit{digit}'
    run_dir.mkdir(parents=True, exist_ok=True)

    train_ds = SingleDigitDataset(
        str(Path(args.data_dir) / 'train.bytes'), train_labels, digit)
    val_ds = SingleDigitDataset(
        str(Path(args.data_dir) / 'val.bytes'), val_labels, digit)

    n_train = len(train_ds)
    n_val = len(val_ds)
    print(f"\n  Digit {digit}: {n_train} train, {n_val} val images")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    model = RoPE2DByteLM(
        vocab_size=256, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, height=28, width=28,
        tie_weights=True, bidirectional=bidirectional,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    ckpt_path = run_dir / 'checkpoint.pt'
    best_path = run_dir / 'best_model.pt'
    start_epoch = 0
    best_val_loss = float('inf')
    best_state = None

    if args.resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        start_epoch = ckpt['epoch']
        best_val_loss = ckpt['best_val_loss']
        best_state = ckpt.get('best_model_state')
        print(f"    Resumed from epoch {start_epoch}, best_loss={best_val_loss:.4f}")

    model.train()
    t0 = time.time()

    for epoch in range(start_epoch, args.epochs):
        total_loss = 0
        n_batches = 0
        for batch in train_loader:
            ids = batch.to(device)
            logits = model(ids)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, 256),
                ids[:, 1:].contiguous().view(-1),
            )
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        avg_train_loss = total_loss / max(n_batches, 1)

        # Validate
        model.eval()
        val_loss_total = 0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch.to(device)
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].contiguous().view(-1, 256),
                    ids[:, 1:].contiguous().view(-1),
                )
                val_loss_total += loss.item()
                val_n += 1
        val_loss = val_loss_total / max(val_n, 1)
        model.train()

        tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            tag = " *BEST*"

        tokens_seen = (epoch + 1) * n_train * seq_len
        tp = tokens_seen / n_params

        if (epoch + 1) % 5 == 0 or epoch == 0 or tag:
            print(f"    ep {epoch+1:>3}: train_loss={avg_train_loss:.4f} "
                  f"val_loss={val_loss:.4f} ppl={math.exp(min(val_loss,20)):.2f} "
                  f"tp={tp:.0f}{tag}", flush=True)

        torch.save({
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_loss': best_val_loss,
            'best_model_state': best_state,
        }, ckpt_path)

    elapsed = time.time() - t0

    if best_state:
        torch.save({
            'model_state': best_state,
            'config': {
                'd_model': args.d_model, 'n_layers': args.n_layers,
                'n_heads': args.n_heads, 'digit': digit,
                'bidirectional': bidirectional,
            },
            'val_loss': best_val_loss,
            'val_ppl': math.exp(min(best_val_loss, 20)),
            'n_train': n_train,
        }, best_path)

    print(f"    Done digit {digit}: best_loss={best_val_loss:.4f} "
          f"ppl={math.exp(min(best_val_loss, 20)):.2f} ({elapsed:.1f}s)")

    return {
        'digit': digit,
        'best_val_loss': best_val_loss,
        'best_ppl': math.exp(min(best_val_loss, 20)),
        'n_train': n_train,
        'elapsed': elapsed,
        'n_params': n_params,
    }


def classify_by_loss(specialists, test_ds, device, max_images=None):
    """Classify test images by picking the specialist with the lowest loss."""
    n_test = len(test_ds) if max_images is None else min(len(test_ds), max_images)

    all_losses = np.zeros((n_test, 10))

    for digit, model in enumerate(specialists):
        model.eval()
        print(f"  Scoring with specialist {digit}...", end="", flush=True)
        t0 = time.time()
        with torch.no_grad():
            for i in range(n_test):
                img, label = test_ds[i]
                ids = img.unsqueeze(0).to(device)
                logits = model(ids)
                loss = F.cross_entropy(
                    logits[:, :-1, :].contiguous().view(-1, 256),
                    ids[:, 1:].contiguous().view(-1),
                )
                all_losses[i, digit] = loss.item()
        print(f" {time.time()-t0:.1f}s")

    predictions = all_losses.argmin(axis=1)
    labels = np.array([test_ds[i][1].item() for i in range(n_test)])
    accuracy = (predictions == labels).mean()

    per_class_acc = {}
    for d in range(10):
        mask = labels == d
        if mask.sum() > 0:
            per_class_acc[d] = float((predictions[mask] == d).mean())

    return accuracy, predictions, labels, all_losses, per_class_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str,
                        default='mps' if torch.backends.mps.is_available() else
                        ('cuda' if torch.cuda.is_available() else 'cpu'))
    parser.add_argument('--data-dir', type=str, default='pile_runs/custom_vocab/byte_mnist')
    parser.add_argument('--output-dir', type=str, default='pile_runs/rope_byte/specialists')
    parser.add_argument('--resume', action='store_true', default=True)
    parser.add_argument('--no-resume', dest='resume', action='store_false')
    parser.add_argument('--classify-only', action='store_true',
                        help='Skip training, just classify with existing models')
    parser.add_argument('--max-test', type=int, default=None,
                        help='Max test images for classification (None = all)')
    parser.add_argument('--bidirectional', action='store_true',
                        help='Train bidirectional (not causal) specialists')
    parser.add_argument('--digits', type=int, nargs='+', default=list(range(10)),
                        help='Which digits to train (default: all 10)')
    args = parser.parse_args()

    mode = "Bidirectional" if args.bidirectional else "Causal"
    if args.bidirectional and args.output_dir == 'pile_runs/rope_byte/specialists':
        args.output_dir = 'pile_runs/rope_byte/bidir_specialists'

    print(f"{'='*70}")
    print(f"  Per-Digit Specialist Training — 10 {mode} Models")
    print(f"{'='*70}")
    print(f"  d_model={args.d_model}, {args.n_layers}L, {args.n_heads}H")
    print(f"  Epochs per specialist: {args.epochs}")
    print(f"  Device: {args.device}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*70}")

    from torchvision import datasets as tv_datasets
    mnist_train = tv_datasets.MNIST(root='data', train=True, download=True)
    mnist_test = tv_datasets.MNIST(root='data', train=False, download=True)
    train_labels = mnist_train.targets.numpy()
    val_labels = mnist_test.targets.numpy()

    device = args.device

    if not args.classify_only:
        print(f"\n  Phase 1: Training 10 specialists")
        print(f"  {'='*50}")
        results = []
        total_t0 = time.time()

        for digit in args.digits:
            r = train_specialist(digit, args, train_labels, val_labels, device,
                                 bidirectional=args.bidirectional)
            results.append(r)

        total_elapsed = time.time() - total_t0
        print(f"\n  All 10 specialists trained in {total_elapsed:.1f}s")
        print(f"  Per-digit PPLs:")
        for r in results:
            print(f"    digit {r['digit']}: ppl={r['best_ppl']:.2f} "
                  f"({r['n_train']} images, {r['elapsed']:.1f}s)")

        with open(Path(args.output_dir) / 'training_results.json', 'w') as f:
            json.dump(results, f, indent=2)

    print(f"\n  Phase 2: Classification by lowest loss")
    print(f"  {'='*50}")

    specialists = []
    for digit in range(10):
        best_path_check = Path(args.output_dir) / f'specialist_digit{digit}' / 'best_model.pt'
        if not best_path_check.exists():
            print(f"  Skipping classification — not all 10 specialists trained yet (missing digit {digit})")
            return

    for digit in range(10):
        best_path = Path(args.output_dir) / f'specialist_digit{digit}' / 'best_model.pt'
        if not best_path.exists():
            print(f"  ERROR: No model for digit {digit} at {best_path}")
            return
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model = RoPE2DByteLM(
            vocab_size=256, d_model=args.d_model, n_layers=args.n_layers,
            n_heads=args.n_heads, height=28, width=28,
            tie_weights=True, bidirectional=args.bidirectional,
        ).to(device)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        specialists.append(model)
        ppl = ckpt.get('val_ppl', math.exp(min(ckpt.get('val_loss', 99), 20)))
        print(f"  Loaded specialist {digit}: ppl={ppl:.2f}")

    test_ds = MNISTByteDataset(
        str(Path(args.data_dir) / 'val.bytes'), val_labels, seq_len=784)

    accuracy, preds, labels, losses, per_class = classify_by_loss(
        specialists, test_ds, device, max_images=args.max_test)

    print(f"\n  {'='*50}")
    print(f"  SPECIALIST ENSEMBLE ACCURACY: {accuracy*100:.2f}%")
    print(f"  {'='*50}")
    print(f"  Per-class accuracy:")
    for d in range(10):
        print(f"    digit {d}: {per_class.get(d, 0)*100:.1f}%")

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(labels, preds)
    print(f"\n  Confusion matrix:")
    print(f"       " + " ".join(f"{d:>5}" for d in range(10)))
    for d in range(10):
        print(f"    {d}: " + " ".join(f"{cm[d,j]:>5}" for j in range(10)))

    results_path = Path(args.output_dir) / 'classification_results.json'
    with open(results_path, 'w') as f:
        json.dump({
            'accuracy': float(accuracy),
            'per_class_accuracy': per_class,
            'n_test': len(labels),
            'confusion_matrix': cm.tolist(),
        }, f, indent=2)

    print(f"\n  Results saved to {results_path}")


if __name__ == '__main__':
    main()
