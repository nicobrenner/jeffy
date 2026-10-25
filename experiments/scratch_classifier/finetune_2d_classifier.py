"""Fine-tune 2D RoPE models for MNIST classification.

Takes a pretrained 2D RoPE byte model and fine-tunes end-to-end
for classification. Also includes training from scratch for comparison.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision

from train_rope_byte_2d import RoPE2DByteLM, MNISTByteDataset


class ByteClassifier(nn.Module):
    """Wraps a RoPE2DByteLM with a classification head."""

    def __init__(self, backbone, n_classes=10, pool='mean', freeze_backbone=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.n_classes = n_classes
        d = backbone.d_model
        if pool == 'both':
            d = d * 2
        self.head = nn.Linear(d, n_classes)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_ids):
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            feats = self.backbone.get_features(input_ids, pool=self.pool)
        return self.head(feats)


def train_classifier(model, train_loader, test_loader, device, epochs=30,
                     lr=1e-3, backbone_lr=None):
    if backbone_lr is not None:
        backbone_params = list(model.backbone.parameters())
        head_params = list(model.head.parameters())
        optimizer = torch.optim.AdamW([
            {'params': backbone_params, 'lr': backbone_lr},
            {'params': head_params, 'lr': lr},
        ], weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best_acc = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        scheduler.step()

        train_acc = correct / total

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = model(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
        test_acc = correct / total

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            marker = '*BEST*'
        else:
            marker = ''

        if (ep + 1) % 5 == 0 or ep == 0 or marker:
            print(f'  ep {ep+1:3d}: train={train_acc*100:.1f}% test={test_acc*100:.1f}% '
                  f'loss={total_loss/total:.4f} {marker}')

    return best_acc, best_state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='mps')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()

    device = args.device

    mnist_train = torchvision.datasets.MNIST(root='/tmp/mnist', train=True, download=True)
    mnist_test = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True)
    train_labels = np.array([l for _, l in mnist_train])
    test_labels = np.array([l for _, l in mnist_test])

    train_ds = MNISTByteDataset('pile_runs/custom_vocab/byte_mnist/train.bytes', train_labels, 784)
    test_ds = MNISTByteDataset('pile_runs/custom_vocab/byte_mnist/val.bytes', test_labels, 784)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, num_workers=0)

    results = {}

    # ============================================================
    # Experiment 1: Fine-tune pretrained 2D CAUSAL model
    # ============================================================
    causal_path = Path('pile_runs/rope_byte/mnist_2d_d64_causal_tp1800/best_model.pt')
    if causal_path.exists():
        print('=' * 60)
        print('  EXPERIMENT 1: Fine-tune pretrained 2D CAUSAL')
        print('=' * 60)

        for pool in ['mean', 'both']:
            backbone = RoPE2DByteLM(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                                    height=28, width=28, bidirectional=False).to(device)
            ckpt = torch.load(causal_path, map_location=device, weights_only=False)
            backbone.load_state_dict(ckpt['model_state'])

            model = ByteClassifier(backbone, n_classes=10, pool=pool).to(device)
            print(f'\n  2D causal, pool={pool}, full fine-tune (backbone_lr=1e-5, head_lr=1e-3)')
            t0 = time.time()
            acc, state = train_classifier(model, train_loader, test_loader, device,
                                          epochs=args.epochs, lr=1e-3, backbone_lr=1e-5)
            elapsed = time.time() - t0
            print(f'  Best: {acc*100:.1f}% ({elapsed:.0f}s)')
            results[f'finetune_causal_{pool}'] = {'test_acc': acc, 'time': elapsed}

            # Also try frozen backbone
            backbone2 = RoPE2DByteLM(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                                     height=28, width=28, bidirectional=False).to(device)
            backbone2.load_state_dict(ckpt['model_state'])
            model2 = ByteClassifier(backbone2, n_classes=10, pool=pool, freeze_backbone=True).to(device)
            print(f'\n  2D causal, pool={pool}, frozen backbone (head only)')
            t0 = time.time()
            acc2, _ = train_classifier(model2, train_loader, test_loader, device,
                                       epochs=args.epochs, lr=1e-3)
            elapsed2 = time.time() - t0
            print(f'  Best: {acc2*100:.1f}% ({elapsed2:.0f}s)')
            results[f'frozen_causal_{pool}'] = {'test_acc': acc2, 'time': elapsed2}
    else:
        print(f'No causal checkpoint at {causal_path} — skipping')

    # ============================================================
    # Experiment 2: Fine-tune pretrained 2D BIDIRECTIONAL model
    # ============================================================
    bidir_path = Path('pile_runs/rope_byte/mnist_2d_d64_bidirectional_tp1800/best_model.pt')
    if bidir_path.exists():
        print('\n' + '=' * 60)
        print('  EXPERIMENT 2: Fine-tune pretrained 2D BIDIRECTIONAL')
        print('=' * 60)

        backbone = RoPE2DByteLM(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                                height=28, width=28, bidirectional=True).to(device)
        ckpt = torch.load(bidir_path, map_location=device, weights_only=False)
        backbone.load_state_dict(ckpt['model_state'])

        model = ByteClassifier(backbone, n_classes=10, pool='mean').to(device)
        print(f'\n  2D bidir, pool=mean, full fine-tune')
        t0 = time.time()
        acc, _ = train_classifier(model, train_loader, test_loader, device,
                                  epochs=args.epochs, lr=1e-3, backbone_lr=1e-5)
        elapsed = time.time() - t0
        print(f'  Best: {acc*100:.1f}% ({elapsed:.0f}s)')
        results['finetune_bidir_mean'] = {'test_acc': acc, 'time': elapsed}

    # ============================================================
    # Experiment 3: Train 2D causal from scratch for classification
    # ============================================================
    print('\n' + '=' * 60)
    print('  EXPERIMENT 3: Train 2D CAUSAL from scratch')
    print('=' * 60)

    backbone = RoPE2DByteLM(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                            height=28, width=28, bidirectional=False).to(device)
    model = ByteClassifier(backbone, n_classes=10, pool='mean').to(device)
    print(f'\n  2D causal from scratch, pool=mean, {args.epochs} epochs')
    t0 = time.time()
    acc, _ = train_classifier(model, train_loader, test_loader, device,
                              epochs=args.epochs, lr=3e-4)
    elapsed = time.time() - t0
    print(f'  Best: {acc*100:.1f}% ({elapsed:.0f}s)')
    results['scratch_causal_mean'] = {'test_acc': acc, 'time': elapsed}

    # Bidirectional from scratch
    backbone = RoPE2DByteLM(vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                            height=28, width=28, bidirectional=True).to(device)
    model = ByteClassifier(backbone, n_classes=10, pool='mean').to(device)
    print(f'\n  2D bidir from scratch, pool=mean, {args.epochs} epochs')
    t0 = time.time()
    acc, _ = train_classifier(model, train_loader, test_loader, device,
                              epochs=args.epochs, lr=3e-4)
    elapsed = time.time() - t0
    print(f'  Best: {acc*100:.1f}% ({elapsed:.0f}s)')
    results['scratch_bidir_mean'] = {'test_acc': acc, 'time': elapsed}

    # ============================================================
    # Summary
    # ============================================================
    print('\n' + '=' * 60)
    print('  SUMMARY')
    print('=' * 60)
    print(f'  {"Method":<40s} {"Test Acc":>8s}')
    print(f'  {"-"*40} {"-"*8}')
    for k, v in sorted(results.items(), key=lambda x: -x[1]['test_acc']):
        print(f'  {k:<40s} {v["test_acc"]*100:>7.1f}%')

    with open('pile_runs/rope_byte/finetune_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\n  Results saved to pile_runs/rope_byte/finetune_results.json')


if __name__ == '__main__':
    main()
