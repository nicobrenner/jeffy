"""MNIST classification evaluation for byte-level models.

Trains MNIST byte models at d=32 and d=64, then evaluates:
1. Feature extraction (mean pool, max pool, last token, concat)
2. Linear probe (logistic regression)
3. MLP classifier on frozen features
4. Frozen backbone + trained classification head

Compares against raw pixel baseline (784-dim logistic regression: 90.4%).
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

from train_rope_byte import RoPEByteLM, ByteDataset, evaluate


device = 'mps' if torch.backends.mps.is_available() else 'cpu'
BASE_DIR = Path('pile_runs/rope_byte')
MNIST_DATA = Path('pile_runs/custom_vocab/byte_mnist')


# ── MNIST image dataset (for classification, not LM) ─────────────────

class MNISTByteDataset(Dataset):
    """Load MNIST byte data as (image_bytes, label) pairs."""
    def __init__(self, bytes_path, labels, seq_len=784):
        self.data = np.fromfile(bytes_path, dtype=np.uint8)
        self.labels = labels
        self.seq_len = seq_len
        self.n_images = len(self.data) // seq_len

    def __len__(self):
        return min(self.n_images, len(self.labels))

    def __getitem__(self, idx):
        start = idx * self.seq_len
        img = torch.from_numpy(
            self.data[start:start + self.seq_len].astype(np.int64))
        return img, self.labels[idx]


# ── Feature extraction methods ────────────────────────────────────────

@torch.no_grad()
def extract_features_batch(model, dataloader, method='mean'):
    """Extract features from a byte model using various pooling methods.

    Methods:
        mean: mean pool all hidden states
        max: max pool all hidden states
        last: last position hidden state (causal LM's "summary")
        concat: concatenate mean + max + last
    """
    model.eval()
    all_feats = []
    all_labels = []

    for imgs, labels in dataloader:
        ids = imgs.to(device)
        x = model.token_emb(ids)
        mask = model.causal_mask
        for layer in model.layers:
            x = layer(x, model.freqs, mask)
        x = model.norm(x)  # (B, S, d_model)

        if method == 'mean':
            feat = x.mean(dim=1)
        elif method == 'max':
            feat = x.max(dim=1).values
        elif method == 'last':
            feat = x[:, -1, :]
        elif method == 'concat':
            feat = torch.cat([
                x.mean(dim=1),
                x.max(dim=1).values,
                x[:, -1, :],
            ], dim=-1)
        else:
            raise ValueError(f"Unknown method: {method}")

        all_feats.append(feat.cpu().numpy())
        all_labels.append(labels.numpy())

    return np.concatenate(all_feats), np.concatenate(all_labels)


# ── Classifiers ───────────────────────────────────────────────────────

def logistic_probe(train_feats, train_labels, val_feats, val_labels):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(train_feats, train_labels)
    return clf.score(train_feats, train_labels), clf.score(val_feats, val_labels)


def mlp_probe(train_feats, train_labels, val_feats, val_labels, hidden=128, epochs=30):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=epochs,
                        learning_rate_init=1e-3, early_stopping=True,
                        validation_fraction=0.1, random_state=42)
    clf.fit(train_feats, train_labels)
    return clf.score(train_feats, train_labels), clf.score(val_feats, val_labels)


class ClassificationHead(nn.Module):
    """Frozen backbone + trainable linear head for classification."""
    def __init__(self, backbone, n_classes=10, pool='last'):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        d = backbone.d_model
        if pool == 'concat':
            d *= 3
        self.head = nn.Linear(d, n_classes)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, input_ids):
        x = self.backbone.token_emb(input_ids)
        mask = self.backbone.causal_mask
        for layer in self.backbone.layers:
            x = layer(x, self.backbone.freqs, mask)
        x = self.backbone.norm(x)

        if self.pool == 'mean':
            feat = x.mean(dim=1)
        elif self.pool == 'max':
            feat = x.max(dim=1).values
        elif self.pool == 'last':
            feat = x[:, -1, :]
        elif self.pool == 'concat':
            feat = torch.cat([x.mean(dim=1), x.max(dim=1).values, x[:, -1, :]], dim=-1)

        return self.head(feat)


def train_classification_head(model, train_loader, val_loader, pool='last',
                              epochs=20, lr=1e-3):
    """Train a classification head on frozen backbone features."""
    clf_model = ClassificationHead(model, n_classes=10, pool=pool).to(device)
    optimizer = torch.optim.Adam(clf_model.head.parameters(), lr=lr)

    best_val_acc = 0
    for epoch in range(epochs):
        clf_model.train()
        correct = total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = clf_model(imgs)
            loss = F.cross_entropy(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)
        train_acc = correct / total

        clf_model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                logits = clf_model(imgs)
                correct += (logits.argmax(1) == labels).sum().item()
                total += len(labels)
        val_acc = correct / total
        best_val_acc = max(best_val_acc, val_acc)

    return train_acc, best_val_acc


# ── Training ──────────────────────────────────────────────────────────

def train_mnist_model(d_model, n_layers, n_heads, tok_param, seq_len=784,
                      batch_size=16, lr=3e-4):
    """Train a byte model on MNIST data. Returns the model and run directory."""
    run_dir = BASE_DIR / f'mnist_d{d_model}_tp{int(tok_param)}'
    run_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = run_dir / 'checkpoint.pt'
    best_path = run_dir / 'best_model.pt'

    # Check if already trained
    if best_path.exists():
        print(f"    Loading existing model from {run_dir}")
        model = RoPEByteLM(vocab_size=256, d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, max_seq_len=seq_len, tie_weights=True).to(device)
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        return model, run_dir

    model = RoPEByteLM(vocab_size=256, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, max_seq_len=seq_len, tie_weights=True).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    total_tokens = int(tok_param * n_params)
    n_steps = total_tokens // (batch_size * seq_len)

    print(f"    Model: {n_params:,} params, {n_steps:,} steps, {total_tokens/1e6:.1f}M tokens")

    train_ds = ByteDataset(str(MNIST_DATA / 'train.bytes'), seq_len, max_tokens=total_tokens)
    val_ds = ByteDataset(str(MNIST_DATA / 'val.bytes'), seq_len, max_tokens=seq_len * 200)
    train_loader = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    start_step = 0
    best_ppl = float('inf')
    best_state = None

    if ckpt_path.exists():
        print(f"    Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt['step']
        best_ppl = ckpt['best_ppl']
        best_state = ckpt.get('best_model_state')
        print(f"    Resumed at step {start_step:,}")

    model.train()
    step = 0
    running_loss = 0
    t0 = time.time()
    eval_every = max(50, n_steps // 20)
    log_every = max(100, n_steps // 10)
    save_every = max(200, n_steps // 5)

    for batch in train_loader:
        step += 1
        if step <= start_step:
            scheduler.step()
            continue

        ids = batch.to(device)
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            ids[:, 1:].contiguous().view(-1),
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        tp = step * batch_size * seq_len / n_params

        if step % eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            tag = ""
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                tag = " *BEST*"
            print(f"      eval step {step}: ppl={val_ppl:.3f} tp={tp:.0f}{tag}")
            model.train()

        if step % log_every == 0:
            avg = running_loss / log_every
            elapsed = time.time() - t0
            print(f"      step {step:>6,}/{n_steps:,} loss={avg:.4f} ppl={math.exp(min(avg,20)):.2f} "
                  f"{step*batch_size*seq_len/elapsed:,.0f} tok/s")
            running_loss = 0

        if step % save_every == 0:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'step': step,
                'best_ppl': best_ppl,
                'best_model_state': best_state,
            }, ckpt_path)

        if step >= n_steps:
            break

    # Final eval
    val_loss, val_ppl = evaluate(model, val_loader, device)
    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    print(f"    Final ppl={val_ppl:.3f}, best={best_ppl:.3f}, time={elapsed:.1f}s")

    # Save results and best model
    results = {
        'model': 'mnist_rope_byte',
        'd_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads,
        'n_params': n_params, 'tok_param': tok_param,
        'best_val_ppl': round(best_ppl, 4),
        'final_val_ppl': round(val_ppl, 4),
        'training_time_s': round(elapsed, 1),
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if best_state:
        model.load_state_dict(best_state)
        torch.save({
            'model_state': best_state,
            'config': {'d_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads},
            'val_ppl': best_ppl,
        }, best_path)

    model.eval()
    return model, run_dir


# ── Main ──────────────────────────────────────────────────────────────

def main():
    import torchvision

    print("=" * 70)
    print("  MNIST CLASSIFICATION — BYTE MODEL EVALUATION")
    print("=" * 70)

    # Load MNIST labels
    train_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=True, download=True)
    val_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True)
    train_labels = np.array([label for _, label in train_ds])
    val_labels = np.array([label for _, label in val_ds])

    # ── Step 1: Train models ──────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  STEP 1: TRAIN MNIST BYTE MODELS")
    print("=" * 70)

    models = {}

    configs = [
        ('d32_1800', 32, 4, 4, 1800),
        ('d32_7200', 32, 4, 4, 7200),
        ('d64_1800', 64, 4, 4, 1800),
        ('d64_7200', 64, 4, 4, 7200),
    ]

    for name, d, nl, nh, tp in configs:
        print(f"\n  [{name}] d={d}, {nl}L, {nh}H, {tp} tok/param")
        model, run_dir = train_mnist_model(d, nl, nh, tp, seq_len=784, batch_size=16)
        models[name] = model

    # ── Step 2: Feature extraction + classification ───────────────────

    print("\n" + "=" * 70)
    print("  STEP 2: FEATURE EXTRACTION + CLASSIFICATION")
    print("=" * 70)

    # Create classification dataloaders (full images as sequences)
    train_cls = MNISTByteDataset(str(MNIST_DATA / 'train.bytes'), train_labels, seq_len=784)
    val_cls = MNISTByteDataset(str(MNIST_DATA / 'val.bytes'), val_labels, seq_len=784)

    # Use smaller subsets for feature extraction (faster)
    n_train = min(10000, len(train_cls))
    n_val = min(10000, len(val_cls))
    train_sub = torch.utils.data.Subset(train_cls, range(n_train))
    val_sub = torch.utils.data.Subset(val_cls, range(n_val))

    train_feat_loader = DataLoader(train_sub, batch_size=32, num_workers=0)
    val_feat_loader = DataLoader(val_sub, batch_size=32, num_workers=0)

    methods = ['mean', 'max', 'last', 'concat']
    results = []

    for model_name, model in models.items():
        for method in methods:
            print(f"\n  [{model_name}] Extracting features ({method})...", end=" ", flush=True)
            train_feats, train_lbls = extract_features_batch(model, train_feat_loader, method)
            val_feats, val_lbls = extract_features_batch(model, val_feat_loader, method)

            feat_dim = train_feats.shape[1]

            # Logistic regression
            lr_train, lr_val = logistic_probe(train_feats, train_lbls, val_feats, val_lbls)
            print(f"LogReg: {lr_val*100:.1f}%", end="", flush=True)
            results.append({
                'model': model_name, 'method': method, 'classifier': 'logreg',
                'feat_dim': feat_dim, 'train_acc': lr_train, 'val_acc': lr_val,
            })

            # MLP
            mlp_train, mlp_val = mlp_probe(train_feats, train_lbls, val_feats, val_lbls,
                                            hidden=min(256, feat_dim * 2))
            print(f"  MLP: {mlp_val*100:.1f}%")
            results.append({
                'model': model_name, 'method': method, 'classifier': 'mlp',
                'feat_dim': feat_dim, 'train_acc': mlp_train, 'val_acc': mlp_val,
            })

    # ── Step 3: Trained classification head ───────────────────────────

    print("\n" + "=" * 70)
    print("  STEP 3: FROZEN BACKBONE + TRAINED CLASSIFICATION HEAD")
    print("=" * 70)

    train_head_loader = DataLoader(train_sub, batch_size=32, shuffle=True, num_workers=0)
    val_head_loader = DataLoader(val_sub, batch_size=32, num_workers=0)

    head_results = []
    for model_name, model in models.items():
        for pool in ['last', 'concat']:
            print(f"\n  [{model_name}] Training head (pool={pool})...", end=" ", flush=True)
            tr_acc, val_acc = train_classification_head(
                model, train_head_loader, val_head_loader,
                pool=pool, epochs=30, lr=1e-3)
            print(f"val={val_acc*100:.1f}%")
            head_results.append({
                'model': model_name, 'pool': pool,
                'train_acc': tr_acc, 'val_acc': val_acc,
            })

    # ── Step 4: Baselines ─────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  BASELINES")
    print("=" * 70)

    # Raw pixel logistic regression
    train_raw = np.fromfile(str(MNIST_DATA / 'train.bytes'), dtype=np.uint8)
    val_raw = np.fromfile(str(MNIST_DATA / 'val.bytes'), dtype=np.uint8)
    train_pix = train_raw[:n_train*784].reshape(n_train, 784).astype(float) / 255.0
    val_pix = val_raw[:n_val*784].reshape(n_val, 784).astype(float) / 255.0

    pix_lr_train, pix_lr_val = logistic_probe(
        train_pix, train_labels[:n_train], val_pix, val_labels[:n_val])
    print(f"  Raw pixels (784-dim) + LogReg: train={pix_lr_train*100:.1f}%, val={pix_lr_val*100:.1f}%")

    pix_mlp_train, pix_mlp_val = mlp_probe(
        train_pix, train_labels[:n_train], val_pix, val_labels[:n_val], hidden=256)
    print(f"  Raw pixels (784-dim) + MLP:    train={pix_mlp_train*100:.1f}%, val={pix_mlp_val*100:.1f}%")

    # ── Summary table ─────────────────────────────────────────────────

    print("\n" + "=" * 70)
    print("  SUMMARY: FEATURE EXTRACTION + LOGISTIC REGRESSION")
    print("=" * 70)
    print(f"  {'Model':<14s} {'Method':<8s} {'Dim':>5s} {'LogReg':>8s} {'MLP':>8s}")
    print(f"  {'-'*14} {'-'*8} {'-'*5} {'-'*8} {'-'*8}")

    for method in methods:
        for model_name in models:
            lr_row = next(r for r in results
                         if r['model'] == model_name and r['method'] == method
                         and r['classifier'] == 'logreg')
            mlp_row = next(r for r in results
                          if r['model'] == model_name and r['method'] == method
                          and r['classifier'] == 'mlp')
            print(f"  {model_name:<14s} {method:<8s} {lr_row['feat_dim']:>5d} "
                  f"{lr_row['val_acc']*100:>7.1f}% {mlp_row['val_acc']*100:>7.1f}%")
        print()

    print(f"  {'raw_pixels':<14s} {'—':<8s} {'784':>5s} {pix_lr_val*100:>7.1f}% {pix_mlp_val*100:>7.1f}%")

    print("\n" + "=" * 70)
    print("  SUMMARY: FROZEN BACKBONE + TRAINED HEAD")
    print("=" * 70)
    print(f"  {'Model':<14s} {'Pool':<8s} {'Val Acc':>8s}")
    print(f"  {'-'*14} {'-'*8} {'-'*8}")
    for r in head_results:
        print(f"  {r['model']:<14s} {r['pool']:<8s} {r['val_acc']*100:>7.1f}%")

    # ── Save all results ──────────────────────────────────────────────

    all_results = {
        'feature_extraction': results,
        'classification_head': head_results,
        'baselines': {
            'raw_pixel_logreg': {'train': pix_lr_train, 'val': pix_lr_val},
            'raw_pixel_mlp': {'train': pix_mlp_train, 'val': pix_mlp_val},
        }
    }
    out_path = BASE_DIR / 'mnist_classification_results.json'
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  All results saved to {out_path}")


if __name__ == '__main__':
    main()
