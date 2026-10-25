"""Fast MNIST classification eval: train d64, evaluate d32 vs d64 vs baselines."""

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


class MNISTByteDataset(Dataset):
    def __init__(self, bytes_path, labels, seq_len=784):
        self.data = np.fromfile(bytes_path, dtype=np.uint8)
        self.labels = labels
        self.seq_len = seq_len
        self.n_images = len(self.data) // seq_len

    def __len__(self):
        return min(self.n_images, len(self.labels))

    def __getitem__(self, idx):
        start = idx * self.seq_len
        img = torch.from_numpy(self.data[start:start+self.seq_len].astype(np.int64))
        return img, self.labels[idx]


@torch.no_grad()
def extract_features_batch(model, dataloader, method='last'):
    model.eval()
    all_feats, all_labels = [], []
    for imgs, labels in dataloader:
        ids = imgs.to(device)
        x = model.token_emb(ids)
        mask = model.causal_mask
        for layer in model.layers:
            x = layer(x, model.freqs, mask)
        x = model.norm(x)
        if method == 'mean':
            feat = x.mean(dim=1)
        elif method == 'max':
            feat = x.max(dim=1).values
        elif method == 'last':
            feat = x[:, -1, :]
        elif method == 'concat':
            feat = torch.cat([x.mean(dim=1), x.max(dim=1).values, x[:, -1, :]], dim=-1)
        all_feats.append(feat.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def logistic_probe(tr_f, tr_l, va_f, va_l):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(tr_f, tr_l)
    return clf.score(tr_f, tr_l), clf.score(va_f, va_l)


def mlp_probe(tr_f, tr_l, va_f, va_l, hidden=128):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=50,
                        learning_rate_init=1e-3, early_stopping=True,
                        validation_fraction=0.1, random_state=42)
    clf.fit(tr_f, tr_l)
    return clf.score(tr_f, tr_l), clf.score(va_f, va_l)


class ClassificationHead(nn.Module):
    def __init__(self, backbone, n_classes=10, pool='last'):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        d = backbone.d_model * (3 if pool == 'concat' else 1)
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


def train_head(model, train_loader, val_loader, pool='last', epochs=30, lr=1e-3):
    clf = ClassificationHead(model, 10, pool).to(device)
    opt = torch.optim.Adam(clf.head.parameters(), lr=lr)
    best_val = 0
    for ep in range(epochs):
        clf.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = F.cross_entropy(clf(imgs), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        clf.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                pred = clf(imgs).argmax(1)
                correct += (pred == labels).sum().item()
                total += len(labels)
        val_acc = correct / total
        best_val = max(best_val, val_acc)
        if (ep+1) % 10 == 0:
            print(f"      epoch {ep+1}: val_acc={val_acc*100:.1f}% (best={best_val*100:.1f}%)")
    return best_val


def train_mnist_model(d_model, n_layers, n_heads, tok_param, seq_len=784, batch_size=16, lr=3e-4):
    run_dir = BASE_DIR / f'mnist_d{d_model}_tp{int(tok_param)}'
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / 'best_model.pt'
    ckpt_path = run_dir / 'checkpoint.pt'

    model = RoPEByteLM(vocab_size=256, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, max_seq_len=seq_len, tie_weights=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if best_path.exists():
        print(f"    Loading existing model from {run_dir}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        return model, run_dir

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
        print(f"    Resumed at step {start_step:,}, best_ppl={best_ppl:.3f}")

    model.train()
    step = 0
    running_loss = 0
    t0 = time.time()
    eval_every = max(50, n_steps // 20)
    log_every = max(100, n_steps // 10)
    save_every = max(500, n_steps // 5)

    for batch in train_loader:
        step += 1
        if step <= start_step:
            scheduler.step()
            continue
        ids = batch.to(device)
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            ids[:, 1:].contiguous().view(-1))
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); scheduler.step()
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
            print(f"      step {step:>6,}/{n_steps:,} loss={avg:.4f} {step*batch_size*seq_len/elapsed:,.0f} tok/s")
            running_loss = 0

        if step % save_every == 0:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'step': step, 'best_ppl': best_ppl, 'best_model_state': best_state,
            }, ckpt_path)

        if step >= n_steps:
            break

    val_loss, val_ppl = evaluate(model, val_loader, device)
    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    print(f"    Final ppl={val_ppl:.3f}, best={best_ppl:.3f}, time={elapsed:.1f}s")

    results = {
        'model': 'mnist_rope_byte', 'd_model': d_model,
        'n_params': n_params, 'tok_param': tok_param,
        'best_val_ppl': round(best_ppl, 4),
    }
    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)

    if best_state:
        model.load_state_dict(best_state)
        torch.save({'model_state': best_state,
                     'config': {'d_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads},
                     'val_ppl': best_ppl}, best_path)

    model.eval()
    return model, run_dir


def main():
    import torchvision

    print("=" * 70)
    print("  MNIST CLASSIFICATION — BYTE MODEL EVALUATION")
    print("=" * 70)

    train_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=True, download=True)
    val_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True)
    train_labels = np.array([label for _, label in train_ds])
    val_labels = np.array([label for _, label in val_ds])

    # ── Train models ──
    print("\n  TRAINING MODELS")
    print("  " + "-" * 40)
    models = {}
    for name, d, nl, nh, tp in [
        ('d32_1800', 32, 4, 4, 1800),
        ('d64_1800', 64, 4, 4, 1800),
    ]:
        print(f"\n  [{name}]")
        model, _ = train_mnist_model(d, nl, nh, tp)
        models[name] = model

    # ── Feature extraction datasets ──
    n_train, n_val = 10000, 10000
    train_cls = MNISTByteDataset(str(MNIST_DATA / 'train.bytes'), train_labels, seq_len=784)
    val_cls = MNISTByteDataset(str(MNIST_DATA / 'val.bytes'), val_labels, seq_len=784)
    train_sub = torch.utils.data.Subset(train_cls, range(n_train))
    val_sub = torch.utils.data.Subset(val_cls, range(n_val))
    train_loader = DataLoader(train_sub, batch_size=32, num_workers=0)
    val_loader = DataLoader(val_sub, batch_size=32, num_workers=0)

    # ── Feature extraction + probes ──
    print("\n" + "=" * 70)
    print("  FEATURE EXTRACTION + CLASSIFICATION PROBES")
    print("=" * 70)

    methods = ['mean', 'max', 'last', 'concat']
    rows = []

    for mname, model in models.items():
        for method in methods:
            print(f"\n  [{mname} / {method}]", end=" ", flush=True)
            t0 = time.time()
            tr_f, tr_l = extract_features_batch(model, train_loader, method)
            va_f, va_l = extract_features_batch(model, val_loader, method)
            dim = tr_f.shape[1]
            lr_tr, lr_va = logistic_probe(tr_f, tr_l, va_f, va_l)
            mlp_tr, mlp_va = mlp_probe(tr_f, tr_l, va_f, va_l, hidden=min(256, dim*2))
            elapsed = time.time() - t0
            print(f"dim={dim} LogReg={lr_va*100:.1f}% MLP={mlp_va*100:.1f}% ({elapsed:.0f}s)")
            rows.append(dict(model=mname, method=method, dim=dim,
                             logreg_val=lr_va, mlp_val=mlp_va))

    # ── Trained classification head ──
    print("\n" + "=" * 70)
    print("  FROZEN BACKBONE + TRAINED HEAD (30 epochs)")
    print("=" * 70)

    train_shuf = DataLoader(train_sub, batch_size=32, shuffle=True, num_workers=0)
    head_rows = []
    for mname, model in models.items():
        for pool in ['last', 'mean', 'concat']:
            print(f"\n  [{mname} / {pool}]")
            val_acc = train_head(model, train_shuf, val_loader, pool=pool, epochs=30)
            head_rows.append(dict(model=mname, pool=pool, val_acc=val_acc))

    # ── Baselines ──
    print("\n" + "=" * 70)
    print("  BASELINES")
    print("=" * 70)

    train_raw = np.fromfile(str(MNIST_DATA / 'train.bytes'), dtype=np.uint8)
    val_raw = np.fromfile(str(MNIST_DATA / 'val.bytes'), dtype=np.uint8)
    tr_pix = train_raw[:n_train*784].reshape(n_train, 784).astype(float) / 255.0
    va_pix = val_raw[:n_val*784].reshape(n_val, 784).astype(float) / 255.0

    pix_lr_tr, pix_lr_va = logistic_probe(tr_pix, train_labels[:n_train], va_pix, val_labels[:n_val])
    pix_mlp_tr, pix_mlp_va = mlp_probe(tr_pix, train_labels[:n_train], va_pix, val_labels[:n_val], hidden=256)
    print(f"  Raw pixels (784d) + LogReg: {pix_lr_va*100:.1f}%")
    print(f"  Raw pixels (784d) + MLP:    {pix_mlp_va*100:.1f}%")

    # ── Summary tables ──
    print("\n" + "=" * 70)
    print("  RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n  Feature extraction + probe:")
    print(f"  {'Model':<12s} {'Pool':<8s} {'Dim':>5s} {'LogReg':>8s} {'MLP':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*5} {'-'*8} {'-'*8}")
    for r in rows:
        print(f"  {r['model']:<12s} {r['method']:<8s} {r['dim']:>5d} "
              f"{r['logreg_val']*100:>7.1f}% {r['mlp_val']*100:>7.1f}%")
    print(f"  {'raw_pixels':<12s} {'—':<8s} {'784':>5s} {pix_lr_va*100:>7.1f}% {pix_mlp_va*100:>7.1f}%")

    print(f"\n  Frozen backbone + trained head:")
    print(f"  {'Model':<12s} {'Pool':<8s} {'Val Acc':>8s}")
    print(f"  {'-'*12} {'-'*8} {'-'*8}")
    for r in head_rows:
        print(f"  {r['model']:<12s} {r['pool']:<8s} {r['val_acc']*100:>7.1f}%")

    # Save
    all_results = {'features': rows, 'heads': head_rows,
                   'baselines': {'pixel_logreg': pix_lr_va, 'pixel_mlp': pix_mlp_va}}
    with open(BASE_DIR / 'mnist_classification_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to {BASE_DIR / 'mnist_classification_results.json'}")


if __name__ == '__main__':
    main()
