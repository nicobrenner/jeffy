"""MNIST classification: train d64, evaluate d32 vs d64 vs baselines.
Trains d64 at 450 tok/param (MNIST converges by tp~500 based on d32 trajectory).
"""

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
SEQ_LEN = 784


class MNISTByteDataset(Dataset):
    def __init__(self, bytes_path, labels):
        self.data = np.fromfile(bytes_path, dtype=np.uint8)
        self.labels = labels
        self.n_images = len(self.data) // SEQ_LEN

    def __len__(self):
        return min(self.n_images, len(self.labels))

    def __getitem__(self, idx):
        start = idx * SEQ_LEN
        img = torch.from_numpy(self.data[start:start+SEQ_LEN].astype(np.int64))
        return img, self.labels[idx]


@torch.no_grad()
def extract_features(model, loader, method='last'):
    model.eval()
    feats, labs = [], []
    for imgs, labels in loader:
        ids = imgs.to(device)
        x = model.token_emb(ids)
        mask = model.causal_mask
        for layer in model.layers:
            x = layer(x, model.freqs, mask)
        x = model.norm(x)
        if method == 'mean': f = x.mean(dim=1)
        elif method == 'max': f = x.max(dim=1).values
        elif method == 'last': f = x[:, -1, :]
        elif method == 'concat':
            f = torch.cat([x.mean(dim=1), x.max(dim=1).values, x[:, -1, :]], dim=-1)
        feats.append(f.cpu().numpy())
        labs.append(labels.numpy())
    return np.concatenate(feats), np.concatenate(labs)


def logreg(tr_f, tr_l, va_f, va_l):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(tr_f, tr_l)
    return clf.score(va_f, va_l)


def mlp(tr_f, tr_l, va_f, va_l, hidden=128):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=50,
                        learning_rate_init=1e-3, early_stopping=True,
                        validation_fraction=0.1, random_state=42)
    clf.fit(tr_f, tr_l)
    return clf.score(va_f, va_l)


class ClsHead(nn.Module):
    def __init__(self, backbone, pool='last'):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        d = backbone.d_model * (3 if pool == 'concat' else 1)
        self.head = nn.Linear(d, 10)
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, ids):
        x = self.backbone.token_emb(ids)
        mask = self.backbone.causal_mask
        for layer in self.backbone.layers:
            x = layer(x, self.backbone.freqs, mask)
        x = self.backbone.norm(x)
        if self.pool == 'last': f = x[:, -1, :]
        elif self.pool == 'mean': f = x.mean(dim=1)
        elif self.pool == 'concat':
            f = torch.cat([x.mean(dim=1), x.max(dim=1).values, x[:, -1, :]], dim=-1)
        return self.head(f)


def train_head(model, train_loader, val_loader, pool='last', epochs=30, lr=1e-3):
    clf = ClsHead(model, pool).to(device)
    opt = torch.optim.Adam(clf.head.parameters(), lr=lr)
    best = 0
    for ep in range(epochs):
        clf.train()
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            loss = F.cross_entropy(clf(imgs), labels)
            opt.zero_grad(); loss.backward(); opt.step()
        clf.eval()
        c = t = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                c += (clf(imgs).argmax(1) == labels).sum().item()
                t += len(labels)
        acc = c / t
        best = max(best, acc)
    return best


def load_or_train(name, d_model, n_layers, n_heads, tok_param, batch_size=32):
    run_dir = BASE_DIR / f'mnist_{name}'
    run_dir.mkdir(parents=True, exist_ok=True)
    best_path = run_dir / 'best_model.pt'
    ckpt_path = run_dir / 'checkpoint.pt'

    model = RoPEByteLM(vocab_size=256, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, max_seq_len=SEQ_LEN, tie_weights=True).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if best_path.exists():
        print(f"  Loading {name} from {best_path}")
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        return model

    total_tokens = int(tok_param * n_params)
    n_steps = total_tokens // (batch_size * SEQ_LEN)
    print(f"  Training {name}: {n_params:,} params, {n_steps:,} steps, {total_tokens/1e6:.1f}M tok")

    train_ds = ByteDataset(str(MNIST_DATA / 'train.bytes'), SEQ_LEN, max_tokens=total_tokens)
    val_ds = ByteDataset(str(MNIST_DATA / 'val.bytes'), SEQ_LEN, max_tokens=SEQ_LEN * 200)
    tl = DataLoader(train_ds, batch_size=batch_size, num_workers=0)
    vl = DataLoader(val_ds, batch_size=batch_size, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    start_step = 0
    best_ppl = float('inf')
    best_state = None

    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt['step']
        best_ppl = ckpt['best_ppl']
        best_state = ckpt.get('best_model_state')
        print(f"  Resumed at step {start_step}")

    model.train()
    step = 0
    t0 = time.time()
    eval_every = max(50, n_steps // 20)
    save_every = max(200, n_steps // 5)

    for batch in tl:
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
        tp = step * batch_size * SEQ_LEN / n_params

        if step % eval_every == 0:
            val_loss, val_ppl = evaluate(model, vl, device)
            tag = ""
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                tag = " *BEST*"
            print(f"    step {step}/{n_steps} ppl={val_ppl:.3f} tp={tp:.0f}{tag}")
            model.train()

        if step % save_every == 0:
            torch.save({'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'step': step, 'best_ppl': best_ppl,
                        'best_model_state': best_state}, ckpt_path)

        if step >= n_steps:
            break

    val_loss, val_ppl = evaluate(model, vl, device)
    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"  Done: best_ppl={best_ppl:.3f}, time={time.time()-t0:.0f}s")

    if best_state:
        model.load_state_dict(best_state)
        torch.save({'model_state': best_state,
                     'config': {'d_model': d_model, 'n_layers': n_layers, 'n_heads': n_heads},
                     'val_ppl': best_ppl}, best_path)
    with open(run_dir / 'results.json', 'w') as f:
        json.dump({'d_model': d_model, 'n_params': n_params,
                   'tok_param': tok_param, 'best_val_ppl': round(best_ppl, 4)}, f, indent=2)

    model.eval()
    return model


def main():
    import torchvision
    print("=" * 70)
    print("  MNIST BYTE MODEL CLASSIFICATION")
    print("=" * 70)

    train_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=True, download=True)
    val_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True)
    train_labels = np.array([label for _, label in train_ds])
    val_labels = np.array([label for _, label in val_ds])

    # ── Train/load models ──
    print("\n--- MODELS ---")
    models = {}

    # d32 already trained at 1800 t/p — reuse that checkpoint
    d32_dir = BASE_DIR / 'mnist_d32_tp1800'
    if d32_dir.exists() and (d32_dir / 'best_model.pt').exists():
        m = RoPEByteLM(256, 32, 4, 4, max_seq_len=SEQ_LEN, tie_weights=True).to(device)
        ckpt = torch.load(d32_dir / 'best_model.pt', map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        models['d32'] = m
        print(f"  Loaded d32 from {d32_dir}")
    else:
        models['d32'] = load_or_train('d32', 32, 4, 4, 1800, batch_size=32)

    # d64: train at 450 t/p (MNIST converges early)
    models['d64'] = load_or_train('d64', 64, 4, 4, 450, batch_size=32)

    # ── Classification datasets ──
    N = 10000
    train_cls = MNISTByteDataset(str(MNIST_DATA / 'train.bytes'), train_labels)
    val_cls = MNISTByteDataset(str(MNIST_DATA / 'val.bytes'), val_labels)
    tr_sub = torch.utils.data.Subset(train_cls, range(N))
    va_sub = torch.utils.data.Subset(val_cls, range(N))
    tr_loader = DataLoader(tr_sub, batch_size=64, num_workers=0)
    va_loader = DataLoader(va_sub, batch_size=64, num_workers=0)

    # ── Feature extraction ──
    print("\n--- FEATURE EXTRACTION + PROBES ---")
    rows = []
    for mname, model in models.items():
        for method in ['mean', 'max', 'last', 'concat']:
            t0 = time.time()
            tr_f, tr_l = extract_features(model, tr_loader, method)
            va_f, va_l = extract_features(model, va_loader, method)
            dim = tr_f.shape[1]
            lr_acc = logreg(tr_f, tr_l, va_f, va_l)
            mlp_acc = mlp(tr_f, tr_l, va_f, va_l, hidden=min(256, dim*2))
            t1 = time.time()
            rows.append(dict(model=mname, method=method, dim=dim,
                             logreg=lr_acc, mlp=mlp_acc))
            print(f"  {mname:>4s}/{method:<7s} dim={dim:>4d}  LogReg={lr_acc*100:5.1f}%  MLP={mlp_acc*100:5.1f}%  ({t1-t0:.0f}s)")

    # ── Trained classification head ──
    print("\n--- FROZEN BACKBONE + TRAINED HEAD (30 epochs) ---")
    tr_shuf = DataLoader(tr_sub, batch_size=64, shuffle=True, num_workers=0)
    head_rows = []
    for mname, model in models.items():
        for pool in ['last', 'concat']:
            t0 = time.time()
            acc = train_head(model, tr_shuf, va_loader, pool=pool, epochs=30)
            head_rows.append(dict(model=mname, pool=pool, val_acc=acc))
            print(f"  {mname:>4s}/{pool:<7s} val={acc*100:5.1f}%  ({time.time()-t0:.0f}s)")

    # ── Baselines ──
    print("\n--- BASELINES ---")
    tr_raw = np.fromfile(str(MNIST_DATA/'train.bytes'), dtype=np.uint8)[:N*784].reshape(N,784).astype(float)/255
    va_raw = np.fromfile(str(MNIST_DATA/'val.bytes'), dtype=np.uint8)[:N*784].reshape(N,784).astype(float)/255
    pix_lr = logreg(tr_raw, train_labels[:N], va_raw, val_labels[:N])
    pix_mlp = mlp(tr_raw, train_labels[:N], va_raw, val_labels[:N], hidden=256)
    print(f"  Raw pixels (784d): LogReg={pix_lr*100:.1f}%  MLP={pix_mlp*100:.1f}%")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  FULL RESULTS TABLE")
    print("=" * 70)
    print(f"  {'Model':>5s} {'Pool':<8s} {'Dim':>5s} {'LogReg':>8s} {'MLP':>8s}")
    print(f"  {'-'*5} {'-'*8} {'-'*5} {'-'*8} {'-'*8}")
    for r in rows:
        print(f"  {r['model']:>5s} {r['method']:<8s} {r['dim']:>5d} "
              f"{r['logreg']*100:>7.1f}% {r['mlp']*100:>7.1f}%")
    print(f"  {'pixels':>5s} {'raw':<8s} {'784':>5s} {pix_lr*100:>7.1f}% {pix_mlp*100:>7.1f}%")

    print(f"\n  {'Model':>5s} {'Pool':<8s} {'Trained Head':>12s}")
    print(f"  {'-'*5} {'-'*8} {'-'*12}")
    for r in head_rows:
        print(f"  {r['model']:>5s} {r['pool']:<8s} {r['val_acc']*100:>11.1f}%")

    # Save
    all_res = {'features': rows, 'heads': head_rows,
               'pixel_logreg': pix_lr, 'pixel_mlp': pix_mlp}
    with open(BASE_DIR / 'mnist_classification_results.json', 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f"\n  Saved to {BASE_DIR / 'mnist_classification_results.json'}")


if __name__ == '__main__':
    main()
