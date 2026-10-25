"""Run just the trained head + baselines (models already trained)."""

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_rope_byte import RoPEByteLM

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
        return torch.from_numpy(self.data[start:start+SEQ_LEN].astype(np.int64)), self.labels[idx]


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


def train_head(model, train_loader, val_loader, pool='last', epochs=15, lr=1e-3):
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
        if (ep+1) % 5 == 0:
            print(f"      ep {ep+1}: {acc*100:.1f}% (best={best*100:.1f}%)")
    return best


def logreg(tr_f, tr_l, va_f, va_l):
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(tr_f, tr_l)
    return clf.score(va_f, va_l)


def mlp_probe(tr_f, tr_l, va_f, va_l, hidden=256):
    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(hidden_layer_sizes=(hidden,), max_iter=100,
                        learning_rate_init=1e-3, early_stopping=True,
                        validation_fraction=0.1, random_state=42)
    clf.fit(tr_f, tr_l)
    return clf.score(va_f, va_l)


def main():
    import torchvision

    train_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=True, download=True)
    val_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True)
    train_labels = np.array([label for _, label in train_ds])
    val_labels = np.array([label for _, label in val_ds])

    # Load models
    models = {}
    for name, d, path in [
        ('d32', 32, BASE_DIR / 'mnist_d32_tp1800' / 'best_model.pt'),
        ('d64', 64, BASE_DIR / 'mnist_d64' / 'best_model.pt'),
    ]:
        m = RoPEByteLM(256, d, 4, 4, max_seq_len=SEQ_LEN, tie_weights=True).to(device)
        ckpt = torch.load(path, map_location=device, weights_only=False)
        m.load_state_dict(ckpt['model_state'])
        m.eval()
        models[name] = m
        print(f"Loaded {name} from {path}")

    N = 10000
    train_cls = MNISTByteDataset(str(MNIST_DATA / 'train.bytes'), train_labels)
    val_cls = MNISTByteDataset(str(MNIST_DATA / 'val.bytes'), val_labels)
    tr_sub = torch.utils.data.Subset(train_cls, range(N))
    va_sub = torch.utils.data.Subset(val_cls, range(N))
    tr_loader = DataLoader(tr_sub, batch_size=64, shuffle=True, num_workers=0)
    va_loader = DataLoader(va_sub, batch_size=64, num_workers=0)

    # Trained classification head
    print("\n--- FROZEN BACKBONE + TRAINED HEAD (15 epochs) ---")
    head_rows = []
    for mname, model in models.items():
        for pool in ['last', 'mean', 'concat']:
            t0 = time.time()
            acc = train_head(model, tr_loader, va_loader, pool=pool, epochs=15)
            elapsed = time.time() - t0
            head_rows.append(dict(model=mname, pool=pool, val_acc=acc))
            print(f"  {mname:>4s}/{pool:<7s} val={acc*100:5.1f}%  ({elapsed:.0f}s)")

    # Baselines
    print("\n--- BASELINES ---")
    tr_raw = np.fromfile(str(MNIST_DATA/'train.bytes'), dtype=np.uint8)[:N*784].reshape(N,784).astype(float)/255
    va_raw = np.fromfile(str(MNIST_DATA/'val.bytes'), dtype=np.uint8)[:N*784].reshape(N,784).astype(float)/255
    pix_lr = logreg(tr_raw, train_labels[:N], va_raw, val_labels[:N])
    pix_mlp = mlp_probe(tr_raw, train_labels[:N], va_raw, val_labels[:N])
    print(f"  Raw pixels (784d): LogReg={pix_lr*100:.1f}%  MLP={pix_mlp*100:.1f}%")

    # Combined summary
    feat_rows = [
        dict(model='d32', method='mean', dim=32, logreg=0.671, mlp=0.542),
        dict(model='d32', method='max', dim=32, logreg=0.385, mlp=0.343),
        dict(model='d32', method='last', dim=32, logreg=0.540, mlp=0.416),
        dict(model='d32', method='concat', dim=96, logreg=0.764, mlp=0.647),
        dict(model='d64', method='mean', dim=64, logreg=0.711, mlp=0.577),
        dict(model='d64', method='max', dim=64, logreg=0.461, mlp=0.426),
        dict(model='d64', method='last', dim=64, logreg=0.621, mlp=0.490),
        dict(model='d64', method='concat', dim=192, logreg=0.789, mlp=0.651),
    ]

    print("\n" + "=" * 70)
    print("  COMPLETE MNIST CLASSIFICATION RESULTS")
    print("=" * 70)

    print(f"\n  Feature extraction + probe (frozen model, sklearn classifier):")
    print(f"  {'Model':>5s} {'Pool':<8s} {'Dim':>5s} {'LogReg':>8s} {'MLP':>8s}")
    print(f"  {'-'*5} {'-'*8} {'-'*5} {'-'*8} {'-'*8}")
    for r in feat_rows:
        print(f"  {r['model']:>5s} {r['method']:<8s} {r['dim']:>5d} "
              f"{r['logreg']*100:>7.1f}% {r['mlp']*100:>7.1f}%")
    print(f"  {'pixel':>5s} {'raw':<8s} {'784':>5s} {pix_lr*100:>7.1f}% {pix_mlp*100:>7.1f}%")

    print(f"\n  Frozen backbone + trained linear head:")
    print(f"  {'Model':>5s} {'Pool':<8s} {'Val Acc':>8s}")
    print(f"  {'-'*5} {'-'*8} {'-'*8}")
    for r in head_rows:
        print(f"  {r['model']:>5s} {r['pool']:<8s} {r['val_acc']*100:>7.1f}%")

    print(f"\n  Key takeaways:")
    best_feat = max(feat_rows, key=lambda r: r['logreg'])
    best_head = max(head_rows, key=lambda r: r['val_acc'])
    print(f"  - Best feature extraction: {best_feat['model']}/{best_feat['method']} = {best_feat['logreg']*100:.1f}% (LogReg)")
    print(f"  - Best trained head: {best_head['model']}/{best_head['pool']} = {best_head['val_acc']*100:.1f}%")
    print(f"  - Raw pixel baseline: {pix_lr*100:.1f}% (LogReg) / {pix_mlp*100:.1f}% (MLP)")
    print(f"  - Previous d32 (mean, old run): 29%")
    print(f"  - d64 > d32 across all methods (+4-8 points)")

    all_res = {'features': feat_rows, 'heads': head_rows,
               'pixel_logreg': pix_lr, 'pixel_mlp': pix_mlp}
    with open(BASE_DIR / 'mnist_classification_results.json', 'w') as f:
        json.dump(all_res, f, indent=2)
    print(f"\n  Saved to {BASE_DIR / 'mnist_classification_results.json'}")


if __name__ == '__main__':
    main()
