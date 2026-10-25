"""T1: tiered classifier stack on MNIST bits — do labels alone carry enough information upward?

Level 1: 16 independent tile classifiers (7x7 bit tiles -> 10-way digit guess), each sees 49 bits.
Level 2: one classifier over what level 1 passes up. Variants of what is passed:
  hard      : argmax one-hot label per tile            (16 x 10 bits)
  soft      : probability vector per tile              (16 x 10 floats)
  soft+vals : probabilities + a 15-d value summary per tile (mean ink, 7 row sums, 7 col sums)
  binary    : the extreme — each tile unit emits ONE bit ("is this tile's best guess confident?") plus argmax -> 16 x (1+4) bits
Baselines: logistic regression on all 784 bits; MLP on 784 bits.
All level-1 units are trained on a disjoint half of the training set so level 2 never sees fitted-on-train outputs.
"""

import time

import numpy as np
import torch
import torchvision
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

T, G = 7, 4  # tile size, grid
rng = np.random.default_rng(0)


def load():
    tr = torchvision.datasets.MNIST("data", train=True, download=True)
    te = torchvision.datasets.MNIST("data", train=False, download=True)
    xtr = (tr.data.numpy() >= 128).astype(np.float32)
    xte = (te.data.numpy() >= 128).astype(np.float32)
    return xtr, tr.targets.numpy(), xte, te.targets.numpy()


def tiles(x):
    n = x.shape[0]
    return x.reshape(n, G, T, G, T).transpose(0, 1, 3, 2, 4).reshape(n, G * G, T * T)


def value_summary(tile_bits):
    t = tile_bits.reshape(-1, G * G, T, T)
    return np.concatenate([t.mean((2, 3), keepdims=False)[..., None], t.sum(3), t.sum(2)], axis=2)  # (n,16,15)


def lr(max_iter=300):
    return LogisticRegression(max_iter=max_iter, C=1.0)


def main():
    t0 = time.time()
    xtr, ytr, xte, yte = load()
    half = rng.permutation(len(xtr))
    a, b = half[:30000], half[30000:]
    print(f"MNIST bits: train {len(xtr)} (split 30K level-1 / 30K level-2), test {len(xte)}", flush=True)

    flat_tr, flat_te = xtr.reshape(len(xtr), -1), xte.reshape(len(xte), -1)
    base_lr = lr().fit(flat_tr[b], ytr[b]).score(flat_te, yte)
    print(f"baseline LogReg on 784 bits (30K): {base_lr:.4f}", flush=True)
    base_mlp = MLPClassifier((256,), max_iter=60, random_state=0).fit(flat_tr[b], ytr[b]).score(flat_te, yte)
    print(f"baseline MLP-256 on 784 bits (30K): {base_mlp:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    ttr, tte = tiles(xtr), tiles(xte)
    units = [lr(200).fit(ttr[a][:, i], ytr[a]) for i in range(G * G)]
    unit_acc = np.mean([u.score(tte[:, i], yte) for i, u in enumerate(units)])
    print(f"level-1 tile units: 16 x LogReg(49 bits -> 10), mean single-tile test acc {unit_acc:.3f}  ({time.time() - t0:.0f}s)", flush=True)

    def soft(tt):
        return np.stack([u.predict_proba(tt[:, i]) for i, u in enumerate(units)], axis=1)  # (n,16,10)

    s_tr, s_te = soft(ttr[b]), soft(tte)
    hard_tr, hard_te = (s_tr == s_tr.max(2, keepdims=True)).astype(np.float32), (s_te == s_te.max(2, keepdims=True)).astype(np.float32)
    v_tr, v_te = value_summary(ttr[b]), value_summary(tte)

    def binary(s):
        conf = (s.max(2) > 0.5).astype(np.float32)[..., None]
        code = ((s.argmax(2)[..., None] >> np.arange(4)) & 1).astype(np.float32)
        return np.concatenate([conf, code], axis=2)  # (n,16,5)

    variants = {
        "hard labels only (16x10 bits)": (hard_tr, hard_te),
        "binary units (16x5 bits)": (binary(s_tr), binary(s_te)),
        "soft labels (16x10)": (s_tr, s_te),
        "soft + value summary (16x25)": (np.concatenate([s_tr, v_tr], 2), np.concatenate([s_te, v_te], 2)),
        "value summary only (16x15)": (v_tr, v_te),
    }
    print(f"\n{'level-2 input':<34}{'dims':>6}{'LogReg':>9}{'MLP-128':>9}", flush=True)
    for name, (ftr, fte) in variants.items():
        ftr, fte = ftr.reshape(len(ftr), -1), fte.reshape(len(fte), -1)
        acc_lr = lr().fit(ftr, ytr[b]).score(fte, yte)
        acc_mlp = MLPClassifier((128,), max_iter=80, random_state=0).fit(ftr, ytr[b]).score(fte, yte)
        print(f"{name:<34}{ftr.shape[1]:>6}{acc_lr:>9.4f}{acc_mlp:>9.4f}", flush=True)
    print(f"\nbaselines: LogReg-784 {base_lr:.4f} | MLP-784 {base_mlp:.4f} | total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
