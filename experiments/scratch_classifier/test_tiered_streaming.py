"""MVE for the streaming-stack hypothesis (builds on T3).

(1) Streaming: after each segment closes, the level-2 model predicts from the prefix seen so far.
    Report accuracy vs fraction of bytes consumed, for the stack and for the pooled-LogReg baseline.
(2) Segmentation: surprise-driven boundaries (level-1 signal as the clock) vs fixed 8-byte segments.

Level 1 = frozen Pile byte LM (d128); level 2 = tiny transformer over ordered segment hidden states (T3's best variant).
Outputs: pile_runs/rope_byte/tiered_streaming_results.json and tiered_streaming_curves.png
"""

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from jeffy.build_pack import DATASETS, load_and_split
from test_tiered_bytelm import DEVICE, MAX_LEN, MAX_SEG, SEG_CAP, SegTransformer, load_lm

TASKS = ["emotion", "banking77", "sst2"]
EPOCHS, BS = 15, 128
FRACS = np.linspace(0.1, 1.0, 10)
OUT_JSON = Path("pile_runs/rope_byte/tiered_streaming_results.json")
OUT_PNG = Path("pile_runs/rope_byte/tiered_streaming_curves.png")


@torch.no_grad()
def level1(model, texts, mode, bs=64):
    """Segment hidden-state features (n,32,130), mask (n,32), and cumulative byte fraction at each segment end (n,32)."""
    n = len(texts)
    vals = np.zeros((n, MAX_SEG, 130), np.float16)
    mask = np.zeros((n, MAX_SEG), bool)
    frac = np.ones((n, MAX_SEG), np.float32)
    for i in range(0, n, bs):
        chunk = texts[i:i + bs]
        ids = torch.zeros(len(chunk), MAX_LEN, dtype=torch.long)
        lens = []
        for j, t in enumerate(chunk):
            b = list(t.encode("utf-8"))[:MAX_LEN]
            lens.append(max(len(b), 2))
            ids[j, :len(b)] = torch.tensor(b)
        ids = ids.to(DEVICE)
        x = model.token_emb(ids)
        for layer in model.layers:
            x = layer(x, model.freqs, model.causal_mask)
        h = model.norm(x)
        probs = F.softmax(model.lm_head(h).float(), dim=-1)
        nll = -torch.log2(probs[:, :-1].gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) + 1e-9)
        h, nll = h.float().cpu().numpy(), nll.cpu().numpy()
        for j, L in enumerate(lens):
            s = nll[j, :L - 1]
            if mode == "surprise":
                thr = np.quantile(s, 0.7)
                cut = lambda p, start: s[p] > thr or p - start + 1 >= SEG_CAP
            else:
                cut = lambda p, start: p - start + 1 >= 8
            segs, start = [], 0
            for p in range(L - 1):
                if cut(p, start) or p == L - 2:
                    segs.append((start, p + 1))
                    start = p + 1
            segs = segs[:MAX_SEG]
            for k, (a, b) in enumerate(segs):
                vals[i + j, k, :128] = h[j, a:b].mean(0)
                vals[i + j, k, 128] = (b - a) / SEG_CAP
                vals[i + j, k, 129] = s[a:b].mean() / 8.0
                mask[i + j, k] = True
                frac[i + j, k] = (b + 1) / L
    return vals, mask, frac


def train_level2(ftr, mtr, ytr, n_classes):
    torch.manual_seed(0)
    model = SegTransformer(ftr.shape[-1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    X, M, Y = torch.tensor(ftr, dtype=torch.float32), torch.tensor(mtr), torch.tensor(ytr)
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            loss = F.cross_entropy(model(X[idx].to(DEVICE), M[idx].to(DEVICE)), Y[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


def pool(f, m):
    return (f.astype(np.float32) * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1)


@torch.no_grad()
def streaming_curve(predict, fte, mte, frac, yte):
    """predict(features, mask) -> labels. Accuracy at each byte fraction, using the prefix of segments closed by then."""
    accs = []
    for fr in FRACS:
        pm = mte & (frac <= fr + 1e-6)
        pm[:, 0] |= mte[:, 0]  # always at least the first segment
        pred = predict(fte, pm)
        accs.append(float((pred == yte).mean()))
    return accs


def main():
    t0 = time.time()
    lm = load_lm()
    results = {}
    fig, axes = plt.subplots(1, len(TASKS), figsize=(5 * len(TASKS), 4), sharey=False)
    print(f"{'dataset':<10}{'mode':<9} | acc @25% @50% @75% @100% of bytes | bytes to 90% of final", flush=True)
    for ax, name in zip(axes, TASKS):
        xtr, ytr_s, xte, yte_s, _ = load_and_split(name, DATASETS[name])
        le = LabelEncoder().fit(ytr_s + yte_s)
        ytr, yte = le.transform(ytr_s), le.transform(yte_s)
        K = len(le.classes_)
        results[name] = {}
        for mode in ("surprise", "fixed"):
            v_tr, m_tr, _ = level1(lm, xtr, mode)
            v_te, m_te, fr_te = level1(lm, xte, mode)
            l2 = train_level2(v_tr, m_tr, ytr, K)

            def stack_predict(f, m):
                X, M = torch.tensor(f, dtype=torch.float32).to(DEVICE), torch.tensor(m).to(DEVICE)
                return torch.cat([l2(X[i:i + 512], M[i:i + 512]).argmax(-1) for i in range(0, len(X), 512)]).cpu().numpy()

            curve = streaming_curve(stack_predict, v_te, m_te, fr_te, yte)
            results[name][f"stack_{mode}"] = curve
            ax.plot(FRACS, curve, marker="o", label=f"stack ({mode} segments)")
            if mode == "surprise":
                lr = LogisticRegression(max_iter=2000).fit(pool(v_tr, m_tr), ytr)
                bag_curve = streaming_curve(lambda f, m: lr.predict(pool(f, m)), v_te, m_te, fr_te, yte)
                results[name]["bag_surprise"] = bag_curve
                ax.plot(FRACS, bag_curve, marker="s", linestyle="--", label="pooled → LogReg (baseline)")
            for label, c in ([(f"stack-{mode}", curve)] + ([("bag", bag_curve)] if mode == "surprise" else [])):
                target = 0.9 * c[-1]
                reach = next(fr for fr, a in zip(FRACS, c) if a >= target)
                print(f"{name:<10}{label:<9} | {c[1]:.3f} {c[4]:.3f} {c[7]:.3f} {c[9]:.3f}            | {reach:.0%}   ({time.time() - t0:.0f}s)", flush=True)
        ax.set_title(name)
        ax.set_xlabel("fraction of bytes seen")
        ax.set_ylabel("test accuracy")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        OUT_JSON.write_text(json.dumps(dict(fracs=FRACS.tolist(), results=results), indent=1))
    fig.suptitle("Streaming classification over a frozen byte LM: accuracy vs bytes consumed")
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=120)
    print(f"\nSaved {OUT_JSON} and {OUT_PNG}. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
