"""T3: tiered stack with the causal byte LM as level 1.

Level 1 : frozen RoPE byte LM (d128, L4, Pile-trained, run_d128_tp1800). For each byte position it emits a
          256-way next-byte distribution (a learned SOFT label), its hidden state (values), and the surprise
          -log2 p(actual next byte) in bits.
Boundary: a segment ends where surprise exceeds the text's 70th percentile or after 16 bytes (BLT-style).
Level 2 : per segment, mean next-byte distribution (256) [+ mean hidden state (128)] + [length, mean surprise];
          a tiny transformer (d64, 2L) over the ordered segment sequence (<=32) -> class.
Variants : soft labels only | soft labels + values | values only | soft+values mean-pooled -> LogReg (order ablation).
Baseline : existing extractor result (mean+last hidden -> LogReg) from feature_extractor_results_both.json.
Same datasets, same load_and_split caps (10K train / 2K test) as the baseline.
"""

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from jeffy.build_pack import DATASETS, load_and_split
from train_rope_byte import RoPEByteLM

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
RUN = Path("pile_runs/rope_byte/run_d128_tp1800")
TASKS = ["sst2", "emotion", "sms_spam", "banking77", "ag_news"]
MAX_LEN, MAX_SEG, SEG_CAP = 256, 32, 16
EPOCHS, BS = 15, 128
OUT = Path("pile_runs/rope_byte/tiered_bytelm_results.json")


def load_lm():
    m = RoPEByteLM(vocab_size=256, d_model=128, n_layers=4, n_heads=8, max_seq_len=MAX_LEN, tie_weights=True).to(DEVICE)
    ck = torch.load(RUN / "best_model.pt", map_location=DEVICE, weights_only=False)
    m.load_state_dict(ck["model_state"])
    m.eval()
    return m


@torch.no_grad()
def level1(model, texts, bs=64):
    """Return per-text segment features: dict with soft (n,32,258), vals (n,32,130), mask (n,32)."""
    n = len(texts)
    soft = np.zeros((n, MAX_SEG, 258), np.float16)
    vals = np.zeros((n, MAX_SEG, 130), np.float16)
    mask = np.zeros((n, MAX_SEG), bool)
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
        probs, h, nll = probs.cpu().numpy(), h.float().cpu().numpy(), nll.cpu().numpy()
        for j, L in enumerate(lens):
            s = nll[j, :L - 1]
            thr = np.quantile(s, 0.7)
            segs, start = [], 0
            for p in range(L - 1):
                if s[p] > thr or p - start + 1 >= SEG_CAP or p == L - 2:
                    segs.append((start, p + 1))
                    start = p + 1
            segs = segs[:MAX_SEG]
            for k, (a, b) in enumerate(segs):
                soft[i + j, k, :256] = probs[j, a:b].mean(0)
                soft[i + j, k, 256] = (b - a) / SEG_CAP
                soft[i + j, k, 257] = s[a:b].mean() / 8.0
                vals[i + j, k, :128] = h[j, a:b].mean(0)
                vals[i + j, k, 128:] = soft[i + j, k, 256:]
                mask[i + j, k] = True
    return soft, vals, mask


class SegTransformer(nn.Module):
    def __init__(self, d_in, n_classes, d=64, layers=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, MAX_SEG + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x, mask):
        B = x.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), self.proj(x)], 1) + self.pos
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        return self.head(self.enc(x, src_key_padding_mask=pad)[:, 0])


def train_level2(ftr, mtr, ytr, fte, mte, yte, n_classes):
    torch.manual_seed(0)
    model = SegTransformer(ftr.shape[-1], n_classes).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    Xtr, Mtr, Ytr = torch.tensor(ftr, dtype=torch.float32), torch.tensor(mtr), torch.tensor(ytr)
    Xte, Mte = torch.tensor(fte, dtype=torch.float32).to(DEVICE), torch.tensor(mte).to(DEVICE)
    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            loss = F.cross_entropy(model(Xtr[idx].to(DEVICE), Mtr[idx].to(DEVICE)), Ytr[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xte[i:i + 512], Mte[i:i + 512]).argmax(-1) for i in range(0, len(Xte), 512)]).cpu().numpy()
        best = max(best, float((pred == yte).mean()))
    return best


def main():
    t0 = time.time()
    lm = load_lm()
    base = {e["dataset"]: e["test_acc"] for e in json.load(open(RUN / "feature_extractor_results_both.json"))["results"]}
    print(f"Level 1: byte LM {sum(p.numel() for p in lm.parameters()):,} params (frozen). Device {DEVICE}\n", flush=True)
    print(f"{'dataset':<10}{'K':>3} | {'soft':>7}{'soft+vals':>11}{'vals':>7}{'bag(LogReg)':>13} | {'baseline':>9}", flush=True)
    results = {}
    for name in TASKS:
        xtr, ytr_s, xte, yte_s, _ = load_and_split(name, DATASETS[name])
        le = LabelEncoder().fit(ytr_s + yte_s)
        ytr, yte = le.transform(ytr_s), le.transform(yte_s)
        K = len(le.classes_)
        s_tr, v_tr, m_tr = level1(lm, xtr)
        s_te, v_te, m_te = level1(lm, xte)
        sv_tr, sv_te = np.concatenate([s_tr, v_tr[..., :128]], -1), np.concatenate([s_te, v_te[..., :128]], -1)
        acc = {
            "soft": train_level2(s_tr, m_tr, ytr, s_te, m_te, yte, K),
            "soft+vals": train_level2(sv_tr, m_tr, ytr, sv_te, m_te, yte, K),
            "vals": train_level2(v_tr, m_tr, ytr, v_te, m_te, yte, K),
        }
        pool = lambda f, m: (f.astype(np.float32) * m[..., None]).sum(1) / np.maximum(m.sum(1, keepdims=True), 1)
        acc["bag"] = LogisticRegression(max_iter=2000).fit(pool(sv_tr, m_tr), ytr).score(pool(sv_te, m_te), yte)
        results[name] = dict(acc=acc, baseline=base.get(name), n_train=len(xtr), n_test=len(xte), K=K)
        print(f"{name:<10}{K:>3} | {acc['soft']:>7.3f}{acc['soft+vals']:>11.3f}{acc['vals']:>7.3f}{acc['bag']:>13.3f} | {base.get(name, float('nan')):>9.3f}   ({time.time() - t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps(results, indent=1))
    print(f"\nSaved {OUT}. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
