"""Does "keep the sequence" transfer to a strong encoder? Level 1 = frozen 11.7M HybridTransformer
(classification+MLM pretrained on the 6 Jeffy tasks); level 2 = tiny transformer over its per-token states.

Variants: tokens as segments (fine clock) | 4-token mean segments (coarse clock) | mean-pool -> LogReg (no order).
Baseline column: the bytecode classifier (pooled post-transformer state -> 2-layer MLP) from
bytecode_poc/all_datasets_testset_results.json. Same datasets, same 10K/2K caps.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from jeffy.build_pack import DATASETS, load_and_split
from train_hybrid import HybridTransformer

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CKPT = "checkpoints/best_hybrid_mask0.15.pt"
TRAINED = {"banking77", "ag_news", "imdb", "emotion", "sst2", "sms_spam"}
DEFAULT_TASKS = ["banking77", "ag_news", "imdb", "emotion", "sst2", "sms_spam", "dbpedia", "rottentomatoes"]
TASKS = [t for t in DATASETS if t not in TRAINED] if sys.argv[1:] == ["unseen"] else (sys.argv[1:] or DEFAULT_TASKS)
MAX_LEN, EPOCHS, BS = 128, 10, 128
OUT = Path("pile_runs/rope_byte/tiered_hybrid_results.json") if sys.argv[1:] != ["unseen"] else Path("pile_runs/rope_byte/tiered_hybrid_unseen_results.json")


class SeqHead(nn.Module):
    def __init__(self, d_in, n_classes, n_slots, d=64, layers=2):
        super().__init__()
        self.proj = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, n_slots + 1, d))
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        enc = nn.TransformerEncoderLayer(d, 4, 4 * d, dropout=0.1, batch_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(enc, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, n_classes)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, x, mask):
        B = x.shape[0]
        x = torch.cat([self.cls.expand(B, -1, -1), self.proj(x)], 1) + self.pos[:, : x.shape[1] + 1]
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        return self.head(self.enc(x, src_key_padding_mask=pad)[:, 0])


def load_encoder():
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    m = HybridTransformer(**ck["config"]).to(DEVICE)
    m.load_state_dict(ck["model_state"])
    m.eval()
    m.transformer.enable_nested_tensor = False  # MPS lacks the nested-tensor fast path
    m.transformer.use_nested_tensor = False
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    vmap = ck["vocab_map"]
    unk = vmap.get(tok.unk_token_id, 0)
    return m, tok, vmap, unk


@torch.no_grad()
def level1(m, tok, vmap, unk, texts, bs=128):
    """Per-token hidden states (n,128,128) fp16 and mask (n,128)."""
    H = np.zeros((len(texts), MAX_LEN, 128), np.float16)
    M = np.zeros((len(texts), MAX_LEN), bool)
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], truncation=True, max_length=MAX_LEN, padding="max_length", return_tensors="pt")
        ids = torch.tensor([[vmap.get(int(t), unk) for t in row] for row in enc["input_ids"]])
        am = enc["attention_mask"]
        x = m._transformer_forward(ids.to(DEVICE), am.to(DEVICE))
        H[i:i + bs] = x.float().cpu().numpy()
        M[i:i + bs] = am.bool().numpy()
    return H, M


def coarsen(H, M, k=4):
    n, L, d = H.shape
    Hc = H[:, : (L // k) * k].reshape(n, L // k, k, d).astype(np.float32)
    Mc = M[:, : (L // k) * k].reshape(n, L // k, k)
    w = Mc[..., None].astype(np.float32)
    return ((Hc * w).sum(2) / np.maximum(w.sum(2), 1)).astype(np.float16), Mc.any(2)


def train_head(ftr, mtr, ytr, fte, mte, yte, K):
    torch.manual_seed(0)
    model = SeqHead(ftr.shape[-1], K, ftr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    X, Mk, Y = torch.tensor(ftr, dtype=torch.float32), torch.tensor(mtr), torch.tensor(ytr)
    Xte, Mte = torch.tensor(fte, dtype=torch.float32).to(DEVICE), torch.tensor(mte).to(DEVICE)
    best = 0.0
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            loss = F.cross_entropy(model(X[idx].to(DEVICE), Mk[idx].to(DEVICE)), Y[idx].to(DEVICE))
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = torch.cat([model(Xte[i:i + 256], Mte[i:i + 256]).argmax(-1) for i in range(0, len(Xte), 256)]).cpu().numpy()
        best = max(best, float((pred == yte).mean()))
    return best


def pool(H, M):
    w = M[..., None].astype(np.float32)
    return (H.astype(np.float32) * w).sum(1) / np.maximum(w.sum(1), 1)


def main():
    t0 = time.time()
    m, tok, vmap, unk = load_encoder()
    base = {e["dataset"]: e["test_acc"] for e in json.load(open("bytecode_poc/all_datasets_testset_results.json"))["results"]}
    print(f"Level 1: frozen HybridTransformer {sum(p.numel() for p in m.parameters()):,} params. Device {DEVICE}\n", flush=True)
    print(f"{'dataset':<16}{'K':>3} {'seen':<5}| {'tokens':>7}{'4-tok seg':>10}{'pooled LR':>10} | {'bytecode MLP':>12}", flush=True)
    results = {}
    for name in TASKS:
        xtr, ytr_s, xte, yte_s, _ = load_and_split(name, DATASETS[name])
        le = LabelEncoder().fit(ytr_s + yte_s)
        ytr, yte = le.transform(ytr_s), le.transform(yte_s)
        K = len(le.classes_)
        Htr, Mtr = level1(m, tok, vmap, unk, xtr)
        Hte, Mte = level1(m, tok, vmap, unk, xte)
        Ctr, CMtr = coarsen(Htr, Mtr)
        Cte, CMte = coarsen(Hte, Mte)
        acc = {
            "tokens": train_head(Htr, Mtr, ytr, Hte, Mte, yte, K),
            "seg4": train_head(Ctr, CMtr, ytr, Cte, CMte, yte, K),
            "pooled_lr": LogisticRegression(max_iter=2000).fit(pool(Htr, Mtr), ytr).score(pool(Hte, Mte), yte),
        }
        results[name] = dict(acc=acc, baseline=base.get(name), seen=name in TRAINED, K=K, n_train=len(xtr), n_test=len(xte))
        print(f"{name:<16}{K:>3} {'yes' if name in TRAINED else 'no':<5}| {acc['tokens']:>7.3f}{acc['seg4']:>10.3f}{acc['pooled_lr']:>10.3f} | {base.get(name, float('nan')):>12.3f}   ({time.time() - t0:.0f}s)", flush=True)
        OUT.write_text(json.dumps(results, indent=1))
    print(f"\nSaved {OUT}. Total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
