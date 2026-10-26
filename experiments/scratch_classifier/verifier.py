"""Reply verifier: given (prompt [+ history], candidate reply) say whether it is coherent/on-thread/in-persona — the judgement the LLM
grader makes (and that matched Nico's blind picks 24/25), learned by a tiny classifier so it can gate a tiny generator (best-of-N,
or "admit ignorance" when nothing passes). Labels: coherence>=2 and no bad tag, from pile_runs/atl/verifier_rows.jsonl.
Two feature sets: (a) TF-IDF text baseline; (b) the Oracle's own hidden states (final-token + mean) at the end of the reply.
Grouped split by prompt id so the same prompt never appears in train and test.   python3 verifier.py --ckpt checkpoints/best_lm_oracle_v2_grown_life8b.pt
"""
import argparse, json, random, numpy as np, torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.sparse import hstack
from atl_probe import load, SYSTEM, DEVICE
from train_lm import CustomTokenizerWrapper

def split(rows, seed=0):
    ids = sorted({r["id"] for r in rows}); rng = random.Random(seed); rng.shuffle(ids); test = set(ids[: len(ids) // 5])
    return [r for r in rows if r["id"] not in test], [r for r in rows if r["id"] in test]

def text_of(r):
    h = "".join(f"<|user|> {t['user']}\n<|oracle|> {t['oracle']}\n" for t in r.get("history") or [])
    return h + f"<|user|> {r['prompt']}\n<|oracle|> {r['response']}"

@torch.no_grad()
def hidden(m, tok, rows, bs=64):
    feats = {}
    def hook(mod, inp, out): feats["h"] = out
    hd = m.norm.register_forward_hook(hook); X = []
    for i in range(0, len(rows), bs):
        batch = rows[i:i + bs]; ids = [tok.encode(SYSTEM + text_of(r))[-m.max_seq_len:] for r in batch]
        L = max(len(x) for x in ids); pad = torch.zeros(len(ids), L, dtype=torch.long)
        for j, x in enumerate(ids): pad[j, :len(x)] = torch.tensor(x)
        m(pad.to(DEVICE)); h = feats["h"].float().cpu()
        for j, x in enumerate(ids):
            last = h[j, len(x) - 1]; mean = h[j, :len(x)].mean(0); X.append(torch.cat([last, mean]).numpy())
    hd.remove(); return np.stack(X)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--rows", default="pile_runs/atl/verifier_rows.jsonl")
    a = ap.parse_args(); rows = [json.loads(l) for l in open(a.rows)]; tr, te = split(rows)
    ytr = np.array([r["label"] for r in tr]); yte = np.array([r["label"] for r in te])
    print(f"rows {len(rows)} train {len(tr)} test {len(te)} positive rate {ytr.mean():.3f}; majority acc {max(yte.mean(), 1-yte.mean()):.3f}")
    # (a) text baseline: separate vectorisers for prompt and response
    vp, vr = TfidfVectorizer(ngram_range=(1, 2), min_df=2), TfidfVectorizer(ngram_range=(1, 2), min_df=2)
    Xtr = hstack([vp.fit_transform([r["prompt"] for r in tr]), vr.fit_transform([r["response"] for r in tr])]); Xte = hstack([vp.transform([r["prompt"] for r in te]), vr.transform([r["response"] for r in te])])
    clf = LogisticRegression(max_iter=2000, C=2.0, class_weight="balanced").fit(Xtr, ytr); p = clf.predict_proba(Xte)[:, 1]
    print(f"TF-IDF text verifier:      acc {accuracy_score(yte, p > 0.5):.3f}  AUC {roc_auc_score(yte, p):.3f}")
    # (b) hidden states of the Oracle itself
    tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt)
    Htr, Hte = hidden(m, tok, tr), hidden(m, tok, te)
    for C in (0.1, 1.0):
        clf2 = LogisticRegression(max_iter=3000, C=C, class_weight="balanced").fit(Htr, ytr); p2 = clf2.predict_proba(Hte)[:, 1]
        print(f"hidden-state verifier C={C}: acc {accuracy_score(yte, p2 > 0.5):.3f}  AUC {roc_auc_score(yte, p2):.3f}")
    # (c) both
    from scipy.sparse import csr_matrix
    Xb_tr = hstack([Xtr, csr_matrix(Htr)]); Xb_te = hstack([Xte, csr_matrix(Hte)])
    clf3 = LogisticRegression(max_iter=3000, C=0.5, class_weight="balanced").fit(Xb_tr, ytr); p3 = clf3.predict_proba(Xb_te)[:, 1]
    print(f"text + hidden verifier:    acc {accuracy_score(yte, p3 > 0.5):.3f}  AUC {roc_auc_score(yte, p3):.3f}")
    # per-lineage view of the best
    for name, sel in (("v2-lineage rows", [i for i, r in enumerate(te) if "v2" in r["model"]]), ("other rows", [i for i, r in enumerate(te) if "v2" not in r["model"]])):
        if len(set(yte[sel])) > 1: print(f"  {name}: n={len(sel)} AUC text {roc_auc_score(yte[sel], p[sel]):.3f} hidden {roc_auc_score(yte[sel], p2[sel]):.3f} both {roc_auc_score(yte[sel], p3[sel]):.3f}")
    import pickle; pickle.dump({"vp": vp, "vr": vr, "clf_text": clf, "clf_hidden": clf2, "clf_both": clf3, "ckpt": a.ckpt}, open("checkpoints/verifier_v1.pkl", "wb")); print("saved checkpoints/verifier_v1.pkl")

if __name__ == "__main__":
    main()
