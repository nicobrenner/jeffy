"""Joint research, Branch 1 x Jeffy: are the Oracle's own hidden states (2.7M, free in the browser) or a Jeffy-style frozen sentence encoder
(+ logistic head, Jeffy's whole recipe) the better base for Branch 1's small classifiers, and how many labelled examples per class does
each need? Task: user-turn intent (7 classes, LLM-labelled turns + guard turns). Learning curve over examples per class; 5 seeds.
Also: coverage and misroute at 0.9 confidence (Jeffy's abstain numbers) for the full-data heads.
  HF_HUB_OFFLINE=1 python3 jeffy_bridge_curve.py"""
import json, glob, numpy as np
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from intent_clf import hidden as ihidden
from atl_probe import load
from train_lm import CustomTokenizerWrapper
INT = ["greeting", "yesno_question", "open_question", "task", "pushback", "share", "reaction"]
TT = {json.loads(l)["id"]: json.loads(l) for l in open("pile_runs/atl/user_turns_unlabelled.jsonl")}
U = [{**TT[x["id"]], "intent": x["intent"]} for f in sorted(glob.glob("pile_runs/atl/user_turns_labels_part*.jsonl")) for x in map(json.loads, open(f)) if x.get("intent") in INT and x["id"] in TT]
U += [{"text": r["user"], "prev_oracle": "", "intent": r["move"]} for r in json.load(open("pile_runs/guard/cands.json"))]
y = np.array([INT.index(u["intent"]) for u in U]); print("turns", len(U), {k: int((y == i).sum()) for i, k in enumerate(INT)})
tok = CustomTokenizerWrapper("tokenizer_chat_v2.json"); m, _ = load("checkpoints/best_lm_oracle_v2_grown_ct2.pt"); m.eval()
F = {"oracle hidden (2.7M, free)": ihidden(m, tok, [{"text": u["text"], "prev_oracle": u.get("prev_oracle", "")} for u in U])}
for name in ["sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-base-en-v1.5"]:
    enc = SentenceTransformer(name, device="cpu"); F[name.split("/")[1]] = enc.encode([u["text"] for u in U], batch_size=64, normalize_embeddings=True)
F["oracle + MiniLM"] = np.hstack([F["oracle hidden (2.7M, free)"] / (np.linalg.norm(F["oracle hidden (2.7M, free)"], axis=1, keepdims=True) + 1e-9), F["all-MiniLM-L6-v2"]])
rows = []
for seed in range(5):
    rng = np.random.RandomState(seed); idx = rng.permutation(len(U)); te = idx[: len(U) // 5]; pool = idx[len(U) // 5:]
    for n in [5, 10, 20, 40, 80, 10000]:
        tr = np.concatenate([[j for j in pool if y[j] == c][:n] for c in range(len(INT))]).astype(int)
        for fname, X in F.items():
            clf = LogisticRegression(C=1.0 if "oracle hidden" not in fname else 0.5, class_weight="balanced", max_iter=4000).fit(X[tr], y[tr]); P = clf.predict_proba(X[te]); pred = clf.classes_[P.argmax(1)]
            acc = (pred == y[te]).mean(); g = y[te] == INT.index("greeting"); gr = ((pred == y[te]) & g).sum() / max(1, g.sum()); conf = P.max(1) >= 0.9
            rows.append((fname, n, acc, gr, conf.mean(), ((pred != y[te]) & conf).sum() / max(1, conf.sum())))
import collections
agg = collections.defaultdict(list)
for r in rows: agg[(r[0], r[1])].append(r[2:])
print(f"{'features':<28} {'n/class':>7} {'acc':>6} {'greet recall':>12} {'cov@0.9':>8} {'misroute@0.9':>12}")
for (f, n), v in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
    a = np.mean(v, 0); print(f"{f:<28} {('all' if n > 1000 else n):>7} {a[0]:6.3f} {a[1]:12.2f} {a[2]:8.2f} {a[3]:12.3f}")
