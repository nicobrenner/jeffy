"""Boundary-conflict detector: which pairs of classes in a training set overlap in the embedding space the head uses?

A high-confidence misroute like "change the address on my account" -> shipping (issue #11) is not a model error when the
phrasing genuinely lives in two classes of the taxonomy. This script names those class pairs BEFORE anyone labels or
routes, from the training texts alone. Three signals per unordered pair (A, B), all computed on the frozen-encoder
embeddings the head consumes:
  nn_cross    share of A's texts whose nearest neighbour (cosine) is in B, plus the reverse, averaged over the pair's size
  cv_confuse  5-fold cross-validated LogisticRegression(C=0.01) confusions A<->B, as a share of the pair's texts
  centroid    cosine similarity between the two class centroids
The conflict score is the mean rank of the three (lower is worse). Output: data/eval_results/boundary_conflicts_<task>.json/.csv.

Training set used for support_ticket: the bitext texts in data/eval_results/support_ticket_v2/split.json (train + test, 2,700
texts, 100 per intent) relabelled to their 11 parent categories via the v2 manifest. This is a proxy for the shipped pack's
training set (same source, categories), without the 24 hand-written account examples added for issue #11.

    PYTHONPATH=src python scripts/boundary_conflicts.py                # support_ticket, CPU
    PYTHONPATH=src python scripts/boundary_conflicts.py --task support_ticket_v2
"""
import argparse, csv, json, re, time
from itertools import combinations
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "src/jeffy/pack"
OUT = REPO / "data/eval_results"


def load_support(level):
    split = json.loads((OUT / "support_ticket_v2/split.json").read_text())
    m = json.loads((PACK / "support_ticket_v2/manifest.json").read_text())
    parent = {i: re.search(r"under category '([^']+)'", d).group(1).lower() for i, d in m["label_descriptions"].items()}
    texts, labels = [], []
    for part in ("train", "test"):
        for intent in sorted(split[part]):
            for t in split[part][intent]:
                texts.append(t); labels.append(intent if level == "intent" else parent[intent])
    return texts, labels


def conflicts(X, labels, texts_global):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import StandardScaler
    labels = np.asarray(labels); classes = sorted(set(labels)); n = len(labels)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    # nearest neighbour (excluding self)
    S = Xn @ Xn.T; np.fill_diagonal(S, -1.0); nn = labels[S.argmax(axis=1)]
    # 5-fold CV confusions with the pack's recipe
    pred = np.empty(n, dtype=object)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=42).split(X, labels):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(C=0.01, solver="newton-cg", max_iter=3000, random_state=42).fit(sc.transform(X[tr]), labels[tr])
        pred[te] = clf.predict(sc.transform(X[te]))
    cent = {c: Xn[labels == c].mean(axis=0) for c in classes}
    rows = []
    # shared phrases: 2-3 word n-grams that occur in >= 5% of both classes' texts (stopword-only grams dropped)
    STOP = set("i my me the a an to of for and or in on it is this that you your can do how would like want need please".split())
    def grams(t):
        w = re.findall(r"[a-z']+", t.lower()); out = set()
        for n in (2, 3):
            for i in range(len(w) - n + 1):
                g = w[i:i + n]
                if not all(x in STOP for x in g): out.add(" ".join(g))
        return out
    G = [grams(t) for t in texts_global]
    per_class = {}
    for c in classes:
        idx = np.where(labels == c)[0]; cnt = {}
        for i in idx:
            for g in G[i]: cnt[g] = cnt.get(g, 0) + 1
        per_class[c] = {g for g, k in cnt.items() if k >= 0.05 * len(idx)}
    for a, b in combinations(classes, 2):
        ia, ib = labels == a, labels == b; size = ia.sum() + ib.sum()
        nn_cross = (np.sum(nn[ia] == b) + np.sum(nn[ib] == a)) / size
        cv_conf = (np.sum(pred[ia] == b) + np.sum(pred[ib] == a)) / size
        ca, cb = cent[a], cent[b]; cos = float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb)))
        shared = per_class[a] & per_class[b]
        rows.append({"pair": f"{a} / {b}", "nn_cross": round(float(nn_cross), 4), "cv_confuse": round(float(cv_conf), 4), "centroid_cos": round(cos, 4),
                     "shared_phrases": len(shared), "shared_examples": sorted(shared, key=len, reverse=True)[:4]})
    for key in ("nn_cross", "cv_confuse", "centroid_cos", "shared_phrases"):
        order = sorted(range(len(rows)), key=lambda i: -rows[i][key])
        for rank, i in enumerate(order): rows[i][f"rank_{key}"] = rank + 1
    for r in rows:
        r["conflict_rank"] = round((r["rank_nn_cross"] + r["rank_cv_confuse"] + r["rank_centroid_cos"] + r["rank_shared_phrases"]) / 4, 2)
    rows.sort(key=lambda r: r["conflict_rank"])
    cv_acc = float(np.mean(pred == labels))
    return rows, cv_acc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--task", default="support_ticket"); ap.add_argument("--device", default="cpu"); ap.add_argument("--top", type=int, default=12)
    a = ap.parse_args()
    from sentence_transformers import SentenceTransformer
    texts, labels = load_support("intent" if a.task == "support_ticket_v2" else "category")
    t0 = time.time(); enc = SentenceTransformer("BAAI/bge-large-en-v1.5", device=a.device)
    X = enc.encode(texts, batch_size=64, show_progress_bar=False, normalize_embeddings=False)
    rows, cv_acc = conflicts(X, labels, texts)
    out = {"task": a.task, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "n_texts": len(texts), "n_classes": len(set(labels)), "cv_accuracy": cv_acc,
           "source": "data/eval_results/support_ticket_v2/split.json train+test relabelled via v2 manifest parent map", "pairs": rows}
    (OUT / f"boundary_conflicts_{a.task}.json").write_text(json.dumps(out, indent=1))
    with (OUT / f"boundary_conflicts_{a.task}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # query mode: real phrasings (the probe set) scored by the pack's recipe; which pair does each straddle?
    qpath = REPO / "data/probes" / f"{a.task}.jsonl"
    if qpath.exists():
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler().fit(X); clf = LogisticRegression(C=0.01, solver="newton-cg", max_iter=3000, random_state=42).fit(sc.transform(X), labels)
        queries = [json.loads(l) for l in qpath.read_text().splitlines() if l.strip()]
        P = clf.predict_proba(sc.transform(enc.encode([q["text"] for q in queries], batch_size=64, show_progress_bar=False)))
        pairs = {}
        qrows = []
        for q, p in zip(queries, P):
            o = np.argsort(-p); top, second = clf.classes_[o[0]], clf.classes_[o[1]]; margin = float(p[o[0]] - p[o[1]])
            key = " / ".join(sorted([top, second])); pairs.setdefault(key, []).append(margin)
            qrows.append({"text": q["text"], "top": top, "second": second, "top_p": round(float(p[o[0]]), 3), "margin": round(margin, 3)})
        qpairs = sorted(({"pair": k, "n_queries": len(v), "n_low_margin": int(sum(m < 0.5 for m in v)), "mean_margin": round(float(np.mean(v)), 3)} for k, v in pairs.items()),
                        key=lambda r: (-r["n_low_margin"], r["mean_margin"]))
        out["queries"] = qrows; out["query_pairs"] = qpairs
        (OUT / f"boundary_conflicts_{a.task}.json").write_text(json.dumps(out, indent=1))
        print(f"\nquery mode ({qpath.name}, {len(queries)} real phrasings): pairs straddled, by low-margin count")
        print(f"{'pair':<28}{'n':>4}{'low<0.5':>9}{'mean_m':>8}")
        for r in qpairs[:8]:
            print(f"{r['pair']:<28}{r['n_queries']:>4}{r['n_low_margin']:>9}{r['mean_margin']:>8.3f}")
    print(f"{a.task}: {len(texts)} texts, {len(set(labels))} classes, 5-fold CV accuracy {cv_acc:.4f}, {time.time()-t0:.0f}s")
    print(f"{'rank':>4}  {'pair':<28}{'nn_cross':>9}{'cv_conf':>9}{'centroid':>9}{'phrases':>8}  shared e.g.")
    for i, r in enumerate(rows[: a.top], 1):
        print(f"{i:>4}  {r['pair']:<28}{r['nn_cross']:>9.3f}{r['cv_confuse']:>9.3f}{r['centroid_cos']:>9.3f}{r['shared_phrases']:>8}  {r['shared_examples'][:2]}")


if __name__ == "__main__":
    main()
