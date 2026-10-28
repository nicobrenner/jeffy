"""Encoder choice, per-head C and per-flag binary heads on a multilingual care-intake email set (issue #10's shape).

Reads a labeled dataset of the form [{"subject", "body", "team", "kind", "category", "desired_outcome", "urgency",
"severity", "emotion", "flags": [..]}, ...] and reports, per encoder:
  - accuracy / macro-F1 per single-label head with the pack's default C=0.01 and with a quick per-head C grid (3-fold CV)
  - encode time per text (CPU)
  - per-flag F1 for one binary head per flag (class_weight=balanced) vs one multi-class head on the flag-set label
  - the Arabic subset's accuracy on the `category` head
Prints one JSON object (aggregates only, no texts). Nothing is written by this script.

    HF_HUB_OFFLINE=1 python scripts/care_intake_sweep.py --data path/to/care_intake_labeled_dataset.json \
        --encoders sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 intfloat/multilingual-e5-small sentence-transformers/LaBSE
"""
import argparse, json, re, time
import numpy as np

HEADS = ["team", "kind", "category", "desired_outcome", "urgency", "severity", "emotion"]
C_GRID = [0.01, 0.1, 1.0]
ARABIC = re.compile(r"[؀-ۿ]")


def fit_lr(X, y, C):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, solver="newton-cg", max_iter=3000, random_state=42).fit(sc.transform(X), y)
    return lambda Z: clf.predict(sc.transform(Z))


def best_c(X, y):
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    counts = np.unique(y, return_counts=True)[1]
    folds = int(min(3, counts.min()))
    if folds < 2:
        return 0.01, None
    best = (None, -1)
    for C in C_GRID:
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, solver="newton-cg", max_iter=3000, random_state=42))
        s = cross_val_score(pipe, X, y, cv=StratifiedKFold(folds, shuffle=True, random_state=42), scoring="f1_macro").mean()
        if s > best[1]:
            best = (C, s)
    return best


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--encoders", nargs="+", required=True); ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import train_test_split
    from sentence_transformers import SentenceTransformer
    rows = json.load(open(a.data))
    texts = [f"{r.get('subject', '')}\n{r.get('body', '')}".strip() for r in rows]
    lang = np.array(["ar" if ARABIC.search(t) else "en" for t in texts])
    idx = np.arange(len(rows))
    tr, te = train_test_split(idx, test_size=0.25, random_state=a.seed, stratify=[r["team"] for r in rows])
    flags_all = sorted({f for r in rows for f in r.get("flags", [])})
    out = {"n": len(rows), "n_train": int(len(tr)), "n_test": int(len(te)), "lang_counts": {k: int(v) for k, v in zip(*np.unique(lang, return_counts=True))},
           "test_lang_counts": {k: int(v) for k, v in zip(*np.unique(lang[te], return_counts=True))}, "flags": flags_all, "encoders": {}}
    for name in a.encoders:
        enc = SentenceTransformer(name, device="cpu")
        t0 = time.time(); X = enc.encode(texts, batch_size=32, show_progress_bar=False); enc_ms = (time.time() - t0) * 1000 / len(texts)
        res = {"dim": int(X.shape[1]), "encode_ms_per_text": round(enc_ms, 1), "heads": {}, "flags_binary": {}, "flags_multiclass": {}, "arabic_category_accuracy": {}}
        for h in HEADS:
            y = np.array([r[h] for r in rows])
            p0 = fit_lr(X[tr], y[tr], 0.01)(X[te])
            C, cvs = best_c(X[tr], y[tr]); p1 = fit_lr(X[tr], y[tr], C)(X[te])
            res["heads"][h] = {"n_classes": int(len(set(y))), "acc_C0.01": round(float(accuracy_score(y[te], p0)), 4), "f1_C0.01": round(float(f1_score(y[te], p0, average="macro")), 4),
                               "best_C": C, "acc_tuned": round(float(accuracy_score(y[te], p1)), 4), "f1_tuned": round(float(f1_score(y[te], p1, average="macro")), 4)}
            if h == "category":
                ar = te[lang[te] == "ar"]
                if len(ar):
                    pa = fit_lr(X[tr], y[tr], 0.01)(X[ar]); res["arabic_category_accuracy"] = {"n": int(len(ar)), "acc_C0.01": round(float(accuracy_score(y[ar], pa)), 4)}
        # flags: one binary head per flag (balanced) vs one multi-class head on the flag-set label
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        fs = np.array(["|".join(sorted(r.get("flags", []))) or "none" for r in rows])
        sc = StandardScaler().fit(X[tr]); mc = LogisticRegression(C=0.01, solver="newton-cg", max_iter=3000, random_state=42).fit(sc.transform(X[tr]), fs[tr])
        pm = mc.predict(sc.transform(X[te]))
        for f in flags_all:
            yb = np.array(["yes" if f in r.get("flags", []) else "no" for r in rows])
            n_pos_tr, n_pos_te = int((yb[tr] == "yes").sum()), int((yb[te] == "yes").sum())
            if n_pos_tr < 2 or n_pos_te == 0:
                res["flags_binary"][f] = {"n_pos_train": n_pos_tr, "n_pos_test": n_pos_te, "f1": None}; continue
            clf = LogisticRegression(C=0.01, solver="newton-cg", max_iter=3000, random_state=42, class_weight="balanced").fit(sc.transform(X[tr]), yb[tr])
            pb = clf.predict(sc.transform(X[te]))
            res["flags_binary"][f] = {"n_pos_train": n_pos_tr, "n_pos_test": n_pos_te, "f1": round(float(f1_score(yb[te], pb, pos_label="yes")), 4)}
            pm_flag = np.array(["yes" if f in s.split("|") else "no" for s in pm])
            res["flags_multiclass"][f] = {"f1": round(float(f1_score(yb[te], pm_flag, pos_label="yes")), 4)}
        out["encoders"][name] = res
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
