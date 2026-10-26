"""Re-read the agent-labeling headlines after the stratification finding.

That experiment reported "agent labels cost 4.9pp, sample size cost 23pp". The
23pp came from comparing a RANDOM draw of 385 against the full 10,000 pool. On
77 classes a random 385 averages 5 examples per class and, by chance, leaves
some classes with one or none. So an unknown part of that 23pp was the sampling
scheme, not the sample size.

This splits the two. Same cached embeddings, same encoder, same test set, same
C as the original run, so the numbers are directly comparable to it rather than
merely similar.

    python scripts/reread_agent_label_numbers.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/eval_results/agent_label_banking77"
RESULT = REPO / "data/eval_results/agent_label_reread.json"

SEEDS = list(range(10))
C = 0.01


def fit_eval(tr_e, tr_l, te_e, te_l):
    s = StandardScaler()
    c = LogisticRegression(C=C, max_iter=3000, solver="newton-cg", random_state=42)
    c.fit(s.fit_transform(tr_e), tr_l)
    return float(accuracy_score(te_l, c.predict(s.transform(te_e))))


def draw(labels, n, seed, stratified):
    rng = np.random.default_rng(seed)
    if not stratified:
        return rng.choice(len(labels), size=n, replace=False).tolist()
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[y].append(i)
    per = n // len(by)
    idx = []
    for y, pool in by.items():
        idx.extend(rng.choice(pool, size=min(per, len(pool)), replace=False).tolist())
    # top up to exactly n from whatever is left, so the budgets match exactly
    rest = list(set(range(len(labels))) - set(idx))
    if len(idx) < n:
        idx.extend(rng.choice(rest, size=n - len(idx), replace=False).tolist())
    return idx[:n]


def coverage(labels, idx, n_classes):
    seen = defaultdict(int)
    for i in idx:
        seen[labels[i]] += 1
    return {"classes_present": len(seen),
            "classes_missing": n_classes - len(seen),
            "min_per_class": min([seen.get(c, 0) for c in range(n_classes)]),
            "median_per_class": float(np.median([seen.get(c, 0) for c in range(n_classes)]))}


def main():
    pool = json.loads((OUT / "pool.json").read_text())
    z = np.load(OUT / "embeddings.npz")
    tr_e, te_e = z["train"], z["test"]
    tr_l = np.array(pool["train_labels"])
    te_l = np.array(pool["test_labels"])
    n_classes = len(set(tr_l.tolist()))

    orig = json.loads((OUT / "results.json").read_text())["runs"]
    full = orig["full_supervised"]["test_accuracy"]
    rand385_orig = orig["fewshot_supervised"]["test_accuracy"]
    agent385 = orig["fewshot_agent_labeled"]["test_accuracy"]
    agent_desc = orig["fewshot_agent_with_descriptions"]["test_accuracy"]

    rows = {}
    for n in (385, 770):
        for strat in (False, True):
            accs, covs = [], []
            for s in SEEDS:
                idx = draw(tr_l, n, s, strat)
                accs.append(fit_eval(tr_e[idx], tr_l[idx], te_e, te_l))
                covs.append(coverage(tr_l.tolist(), idx, n_classes))
            key = f"{'stratified' if strat else 'random'}_{n}"
            rows[key] = {
                "n": n, "sampling": "stratified" if strat else "random",
                "accuracy_mean": round(float(np.mean(accs)), 4),
                "accuracy_sd": round(float(np.std(accs)), 4),
                "classes_missing_mean": round(float(np.mean([c["classes_missing"] for c in covs])), 1),
                "min_per_class_mean": round(float(np.mean([c["min_per_class"] for c in covs])), 2),
                "seeds": SEEDS,
            }
            print(f"{key:>18}  acc {rows[key]['accuracy_mean']:.4f} "
                  f"(sd {rows[key]['accuracy_sd']:.4f})  "
                  f"classes missing {rows[key]['classes_missing_mean']}")

    r385, s385 = rows["random_385"]["accuracy_mean"], rows["stratified_385"]["accuracy_mean"]
    strat_cost = s385 - r385
    size_cost = full - s385
    old_claim = full - rand385_orig

    print(f"\nORIGINAL CLAIM: sample size cost = {old_claim:.4f} ({old_claim*100:.1f}pp)")
    print(f"  that number compared full 10,000 against a RANDOM 385")
    print(f"\nSPLIT:")
    print(f"  sampling scheme (random -> stratified at n=385)  {strat_cost*100:5.1f}pp")
    print(f"  genuine sample size (stratified 385 -> 10,000)   {size_cost*100:5.1f}pp")
    print(f"  sum {(strat_cost+size_cost)*100:.1f}pp vs claimed {old_claim*100:.1f}pp")

    print(f"\nUNAFFECTED, because both arms used the same random draw:")
    print(f"  label-quality cost  = {rand385_orig:.4f} - {agent385:.4f} "
          f"= {(rand385_orig-agent385)*100:.2f}pp")
    print(f"  with descriptions   = {rand385_orig:.4f} - {agent_desc:.4f} "
          f"= {(rand385_orig-agent_desc)*100:.2f}pp")

    doc = {
        "_what": "Re-read of the agent-labeling headline numbers after finding that random "
                 "sampling, not sample size, drove most of the reported 23pp gap.",
        "_comparable": "Same embeddings.npz, encoder, test set and C as the original run, "
                       "so these replace rather than merely resemble those numbers.",
        "n_classes": n_classes, "n_test": int(len(te_l)), "seeds": SEEDS,
        "curves": rows,
        "full_pool_accuracy": full,
        "original_claim": {
            "statement": "sample size cost 23pp",
            "value_pp": round(old_claim * 100, 1),
            "problem": "compared the full pool against a RANDOM draw of 385, which on 77 "
                       "classes leaves classes with one example or none",
        },
        "corrected": {
            "sampling_scheme_cost_pp": round(strat_cost * 100, 1),
            "true_sample_size_cost_pp": round(size_cost * 100, 1),
        },
        "unaffected": {
            "_why": "the supervised and agent-labeled arms used the SAME random indices, so "
                    "the label-quality comparison is internally consistent and stands",
            "label_quality_cost_pp": round((rand385_orig - agent385) * 100, 2),
            "label_quality_cost_with_descriptions_pp": round((rand385_orig - agent_desc) * 100, 2),
        },
        "implication": (
            "The headline 'sample size costs 23pp' should not be published. Most of it is a "
            "sampling artifact and is free to fix: stratify. The label-quality findings, "
            "including the collapse from 4.9pp to 0.55pp with label descriptions, are "
            "unaffected and remain the result worth publishing."),
    }
    RESULT.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {RESULT}")


if __name__ == "__main__":
    main()
