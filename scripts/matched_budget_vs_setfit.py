"""At the same number of training examples, does SetFit beat a frozen head?

The v2 benchmark compares Jeffy at 10,003 banking77 examples against SetFit at
1,540 (20 shots per class), because SetFit's contrastive stage is quadratic in
training-set size and cannot be run on the full split in bounded time. That is
an honest protocol limit, but it is not a comparison: it confounds "does
adapting the encoder help" with "does more data help".

This sweeps the frozen head over the same budgets SetFit can actually reach, so
the two can be read at the same n. Stratified draws, so every class is present
at every budget, averaged over 5 seeds.

    python scripts/matched_budget_vs_setfit.py

Reuses the cached bge-large embeddings from the agent-labeling experiment, so
it costs seconds rather than an encode. NOTE the caveats printed at the end:
this is a different encoder and a different test subset from the benchmark run,
so it answers "how does a frozen head scale with n" and NOT "Jeffy beats SetFit
by X". Settling that needs both arms re-run under benchmarks/.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/eval_results/agent_label_banking77"
RESULT = REPO / "data/eval_results/matched_budget_vs_setfit.json"

# shots per class -> n for 77 classes. 20/class = 1,540 is the SetFit budget.
SHOTS = [2, 5, 10, 20, 30, 50]
SEEDS = [0, 1, 2, 3, 4]
C = 0.01

# From data/eval_results/setfit_fasttext_benchmark_rerun.json
SETFIT = {"shots": 20, "n": 1540, "full_split_acc": 0.8497, "train_s": 982.0,
          "model_mb": 438.4, "latency_p50_ms": 46.14}


def stratified(labels, shots, rng):
    by = defaultdict(list)
    for i, y in enumerate(labels):
        by[y].append(i)
    idx = []
    for y, pool in by.items():
        take = min(shots, len(pool))
        idx.extend(rng.choice(pool, size=take, replace=False).tolist())
    return idx


def fit_eval(tr_e, tr_l, te_e, te_l):
    s = StandardScaler()
    X = s.fit_transform(tr_e)
    c = LogisticRegression(C=C, max_iter=3000, solver="newton-cg", random_state=42)
    c.fit(X, tr_l)
    p = c.predict(s.transform(te_e))
    return float(accuracy_score(te_l, p)), float(f1_score(te_l, p, average="macro", zero_division=0))


def main():
    pool = json.loads((OUT / "pool.json").read_text())
    z = np.load(OUT / "embeddings.npz")
    tr_e, te_e = z["train"], z["test"]
    tr_l = np.array(pool["train_labels"])
    te_l = np.array(pool["test_labels"])

    rows = []
    print(f"{'shots/cls':>10} {'n':>6} {'acc':>8} {'sd':>6} {'macroF1':>8}")
    for shots in SHOTS:
        accs, f1s, ns = [], [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            idx = stratified(tr_l, shots, rng)
            a, f = fit_eval(tr_e[idx], tr_l[idx], te_e, te_l)
            accs.append(a); f1s.append(f); ns.append(len(idx))
        row = {"shots_per_class": shots, "n_train": int(np.mean(ns)),
               "accuracy_mean": round(float(np.mean(accs)), 4),
               "accuracy_sd": round(float(np.std(accs)), 4),
               "macro_f1_mean": round(float(np.mean(f1s)), 4),
               "seeds": SEEDS}
        rows.append(row)
        print(f"{shots:>10} {row['n_train']:>6} {row['accuracy_mean']:>8.4f} "
              f"{row['accuracy_sd']:>6.4f} {row['macro_f1_mean']:>8.4f}")

    # full split, for the ceiling
    a_full, f_full = fit_eval(tr_e, tr_l, te_e, te_l)
    print(f"{'ALL':>10} {len(tr_l):>6} {a_full:>8.4f} {'-':>6} {f_full:>8.4f}")

    at20 = next(r for r in rows if r["shots_per_class"] == 20)
    delta = at20["accuracy_mean"] - SETFIT["full_split_acc"]
    print(f"\nAt SetFit's own budget ({SETFIT['n']} examples, 20/class):")
    print(f"  frozen head  {at20['accuracy_mean']:.4f}")
    print(f"  SetFit       {SETFIT['full_split_acc']:.4f}  (v2 benchmark)")
    print(f"  difference   {delta:+.4f}")
    print(f"  training     seconds vs {SETFIT['train_s']:.0f}s")

    doc = {
        "_what": "Frozen-encoder LR head swept over the training budgets SetFit can reach.",
        "_caveat": (
            "NOT directly comparable to the v2 benchmark SetFit number. Different encoder "
            "(BAAI/bge-large-en-v1.5 vs paraphrase-mpnet-base-v2), different test set "
            "(2,000-example subset vs the full 3,080 split), different preprocessing. This "
            "measures how a frozen head scales with n, and says the budget gap explains a "
            "large part of the published difference. Settling it needs both arms re-run "
            "inside benchmarks/classifier_baselines.py."),
        "encoder": "BAAI/bge-large-en-v1.5",
        "head": f"LogisticRegression(C={C}, newton-cg), StandardScaler",
        "dataset": "banking77, 77 classes",
        "n_test": int(len(te_l)),
        "n_train_pool": int(len(tr_l)),
        "sampling": "stratified, every class present at every budget",
        "seeds": SEEDS,
        "curve": rows,
        "full_pool": {"n_train": int(len(tr_l)), "accuracy": round(a_full, 4),
                      "macro_f1": round(f_full, 4)},
        "setfit_reference": SETFIT,
        "at_setfit_budget": {"frozen_head": at20["accuracy_mean"],
                             "setfit": SETFIT["full_split_acc"],
                             "difference": round(delta, 4)},
    }
    RESULT.write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {RESULT}")


if __name__ == "__main__":
    main()
