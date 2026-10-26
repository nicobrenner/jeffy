"""Does giving the labeler written label descriptions improve its labels?

The first run (scripts/agent_label_banking77.py) gave the agent 77 bare label
names and nothing else: no task description, no definition of any class, no
disambiguation rules. It agreed with the crowdworkers on 82.6% of 385 random
training examples. Nico's point was that this undersells the method, because a
real operator hands over all of that for free.

Design note on contamination. The obvious test is to relabel the same 385 with
descriptions added, but the same labeler has already seen those texts, so a
within-subject rerun is confounded and cannot be fixed by caveats. Instead this
draws a FRESH 385 from the same pool, disjoint from rounds 1 and 2, and labels
it with descriptions. Both samples are random draws from the same pool, so the
agreement rates are comparable in expectation and the labeler has not seen the
new texts.

    python scripts/agent_label_with_descriptions.py emit
    python scripts/agent_label_with_descriptions.py score
"""

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/eval_results/agent_label_banking77"
DESC = OUT / "banking77_label_descriptions.json"
BATCH = OUT / "batch_round3_unlabeled.json"
LABELS = OUT / "labels_round3.json"
N = 385
SEED = 1337  # different draw from the seed-42 rounds


def load():
    pool = json.loads((OUT / "pool.json").read_text())
    z = np.load(OUT / "embeddings.npz")
    return pool, z["train"], z["test"]


def used_indices():
    seen = set()
    for f in ("labels_round1.json", "labels_round2.json"):
        p = OUT / f
        if p.exists():
            seen |= {int(i) for i in json.loads(p.read_text())["labels"]}
    return seen


def stage_emit():
    pool, _, _ = load()
    if not DESC.exists():
        raise SystemExit(f"write {DESC} first: {{intent_name: description}} for all 77")
    desc = json.loads(DESC.read_text())
    names = set(pool["label_names"].values())
    missing = names - set(desc)
    if missing:
        raise SystemExit(f"{len(missing)} descriptions missing: {sorted(missing)[:8]}")

    seen = used_indices()
    rng = np.random.default_rng(SEED)
    pick, i = [], 0
    order = rng.permutation(len(pool["train_texts"]))
    while len(pick) < N and i < len(order):
        j = int(order[i]); i += 1
        if j not in seen:
            pick.append(j)
    assert len(pick) == N, len(pick)

    BATCH.write_text(json.dumps({
        "instructions": ("Label each item with exactly one label from label_descriptions. "
                         "The descriptions are the taxonomy owner's definitions: prefer them "
                         "over your reading of the bare label name."),
        "label_descriptions": desc,
        "items": [{"id": j, "text": pool["train_texts"][j]} for j in pick],
    }, indent=2))
    print(f"wrote {BATCH}: {N} fresh texts, disjoint from {len(seen)} already labeled")


def fit_eval(tr_e, tr_l, te_e, te_l):
    s = StandardScaler(); X = s.fit_transform(tr_e)
    c = LogisticRegression(C=0.01, max_iter=3000, solver="newton-cg", random_state=42).fit(X, tr_l)
    p = c.predict(s.transform(te_e))
    return (round(float(accuracy_score(te_l, p)), 4),
            round(float(f1_score(te_l, p, average="macro", zero_division=0)), 4),
            len(set(tr_l)))


def stage_score():
    pool, tr_e, te_e = load()
    names = pool["label_names"]
    id_of = {v: k for k, v in names.items()}
    doc = json.loads(LABELS.read_text())
    agent = doc["labels"]
    idx = [int(i) for i in agent]
    bad = {v for v in agent.values() if v not in id_of}
    if bad:
        raise SystemExit(f"labels outside the taxonomy: {sorted(bad)[:8]}")

    mine = [id_of[agent[str(i)]] for i in idx]
    truth = [pool["train_labels"][i] for i in idx]
    agree = sum(a == b for a, b in zip(mine, truth)) / len(idx)

    acc, f1, cov = fit_eval(tr_e[idx], mine, te_e, pool["test_labels"])

    base = json.loads((OUT / "results.json").read_text())["runs"]
    b_agree = base["fewshot_agent_labeled"]["agent_human_agreement"]
    b_acc = base["fewshot_agent_labeled"]["test_accuracy"]
    sup = base["fewshot_supervised"]["test_accuracy"]

    print(f"\nWITH descriptions, {len(idx)} fresh examples")
    print(f"  agreement        {agree:.4f}   (no descriptions: {b_agree:.4f})   "
          f"delta {agree - b_agree:+.4f}")
    print(f"  test_accuracy    {acc}   (no descriptions: {b_acc})   delta {acc - b_acc:+.4f}")
    print(f"  classes covered  {cov}/77")
    print(f"  few-shot supervised ceiling at this n: {sup}")
    print(f"  label-quality cost {sup - acc:+.4f}  (was {sup - b_acc:+.4f})")

    pairs = {}
    for a, t in zip(mine, truth):
        if a != t:
            pairs[(names[t], names[a])] = pairs.get((names[t], names[a]), 0) + 1
    top = sorted(pairs.items(), key=lambda x: -x[1])[:15]
    print("\n  top disagreements (human -> agent):")
    for (t, a), n in top:
        print(f"    {n:>2}x  {t}  ->  {a}")

    res = json.loads((OUT / "results.json").read_text())
    res["runs"]["fewshot_agent_with_descriptions"] = {
        "n_train": len(idx), "sample": "fresh draw, seed 1337, disjoint from rounds 1 and 2",
        "labeler": doc["labeler"], "agent_human_agreement": round(agree, 4),
        "agreement_delta_vs_no_descriptions": round(agree - b_agree, 4),
        "test_accuracy": acc, "test_macro_f1": f1, "n_classes_seen": cov,
        "accuracy_delta_vs_no_descriptions": round(acc - b_acc, 4),
        "label_quality_cost": round(sup - acc, 4),
        "top_disagreements": [{"human": t, "agent": a, "n": n} for (t, a), n in top],
    }
    (OUT / "results.json").write_text(json.dumps(res, indent=2))
    with open(OUT / "headline.csv", "a") as f:
        f.write(f"fewshot_agent_with_descriptions,{len(idx)},agent,{doc['labeler']['model_id']},"
                f"{cov},{agree:.4f},{acc},{f1}\n")
    print(f"\nwrote {OUT / 'results.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["emit", "score"])
    {"emit": stage_emit, "score": stage_score}[ap.parse_args().stage]()


if __name__ == "__main__":
    main()
