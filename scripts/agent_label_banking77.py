"""Can an agent label its way to a usable classifier?

Experiment: strip the labels off banking77's train split, have the agent (Claude
Code, via a JSON handoff on disk) label a small sample, train a Jeffy head on
those labels, and score it on the SAME 2,000-example test split that produced
the pack's published 94.3%.

The point of the design is the two controls. Three numbers, not one:

  1. full supervised   10,000 true labels   -> the published ceiling
  2. few-shot supervised  N true labels     -> isolates sample size
  3. few-shot agent-labeled  N agent labels -> isolates label quality

(1) minus (2) is what you give up by labeling less. (2) minus (3) is what you
give up by letting an agent do the labeling. Reporting only (1) minus (3) would
blame the agent for both.

Recipe is copied from src/jeffy/build_pack.py so the baseline is comparable to
the decimal: bge-large-en-v1.5, StandardScaler, LogisticRegression(C=0.01,
solver=newton-cg, max_iter=3000, random_state=42), seed-42 caps at 10,000 train
and 2,000 test.

Stages:
    python scripts/agent_label_banking77.py embed      # cache embeddings (slow, once)
    python scripts/agent_label_banking77.py baseline   # numbers 1 and 2
    python scripts/agent_label_banking77.py emit       # write the unlabeled batch
    python scripts/agent_label_banking77.py score      # number 3, + next round's batch
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/eval_results/agent_label_banking77"
CACHE = OUT / "embeddings.npz"
POOL = OUT / "pool.json"
BATCH = OUT / "batch_round1_unlabeled.json"
RESULTS = OUT / "results.json"

ENCODER = "BAAI/bge-large-en-v1.5"
C = 0.01
SEED = 42

# How many the agent labels in round 1. 77 classes, so this is ~5 per class in
# expectation. Deliberately NOT stratified: a real cold start cannot stratify,
# because stratifying requires the labels you do not have yet.
N_ROUND1 = 385


def load_pool():
    """banking77 train (capped 10k) and test (capped 2k), exactly as build_pack does."""
    from datasets import load_dataset

    ds = load_dataset("legacy-datasets/banking77")
    train_ds, test_ds = ds["train"], ds["test"]

    train_texts = list(train_ds["text"])
    train_labels = [str(x) for x in train_ds["label"]]
    test_texts = list(test_ds["text"])
    test_labels = [str(x) for x in test_ds["label"]]

    np.random.seed(SEED)
    idx = np.random.choice(len(train_texts), 10000, replace=False)
    train_texts = [train_texts[i] for i in idx]
    train_labels = [train_labels[i] for i in idx]

    np.random.seed(SEED)
    idx = np.random.choice(len(test_texts), 2000, replace=False)
    test_texts = [test_texts[i] for i in idx]
    test_labels = [test_labels[i] for i in idx]

    names = ds["train"].features["label"].names
    label_names = {str(i): n for i, n in enumerate(names)}
    return train_texts, train_labels, test_texts, test_labels, label_names


def stage_embed():
    OUT.mkdir(parents=True, exist_ok=True)
    tr_t, tr_l, te_t, te_l, names = load_pool()
    POOL.write_text(
        json.dumps(
            {
                "train_texts": tr_t,
                "train_labels": tr_l,
                "test_texts": te_t,
                "test_labels": te_l,
                "label_names": names,
            }
        )
    )
    print(f"pool: {len(tr_t)} train, {len(te_t)} test, {len(names)} classes")

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(ENCODER)
    t0 = time.perf_counter()
    tr_e = enc.encode(tr_t, batch_size=256, show_progress_bar=True)
    te_e = enc.encode(te_t, batch_size=256, show_progress_bar=True)
    dt = time.perf_counter() - t0
    np.savez_compressed(CACHE, train=tr_e, test=te_e)
    print(f"encoded {len(tr_t) + len(te_t)} texts in {dt:.1f}s -> {CACHE}")


def load_cached():
    pool = json.loads(POOL.read_text())
    z = np.load(CACHE)
    return pool, z["train"], z["test"]


def fit_eval(train_emb, train_labels, test_emb, test_labels):
    """One train+eval cycle on the shared recipe. Returns metrics plus the model."""
    scaler = StandardScaler()
    X = scaler.fit_transform(train_emb)
    Xt = scaler.transform(test_emb)
    t0 = time.perf_counter()
    clf = LogisticRegression(C=C, max_iter=3000, solver="newton-cg", random_state=SEED)
    clf.fit(X, train_labels)
    train_s = time.perf_counter() - t0
    preds = clf.predict(Xt)
    return {
        "n_train": len(train_labels),
        "n_classes_seen": len(set(train_labels)),
        "train_accuracy": round(float(accuracy_score(train_labels, clf.predict(X))), 4),
        "test_accuracy": round(float(accuracy_score(test_labels, preds)), 4),
        "test_macro_f1": round(float(f1_score(test_labels, preds, average="macro", zero_division=0)), 4),
        "train_seconds": round(train_s, 2),
    }, clf, scaler


def stage_baseline():
    pool, tr_e, te_e = load_cached()
    tr_l, te_l = pool["train_labels"], pool["test_labels"]

    results = {"encoder": ENCODER, "C": C, "seed": SEED, "runs": {}}

    print("\n[1] full supervised, 10,000 true labels")
    full, _, _ = fit_eval(tr_e, tr_l, te_e, te_l)
    results["runs"]["full_supervised"] = full
    print(f"    test_accuracy {full['test_accuracy']}  (pack manifest says 0.943)")

    # The agent's round-1 sample, drawn now so the supervised control uses the
    # exact same texts. Same indices, true labels instead of agent labels.
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(tr_l), N_ROUND1, replace=False)
    results["round1_indices"] = [int(i) for i in idx]

    print(f"\n[2] few-shot supervised, {N_ROUND1} TRUE labels (same texts the agent gets)")
    fs, _, _ = fit_eval(tr_e[idx], [tr_l[i] for i in idx], te_e, te_l)
    results["runs"]["fewshot_supervised"] = fs
    print(f"    test_accuracy {fs['test_accuracy']}  classes covered {fs['n_classes_seen']}/77")

    # Learning curve, true labels, so we know the shape of the sample-size axis
    # before the agent labels anything.
    print("\n[curve] true labels at several sizes")
    curve = []
    for n in (77, 154, 385, 770, 1540, 3080, 10000):
        if n > len(tr_l):
            continue
        rr = np.random.default_rng(SEED)
        ii = rr.choice(len(tr_l), n, replace=False)
        m, _, _ = fit_eval(tr_e[ii], [tr_l[i] for i in ii], te_e, te_l)
        curve.append({"n": n, **m})
        print(f"    n={n:>6}  classes {m['n_classes_seen']:>2}/77  test_acc {m['test_accuracy']}")
    results["learning_curve_true_labels"] = curve

    RESULTS.write_text(json.dumps(results, indent=2))
    with open(OUT / "learning_curve_true_labels.csv", "w") as f:
        f.write("n,n_classes_seen,train_accuracy,test_accuracy,test_macro_f1,train_seconds\n")
        for r in curve:
            f.write(
                f"{r['n']},{r['n_classes_seen']},{r['train_accuracy']},"
                f"{r['test_accuracy']},{r['test_macro_f1']},{r['train_seconds']}\n"
            )
    print(f"\nwrote {RESULTS}")


def stage_emit():
    """Write the unlabeled batch for the agent. No true labels in this file."""
    pool = json.loads(POOL.read_text())
    results = json.loads(RESULTS.read_text())
    idx = results["round1_indices"]
    items = [{"id": int(i), "text": pool["train_texts"][i]} for i in idx]
    BATCH.write_text(
        json.dumps(
            {
                "instructions": (
                    "Label each item with exactly one label from label_names.values(). "
                    "Write {id: label} to labels_round1.json. Do not guess from an "
                    "index; read the text."
                ),
                "label_names": sorted(pool["label_names"].values()),
                "items": items,
            },
            indent=2,
        )
    )
    print(f"wrote {BATCH}: {len(items)} unlabeled texts, {len(pool['label_names'])} candidate labels")


def stage_score():
    """Number 3: train on the agent's labels, and diagnose where they went wrong."""
    pool, tr_e, te_e = load_cached()
    results = json.loads(RESULTS.read_text())
    names = pool["label_names"]
    id_of = {v: k for k, v in names.items()}

    doc = json.loads((OUT / "labels_round1.json").read_text())
    # The labeler is a first-class variable, not a footnote. A frontier model, a
    # local 4B and a crowdworker will not agree, and the whole point of the skill
    # is that you get to choose. Record which one produced these labels.
    if "labeler" not in doc or "labels" not in doc:
        raise SystemExit("labels_round1.json must be {'labeler': {...}, 'labels': {id: label}}")
    labeler = doc["labeler"]
    for k in ("model", "model_id", "harness", "method"):
        if not labeler.get(k):
            raise SystemExit(f"labeler.{k} is required so the result is attributable")
    print(f"\n    labeler: {labeler['model']} ({labeler['model_id']}) via {labeler['harness']}")

    agent = doc["labels"]
    idx = [int(i) for i in agent.keys()]
    agent_labels = []
    unknown = []
    for i in idx:
        name = agent[str(i)]
        if name not in id_of:
            unknown.append(name)
        agent_labels.append(id_of.get(name, name))
    if unknown:
        raise SystemExit(f"agent used {len(set(unknown))} labels not in the taxonomy: {sorted(set(unknown))[:10]}")

    true_labels = [pool["train_labels"][i] for i in idx]
    agree = sum(a == t for a, t in zip(agent_labels, true_labels))
    agreement = agree / len(idx)

    print(f"\n[3] few-shot AGENT-labeled, {len(idx)} labels")
    print(f"    agent/human agreement on the labels themselves: {agree}/{len(idx)} = {agreement:.4f}")

    m, clf, scaler = fit_eval(tr_e[idx], agent_labels, te_e, pool["test_labels"])
    print(f"    downstream test_accuracy {m['test_accuracy']}  classes covered {m['n_classes_seen']}/77")

    fs = results["runs"]["fewshot_supervised"]
    full = results["runs"]["full_supervised"]
    print(f"\n    vs few-shot supervised (same texts, true labels): {fs['test_accuracy']}"
          f"  -> label-quality cost {fs['test_accuracy'] - m['test_accuracy']:+.4f}")
    print(f"    vs full supervised (10,000 true labels):          {full['test_accuracy']}"
          f"  -> {m['test_accuracy'] / full['test_accuracy']:.1%} of the ceiling")

    # Where the agent and the humans disagreed, most frequent pairs first.
    pairs = {}
    for a, t in zip(agent_labels, true_labels):
        if a != t:
            pairs[(names[t], names[a])] = pairs.get((names[t], names[a]), 0) + 1
    top = sorted(pairs.items(), key=lambda x: -x[1])[:20]
    print("\n    top disagreements (human label -> agent label):")
    for (t, a), n in top:
        print(f"      {n:>2}x  {t}  ->  {a}")

    # Active learning: the pool examples this classifier is least sure about.
    # Margin is free from the LR, so this round costs nothing extra to select.
    seen = set(idx)
    rest = [i for i in range(len(pool["train_texts"])) if i not in seen]
    probs = clf.predict_proba(scaler.transform(tr_e[rest]))
    srt = np.sort(probs, axis=1)
    margin = srt[:, -1] - srt[:, -2]
    order = np.argsort(margin)[:N_ROUND1]
    nxt = [{"id": int(rest[j]), "text": pool["train_texts"][rest[j]],
            "margin": round(float(margin[j]), 4),
            "model_guess": names[clf.classes_[int(np.argmax(probs[j]))]]}
           for j in order]
    (OUT / "batch_round2_unlabeled.json").write_text(json.dumps({
        "instructions": "Same taxonomy. These are the pool texts the round-1 head is least "
                        "sure about. model_guess is what it would say; correct it or confirm it.",
        "label_names": sorted(names.values()),
        "items": nxt,
    }, indent=2))

    results["runs"]["fewshot_agent_labeled"] = {
        **m,
        "labeler": labeler,
        "agent_human_agreement": round(agreement, 4),
        "label_quality_cost": round(fs["test_accuracy"] - m["test_accuracy"], 4),
        "fraction_of_ceiling": round(m["test_accuracy"] / full["test_accuracy"], 4),
        "top_disagreements": [{"human": t, "agent": a, "n": n} for (t, a), n in top],
    }
    RESULTS.write_text(json.dumps(results, indent=2))

    with open(OUT / "headline.csv", "w") as f:
        f.write("condition,n_train,labels_from,labeler_model,n_classes_seen,"
                "label_agreement,test_accuracy,test_macro_f1\n")
        f.write(f"full_supervised,{full['n_train']},human,banking77_crowdworkers,"
                f"{full['n_classes_seen']},1.0,{full['test_accuracy']},{full['test_macro_f1']}\n")
        f.write(f"fewshot_supervised,{fs['n_train']},human,banking77_crowdworkers,"
                f"{fs['n_classes_seen']},1.0,{fs['test_accuracy']},{fs['test_macro_f1']}\n")
        f.write(f"fewshot_agent_labeled,{m['n_train']},agent,{labeler['model_id']},"
                f"{m['n_classes_seen']},{agreement:.4f},{m['test_accuracy']},{m['test_macro_f1']}\n")
    print(f"\nwrote {RESULTS} and {OUT / 'headline.csv'}")
    print(f"round 2 batch ready: {OUT / 'batch_round2_unlabeled.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["embed", "baseline", "emit", "score"])
    args = ap.parse_args()
    {"embed": stage_embed, "baseline": stage_baseline,
     "emit": stage_emit, "score": stage_score}[args.stage]()


if __name__ == "__main__":
    main()
