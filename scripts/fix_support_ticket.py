"""Test whether the support_ticket account-vs-shipping defect (issue #11) is a
sample-size problem or a label-overlap problem.

Trains the same frozen-encoder + LR head at several examples-per-class values
and scores a fixed probe set of account-update phrasings.

Writes results to data/eval_results/support_ticket_issue11.json
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "data/eval_results/support_ticket_issue11.json"
# Embedding cache: ~25k bge-large vectors take ~25 min to compute on CPU,
# so keep them out of the repo but reuse across runs.
CACHE = Path(os.environ.get("JEFFY_EMBED_CACHE",
                            Path.home() / ".cache/jeffy-experiments")) / "bitext_embeddings.npz"

# The phrasings that misroute today, plus controls that already work.
PROBES = [
    ("change the address on my account", "account"),
    ("how do I change my email address", "account"),
    ("update my contact details", "account"),
    ("I need to edit my profile", "account"),
    ("change my email", "account"),
    ("I cannot log into my account", "account"),
    ("reset my password", "account"),
    ("I want to delete my account", "account"),
    ("where is my package", "delivery"),
    ("I have to change the delivery address", "shipping"),
    ("I need to edit my shipping address", "shipping"),
]

CLASS_MAP = {
    "ACCOUNT": "account",
    "CANCELLATION_FEE": "cancel",
    "CONTACT": "contact",
    "DELIVERY": "delivery",
    "FEEDBACK": "feedback",
    "INVOICE": "invoice",
    "ORDER": "order",
    "PAYMENT": "payment",
    "REFUND": "refund",
    "SHIPPING": "shipping",
    "SUBSCRIPTION": "subscription",
}


def load_rows():
    from datasets import load_dataset

    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train"
    )
    by_class = defaultdict(list)
    for row in ds:
        label = CLASS_MAP.get(row["category"])
        if label:
            by_class[label].append(row["instruction"])
    return {k: v for k, v in by_class.items()}


def main():
    rng = np.random.default_rng(0)
    by_class = load_rows()
    print("available per class:", {k: len(v) for k, v in sorted(by_class.items())})

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

    # Hold out a disjoint test set per class before sampling any training data.
    test_texts, test_labels, pools = [], [], {}
    for label, texts in sorted(by_class.items()):
        idx = rng.permutation(len(texts))
        held = idx[:120]
        pools[label] = [texts[i] for i in idx[120:]]
        test_texts += [texts[i] for i in held]
        test_labels += [label] * len(held)

    all_texts = sorted({t for pool in pools.values() for t in pool} | set(test_texts))
    if CACHE.exists():
        blob = np.load(CACHE, allow_pickle=True)
        emb = {t: v for t, v in zip(blob["texts"], blob["vecs"])}
        print(f"loaded {len(emb)} cached embeddings")
    else:
        emb = {}
    missing = [t for t in all_texts if t not in emb]
    if missing:
        print(f"encoding {len(missing)} texts...")
        vecs = model.encode(missing, batch_size=64, show_progress_bar=False,
                            normalize_embeddings=True)
        emb.update(dict(zip(missing, vecs)))
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(CACHE, texts=np.array(list(emb.keys()), dtype=object),
                            vecs=np.array(list(emb.values())))
        print(f"cached {len(emb)} embeddings to {CACHE}")

    probe_vecs = model.encode([p[0] for p in PROBES], normalize_embeddings=True)
    X_test = np.array([emb[t] for t in test_texts])

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    results = {"probes": [p[0] for p in PROBES], "runs": []}

    for n_per_class in (50, 200, 500, "all"):
        X, y = [], []
        for label, pool in sorted(pools.items()):
            take = len(pool) if n_per_class == "all" else min(n_per_class, len(pool))
            for t in pool[:take]:
                X.append(emb[t])
                y.append(label)
        X = np.array(X)

        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(scaler.transform(X), y)

        test_acc = float(clf.score(scaler.transform(X_test), test_labels))
        probs = clf.predict_proba(scaler.transform(probe_vecs))
        classes = list(clf.classes_)

        run = {
            "n_per_class": n_per_class,
            "n_train": int(len(X)),
            "test_accuracy": round(test_acc, 4),
            "probe_results": [],
        }
        for (text, expected), row in zip(PROBES, probs):
            top = int(np.argmax(row))
            run["probe_results"].append({
                "text": text,
                "expected": expected,
                "predicted": classes[top],
                "confidence": round(float(row[top]), 4),
                "correct": classes[top] == expected,
                "p_expected": round(float(row[classes.index(expected)]), 4),
            })
        n_ok = sum(p["correct"] for p in run["probe_results"])
        run["probes_correct"] = f"{n_ok}/{len(PROBES)}"
        results["runs"].append(run)

        print(f"\n=== {n_per_class} per class (n={len(X)}) "
              f"test_acc={test_acc:.4f} probes={n_ok}/{len(PROBES)}")
        for p in run["probe_results"]:
            flag = "ok  " if p["correct"] else "WRONG"
            print(f"  {flag} {p['predicted']:13s} {p['confidence']:.4f} "
                  f"(expected {p['expected']}, p={p['p_expected']:.4f})  {p['text']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
