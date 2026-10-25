"""Follow-up to fix_support_ticket.py.

That run showed more data does not fix issue #11: at 24,722 examples per class
"change the address on my account" is predicted shipping at 0.9997, worse than
at 50 per class, while held-out accuracy reaches 0.9992. So the model is
learning the dataset correctly and the dataset's label boundary is the problem.

This tests the remaining hypothesis: adding the missing phrasing to the account
class fixes the probes without hurting accuracy on the dataset's own test split.

Writes data/eval_results/support_ticket_issue11_fix.json
"""
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

OUT = REPO / "data/eval_results/support_ticket_issue11_fix.json"
# Embedding cache: ~25k bge-large vectors take ~25 min to compute on CPU,
# so keep them out of the repo but reuse across runs.
CACHE = Path(os.environ.get("JEFFY_EMBED_CACHE",
                            Path.home() / ".cache/jeffy-experiments")) / "bitext_embeddings.npz"

# The phrasing Bitext's account class is missing: changing a contact field on an
# existing account, as opposed to changing the account's plan or type.
ACCOUNT_PATCH = [
    "change the address on my account",
    "how do I change my email address",
    "update my contact details",
    "change my email",
    "I need to update the email on my account",
    "how can I edit my phone number",
    "change my contact email please",
    "update the address on my profile",
    "I want to change my registered address",
    "edit my personal details",
    "update my personal information",
    "how do I change the phone number on my account",
    "change my mailing address on file",
    "I need to correct my name on the account",
    "update my email preferences and address",
    "can I change the email I signed up with",
    "my contact details are out of date",
    "please update my account address",
    "how to edit account contact information",
    "change the phone number linked to my account",
    "I moved, update my address on the account",
    "fix the email address on my account",
    "amend my contact information",
    "I need my account details changed",
]

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
    # Held-back probes, not in ACCOUNT_PATCH, to check this generalizes rather
    # than just memorizing the patch.
    ("how do I update the email on file", "account"),
    ("I need to change my home address", "account"),
    ("where do I edit my contact number", "account"),
]

CLASS_MAP = {
    "ACCOUNT": "account", "CANCELLATION_FEE": "cancel", "CONTACT": "contact",
    "DELIVERY": "delivery", "FEEDBACK": "feedback", "INVOICE": "invoice",
    "ORDER": "order", "PAYMENT": "payment", "REFUND": "refund",
    "SHIPPING": "shipping", "SUBSCRIPTION": "subscription",
}

UNSEEN = {"how do I update the email on file", "I need to change my home address",
          "where do I edit my contact number"}


def main():
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    ds = load_dataset(
        "bitext/Bitext-customer-support-llm-chatbot-training-dataset", split="train")
    by_class = defaultdict(list)
    for row in ds:
        label = CLASS_MAP.get(row["category"])
        if label:
            by_class[label].append(row["instruction"])

    blob = np.load(CACHE, allow_pickle=True)
    emb = {t: v for t, v in zip(blob["texts"], blob["vecs"])}
    print(f"loaded {len(emb)} cached embeddings")

    model = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")
    extra = [t for t in ACCOUNT_PATCH + [p[0] for p in PROBES] if t not in emb]
    if extra:
        print(f"encoding {len(extra)} new texts...")
        vecs = model.encode(extra, batch_size=32, normalize_embeddings=True)
        emb.update(dict(zip(extra, vecs)))

    # Same holdout as the first run, so test accuracy is comparable.
    test_texts, test_labels, pools = [], [], {}
    for label, texts in sorted(by_class.items()):
        idx = rng.permutation(len(texts))
        pools[label] = [texts[i] for i in idx[120:]]
        test_texts += [texts[i] for i in idx[:120]]
        test_labels += [label] * 120
    X_test = np.array([emb[t] for t in test_texts])

    results = {"runs": []}
    N_PER_CLASS = 500

    for patched in (False, True):
        X, y = [], []
        for label, pool in sorted(pools.items()):
            for t in pool[:N_PER_CLASS]:
                X.append(emb[t]); y.append(label)
        if patched:
            for t in ACCOUNT_PATCH:
                X.append(emb[t]); y.append("account")
        X = np.array(X)

        scaler = StandardScaler().fit(X)
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(scaler.transform(X), y)

        test_acc = float(clf.score(scaler.transform(X_test), test_labels))
        probe_vecs = np.array([emb[p[0]] for p in PROBES])
        probs = clf.predict_proba(scaler.transform(probe_vecs))
        classes = list(clf.classes_)

        run = {"patched": patched, "n_train": int(len(X)),
               "n_patch": len(ACCOUNT_PATCH) if patched else 0,
               "test_accuracy": round(test_acc, 4), "probes": []}
        label = f"+{len(ACCOUNT_PATCH)} account examples" if patched else "baseline"
        print(f"\n=== {label}: n={len(X)} test_acc={test_acc:.4f}")
        for (text, expected), row in zip(PROBES, probs):
            top = int(np.argmax(row))
            ok = classes[top] == expected
            run["probes"].append({
                "text": text, "expected": expected, "predicted": classes[top],
                "confidence": round(float(row[top]), 4), "correct": ok,
                "held_back": text in UNSEEN,
            })
            mark = "ok   " if ok else "WRONG"
            tag = " [unseen]" if text in UNSEEN else ""
            print(f"  {mark} {classes[top]:10s} {row[top]:.4f}  {text}{tag}")
        n_ok = sum(p["correct"] for p in run["probes"])
        run["probes_correct"] = f"{n_ok}/{len(PROBES)}"
        print(f"  -> {n_ok}/{len(PROBES)} probes correct")
        results["runs"].append(run)

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
