"""Build support_ticket_v2 on bitext's 27 INTENTS instead of its 11 CATEGORIES.

The shipped support_ticket pack trains on bitext's `category` column. But bitext
ships a two-level taxonomy: 27 fine-grained `intent` values grouped under 11
`category` values. Training on the categories throws the useful level away, and
the category names do not describe what they contain:

    CANCEL        <- check_cancellation_fee only; cancel_order lives in ORDER
    SHIPPING      <- change_shipping_address, set_up_shipping_address only
    SUBSCRIPTION  <- newsletter_subscription only

So "cancel my order" routing to `order` was correct all along, and the
11-class pack cannot express the distinction a caller actually wants.

The decisive comparison is not v2's accuracy on 27 classes, which is a
different and harder task. It is: take v2's prediction, map the intent up to
its parent category, and score that against the SAME test texts the 11-class
pack sees. If the mapped-up v2 beats v1 on v1's own task, the collapse was
actively harmful rather than merely lossy.

Recipe matches the shipped pack so the comparison is honest: bge-large-en-v1.5,
StandardScaler, LogisticRegression(C=0.01, solver=newton-cg, max_iter=3000,
random_state=42), 50 train and 50 test per class, disjoint.

    python scripts/build_support_ticket_v2.py           # measure only
    python scripts/build_support_ticket_v2.py --write   # also write the pack
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
OUT = REPO / "data/eval_results/support_ticket_v2"
PACK = REPO / "src/jeffy/pack"
ENCODER = "BAAI/bge-large-en-v1.5"
C = 0.01
SEED = 42
N_PER_CLASS = 50

# Shital's reported mismatches plus the phrasings that probe the same seams.
PROBES = [
    "I want to cancel my subscription",
    "cancel my order",
    "how do I cancel",
    "I need to cancel a purchase I made",
    "where is my order",
    "when will my order arrive",
    "I want to check the status of my order",
    "I need to place an order",
    "I need to change the shipping address",
    "track my shipment",
    "what are your shipping options",
    "how much does shipping cost",
    "where is my package",
]


def load_bitext():
    """Group bitext by intent, and keep the intent -> category map."""
    from datasets import load_dataset

    ds = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")["train"]
    by_intent, parent = {}, {}
    for text, intent, category in zip(ds["instruction"], ds["intent"], ds["category"]):
        by_intent.setdefault(intent, []).append(text)
        parent[intent] = category.lower()
    return by_intent, parent


def split(by_intent):
    """50 train and 50 test per intent, disjoint, deterministic."""
    rng = np.random.default_rng(SEED)
    train, test = {}, {}
    for intent, texts in sorted(by_intent.items()):
        uniq = sorted(set(texts))
        idx = rng.permutation(len(uniq))
        train[intent] = [uniq[i] for i in idx[:N_PER_CLASS]]
        test[intent] = [uniq[i] for i in idx[N_PER_CLASS:N_PER_CLASS * 2]]
    return train, test


def flatten(d):
    texts, labels = [], []
    for k in sorted(d):
        for t in d[k]:
            texts.append(t)
            labels.append(k)
    return texts, labels


def fit(train_emb, train_labels, test_emb, test_labels):
    scaler = StandardScaler()
    X = scaler.fit_transform(train_emb)
    clf = LogisticRegression(C=C, max_iter=3000, solver="newton-cg", random_state=SEED)
    clf.fit(X, train_labels)
    preds = clf.predict(scaler.transform(test_emb))
    return {
        "n_train": len(train_labels),
        "n_classes": len(set(train_labels)),
        "n_test": len(test_labels),
        "train_accuracy": round(float(accuracy_score(train_labels, clf.predict(X))), 4),
        "test_accuracy": round(float(accuracy_score(test_labels, preds)), 4),
        "test_macro_f1": round(float(f1_score(test_labels, preds, average="macro", zero_division=0)), 4),
    }, clf, scaler, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="write the pack to src/jeffy/pack")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    by_intent, parent = load_bitext()
    print(f"{len(by_intent)} intents under {len(set(parent.values()))} categories")
    train, test = split(by_intent)
    tr_t, tr_intent = flatten(train)
    te_t, te_intent = flatten(test)
    overlap = set(tr_t) & set(te_t)
    if overlap:
        raise SystemExit(f"train/test overlap on {len(overlap)} texts, aborting")
    tr_cat = [parent[i] for i in tr_intent]
    te_cat = [parent[i] for i in te_intent]
    print(f"train {len(tr_t)}, test {len(te_t)}, no overlap")

    from sentence_transformers import SentenceTransformer

    enc = SentenceTransformer(ENCODER)
    t0 = time.perf_counter()
    tr_e = enc.encode(tr_t, batch_size=256, show_progress_bar=False)
    te_e = enc.encode(te_t, batch_size=256, show_progress_bar=False)
    pr_e = enc.encode(PROBES, batch_size=32, show_progress_bar=False)
    print(f"encoded {len(tr_t) + len(te_t) + len(PROBES)} texts in {time.perf_counter() - t0:.1f}s")

    results = {"encoder": ENCODER, "C": C, "seed": SEED, "n_per_class": N_PER_CLASS,
               "intent_to_category": parent, "runs": {}}

    # v2: 27 intents, the harder task
    v2, clf2, sc2, p2_intent = fit(tr_e, tr_intent, te_e, te_intent)
    results["runs"]["v2_27_intents"] = v2
    print(f"\nv2, 27 intents          acc {v2['test_accuracy']}  macroF1 {v2['test_macro_f1']}")

    # v1-style: the same texts collapsed to 11 categories, which is what ships
    v1, clf1, sc1, p1_cat = fit(tr_e, tr_cat, te_e, te_cat)
    results["runs"]["v1_11_categories"] = v1
    print(f"v1, 11 categories       acc {v1['test_accuracy']}  macroF1 {v1['test_macro_f1']}")

    # The decisive one: v2's intent prediction mapped up to its parent category,
    # scored on the identical task v1 was scored on.
    mapped = [parent[i] for i in p2_intent]
    acc = round(float(accuracy_score(te_cat, mapped)), 4)
    f1 = round(float(f1_score(te_cat, mapped, average="macro", zero_division=0)), 4)
    results["runs"]["v2_mapped_to_categories"] = {
        "n_test": len(te_cat), "test_accuracy": acc, "test_macro_f1": f1,
        "delta_vs_v1": round(acc - v1["test_accuracy"], 4)}
    print(f"v2 mapped -> categories acc {acc}  macroF1 {f1}   "
          f"delta vs v1 {acc - v1['test_accuracy']:+.4f}")

    # Shital's probes, both heads
    probes = []
    for i, text in enumerate(PROBES):
        pi = clf2.predict(sc2.transform(pr_e[i:i + 1]))[0]
        pc = clf1.predict(sc1.transform(pr_e[i:i + 1]))[0]
        conf2 = float(np.max(clf2.predict_proba(sc2.transform(pr_e[i:i + 1]))))
        probes.append({"text": text, "v2_intent": pi, "v2_mapped": parent[pi],
                       "v2_confidence": round(conf2, 4), "v1_category": pc})
        print(f"  {text:<42} v2={pi:<24} v1={pc}")
    results["probes"] = probes

    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    with open(OUT / "headline.csv", "w") as f:
        f.write("condition,n_classes,n_train,n_test,test_accuracy,test_macro_f1\n")
        for k in ("v2_27_intents", "v1_11_categories", "v2_mapped_to_categories"):
            r = results["runs"][k]
            f.write(f"{k},{r.get('n_classes', 11)},{r.get('n_train', len(tr_t))},"
                    f"{r['n_test']},{r['test_accuracy']},{r['test_macro_f1']}\n")
    (OUT / "split.json").write_text(json.dumps({"train": train, "test": test}, indent=2))
    print(f"\nwrote {OUT}/results.json, headline.csv, split.json")

    if args.write:
        from jeffy.model_pack import ArtifactManifest, coef_hash, save_artifact, scaler_hash
        labels = {i: i for i in sorted(by_intent)}
        manifest = ArtifactManifest(
            dataset="support_ticket_v2",
            description="Customer support ticket routing, 27 fine-grained intents",
            n_classes=len(labels), task_type="choice", encoder=ENCODER,
            encoder_dim=int(tr_e.shape[1]), labels=labels,
            class_order=list(clf2.classes_),
            hf_source="bitext/Bitext-customer-support-llm-chatbot-training-dataset (intent column, not category)",
            hf_config=None, hf_revision=None, license="CC BY 4.0",
            train_examples=v2["n_train"], train_accuracy=v2["train_accuracy"],
            test_examples=v2["n_test"], test_accuracy=v2["test_accuracy"],
            scaler_mean_hash=scaler_hash(sc2), classifier_coef_hash=coef_hash(clf2),
            label_descriptions={i: f"bitext intent '{i}', under category '{parent[i]}'"
                                for i in sorted(by_intent)},
        )
        save_artifact(PACK, "support_ticket_v2", clf2, sc2, manifest)
        print(f"wrote pack to {PACK / 'support_ticket_v2'}")


if __name__ == "__main__":
    main()
