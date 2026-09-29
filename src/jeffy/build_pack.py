"""Build a verified model pack from source datasets.

Trains each dataset independently, saves artifacts with explicit identity,
recovers label mappings from the dataset features, and verifies roundtrip
prediction fidelity.

Usage:
    python -m jeffy.build_pack
    python -m jeffy.build_pack --datasets banking77 ag_news
    python -m jeffy.build_pack --out data/model_pack
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .catalog import ENCODER, ENCODER_DIM, LICENSES
from .model_pack import ArtifactManifest, coef_hash, save_artifact, scaler_hash

# Dataset definitions with label recovery instructions
DATASETS = {
    "banking77": {
        "hf": "legacy-datasets/banking77",
        "text": "text", "label": "label",
        "description": "Banking customer service intent detection",
        "task_type": "choice",
        "label_source": "features",  # labels from dataset.features['label'].names
    },
    "clinc_oos": {
        "hf": "clinc_oos", "hf_config": "plus",
        "text": "text", "label": "intent",
        "description": "Intent detection with out-of-scope",
        "task_type": "choice",
        "label_source": "features",
    },
    "massive_intent": {
        "hf": "mteb/amazon_massive_intent", "hf_config": "en",
        "text": "text", "label": "label",
        "description": "Amazon MASSIVE voice command intents",
        "task_type": "choice",
        "label_source": "string_values",  # labels ARE the string values
    },
    "ag_news": {
        "hf": "fancyzhx/ag_news",
        "text": "text", "label": "label",
        "description": "News article topic classification",
        "task_type": "choice",
        "label_source": "features",
    },
    "dbpedia": {
        "hf": "fancyzhx/dbpedia_14",
        "text": "content", "label": "label",
        "description": "Wikipedia article ontology classification",
        "task_type": "choice",
        "label_source": "features",
    },
    "sst2": {
        "hf": "stanfordnlp/sst2",
        "text": "sentence", "label": "label",
        "split_test": "validation",
        "description": "Movie review sentiment (positive/negative)",
        "task_type": "noul",
        "label_source": "manual",
        "manual_labels": {"0": "negative", "1": "positive"},
    },
    "emotion": {
        "hf": "dair-ai/emotion",
        "text": "text", "label": "label",
        "description": "Text emotion detection",
        "task_type": "choice",
        "label_source": "features",
    },
    "imdb": {
        "hf": "stanfordnlp/imdb",
        "text": "text", "label": "label",
        "description": "Movie review sentiment (positive/negative)",
        "task_type": "noul",
        "label_source": "manual",
        "manual_labels": {"0": "negative", "1": "positive"},
    },
    "sms_spam": {
        "hf": "ucirvine/sms_spam",
        "text": "sms", "label": "label",
        "description": "SMS spam detection",
        "task_type": "noul",
        "label_source": "manual",
        "manual_labels": {"0": "ham", "1": "spam"},
    },
    "snli": {
        "hf": "stanfordnlp/snli",
        "text": ["premise", "hypothesis"], "label": "label",
        "filter_label": -1,
        "description": "Natural language inference",
        "task_type": "choice",
        "label_source": "manual",
        "manual_labels": {"0": "entailment", "1": "neutral", "2": "contradiction"},
    },
    "tweet_eval_sentiment": {
        "hf": "cardiffnlp/tweet_eval", "hf_config": "sentiment",
        "text": "text", "label": "label",
        "description": "Tweet sentiment analysis",
        "task_type": "choice",
        "label_source": "manual",
        "manual_labels": {"0": "negative", "1": "neutral", "2": "positive"},
    },
    "tweet_eval_emotion": {
        "hf": "cardiffnlp/tweet_eval", "hf_config": "emotion",
        "text": "text", "label": "label",
        "description": "Tweet emotion detection",
        "task_type": "choice",
        "label_source": "manual",
        "manual_labels": {"0": "anger", "1": "joy", "2": "optimism", "3": "sadness"},
    },
    "tweet_eval_offensive": {
        "hf": "cardiffnlp/tweet_eval", "hf_config": "offensive",
        "text": "text", "label": "label",
        "description": "Offensive language detection",
        "task_type": "noul",
        "label_source": "manual",
        "manual_labels": {"0": "not_offensive", "1": "offensive"},
    },
}


def recover_labels(ds, config) -> dict[str, str]:
    """Recover human-readable label mapping from the dataset source."""
    source = config["label_source"]

    if source == "manual":
        return config["manual_labels"]

    if source == "features":
        label_col = config["label"]
        feat = ds["train"].features[label_col]
        if hasattr(feat, "names"):
            return {str(i): name for i, name in enumerate(feat.names)}
        raise ValueError(f"No .names on feature {label_col}")

    if source == "string_values":
        # Labels are the string values themselves
        train_labels = sorted(set(ds["train"][config["label"]]))
        return {label: label for label in train_labels}

    raise ValueError(f"Unknown label_source: {source}")


def load_and_split(ds_name, config, max_train=10000, max_test=2000):
    """Load dataset, extract texts and string labels."""
    hf_config = config.get("hf_config")
    ds = load_dataset(config["hf"], hf_config) if hf_config else load_dataset(config["hf"])

    split_test = config.get("split_test", "test")
    if split_test in ds:
        train_ds, test_ds = ds["train"], ds[split_test]
    elif "validation" in ds:
        train_ds, test_ds = ds["train"], ds["validation"]
    else:
        split = ds["train"].train_test_split(test_size=0.2, seed=42)
        train_ds, test_ds = split["train"], split["test"]

    text_col = config["text"]
    if isinstance(text_col, list):
        train_texts = [" [SEP] ".join(train_ds[c][i] for c in text_col) for i in range(len(train_ds))]
        test_texts = [" [SEP] ".join(test_ds[c][i] for c in text_col) for i in range(len(test_ds))]
    else:
        train_texts = list(train_ds[text_col])
        test_texts = list(test_ds[text_col])

    train_labels = [str(x) for x in train_ds[config["label"]]]
    test_labels = [str(x) for x in test_ds[config["label"]]]

    # Filter invalid
    if "filter_label" in config:
        fl = str(config["filter_label"])
        mask = [l != fl for l in train_labels]
        train_texts = [t for t, m in zip(train_texts, mask) if m]
        train_labels = [l for l, m in zip(train_labels, mask) if m]
        mask = [l != fl for l in test_labels]
        test_texts = [t for t, m in zip(test_texts, mask) if m]
        test_labels = [l for l, m in zip(test_labels, mask) if m]

    # Cap sizes
    if len(train_texts) > max_train:
        np.random.seed(42)
        idx = np.random.choice(len(train_texts), max_train, replace=False)
        train_texts = [train_texts[i] for i in idx]
        train_labels = [train_labels[i] for i in idx]

    if len(test_texts) > max_test:
        np.random.seed(42)
        idx = np.random.choice(len(test_texts), max_test, replace=False)
        test_texts = [test_texts[i] for i in idx]
        test_labels = [test_labels[i] for i in idx]

    labels = recover_labels(ds, config)
    return train_texts, train_labels, test_texts, test_labels, labels


def main():
    parser = argparse.ArgumentParser(description="Build verified model pack")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS.keys()))
    parser.add_argument("--out", default="data/model_pack")
    parser.add_argument("--C", type=float, default=0.01)
    parser.add_argument("--max-train", type=int, default=10000)
    parser.add_argument("--max-test", type=int, default=2000)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Building model pack: {len(args.datasets)} datasets")
    print(f"Output: {out_dir}")
    print(f"Encoder: {ENCODER}")
    print()

    encoder = SentenceTransformer(ENCODER)
    results = []

    for ds_name in args.datasets:
        if ds_name not in DATASETS:
            print(f"Unknown dataset: {ds_name}")
            continue

        config = DATASETS[ds_name]
        print(f"{'='*60}")
        print(f"{ds_name}: {config['description']}")

        try:
            train_texts, train_labels, test_texts, test_labels, labels = \
                load_and_split(ds_name, config, args.max_train, args.max_test)
        except Exception as e:
            print(f"  FAILED to load: {e}")
            continue

        n_classes = len(set(train_labels))
        print(f"  Train: {len(train_texts)}, Test: {len(test_texts)}, Classes: {n_classes}")
        print(f"  Labels: {list(labels.items())[:5]}...")

        # Embed
        t0 = time.perf_counter()
        train_emb = encoder.encode(train_texts, batch_size=256, show_progress_bar=False)
        test_emb = encoder.encode(test_texts, batch_size=256, show_progress_bar=False)
        t_embed = time.perf_counter() - t0

        # Train
        t1 = time.perf_counter()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_emb)
        X_test = scaler.transform(test_emb)

        clf = LogisticRegression(C=args.C, max_iter=3000, solver="newton-cg", random_state=42)
        clf.fit(X_train, train_labels)
        t_train = time.perf_counter() - t1

        # Evaluate
        train_preds = clf.predict(X_train)
        test_preds = clf.predict(X_test)
        train_acc = accuracy_score(train_labels, train_preds)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)

        # Verify probability mapping roundtrip
        probs = clf.predict_proba(X_test[:3])
        for i in range(min(3, len(probs))):
            pred_from_probs = clf.classes_[np.argmax(probs[i])]
            pred_from_predict = test_preds[i]
            assert pred_from_probs == pred_from_predict, \
                f"Probability mapping mismatch at test[{i}]: {pred_from_probs} vs {pred_from_predict}"

        # Verify all class IDs have label mappings
        for cls_id in clf.classes_:
            if str(cls_id) not in labels:
                print(f"  WARNING: class {cls_id} has no label mapping")

        manifest = ArtifactManifest(
            dataset=ds_name,
            description=config["description"],
            n_classes=n_classes,
            task_type=config["task_type"],
            encoder=ENCODER,
            encoder_dim=ENCODER_DIM,
            labels=labels,
            class_order=list(clf.classes_),
            hf_source=config["hf"],
            hf_config=config.get("hf_config"),
            hf_revision=None,  # TODO: pin revision
            license=LICENSES.get(ds_name, "see source"),
            train_examples=len(train_texts),
            train_accuracy=round(train_acc, 6),
            test_examples=len(test_texts),
            test_accuracy=round(test_acc, 6),
            scaler_mean_hash=scaler_hash(scaler),
            classifier_coef_hash=coef_hash(clf),
        )

        save_artifact(out_dir, ds_name, clf, scaler, manifest)

        print(f"  Train acc: {train_acc:.4f}")
        print(f"  Test acc:  {test_acc:.4f}")
        print(f"  Test F1:   {test_f1:.4f}")
        print(f"  Time: embed={t_embed:.1f}s train={t_train:.1f}s")
        print(f"  Saved to {out_dir / ds_name}/")

        results.append({
            "dataset": ds_name,
            "n_classes": n_classes,
            "train_size": len(train_texts),
            "test_size": len(test_texts),
            "train_acc": round(train_acc, 6),
            "test_acc": round(test_acc, 6),
            "test_f1": round(test_f1, 6),
            "embed_time": round(t_embed, 1),
            "train_time": round(t_train, 1),
        })
        print()

    # Summary
    print(f"\n{'='*80}")
    print("MODEL PACK SUMMARY")
    print(f"{'='*80}")
    print(f"{'Dataset':<25s} {'Classes':>7s} {'Train':>7s} {'Test':>7s} {'Test Acc':>9s} {'F1':>8s}")
    print("-" * 67)
    for r in sorted(results, key=lambda x: -x["test_acc"]):
        print(f"{r['dataset']:<25s} {r['n_classes']:>7d} {r['train_size']:>7d} "
              f"{r['test_size']:>7d} {r['test_acc']:>9.4f} {r['test_f1']:>8.4f}")

    avg_acc = np.mean([r["test_acc"] for r in results])
    avg_f1 = np.mean([r["test_f1"] for r in results])
    print(f"\n{'Average':<25s} {'':>7s} {'':>7s} {'':>7s} {avg_acc:>9.4f} {avg_f1:>8.4f}")

    # Save pack manifest
    with open(out_dir / "pack_manifest.json", "w") as f:
        json.dump({
            "encoder": ENCODER,
            "encoder_dim": ENCODER_DIM,
            "classifier": "LogisticRegression(C=0.01, solver=newton-cg)",
            "datasets": results,
            "build_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "schema_version": 1,
        }, f, indent=2)

    print(f"\nPack manifest: {out_dir / 'pack_manifest.json'}")
    print(f"Total: {len(results)} artifacts")


if __name__ == "__main__":
    main()
