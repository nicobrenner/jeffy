"""Train a custom classifier from user-provided examples.

Usage:
    # From Python
    from jeffy.train import train_classifier
    clf = train_classifier(
        texts=["great product", "terrible service", ...],
        labels=["positive", "negative", ...],
        task_id="my_sentiment",
    )
    result = clf.predict("this is wonderful")

    # From CLI
    jeffy-train --input data.csv --text-col text --label-col label --task-id my_task
    jeffy-train --input data.jsonl --text-key text --label-key label --task-id my_task
"""

import argparse
import csv
import json
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.preprocessing import StandardScaler

from .catalog import ENCODER, ENCODER_DIM
from .model_pack import ArtifactManifest, coef_hash, save_artifact, scaler_hash


class TrainedClassifier:
    """A trained classifier ready for prediction."""

    def __init__(self, clf, scaler, encoder, labels, task_id):
        self.clf = clf
        self.scaler = scaler
        self.encoder = encoder
        self.labels = labels  # {class_id: human_label}
        self.task_id = task_id
        self._reverse = {v: k for k, v in labels.items()}

    def predict(self, text: str) -> dict:
        """Classify a single text."""
        t0 = time.perf_counter()
        emb = self.encoder.encode([text], show_progress_bar=False)
        t_embed = time.perf_counter() - t0

        t1 = time.perf_counter()
        X = self.scaler.transform(emb)
        probs = self.clf.predict_proba(X)[0]
        pred_idx = np.argmax(probs)
        pred_id = self.clf.classes_[pred_idx]
        t_clf = time.perf_counter() - t1

        prob_dict = {}
        for cls, prob in zip(self.clf.classes_, probs):
            human = self.labels.get(str(cls), str(cls))
            prob_dict[human] = round(float(prob), 4)
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: -x[1]))

        return {
            "label": self.labels.get(str(pred_id), str(pred_id)),
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": prob_dict,
            "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            "embedding_ms": round(t_embed * 1000, 2),
            "classifier_ms": round(t_clf * 1000, 4),
        }

    def save(self, out_dir: str):
        """Save as a model pack artifact."""
        out_path = Path(out_dir) / self.task_id
        manifest = ArtifactManifest(
            dataset=self.task_id,
            description=f"Custom classifier: {self.task_id}",
            n_classes=len(self.labels),
            task_type="choice" if len(self.labels) > 2 else "noul",
            encoder=ENCODER,
            encoder_dim=ENCODER_DIM,
            labels=self.labels,
            class_order=list(self.clf.classes_),
            hf_source="user-provided",
            hf_config=None,
            hf_revision=None,
            license="user-provided",
            train_examples=0,  # filled below
            train_accuracy=0,
            test_examples=0,
            test_accuracy=0,
            scaler_mean_hash=scaler_hash(self.scaler),
            classifier_coef_hash=coef_hash(self.clf),
        )
        save_artifact(out_path.parent, self.task_id, self.clf, self.scaler, manifest)
        return str(out_path)


def train_classifier(
    texts: list[str],
    labels: list[str],
    task_id: str = "custom",
    C: float = 0.01,
    test_size: float = 0.2,
    cv_folds: int = 3,
    verbose: bool = True,
) -> TrainedClassifier:
    """Train a classifier from user-provided examples.

    Args:
        texts: Input texts.
        labels: Corresponding labels (strings).
        task_id: Name for this classifier.
        C: Regularization strength (smaller = more regularization).
        test_size: Fraction held out for evaluation (0 to skip).
        cv_folds: Cross-validation folds for reported accuracy (0 to skip).
        verbose: Print training progress.

    Returns:
        TrainedClassifier ready for prediction and saving.
    """
    from sentence_transformers import SentenceTransformer

    if len(texts) != len(labels):
        raise ValueError(f"texts ({len(texts)}) and labels ({len(labels)}) must have same length")
    if len(texts) < 2:
        raise ValueError("Need at least 2 examples")

    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    label_map = {label: label for label in unique_labels}

    if verbose:
        print(f"Training '{task_id}': {len(texts)} examples, {n_classes} classes")
        for label in unique_labels:
            count = labels.count(label)
            print(f"  {label}: {count} examples")

    # Split if requested
    train_texts, train_labels = texts, labels
    test_texts, test_labels = None, None

    if test_size > 0 and len(texts) >= 10:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(sss.split(texts, labels))
        train_texts = [texts[i] for i in train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_texts = [texts[i] for i in test_idx]
        test_labels = [labels[i] for i in test_idx]
        if verbose:
            print(f"  Split: {len(train_texts)} train, {len(test_texts)} test")

    # Encode
    if verbose:
        print(f"  Encoding with {ENCODER}...")
    t0 = time.perf_counter()
    encoder = SentenceTransformer(ENCODER, device="cpu")
    train_embs = encoder.encode(train_texts, batch_size=64, show_progress_bar=verbose)
    t_encode = time.perf_counter() - t0
    if verbose:
        print(f"  Encoded {len(train_texts)} texts in {t_encode:.1f}s")

    # Train
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_embs)
    clf = LogisticRegression(C=C, max_iter=3000, solver="newton-cg", random_state=42)
    clf.fit(X_train, train_labels)
    t_train = time.perf_counter() - t0

    train_acc = accuracy_score(train_labels, clf.predict(X_train))
    if verbose:
        print(f"  Train accuracy: {train_acc:.1%} ({t_train:.1f}s)")

    # Cross-validation
    if cv_folds > 0 and len(train_texts) >= cv_folds * 2:
        cv_scores = cross_val_score(
            LogisticRegression(C=C, max_iter=3000, solver="newton-cg", random_state=42),
            X_train, train_labels, cv=cv_folds, scoring="accuracy")
        if verbose:
            print(f"  {cv_folds}-fold CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    # Test evaluation
    if test_texts:
        test_embs = encoder.encode(test_texts, batch_size=64, show_progress_bar=False)
        X_test = scaler.transform(test_embs)
        test_preds = clf.predict(X_test)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        if verbose:
            print(f"  Test accuracy: {test_acc:.1%}")
            print(f"  Test macro F1: {test_f1:.1%}")

    result = TrainedClassifier(clf, scaler, encoder, label_map, task_id)
    if verbose:
        print(f"  Ready. Use .predict(text) or .save(dir)")
    return result


def main():
    """CLI entry point for training from CSV or JSONL files."""
    parser = argparse.ArgumentParser(
        description="Train a Jeffy classifier from a data file",
        epilog="Example: jeffy-train --input emails.csv --text-col body --label-col category --task-id email_routing")
    parser.add_argument("--input", required=True, help="CSV or JSONL file with text and labels")
    parser.add_argument("--text-col", "--text-key", default="text", help="Column/key for text (default: text)")
    parser.add_argument("--label-col", "--label-key", default="label", help="Column/key for labels (default: label)")
    parser.add_argument("--task-id", default="custom", help="Name for this classifier")
    parser.add_argument("--C", type=float, default=0.01, help="Regularization (default: 0.01)")
    parser.add_argument("--save-dir", default=None, help="Save trained model to this directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {input_path} not found")
        return

    # Load data
    texts, labels = [], []
    if input_path.suffix == ".jsonl":
        with open(input_path) as f:
            for line in f:
                d = json.loads(line)
                texts.append(str(d[args.text_col]))
                labels.append(str(d[args.label_col]))
    elif input_path.suffix in (".csv", ".tsv"):
        delimiter = "\t" if input_path.suffix == ".tsv" else ","
        with open(input_path) as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                texts.append(row[args.text_col])
                labels.append(row[args.label_col])
    else:
        print(f"Unsupported format: {input_path.suffix}. Use .csv, .tsv, or .jsonl")
        return

    print(f"Loaded {len(texts)} examples from {input_path}")

    clf = train_classifier(
        texts=texts, labels=labels, task_id=args.task_id,
        C=args.C, test_size=args.test_size)

    if args.save_dir:
        path = clf.save(args.save_dir)
        print(f"\nSaved to {path}/")
        print(f"Load with: JEFFY_PACK_DIR={args.save_dir} jeffy-serve")

    # Interactive test
    print("\nTest predictions (Ctrl+C to exit):")
    try:
        while True:
            text = input("> ")
            if not text.strip():
                continue
            r = clf.predict(text)
            print(f"  {r['label']} ({r['confidence']:.0%})")
            top3 = list(r["probabilities"].items())[:3]
            for label, prob in top3:
                print(f"    {prob:.1%} {label}")
    except (KeyboardInterrupt, EOFError):
        print()


if __name__ == "__main__":
    main()
