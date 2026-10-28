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

from .catalog import ENCODER, ENCODER_DIM, FEATURES_ENCODER
from .model_pack import ArtifactManifest, coef_hash, save_artifact, scaler_hash


def _check_encoder_path(encoder_name: str):
    """A path-like encoder (local checkpoint dir) must exist before we hand it to SentenceTransformer."""
    if encoder_name.startswith(("./", "../", "/")) and not Path(encoder_name).is_dir():
        raise ValueError(f"Encoder path not found: {encoder_name}")


class TrainedClassifier:
    """A trained classifier ready for prediction."""

    def __init__(self, clf, scaler, encoder, labels, task_id, encoder_name: str | None = None):
        self.clf = clf
        self.scaler = scaler
        self.encoder = encoder
        self.labels = labels  # {class_id: human_label}
        self.task_id = task_id
        self.encoder_name = encoder_name or ENCODER
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
        encoder_dim = self.scaler.mean_.shape[0]
        manifest = ArtifactManifest(
            dataset=self.task_id,
            description=f"Custom classifier: {self.task_id}",
            n_classes=len(self.labels),
            task_type="choice" if len(self.labels) > 2 else "noul",
            encoder=self.encoder_name,
            encoder_dim=encoder_dim,
            labels=self.labels,
            class_order=list(self.clf.classes_),
            hf_source="user-provided",
            hf_config=None,
            hf_revision=None,
            license="user-provided",
            train_examples=0,
            train_accuracy=0,
            test_examples=0,
            test_accuracy=0,
            scaler_mean_hash=scaler_hash(self.scaler),
            classifier_coef_hash=coef_hash(self.clf),
        )
        save_artifact(out_path.parent, self.task_id, self.clf, self.scaler, manifest)
        if getattr(self, "support", None):
            from .support import save_support
            save_support(out_path, self.support)
        return str(out_path)


def train_classifier(
    texts: list[str],
    labels: list[str],
    task_id: str = "custom",
    C: float = 0.01,
    test_size: float = 0.2,
    cv_folds: int = 3,
    verbose: bool = True,
    encoder: str | None = None,
    tune_C: bool = False,
) -> TrainedClassifier:
    """Train a classifier from user-provided examples.

    Args:
        texts: Input texts.
        labels: Corresponding labels (strings).
        task_id: Name for this classifier.
        C: Regularization strength (smaller = more regularization). Ignored when tune_C=True.
        test_size: Fraction held out for evaluation (0 to skip).
        cv_folds: Cross-validation folds for reported accuracy (0 to skip).
        verbose: Print training progress.
        encoder: SentenceTransformer model name, HF model id, or local checkpoint
                 path (plain transformers checkpoints get mean pooling automatically).
                 Defaults to BAAI/bge-large-en-v1.5.
        tune_C: When True, grid-search over [0.001, 0.01, 0.1, 1.0, 10.0] using 3-fold CV.

    Returns:
        TrainedClassifier ready for prediction and saving.
    """
    from sentence_transformers import SentenceTransformer

    if len(texts) != len(labels):
        raise ValueError(f"texts ({len(texts)}) and labels ({len(labels)}) must have same length")
    if len(texts) < 2:
        raise ValueError("Need at least 2 examples")

    encoder_name = encoder or ENCODER
    _check_encoder_path(encoder_name)
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
        print(f"  Encoding with {encoder_name}...")
    t0 = time.perf_counter()
    st_encoder = SentenceTransformer(encoder_name, device="cpu")
    train_embs = st_encoder.encode(train_texts, batch_size=64, show_progress_bar=verbose)
    t_encode = time.perf_counter() - t0
    if verbose:
        print(f"  Encoded {len(train_texts)} texts in {t_encode:.1f}s")

    # Train
    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_embs)

    if tune_C and len(train_texts) >= 6:
        C_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
        best_C, best_score = C, 0.0
        n_folds = min(3, len(set(train_labels)))
        for c_val in C_grid:
            try:
                scores = cross_val_score(
                    LogisticRegression(C=c_val, max_iter=3000, solver="newton-cg", random_state=42),
                    X_train, train_labels, cv=n_folds, scoring="accuracy")
                mean_score = scores.mean()
                if mean_score > best_score:
                    best_score = mean_score
                    best_C = c_val
            except ValueError:
                continue
        C = best_C
        if verbose:
            print(f"  tune_C: best C={C} (CV accuracy={best_score:.1%})")

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
        test_embs = st_encoder.encode(test_texts, batch_size=64, show_progress_bar=False)
        X_test = scaler.transform(test_embs)
        test_preds = clf.predict(X_test)
        test_acc = accuracy_score(test_labels, test_preds)
        test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        if verbose:
            print(f"  Test accuracy: {test_acc:.1%}")
            print(f"  Test macro F1: {test_f1:.1%}")

    result = TrainedClassifier(clf, scaler, st_encoder, label_map, task_id, encoder_name=encoder_name)
    try:
        from .support import build_support
        result.support = build_support(train_embs, train_labels)
    except Exception:
        result.support = None
    if verbose:
        print(f"  Ready. Use .predict(text) or .save(dir)")
    return result


def train_binary_flags(
    texts: list[str],
    flags: dict[str, list[str]],
    task_id: str = "custom",
    C: float = 0.01,
    test_size: float = 0.2,
    verbose: bool = True,
    encoder: str | None = None,
    tune_C: bool = False,
    class_weight: str | dict | None = "balanced",
    tune_threshold: bool = False,
    min_threshold_positives: int = 8,
) -> dict[str, TrainedClassifier]:
    """Train one binary classifier per flag.

    Each flag gets its own binary LR. Flags can fire independently, so an input
    can trigger multiple flags at once.

    Args:
        texts: Input texts.
        flags: {flag_name: [label_per_text]}. Each label list must be "yes"/"no"
               (or "true"/"false", "1"/"0").
        task_id: Base name; each flag becomes {task_id}_flag_{flag_name}.
        C: Regularization strength. Ignored when tune_C=True.
        test_size: Fraction held out for evaluation.
        verbose: Print training progress.
        encoder: SentenceTransformer model name, HF model id, or local checkpoint
                 path (plain transformers checkpoints get mean pooling automatically).
                 Defaults to BAAI/bge-large-en-v1.5.
        tune_C: Grid-search C per flag independently, scored by positive-class F1.
        class_weight: Passed to LogisticRegression. "balanced" (the default)
                      maximizes recall on rare flags, which is usually what you
                      want for triage. Pass None for the unweighted fit, which
                      raises far fewer flags at higher precision, or a dict for
                      explicit weights.
        tune_threshold: Also pick a per-flag decision threshold on a held-out dev
                        slice, scored by positive-class F1. The chosen threshold
                        is folded into the model's intercept, so it survives
                        save() and needs no special handling at predict time.
        min_threshold_positives: Skip threshold tuning for a flag whose dev slice
                                 has fewer positives than this. Below roughly 8
                                 the threshold fits noise rather than signal.

    Returns:
        Dict of {flag_name: TrainedClassifier}.

    Note:
        Positive-class F1 is used for tuning rather than macro F1. For a rare
        yes/no flag the "no" class scores near 1.0, so macro F1 is dominated by
        the easy class and can select a different C than the flag's own
        performance warrants.
    """
    from sklearn.metrics import make_scorer
    from sentence_transformers import SentenceTransformer

    _YES = {"yes", "true", "1", "y"}
    _NO = {"no", "false", "0", "n"}

    for flag_name, labels in flags.items():
        if len(labels) != len(texts):
            raise ValueError(f"Flag '{flag_name}' has {len(labels)} labels but {len(texts)} texts")
        normed = {l.lower().strip() for l in labels}
        if not normed <= (_YES | _NO):
            raise ValueError(f"Flag '{flag_name}' has invalid labels: {normed - _YES - _NO}. Use yes/no")

    encoder_name = encoder or ENCODER
    _check_encoder_path(encoder_name)
    if verbose:
        print(f"Encoding {len(texts)} texts with {encoder_name}...")
    st_encoder = SentenceTransformer(encoder_name, device="cpu")
    all_embs = st_encoder.encode(texts, batch_size=64, show_progress_bar=verbose)

    results = {}
    for flag_name, raw_labels in flags.items():
        binary_labels = ["yes" if l.lower().strip() in _YES else "no" for l in raw_labels]
        flag_task_id = f"{task_id}_flag_{flag_name}"

        if verbose:
            pos = binary_labels.count("yes")
            neg = binary_labels.count("no")
            print(f"  Flag '{flag_name}': {pos} yes, {neg} no")

        train_embs, train_labels = all_embs, binary_labels
        test_embs, test_labels = None, None

        if test_size > 0 and len(texts) >= 10:
            sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
            try:
                train_idx, test_idx = next(sss.split(all_embs, binary_labels))
                train_embs = all_embs[train_idx]
                train_labels = [binary_labels[i] for i in train_idx]
                test_embs = all_embs[test_idx]
                test_labels = [binary_labels[i] for i in test_idx]
            except ValueError:
                pass

        scaler = StandardScaler()
        X_train = scaler.fit_transform(train_embs)

        n_pos = train_labels.count("yes")
        n_neg = train_labels.count("no")

        def _make_lr(c_val):
            return LogisticRegression(
                C=c_val, max_iter=3000, solver="newton-cg",
                class_weight=class_weight, random_state=42)

        C_grid = [0.001, 0.01, 0.1, 1.0, 10.0]
        DEV_FRACTION = 0.25
        flag_C = C
        flag_threshold = None
        dev_pos = 0

        # A threshold tuned on the data it was fit on is meaningless, so this
        # needs its own dev slice carved out of train (never out of test).
        # The dev slice is only a quarter of train, so clearing
        # min_threshold_positives there takes 1/DEV_FRACTION times as many
        # positives overall.
        dev_ready = (
            tune_threshold
            and n_pos >= min_threshold_positives / DEV_FRACTION
            and n_neg >= 4
            and len(train_labels) >= 20
        )

        if dev_ready:
            sss_dev = StratifiedShuffleSplit(
                n_splits=1, test_size=DEV_FRACTION, random_state=42)
            try:
                fit_idx, dev_idx = next(sss_dev.split(X_train, train_labels))
            except ValueError:
                dev_ready = False

        if dev_ready:
            dev_labels = [train_labels[i] for i in dev_idx]
            dev_pos = dev_labels.count("yes")
            if dev_pos < min_threshold_positives:
                dev_ready = False

        if dev_ready:
            X_fit = X_train[fit_idx]
            fit_labels = [train_labels[i] for i in fit_idx]
            X_dev = X_train[dev_idx]

            # Pick C and the threshold together: the best threshold depends on how
            # the probabilities are spread, which C changes.
            best = (C, 0.5, -1.0)
            for c_val in C_grid if tune_C else [C]:
                try:
                    probe = _make_lr(c_val).fit(X_fit, fit_labels)
                except ValueError:
                    continue
                yes_col = list(probe.classes_).index("yes")
                p_dev = probe.predict_proba(X_dev)[:, yes_col]
                for t in [round(0.05 * i, 2) for i in range(2, 19)]:
                    preds = ["yes" if p >= t else "no" for p in p_dev]
                    score = f1_score(dev_labels, preds, pos_label="yes", zero_division=0)
                    if score > best[2]:
                        best = (c_val, t, score)
            flag_C, flag_threshold, best_score = best
            if verbose:
                print(f"    tuned on {dev_pos} dev positives: C={flag_C}, "
                      f"threshold={flag_threshold} (dev F1+={best_score:.1%})")
        elif tune_C and len(train_labels) >= 6 and n_pos >= 2 and n_neg >= 2:
            # Fold count must not exceed the rarer class, or StratifiedKFold
            # produces folds with no positives at all.
            n_folds = max(2, min(3, n_pos, n_neg))
            pos_f1 = make_scorer(f1_score, pos_label="yes", zero_division=0)
            best_C, best_score = C, -1.0
            for c_val in C_grid:
                try:
                    scores = cross_val_score(
                        _make_lr(c_val), X_train, train_labels,
                        cv=n_folds, scoring=pos_f1)
                except ValueError:
                    continue
                if scores.mean() > best_score:
                    best_score = scores.mean()
                    best_C = c_val
            flag_C = best_C
            if verbose:
                print(f"    tune_C: best C={flag_C} "
                      f"({n_folds}-fold CV F1+={max(best_score, 0.0):.1%})")
        elif tune_C and verbose:
            print(f"    tune_C: skipped, only {n_pos} positive and {n_neg} "
                  f"negative training examples")

        if tune_threshold and flag_threshold is None and verbose:
            needed = int(np.ceil(min_threshold_positives / DEV_FRACTION))
            print(f"    threshold: left at 0.5, needs {min_threshold_positives} "
                  f"positives in the dev slice (got {dev_pos}); that takes about "
                  f"{needed} in train and this flag has {n_pos}")

        clf = _make_lr(flag_C)
        clf.fit(X_train, train_labels)

        # Folding the threshold into the intercept keeps the decision boundary
        # inside the linear model, so it survives save() and ONNX export.
        # argmax on the shifted model == p(yes) >= threshold on the original.
        if flag_threshold is not None and flag_threshold != 0.5:
            clf.intercept_ = clf.intercept_ - np.log(
                flag_threshold / (1.0 - flag_threshold))

        if verbose and test_embs is not None:
            X_test = scaler.transform(test_embs)
            test_preds = clf.predict(X_test)
            test_pos_f1 = f1_score(test_labels, test_preds, pos_label="yes", zero_division=0)
            test_macro_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
            n_raised = list(test_preds).count("yes")
            print(f"    Test F1+: {test_pos_f1:.1%} (macro {test_macro_f1:.1%}), "
                  f"raised {n_raised}/{len(test_labels)}")

        label_map = {"yes": "yes", "no": "no"}
        results[flag_name] = TrainedClassifier(
            clf, scaler, st_encoder, label_map, flag_task_id, encoder_name=encoder_name)

    if verbose:
        print(f"  Trained {len(results)} binary flag classifiers")
    return results


class TrainedFeatureClassifier:
    """A trained classifier for numeric feature vectors."""

    def __init__(self, clf, scaler, labels, task_id, n_features, feature_layout=None):
        self.clf = clf
        self.scaler = scaler
        self.labels = labels
        self.task_id = task_id
        self.n_features = n_features
        self.feature_layout = feature_layout

    def predict(self, features: list[float]) -> dict:
        t0 = time.perf_counter()
        X = np.array(features, dtype=np.float64).reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        probs = self.clf.predict_proba(X_scaled)[0]
        pred_idx = np.argmax(probs)
        pred_id = self.clf.classes_[pred_idx]
        t_clf = time.perf_counter() - t0

        prob_dict = {}
        for cls, prob in zip(self.clf.classes_, probs):
            human = self.labels.get(str(cls), str(cls))
            prob_dict[human] = round(float(prob), 4)
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: -x[1]))

        return {
            "label": self.labels.get(str(pred_id), str(pred_id)),
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": prob_dict,
            "latency_ms": round(t_clf * 1000, 2),
        }

    def save(self, out_dir: str):
        out_path = Path(out_dir) / self.task_id
        manifest = ArtifactManifest(
            dataset=self.task_id,
            description=f"Custom feature classifier: {self.task_id}",
            n_classes=len(self.labels),
            task_type="choice" if len(self.labels) > 2 else "noul",
            encoder=FEATURES_ENCODER,
            encoder_dim=self.n_features,
            labels=self.labels,
            class_order=list(self.clf.classes_),
            hf_source="user-provided",
            hf_config=None,
            hf_revision=None,
            license="user-provided",
            train_examples=0,
            train_accuracy=0,
            test_examples=0,
            test_accuracy=0,
            scaler_mean_hash=scaler_hash(self.scaler),
            classifier_coef_hash=coef_hash(self.clf),
            n_features=self.n_features,
            feature_layout=self.feature_layout,
        )
        save_artifact(out_path.parent, self.task_id, self.clf, self.scaler, manifest)
        return str(out_path)


def train_feature_classifier(
    features: np.ndarray | list[list[float]],
    labels: list[str],
    task_id: str = "custom",
    feature_layout: dict[str, str] | None = None,
    C: float = 0.01,
    test_size: float = 0.2,
    cv_folds: int = 3,
    verbose: bool = True,
) -> TrainedFeatureClassifier:
    """Train a classifier from numeric feature vectors.

    Args:
        features: 2D array of shape (n_samples, n_features).
        labels: Corresponding labels (strings).
        task_id: Name for this classifier.
        feature_layout: Optional mapping of feature index to name.
        C: Regularization strength.
        test_size: Fraction held out for evaluation.
        cv_folds: Cross-validation folds.
        verbose: Print training progress.
    """
    X = np.asarray(features, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"features must be 2D, got {X.ndim}D")
    if len(X) != len(labels):
        raise ValueError(f"features ({len(X)}) and labels ({len(labels)}) must have same length")
    if len(X) < 2:
        raise ValueError("Need at least 2 examples")

    n_features = X.shape[1]
    unique_labels = sorted(set(labels))
    n_classes = len(unique_labels)
    label_map = {label: label for label in unique_labels}

    if verbose:
        print(f"Training '{task_id}': {len(X)} examples, {n_features} features, {n_classes} classes")

    train_X, train_labels = X, labels
    test_X, test_labels = None, None

    if test_size > 0 and len(X) >= 10:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        train_idx, test_idx = next(sss.split(X, labels))
        train_X = X[train_idx]
        train_labels = [labels[i] for i in train_idx]
        test_X = X[test_idx]
        test_labels = [labels[i] for i in test_idx]
        if verbose:
            print(f"  Split: {len(train_X)} train, {len(test_X)} test")

    t0 = time.perf_counter()
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_X)
    clf = LogisticRegression(C=C, max_iter=3000, solver="newton-cg", random_state=42)
    clf.fit(X_train_scaled, train_labels)
    t_train = time.perf_counter() - t0

    train_acc = accuracy_score(train_labels, clf.predict(X_train_scaled))
    if verbose:
        print(f"  Train accuracy: {train_acc:.1%} ({t_train:.1f}s)")

    if cv_folds > 0 and len(train_X) >= cv_folds * 2:
        cv_scores = cross_val_score(
            LogisticRegression(C=C, max_iter=3000, solver="newton-cg", random_state=42),
            X_train_scaled, train_labels, cv=cv_folds, scoring="accuracy")
        if verbose:
            print(f"  {cv_folds}-fold CV: {cv_scores.mean():.1%} ± {cv_scores.std():.1%}")

    if test_X is not None:
        X_test_scaled = scaler.transform(test_X)
        test_preds = clf.predict(X_test_scaled)
        test_acc = accuracy_score(test_labels, test_preds)
        if verbose:
            print(f"  Test accuracy: {test_acc:.1%}")

    result = TrainedFeatureClassifier(clf, scaler, label_map, task_id, n_features, feature_layout)
    if verbose:
        print(f"  Ready. Use .predict(features) or .save(dir)")
    return result


def main():
    """CLI entry point for training from CSV or JSONL files."""
    parser = argparse.ArgumentParser(
        description="Train a Jeffy classifier from a data file",
        epilog="Example: jeffy-train --input emails.csv --text-col body --label-col category --task-id email_routing")
    parser.add_argument("--input", help="CSV or JSONL file with text and labels")
    parser.add_argument("--example", action="store_true",
                        help="Train from the bundled reviews.csv (no --input needed)")
    parser.add_argument("--text-col", "--text-key", default="text", help="Column/key for text (default: text)")
    parser.add_argument("--label-col", "--label-key", default="label", help="Column/key for labels (default: label)")
    parser.add_argument("--task-id", default="custom", help="Name for this classifier")
    parser.add_argument("--C", type=float, default=0.01, help="Regularization (default: 0.01)")
    parser.add_argument("--save-dir", default=None, help="Save trained model to this directory")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split fraction (default: 0.2)")
    args = parser.parse_args()

    if args.example:
        from .examples import REVIEWS_CSV
        import importlib.resources
        input_path = Path(str(REVIEWS_CSV))
        if not args.task_id or args.task_id == "custom":
            args.task_id = "reviews"
    elif args.input:
        input_path = Path(args.input)
    else:
        parser.error("--input or --example is required")
        return

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
