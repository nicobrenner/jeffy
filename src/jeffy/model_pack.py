"""Model pack: explicitly identified pretrained artifacts.

Each artifact is bound to its dataset name, encoder, label mapping,
and classifier class ordering. No guessing by class count.

The pack is built by `build_pack.py` which trains from source datasets
and saves all metadata needed to serve predictions without ambiguity.
"""

import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path

import joblib
import numpy as np


@dataclass
class ArtifactManifest:
    """Everything needed to identify and use a pretrained classifier."""
    dataset: str                      # e.g. "banking77"
    description: str                  # human readable task description
    n_classes: int
    task_type: str                    # "choice" or "noul"
    encoder: str                      # e.g. "BAAI/bge-large-en-v1.5"
    encoder_dim: int                  # e.g. 1024
    labels: dict[str, str]            # class_id -> human label (from dataset features)
    class_order: list[str]            # classifier.classes_ order (for probability mapping)
    hf_source: str                    # HuggingFace dataset ID
    hf_config: str | None             # HuggingFace config name
    hf_revision: str | None           # pinned revision (if known)
    license: str
    train_examples: int
    train_accuracy: float
    test_examples: int
    test_accuracy: float
    scaler_mean_hash: str             # first 8 chars of sha256 of scaler mean vector
    classifier_coef_hash: str         # first 8 chars of sha256 of classifier coef
    schema_version: int = 1


def save_artifact(
    out_dir: Path,
    dataset: str,
    clf,
    scaler,
    manifest: ArtifactManifest,
):
    """Save a single pretrained artifact with full metadata."""
    artifact_dir = out_dir / dataset
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save classifier + scaler
    joblib.dump({"classifier": clf, "scaler": scaler}, artifact_dir / "model.pkl")

    # Save manifest
    with open(artifact_dir / "manifest.json", "w") as f:
        json.dump(asdict(manifest), f, indent=2)


def load_artifact(artifact_dir: Path) -> tuple[object, object, ArtifactManifest]:
    """Load a pretrained artifact with its manifest.

    Returns (classifier, scaler, manifest).
    Raises ValueError if integrity checks fail.
    """
    manifest_path = artifact_dir / "manifest.json"
    model_path = artifact_dir / "model.pkl"

    if not manifest_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing manifest or model in {artifact_dir}")

    with open(manifest_path) as f:
        raw = json.load(f)
    manifest = ArtifactManifest(**raw)

    data = joblib.load(model_path)
    clf = data["classifier"]
    scaler = data["scaler"]

    # Verify class order matches
    clf_classes = list(clf.classes_)
    if clf_classes != manifest.class_order:
        raise ValueError(
            f"Classifier classes {clf_classes[:5]}... don't match "
            f"manifest class_order {manifest.class_order[:5]}..."
        )

    # Verify scaler integrity
    scaler_hash = hashlib.sha256(scaler.mean_.tobytes()).hexdigest()[:8]
    if scaler_hash != manifest.scaler_mean_hash:
        raise ValueError(
            f"Scaler hash mismatch: got {scaler_hash}, "
            f"expected {manifest.scaler_mean_hash}"
        )

    return clf, scaler, manifest


def coef_hash(clf) -> str:
    """Hash of classifier coefficients for integrity checking."""
    return hashlib.sha256(clf.coef_.tobytes()).hexdigest()[:8]


def scaler_hash(scaler) -> str:
    """Hash of scaler mean for integrity checking."""
    return hashlib.sha256(scaler.mean_.tobytes()).hexdigest()[:8]
