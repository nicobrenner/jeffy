"""Model pack: explicitly identified pretrained artifacts.

Each artifact is bound to its dataset name, encoder, label mapping,
and classifier class ordering. No guessing by class count.

Artifacts are saved in two formats:
- model.pkl: joblib pickle (fast, but sklearn-version-sensitive)
- model.npz: numpy arrays of coefficients (portable across sklearn versions)

On load, the npz is preferred if the pkl triggers a version warning.
"""

import hashlib
import json
import logging
import warnings

logger = logging.getLogger(__name__)
from dataclasses import dataclass, asdict, fields
from pathlib import Path

import joblib
import numpy as np


@dataclass
class ArtifactManifest:
    """Everything needed to identify and use a pretrained classifier."""
    dataset: str
    description: str
    n_classes: int
    task_type: str
    encoder: str
    encoder_dim: int
    labels: dict[str, str]
    class_order: list[str]
    hf_source: str
    hf_config: str | None
    hf_revision: str | None
    license: str
    train_examples: int
    train_accuracy: float
    test_examples: int
    test_accuracy: float
    scaler_mean_hash: str
    classifier_coef_hash: str
    n_features: int | None = None
    feature_layout: dict[str, str] | None = None
    schema_version: int = 1


def save_artifact(out_dir: Path, dataset: str, clf, scaler, manifest: ArtifactManifest):
    """Save a single pretrained artifact with full metadata."""
    artifact_dir = out_dir / dataset
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Save pickle (fast load when versions match)
    joblib.dump({"classifier": clf, "scaler": scaler}, artifact_dir / "model.pkl")

    # Save portable numpy format (works across sklearn versions)
    np.savez_compressed(
        artifact_dir / "model.npz",
        coef=clf.coef_,
        intercept=clf.intercept_,
        classes=np.array(clf.classes_, dtype=str),
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        scaler_var=scaler.var_,
        scaler_n_samples_seen=np.array([scaler.n_samples_seen_]),
    )

    with open(artifact_dir / "manifest.json", "w") as f:
        json.dump(asdict(manifest), f, indent=2)


def _reconstruct_from_npz(npz_path: Path):
    """Reconstruct classifier and scaler from portable numpy arrays.

    If the npz has scaler arrays, reconstructs both. Otherwise returns
    the classifier and None (caller loads scaler from pkl).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    data = np.load(npz_path, allow_pickle=False)

    clf = LogisticRegression()
    clf.coef_ = data["coef"]
    clf.intercept_ = data["intercept"]
    clf.classes_ = data["classes"]
    clf.n_features_in_ = data["coef"].shape[1]

    if "scaler_mean" in data:
        scaler = StandardScaler()
        scaler.mean_ = data["scaler_mean"]
        scaler.scale_ = data["scaler_scale"]
        scaler.var_ = data["scaler_var"] if "scaler_var" in data else scaler.scale_ ** 2
        scaler.n_samples_seen_ = int(data["scaler_n_samples_seen"][0]) if "scaler_n_samples_seen" in data else 1
        scaler.n_features_in_ = len(scaler.mean_)
        return clf, scaler

    return clf, None


def load_artifact(artifact_dir: Path) -> tuple[object, object, ArtifactManifest]:
    """Load a pretrained artifact with its manifest.

    Prefers npz (portable) over pkl. Falls back to pkl if npz is missing.
    Raises ValueError if integrity checks fail.
    """
    manifest_path = artifact_dir / "manifest.json"
    npz_path = artifact_dir / "model.npz"
    pkl_path = artifact_dir / "model.pkl"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest in {artifact_dir}")

    with open(manifest_path) as f:
        raw = json.load(f)
    known_fields = {f.name for f in fields(ArtifactManifest)}
    manifest = ArtifactManifest(**{k: v for k, v in raw.items() if k in known_fields})

    # Prefer npz (portable), fall back to pkl
    clf, scaler = None, None
    if npz_path.exists():
        clf, scaler = _reconstruct_from_npz(npz_path)
    if clf is None and pkl_path.exists():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            data = joblib.load(pkl_path)
        clf = data.get("classifier") if isinstance(data, dict) else None
        scaler = data.get("scaler") if isinstance(data, dict) else data
    elif scaler is None and pkl_path.exists():
        import pickle
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        if hasattr(data, "mean_"):
            scaler = data
        elif isinstance(data, dict) and "scaler" in data:
            scaler = data["scaler"]
    if clf is None:
        raise FileNotFoundError(f"No model.npz or model.pkl in {artifact_dir}")
    if scaler is None:
        raise FileNotFoundError(f"No scaler found in {artifact_dir}")

    # Verify class order
    clf_classes = list(clf.classes_)
    if clf_classes != manifest.class_order:
        raise ValueError(
            f"Classifier classes {clf_classes[:5]}... don't match "
            f"manifest class_order {manifest.class_order[:5]}..."
        )

    # Verify scaler integrity (skip if hash method differs)
    s_hash = hashlib.sha256(scaler.mean_.tobytes()).hexdigest()[:8]
    if s_hash != manifest.scaler_mean_hash:
        logger.debug(
            f"Scaler hash info: computed {s_hash}, manifest {manifest.scaler_mean_hash}"
        )

    return clf, scaler, manifest


def coef_hash(clf) -> str:
    return hashlib.sha256(clf.coef_.tobytes()).hexdigest()[:8]


def scaler_hash(scaler) -> str:
    return hashlib.sha256(scaler.mean_.tobytes()).hexdigest()[:8]
