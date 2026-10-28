"""Support radius: how far is an input from the data a head was trained on?

A logistic head has no notion of "far from everything it has seen", so it is confidently wrong on text that sits
0.4 cosine away from all training examples. We keep, per head, a compact description of the training data in the
encoder space: the L2-normalised class centroids, plus the sorted distribution of each training text's cosine distance
to its nearest centroid. At prediction time the input's nearest-centroid distance is placed on that distribution and
returned as `support_percentile` (0–100: the share of training texts that were closer to the data than this input;
100 means farther than every training text). It composes with the confidence cutoff rather than replacing it.

The file is `support.npz` next to `manifest.json`: centroids (float16, n_classes × dim) and quantiles (float32, 1001).
"""
from pathlib import Path
import numpy as np

N_QUANTILES = 1001


def _normalise(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def build_support(embeddings: np.ndarray, labels) -> dict:
    """Centroids per label and the training distribution of nearest-centroid cosine distance."""
    X = _normalise(embeddings); labels = np.asarray(labels)
    classes = sorted(set(labels.tolist()))
    cent = _normalise(np.stack([X[labels == c].mean(axis=0) for c in classes]))
    d = 1.0 - (X @ cent.T).max(axis=1)
    q = np.quantile(d, np.linspace(0, 1, N_QUANTILES)).astype(np.float32)
    return {"classes": classes, "centroids": cent.astype(np.float16), "quantiles": q}


def save_support(artifact_dir: Path, support: dict):
    np.savez_compressed(Path(artifact_dir) / "support.npz", centroids=support["centroids"], quantiles=support["quantiles"], classes=np.array(support["classes"], dtype=object))


def load_support(artifact_dir: Path):
    p = Path(artifact_dir) / "support.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return {"centroids": z["centroids"].astype(np.float32), "quantiles": z["quantiles"].astype(np.float32), "classes": z["classes"].tolist()}


def support_of(support: dict, embedding: np.ndarray) -> dict:
    """Distance to the nearest centroid and its percentile on the training distribution."""
    e = _normalise(np.asarray(embedding).reshape(1, -1))[0]
    sims = support["centroids"] @ e
    d = float(1.0 - sims.max())
    q = support["quantiles"]
    pct = float(np.searchsorted(q, d, side="right")) / (len(q) - 1) * 100.0
    return {"support_distance": round(d, 4), "support_percentile": round(min(100.0, pct), 1)}
