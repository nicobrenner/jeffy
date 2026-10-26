"""Pretrained decision engine.

Loads verified model pack artifacts and serves predictions
with capability metadata in every response.
"""

import importlib.resources
import logging
import os
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .catalog import ENCODER, FEATURES_ENCODER, KNOWN_ENCODERS, Capability, get_encoder_size_mb
from .model_pack import load_artifact

logger = logging.getLogger(__name__)

try:
    import onnxruntime  # noqa: F401
    _ONNX_AVAILABLE = True
except ImportError:
    _ONNX_AVAILABLE = False


def _resolve_backend() -> str:
    choice = os.environ.get("JEFFY_BACKEND", "auto").lower()
    if choice == "onnx":
        if not _ONNX_AVAILABLE:
            logger.warning("JEFFY_BACKEND=onnx but onnxruntime not installed, falling back to torch")
            return "torch"
        return "onnx"
    if choice == "torch":
        return "torch"
    return "onnx" if _ONNX_AVAILABLE else "torch"


def _bundled_pack_dir() -> Path | None:
    """Return the path to the model pack bundled inside the installed package."""
    try:
        ref = importlib.resources.files("jeffy") / "pack"
        # Traverse to get a real filesystem path
        p = Path(str(ref))
        if p.is_dir():
            return p
    except Exception:
        pass
    return None


def default_pack_dir() -> str:
    """Resolve the default model pack directory.

    Priority:
    1. JEFFY_PACK_DIR environment variable
    2. Bundled pack inside the installed jeffy package
    3. data/model_pack in the current working directory (development fallback)
    """
    import os
    env = os.environ.get("JEFFY_PACK_DIR")
    if env:
        return env

    bundled = _bundled_pack_dir()
    if bundled is not None:
        return str(bundled)

    return "data/model_pack"


class Engine:
    """Loads pretrained classifiers from a model pack and serves predictions."""

    def __init__(self, pack_dir: str | None = None, device: str = "cpu"):
        if pack_dir is None:
            pack_dir = default_pack_dir()
        self.pack_dir = Path(pack_dir)
        self.device = device
        self._backend = _resolve_backend()
        self._encoders: dict[str, SentenceTransformer] = {}
        self._capabilities: dict[str, Capability] = {}
        self._classifiers: dict[str, object] = {}
        self._scalers: dict[str, object] = {}
        self._encoder_load_time: float = 0.0

    def load(self):
        """Load encoder and all pretrained classifiers from the model pack.

        Raises FileNotFoundError if the pack directory does not exist.
        """
        if not self.pack_dir.exists():
            raise FileNotFoundError(
                f"Model pack not found at '{self.pack_dir}'. "
                f"Set JEFFY_PACK_DIR to the model pack location, or run "
                f"'jeffy-build' to create one from source datasets."
            )

        t0 = time.perf_counter()
        self._encoders[ENCODER] = self._load_encoder(ENCODER)
        self._encoder_load_time = time.perf_counter() - t0
        logger.info(f"Encoder backend: {self._backend} (load: {self._encoder_load_time:.1f}s)")

        quarantined = []

        for artifact_dir in sorted(self.pack_dir.iterdir()):
            if not artifact_dir.is_dir():
                continue
            manifest_path = artifact_dir / "manifest.json"
            if not manifest_path.exists():
                continue

            try:
                clf, scaler, manifest = load_artifact(artifact_dir)
            except (FileNotFoundError, ValueError, Exception) as e:
                quarantined.append((artifact_dir.name, str(e)))
                continue

            ds_name = manifest.dataset

            if manifest.encoder not in KNOWN_ENCODERS:
                encoder_dim = getattr(manifest, "encoder_dim", None)
                if not encoder_dim:
                    quarantined.append((ds_name, f"unknown encoder with no encoder_dim: {manifest.encoder}"))
                    continue
                KNOWN_ENCODERS[manifest.encoder] = encoder_dim

            self._classifiers[ds_name] = clf
            self._scalers[ds_name] = scaler

            cap = Capability(
                task_id=ds_name,
                name=manifest.description,
                description=manifest.description,
                dataset=ds_name,
                labels=manifest.labels,
                n_classes=manifest.n_classes,
                task_type=manifest.task_type,
                hf_source=manifest.hf_source,
                license=manifest.license,
                encoder=manifest.encoder,
                train_examples=manifest.train_examples,
                train_accuracy=manifest.train_accuracy,
                test_accuracy=manifest.test_accuracy,
                n_features=getattr(manifest, 'n_features', None),
                feature_layout=getattr(manifest, 'feature_layout', None),
                task_signature=None,
                encoder_size_mb=get_encoder_size_mb(manifest.encoder),
                label_descriptions=getattr(manifest, 'label_descriptions', None),
            )
            self._capabilities[ds_name] = cap

        if quarantined:
            for name, reason in quarantined:
                logger.warning(f"Quarantined artifact '{name}': {reason}")

        if not self._capabilities:
            logger.warning(f"No capabilities loaded from '{self.pack_dir}'")

    def register_trained(self, task_id: str, clf, scaler, manifest):
        """Hot-load a freshly trained classifier into the running engine."""
        from .catalog import KNOWN_ENCODERS
        if manifest.encoder not in KNOWN_ENCODERS:
            KNOWN_ENCODERS[manifest.encoder] = manifest.encoder_dim
        self._classifiers[task_id] = clf
        self._scalers[task_id] = scaler
        self._capabilities[task_id] = Capability(
            task_id=task_id,
            name=manifest.description,
            description=manifest.description,
            dataset=task_id,
            labels=manifest.labels,
            n_classes=manifest.n_classes,
            task_type=manifest.task_type,
            hf_source=manifest.hf_source,
            license=manifest.license,
            encoder=manifest.encoder,
            train_examples=manifest.train_examples,
            train_accuracy=manifest.train_accuracy,
            test_accuracy=manifest.test_accuracy,
            n_features=getattr(manifest, 'n_features', None),
            feature_layout=getattr(manifest, 'feature_layout', None),
            task_signature=None,
            encoder_size_mb=get_encoder_size_mb(manifest.encoder),
            label_descriptions=getattr(manifest, 'label_descriptions', None),
        )
        logger.info(f"Registered trained classifier '{task_id}' ({manifest.n_classes} classes)")

    @property
    def capabilities(self) -> dict[str, Capability]:
        return self._capabilities

    @property
    def encoder_name(self) -> str:
        return ENCODER

    @property
    def encoder_load_time(self) -> float:
        return self._encoder_load_time

    @property
    def backend(self) -> str:
        return self._backend

    def _load_encoder(self, encoder_name: str) -> SentenceTransformer:
        kwargs: dict = {"device": self.device}
        if self._backend == "onnx":
            kwargs["backend"] = "onnx"
        return SentenceTransformer(encoder_name, **kwargs)

    def _classify(self, task_id: str, X: np.ndarray, t0: float, t_embed: float,
                  threshold: float | None = None) -> dict:
        """Shared classification logic for both text and feature inputs."""
        cap = self._capabilities[task_id]
        clf = self._classifiers[task_id]
        scaler = self._scalers[task_id]

        t1 = time.perf_counter()
        X_scaled = scaler.transform(X)
        probs = clf.predict_proba(X_scaled)[0]
        classes = clf.classes_
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        t_classify = time.perf_counter() - t1

        total_ms = (time.perf_counter() - t0) * 1000

        prob_dict = {}
        for cls, prob in zip(classes, probs):
            human = cap.labels.get(str(cls), str(cls))
            prob_dict[human] = round(float(prob), 4)
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: -x[1]))

        max_conf = float(probs[pred_idx])
        sorted_probs = np.sort(probs)[::-1]
        margin = float(sorted_probs[0] - sorted_probs[1]) if len(sorted_probs) > 1 else 1.0

        entropy = -float(np.sum(probs * np.log(probs + 1e-10)))
        max_entropy = float(np.log(len(classes)))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        is_oos = False
        if threshold is not None and max_conf < threshold:
            is_oos = True

        human_label = cap.labels.get(str(pred_label), str(pred_label))

        result = {
            "label": "out_of_scope" if is_oos else human_label,
            "label_id": "" if is_oos else str(pred_label),
            "confidence": round(max_conf, 4),
            "out_of_scope": is_oos,
            "entropy": round(normalized_entropy, 4),
            "margin": round(margin, 4),
            "probabilities": prob_dict,
            "latency_ms": round(total_ms, 2),
            "embedding_ms": round(t_embed * 1000, 2),
            "classifier_ms": round(t_classify * 1000, 4),
            "capability": {
                "task_id": task_id,
                "name": cap.name,
                "encoder": cap.encoder,
                "backend": self._backend,
                "n_classes": cap.n_classes,
                "train_examples": cap.train_examples,
                "train_accuracy": cap.train_accuracy,
                "test_accuracy": cap.test_accuracy,
            },
        }
        if is_oos:
            result["best_guess"] = human_label
        if threshold is not None:
            result["threshold"] = threshold
        if cap.is_feature_based:
            result["capability"]["n_features"] = cap.n_features
        return result

    def predict(self, task_id: str, text: str, threshold: float | None = None) -> dict:
        """Run prediction for a text-based pretrained capability."""
        if task_id not in self._classifiers:
            return {"error": f"No pretrained capability '{task_id}'. "
                    f"Available: {list(self._capabilities.keys())}"}

        cap = self._capabilities[task_id]
        if cap.is_feature_based:
            return {"error": f"Capability '{task_id}' is feature-based. "
                    f"Use predict_features() with a numeric vector instead of text."}

        t0 = time.perf_counter()
        encoder = self._get_encoder(cap.encoder)
        embedding = encoder.encode([text], show_progress_bar=False)
        t_embed = time.perf_counter() - t0

        return self._classify(task_id, embedding, t0, t_embed, threshold=threshold)

    def predict_batch(self, task_id: str, texts: list[str],
                      threshold: float | None = None) -> list[dict]:
        if task_id not in self._classifiers:
            return [{"error": f"No pretrained capability '{task_id}'. "
                     f"Available: {list(self._capabilities.keys())}"}]

        cap = self._capabilities[task_id]
        if cap.is_feature_based:
            return [{"error": f"Capability '{task_id}' is feature-based. "
                     f"Use predict_features() with a numeric vector instead of text."}]

        t0 = time.perf_counter()
        encoder = self._get_encoder(cap.encoder)
        embeddings = encoder.encode(texts, show_progress_bar=False)
        t_embed = time.perf_counter() - t0

        results = []
        for i, text in enumerate(texts):
            row = embeddings[i:i+1]
            results.append(self._classify(task_id, row, t0, t_embed, threshold=threshold))
        return results

    def _get_encoder(self, encoder_name: str) -> SentenceTransformer:
        if encoder_name not in self._encoders:
            logger.info(f"Loading encoder '{encoder_name}' (backend={self._backend})...")
            self._encoders[encoder_name] = self._load_encoder(encoder_name)
        return self._encoders[encoder_name]

    def predict_features(self, task_id: str, features: list[float],
                         threshold: float | None = None) -> dict:
        """Run prediction for a feature-based pretrained capability."""
        if task_id not in self._classifiers:
            return {"error": f"No pretrained capability '{task_id}'. "
                    f"Available: {list(self._capabilities.keys())}"}

        cap = self._capabilities[task_id]
        if not cap.is_feature_based:
            return {"error": f"Capability '{task_id}' is text-based. "
                    f"Use predict() with text instead of features."}

        if cap.n_features and len(features) != cap.n_features:
            return {"error": f"Expected {cap.n_features} features, got {len(features)}"}

        t0 = time.perf_counter()
        X = np.array(features, dtype=np.float64).reshape(1, -1)

        return self._classify(task_id, X, t0, 0.0, threshold=threshold)
