"""Pretrained decision engine.

Loads verified model pack artifacts and serves predictions
with capability metadata in every response.
"""

import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .catalog import ENCODER, Capability
from .model_pack import load_artifact


class Engine:
    """Loads pretrained classifiers from a model pack and serves predictions."""

    def __init__(self, pack_dir: str = "data/model_pack", device: str = "cpu"):
        self.pack_dir = Path(pack_dir)
        self.device = device
        self._encoder: SentenceTransformer | None = None
        self._capabilities: dict[str, Capability] = {}
        self._classifiers: dict[str, object] = {}
        self._scalers: dict[str, object] = {}
        self._encoder_load_time: float = 0.0

    def load(self):
        """Load encoder and all pretrained classifiers from the model pack."""
        t0 = time.perf_counter()
        self._encoder = SentenceTransformer(ENCODER, device=self.device)
        self._encoder_load_time = time.perf_counter() - t0

        if not self.pack_dir.exists():
            return

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

            # Verify encoder matches
            if manifest.encoder != ENCODER:
                quarantined.append((ds_name, f"encoder mismatch: {manifest.encoder} vs {ENCODER}"))
                continue

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
                task_signature=None,
            )
            self._capabilities[ds_name] = cap

        if quarantined:
            import logging
            logger = logging.getLogger(__name__)
            for name, reason in quarantined:
                logger.warning(f"Quarantined artifact '{name}': {reason}")

    @property
    def capabilities(self) -> dict[str, Capability]:
        return self._capabilities

    @property
    def encoder_name(self) -> str:
        return ENCODER

    @property
    def encoder_load_time(self) -> float:
        return self._encoder_load_time

    def predict(self, task_id: str, text: str) -> dict:
        """Run prediction for a specific pretrained capability."""
        if task_id not in self._classifiers:
            return {"error": f"No pretrained capability '{task_id}'. "
                    f"Available: {list(self._capabilities.keys())}"}

        cap = self._capabilities[task_id]
        clf = self._classifiers[task_id]
        scaler = self._scalers[task_id]

        t0 = time.perf_counter()
        embedding = self._encoder.encode([text], show_progress_bar=False)
        t_embed = time.perf_counter() - t0

        t1 = time.perf_counter()
        X = scaler.transform(embedding)
        probs = clf.predict_proba(X)[0]
        classes = clf.classes_
        pred_idx = np.argmax(probs)
        pred_label = classes[pred_idx]
        t_classify = time.perf_counter() - t1

        total_ms = (time.perf_counter() - t0) * 1000

        # Map probabilities using classifier's class order and manifest labels
        prob_dict = {}
        for cls, prob in zip(classes, probs):
            human = cap.labels.get(str(cls), str(cls))
            prob_dict[human] = round(float(prob), 4)
        prob_dict = dict(sorted(prob_dict.items(), key=lambda x: -x[1]))

        human_label = cap.labels.get(str(pred_label), str(pred_label))

        return {
            "label": human_label,
            "label_id": str(pred_label),
            "confidence": round(float(probs[pred_idx]), 4),
            "probabilities": prob_dict,
            "latency_ms": round(total_ms, 2),
            "embedding_ms": round(t_embed * 1000, 2),
            "classifier_ms": round(t_classify * 1000, 4),
            "capability": {
                "task_id": task_id,
                "name": cap.name,
                "encoder": cap.encoder,
                "n_classes": cap.n_classes,
                "train_examples": cap.train_examples,
                "train_accuracy": cap.train_accuracy,
                "test_accuracy": cap.test_accuracy,
            },
        }
