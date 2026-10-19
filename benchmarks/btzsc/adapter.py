"""Jeffy adapter for the BTZSC benchmark pipeline."""

import json
import numpy as np
from btzsc.models import BaseModel

from pathlib import Path as _Path
_HERE = _Path(__file__).resolve().parent

PACK_DIR = str(_HERE.parent.parent / "data" / "model_pack")
MAPPINGS_PATH = str(_HERE / "mappings.json")

BTZSC_TO_JEFFY = {
    # sentiment (6)
    "imdb": "imdb",
    "amazonpolarity": "amazonpolarity",
    "appreviews": "appreviews",
    "yelpreviews": "yelpreviews",
    "rottentomatoes": "rottentomatoes",
    "financialphrasebank": "financialphrasebank",
    # emotion (2)
    "emotiondair": "emotion",
    "empathetic": "empathetic",
    # intent (3)
    "banking77": "banking77",
    "massive": "massive_intent",
    "biasframes_intent": "biasframes_intent",
    # topic (11)
    "agnews": "ag_news",
    "yahootopics": "yahootopics",
    "trueteacher": "trueteacher",
    "capsotu": "capsotu",
    "manifesto": "manifesto",
    "biasframes_offensive": "biasframes_offensive",
    "biasframes_sex": "biasframes_sex",
    "wikitoxic_toxicaggregated": "wikitoxic_toxicaggregated",
    "wikitoxic_obscene": "wikitoxic_obscene",
    "wikitoxic_threat": "wikitoxic_threat",
    "wikitoxic_insult": "wikitoxic_insult",
}


class JeffyBTZSCModel(BaseModel):
    model_type = "custom"
    model_name = "jeffy (BGE-large + logreg)"

    def __init__(self, device="cpu"):
        from jeffy.engine import Engine
        self._engine = Engine(pack_dir=PACK_DIR, device=device)
        self._engine.load()

        with open(MAPPINGS_PATH) as f:
            cached = json.load(f)

        # Build lookup: tuple(hypothesis labels) -> (jeffy_task_id, {jeffy_key: btzsc_pos})
        self._mappings = {}
        for btzsc_name, info in cached.items():
            jeffy_task = BTZSC_TO_JEFFY.get(btzsc_name)
            if not jeffy_task or jeffy_task not in self._engine.capabilities:
                continue

            labels_key = tuple(info["labels"])
            class_to_pos = info["class_to_pos"]

            manifest_path = f"{PACK_DIR}/{jeffy_task}/manifest.json"
            with open(manifest_path) as f:
                manifest = json.load(f)

            jeffy_to_btzsc = {}
            for key, label_name in manifest["labels"].items():
                norm_j = label_name.lower().replace(" ", "_").replace("/", "_")
                for btzsc_class, pos in class_to_pos.items():
                    norm_b = btzsc_class.lower().replace(" ", "_").replace("/", "_")
                    if norm_j == norm_b:
                        jeffy_to_btzsc[key] = pos
                        break
                else:
                    for btzsc_class, pos in class_to_pos.items():
                        norm_b = btzsc_class.lower().replace(" ", "_").replace("/", "_")
                        if norm_j in norm_b or norm_b in norm_j:
                            jeffy_to_btzsc[key] = pos
                            break

            self._mappings[labels_key] = (jeffy_task, jeffy_to_btzsc)
            print(f"  Ready: {btzsc_name}->{jeffy_task}: {len(jeffy_to_btzsc)}/{manifest['n_classes']} labels mapped")

    def _get_mapping(self, labels):
        key = tuple(labels)
        if key not in self._mappings:
            raise KeyError(f"No Jeffy head for dataset with {len(labels)} labels")
        return self._mappings[key]

    def predict(self, texts, labels, batch_size=32):
        jeffy_task, label_map = self._get_mapping(labels)
        scores = self._compute_scores(texts, jeffy_task, label_map, len(labels), batch_size)
        return scores.argmax(axis=1)

    def predict_scores(self, texts, labels, batch_size=32):
        jeffy_task, label_map = self._get_mapping(labels)
        return self._compute_scores(texts, jeffy_task, label_map, len(labels), batch_size)

    def _compute_scores(self, texts, jeffy_task, label_map, n_labels, batch_size):
        clf = self._engine._classifiers[jeffy_task]
        scaler = self._engine._scalers[jeffy_task]
        classes = clf.classes_

        embeddings = self._engine.encode(texts, batch_size=batch_size)
        X = scaler.transform(embeddings)
        all_probs = clf.predict_proba(X)

        scores = np.zeros((len(texts), n_labels), dtype=np.float64)
        for i, probs_row in enumerate(all_probs):
            for cls, prob in zip(classes, probs_row):
                key = str(cls)
                if key in label_map:
                    scores[i, label_map[key]] = prob

        return scores
