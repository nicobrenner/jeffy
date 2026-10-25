"""Tests for new training features: encoder selection, binary flags, C tuning, encoder size.

Run:
    python -m pytest tests/test_train_features.py -v
"""

import json
import pytest
from fastapi.testclient import TestClient

from jeffy.server import app, get_engine
from jeffy.train import train_classifier, train_binary_flags
from jeffy.catalog import get_encoder_size_mb, ENCODER, ENCODER_SIZES_MB
from jeffy.engine import Engine
from jeffy.model_pack import load_artifact

client = TestClient(app)

SENTIMENT_DATA = {
    "texts": [
        "this is great", "love it", "amazing product", "best ever", "wonderful experience",
        "terrible service", "awful product", "worst purchase", "hate it", "complete waste",
    ],
    "labels": ["positive", "positive", "positive", "positive", "positive",
               "negative", "negative", "negative", "negative", "negative"],
}

FLAG_DATA = {
    "texts": [
        "I will sue you if this isn't resolved",
        "My elderly mother is struggling with the service",
        "The staff member was incredibly rude to me",
        "Just checking on my order status",
        "When will my package arrive?",
        "Your employee named John was terrible",
        "I'm going to contact my lawyer about this",
        "My disabled child needs this urgently",
        "Everything is fine, just a question",
        "Simple billing inquiry",
    ],
    "flags": {
        "legal_threat": ["yes", "no", "no", "no", "no", "no", "yes", "no", "no", "no"],
        "vulnerable_person": ["no", "yes", "no", "no", "no", "no", "no", "yes", "no", "no"],
        "staff_blamed": ["no", "no", "yes", "no", "no", "yes", "no", "no", "no", "no"],
    },
}


@pytest.fixture(scope="module", autouse=True)
def _load_engine():
    get_engine()


class TestEncoderSelection:
    """Feature 1: encoder parameter in train_classifier and /v1/train."""

    def test_default_encoder(self):
        tc = train_classifier(
            texts=SENTIMENT_DATA["texts"],
            labels=SENTIMENT_DATA["labels"],
            task_id="test_default_enc",
            test_size=0, cv_folds=0, verbose=False,
        )
        assert tc.encoder_name == ENCODER

    def test_custom_encoder(self):
        tc = train_classifier(
            texts=SENTIMENT_DATA["texts"],
            labels=SENTIMENT_DATA["labels"],
            task_id="test_custom_enc",
            encoder="all-MiniLM-L6-v2",
            test_size=0, cv_folds=0, verbose=False,
        )
        assert tc.encoder_name == "all-MiniLM-L6-v2"
        result = tc.predict("this is amazing")
        assert result["label"] in ("positive", "negative")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_custom_encoder_dimension(self):
        tc = train_classifier(
            texts=SENTIMENT_DATA["texts"],
            labels=SENTIMENT_DATA["labels"],
            task_id="test_enc_dim",
            encoder="all-MiniLM-L6-v2",
            test_size=0, cv_folds=0, verbose=False,
        )
        assert tc.scaler.mean_.shape[0] == 384

    def test_api_encoder_param(self):
        r = client.post("/v1/train", json={
            "task_id": "test_api_enc",
            "examples": {
                "positive": SENTIMENT_DATA["texts"][:5],
                "negative": SENTIMENT_DATA["texts"][5:],
            },
            "encoder": "all-MiniLM-L6-v2",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "trained"
        assert data["encoder"] == "all-MiniLM-L6-v2"

        r2 = client.post("/v1/predict", json={"text": "this is wonderful", "task": "test_api_enc"})
        assert r2.status_code == 200
        assert r2.json()["label"] in ("positive", "negative")

    def test_api_default_encoder(self):
        r = client.post("/v1/train", json={
            "task_id": "test_api_default_enc",
            "examples": {
                "positive": SENTIMENT_DATA["texts"][:5],
                "negative": SENTIMENT_DATA["texts"][5:],
            },
        })
        assert r.status_code == 200
        assert r.json()["encoder"] == "default"


class TestBinaryFlags:
    """Feature 2: per-flag binary classifiers."""

    def test_train_binary_flags_python(self):
        results = train_binary_flags(
            texts=FLAG_DATA["texts"],
            flags=FLAG_DATA["flags"],
            task_id="test_flags",
            test_size=0, verbose=False,
        )
        assert set(results.keys()) == {"legal_threat", "vulnerable_person", "staff_blamed"}
        for flag_name, tc in results.items():
            assert tc.task_id == f"test_flags_flag_{flag_name}"
            assert set(tc.labels.values()) == {"yes", "no"}
            result = tc.predict("I will contact my lawyer immediately")
            assert result["label"] in ("yes", "no")
            assert 0.0 <= result["confidence"] <= 1.0

    def test_binary_flags_class_weighted(self):
        results = train_binary_flags(
            texts=FLAG_DATA["texts"],
            flags=FLAG_DATA["flags"],
            task_id="test_balanced",
            test_size=0, verbose=False,
        )
        for tc in results.values():
            assert hasattr(tc.clf, 'class_weight')

    def test_binary_flags_invalid_labels(self):
        with pytest.raises(ValueError, match="invalid labels"):
            train_binary_flags(
                texts=["a", "b"],
                flags={"bad": ["yes", "maybe"]},
                task_id="test_bad",
                verbose=False,
            )

    def test_binary_flags_length_mismatch(self):
        with pytest.raises(ValueError, match="labels but"):
            train_binary_flags(
                texts=["a", "b", "c"],
                flags={"f": ["yes", "no"]},
                task_id="test_mismatch",
                verbose=False,
            )

    def test_api_binary_flags(self):
        r = client.post("/v1/train", json={
            "task_id": "test_api_flags",
            "mode": "binary-flags",
            "texts": FLAG_DATA["texts"],
            "flags": FLAG_DATA["flags"],
        })
        assert r.status_code == 200
        data = r.json()
        assert data["mode"] == "binary-flags"
        assert data["n_flags"] == 3
        assert len(data["flag_task_ids"]) == 3
        assert "test_api_flags_flag_legal_threat" in data["flag_task_ids"]

        r2 = client.post("/v1/predict", json={
            "text": "I'm going to sue you",
            "task": "test_api_flags_flag_legal_threat",
        })
        assert r2.status_code == 200
        assert r2.json()["label"] in ("yes", "no")

    def test_api_binary_flags_missing_texts(self):
        r = client.post("/v1/train", json={
            "task_id": "test_missing",
            "mode": "binary-flags",
            "flags": {"f": ["yes", "no"]},
        })
        assert r.status_code == 400

    def test_api_binary_flags_with_encoder(self):
        r = client.post("/v1/train", json={
            "task_id": "test_flags_enc",
            "mode": "binary-flags",
            "texts": FLAG_DATA["texts"],
            "flags": {"legal_threat": FLAG_DATA["flags"]["legal_threat"]},
            "encoder": "all-MiniLM-L6-v2",
        })
        assert r.status_code == 200
        assert r.json()["encoder"] == "all-MiniLM-L6-v2"


class TestTuneC:
    """Feature 3: per-head C tuning via grid search."""

    def test_tune_c_python(self):
        tc = train_classifier(
            texts=SENTIMENT_DATA["texts"],
            labels=SENTIMENT_DATA["labels"],
            task_id="test_tune_c",
            tune_C=True,
            test_size=0, cv_folds=0, verbose=False,
        )
        assert tc.clf.C in [0.001, 0.01, 0.1, 1.0, 10.0]
        result = tc.predict("this is terrible")
        assert result["label"] in ("positive", "negative")

    def test_tune_c_overrides_explicit_c(self):
        tc = train_classifier(
            texts=SENTIMENT_DATA["texts"],
            labels=SENTIMENT_DATA["labels"],
            task_id="test_tune_override",
            C=999.0,
            tune_C=True,
            test_size=0, cv_folds=0, verbose=False,
        )
        assert tc.clf.C in [0.001, 0.01, 0.1, 1.0, 10.0]

    def test_api_tune_c(self):
        r = client.post("/v1/train", json={
            "task_id": "test_api_tune",
            "examples": {
                "positive": SENTIMENT_DATA["texts"][:5],
                "negative": SENTIMENT_DATA["texts"][5:],
            },
            "tune_C": True,
        })
        assert r.status_code == 200
        assert r.json()["tune_C"] is True


class TestEncoderSize:
    """Feature 4: encoder size in capability metadata."""

    def test_known_encoder_sizes(self):
        # On-disk fp32 sizes, not parameter counts (bge-large measured 1277 MB in HF cache)
        assert get_encoder_size_mb(ENCODER) == 1340.0
        assert get_encoder_size_mb("all-MiniLM-L6-v2") == 90.0
        assert get_encoder_size_mb("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2") == 471.0
        assert get_encoder_size_mb("intfloat/multilingual-e5-large") == 2240.0

    def test_unknown_encoder_returns_none(self):
        assert get_encoder_size_mb("nonexistent/model") is None

    def test_capabilities_include_encoder_size(self):
        r = client.get("/v1/capabilities")
        assert r.status_code == 200
        caps = r.json()["capabilities"]
        bge_caps = [c for c in caps if c["encoder"] == ENCODER]
        assert len(bge_caps) > 0
        for cap in bge_caps:
            assert cap.get("encoder_size_mb") == 1340.0

    def test_capability_detail_includes_encoder_size(self):
        r = client.get("/v1/capabilities/banking77")
        assert r.status_code == 200
        assert r.json().get("encoder_size_mb") == 1340.0

    def test_trained_classifier_has_encoder_size(self):
        r = client.post("/v1/train", json={
            "task_id": "test_size_check",
            "examples": {
                "positive": SENTIMENT_DATA["texts"][:5],
                "negative": SENTIMENT_DATA["texts"][5:],
            },
            "encoder": "all-MiniLM-L6-v2",
        })
        assert r.status_code == 200

        r2 = client.get("/v1/capabilities/test_size_check")
        assert r2.status_code == 200
        assert r2.json().get("encoder_size_mb") == 90.0


@pytest.fixture(scope="module")
def local_hf_checkpoint(tmp_path_factory):
    """A plain transformers checkpoint dir (no sentence-transformers config),
    simulating a fine-tuned BERT-style encoder."""
    from transformers import AutoModel, AutoTokenizer

    model_dir = tmp_path_factory.mktemp("models") / "finetuned-minilm"
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    return str(model_dir)


class TestLocalEncoderPaths:
    """Feature: encoder= accepts a local checkpoint path or any HF model."""

    def _train(self, encoder_path, task_id):
        return train_classifier(
            texts=SENTIMENT_DATA["texts"],
            labels=SENTIMENT_DATA["labels"],
            task_id=task_id,
            encoder=encoder_path,
            test_size=0, cv_folds=0, verbose=False,
        )

    def test_train_with_local_path(self, local_hf_checkpoint):
        tc = self._train(local_hf_checkpoint, "test_local_enc")
        assert tc.encoder_name == local_hf_checkpoint
        result = tc.predict("this is wonderful")
        assert result["label"] in ("positive", "negative")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_binary_flags_with_local_path(self, local_hf_checkpoint):
        results = train_binary_flags(
            texts=FLAG_DATA["texts"],
            flags={"legal_threat": FLAG_DATA["flags"]["legal_threat"]},
            task_id="test_local_flags",
            encoder=local_hf_checkpoint,
            test_size=0, verbose=False,
        )
        r = results["legal_threat"].predict("I will contact my lawyer")
        assert r["label"] in ("yes", "no")

    def test_save_and_register_roundtrip(self, local_hf_checkpoint, tmp_path):
        tc = self._train(local_hf_checkpoint, "test_local_roundtrip")
        tc.save(str(tmp_path))
        clf, scaler, manifest = load_artifact(tmp_path / "test_local_roundtrip")
        assert manifest.encoder == local_hf_checkpoint
        engine = get_engine()
        engine.register_trained("test_local_roundtrip", clf, scaler, manifest)
        result = engine.predict("test_local_roundtrip", "absolutely great")
        assert result["label"] in ("positive", "negative")

    def test_engine_load_does_not_quarantine_local_encoder(self, local_hf_checkpoint, tmp_path):
        tc = self._train(local_hf_checkpoint, "test_local_pack")
        tc.save(str(tmp_path))
        engine = Engine(pack_dir=str(tmp_path))
        engine.load()
        assert "test_local_pack" in engine.capabilities
        result = engine.predict("test_local_pack", "truly awful experience")
        assert result["label"] in ("positive", "negative")

    def test_nonexistent_path_raises(self):
        with pytest.raises(ValueError, match="Encoder path not found"):
            self._train("./does-not-exist-anywhere", "test_bad_path")

    def test_local_dir_encoder_size(self, local_hf_checkpoint):
        size = get_encoder_size_mb(local_hf_checkpoint)
        assert isinstance(size, float)
        assert size > 0
