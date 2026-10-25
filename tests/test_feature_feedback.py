"""Tests for feature-vector feedback on /v1/feedback.

/v1/predict already accepted a `features` vector for heads that run on
caller-supplied embeddings; /v1/feedback only accepted `text`, so there was no
way to correct such a head. These cover the vector path, the text path it must
not break, and the validation around mixing the two.
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from jeffy.server import app, _feedback_store

client = TestClient(app)

N_FEATURES = 12


@pytest.fixture(autouse=True)
def _enable_training(monkeypatch, tmp_path):
    """The write endpoints are disabled by default; these exercise them."""
    monkeypatch.setenv("JEFFY_ENABLE_TRAINING", "1")
    monkeypatch.setenv("JEFFY_TRAINED_DIR", str(tmp_path / "trained"))
    monkeypatch.setenv("JEFFY_FEEDBACK_DIR", str(tmp_path / "feedback"))
    monkeypatch.setenv("JEFFY_RETRAIN_THRESHOLD", "10")
    _feedback_store.clear()
    yield
    _feedback_store.clear()


def _register_feature_head(task_id, tmp_path):
    """Train and register a small feature-vector head to give feedback on."""
    from pathlib import Path

    from jeffy.model_pack import load_artifact
    from jeffy.server import get_engine
    from jeffy.train import train_feature_classifier

    rng = np.random.default_rng(0)
    X, y = [], []
    for i in range(40):
        label = "urgent" if i % 2 == 0 else "routine"
        vec = rng.normal(size=N_FEATURES)
        vec[0] += 2.0 if label == "urgent" else -2.0
        X.append(vec.tolist())
        y.append(label)

    tc = train_feature_classifier(
        features=X, labels=y, task_id=task_id, test_size=0, verbose=False)
    path = tc.save(str(tmp_path / "packs"))
    clf, scaler, manifest = load_artifact(Path(path))
    get_engine().register_trained(task_id, clf, scaler, manifest)
    return X, y


class TestFeatureFeedback:
    def test_feature_vector_feedback_is_accepted(self, tmp_path):
        _register_feature_head("fb_feat_accept", tmp_path)
        r = client.post("/v1/feedback", json={
            "task_id": "fb_feat_accept",
            "features": [0.1] * N_FEATURES,
            "correct_label": "urgent",
        })
        assert r.status_code == 200, r.text
        assert r.json()["status"] == "stored"
        assert r.json()["feedback_count"] == 1

    def test_wrong_feature_length_is_rejected(self, tmp_path):
        _register_feature_head("fb_feat_len", tmp_path)
        r = client.post("/v1/feedback", json={
            "task_id": "fb_feat_len",
            "features": [0.1] * (N_FEATURES + 3),
            "correct_label": "urgent",
        })
        assert r.status_code == 400
        assert "expects" in r.json()["detail"]

    def test_neither_text_nor_features_is_rejected(self, tmp_path):
        _register_feature_head("fb_feat_neither", tmp_path)
        r = client.post("/v1/feedback", json={
            "task_id": "fb_feat_neither",
            "correct_label": "urgent",
        })
        assert r.status_code == 400
        assert "either" in r.json()["detail"]

    def test_both_text_and_features_is_rejected(self, tmp_path):
        _register_feature_head("fb_feat_both", tmp_path)
        r = client.post("/v1/feedback", json={
            "task_id": "fb_feat_both",
            "text": "something",
            "features": [0.1] * N_FEATURES,
            "correct_label": "urgent",
        })
        assert r.status_code == 400
        assert "not both" in r.json()["detail"]

    def test_mixing_kinds_on_one_task_is_rejected(self, tmp_path):
        _register_feature_head("fb_feat_mix", tmp_path)
        ok = client.post("/v1/feedback", json={
            "task_id": "fb_feat_mix",
            "features": [0.1] * N_FEATURES,
            "correct_label": "urgent",
        })
        assert ok.status_code == 200

        clash = client.post("/v1/feedback", json={
            "task_id": "fb_feat_mix",
            "text": "this is urgent",
            "correct_label": "urgent",
        })
        assert clash.status_code == 409
        assert "cannot also take" in clash.json()["detail"]

    def test_invalid_label_still_rejected(self, tmp_path):
        _register_feature_head("fb_feat_label", tmp_path)
        r = client.post("/v1/feedback", json={
            "task_id": "fb_feat_label",
            "features": [0.1] * N_FEATURES,
            "correct_label": "not_a_label",
        })
        assert r.status_code == 400

    def test_feature_feedback_triggers_a_feature_retrain(self, tmp_path):
        """At the threshold the retrain must use the feature path, not try to
        encode text that is not there."""
        X, y = _register_feature_head("fb_feat_retrain", tmp_path)

        last = None
        for i in range(10):
            last = client.post("/v1/feedback", json={
                "task_id": "fb_feat_retrain",
                "features": X[i],
                "correct_label": y[i],
            })
            assert last.status_code == 200, last.text

        assert last.json()["status"] == "retrained", last.json()
        assert "10 feedback examples" in last.json()["message"]

        # The retrained head must still serve feature predictions.
        r = client.post("/v1/predict", json={
            "task": "fb_feat_retrain",
            "features": X[0],
        })
        assert r.status_code == 200
        assert r.json()["label"] in ("urgent", "routine")

    def test_feature_feedback_is_disabled_without_opt_in(self, monkeypatch, tmp_path):
        _register_feature_head("fb_feat_gate", tmp_path)
        monkeypatch.delenv("JEFFY_ENABLE_TRAINING", raising=False)
        r = client.post("/v1/feedback", json={
            "task_id": "fb_feat_gate",
            "features": [0.1] * N_FEATURES,
            "correct_label": "urgent",
        })
        assert r.status_code == 404

    def test_traversal_task_id_still_rejected(self):
        r = client.post("/v1/feedback", json={
            "task_id": "../../../etc/cron.d/x",
            "features": [0.1] * N_FEATURES,
            "correct_label": "urgent",
        })
        assert r.status_code == 400
