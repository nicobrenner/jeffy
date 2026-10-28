"""The playground teach box and the /v1/train caps behind it.

- flag off: the page carries `const TEACH=false` (the box never renders) and /v1/train is 404, exactly as before
- flag on: a four-row, two-label head trains through /v1/train and the next prediction runs on it
- caps: too many rows or too many bytes -> 413; too many trains per client per hour -> 429
The flag-on test needs the encoder (same as tests/test_write_endpoint_security.py).
"""
import importlib
import os

import pytest
from fastapi.testclient import TestClient

import jeffy.server as server

ROWS = {"ok": ["the battery lasts all day", "arrived early and works"], "bad": ["screen cracked on day one", "stopped charging after a week"]}


def _client():
    return TestClient(server.app, raise_server_exceptions=False)


def test_flag_off_same_playground_and_train_404(monkeypatch):
    monkeypatch.delenv("JEFFY_ENABLE_TRAINING", raising=False)
    monkeypatch.delenv("JEFFY_PLAYGROUND_TEACH", raising=False)
    c = _client()
    html = c.get("/").text
    assert "const TEACH=false;" in html
    assert html.replace("const TEACH=false;", "const TEACH=__TEACH__;") == server.PLAYGROUND_HTML
    assert c.post("/v1/train", json={"task_id": "my_x", "examples": ROWS, "test_size": 0}).status_code == 404


def test_teach_flag_follows_training_and_override(monkeypatch):
    monkeypatch.setenv("JEFFY_ENABLE_TRAINING", "1"); monkeypatch.delenv("JEFFY_PLAYGROUND_TEACH", raising=False)
    assert server._teach_enabled() is True
    monkeypatch.setenv("JEFFY_PLAYGROUND_TEACH", "off"); assert server._teach_enabled() is False
    monkeypatch.delenv("JEFFY_ENABLE_TRAINING"); monkeypatch.setenv("JEFFY_PLAYGROUND_TEACH", "on"); assert server._teach_enabled() is True
    assert "const TEACH=true;" in _client().get("/").text


def test_caps_reject_oversize_and_rate(monkeypatch, tmp_path):
    monkeypatch.setenv("JEFFY_ENABLE_TRAINING", "1"); monkeypatch.setenv("JEFFY_TRAINED_DIR", str(tmp_path))
    monkeypatch.setattr(server, "_TRAIN_MAX_ROWS", 3)
    c = _client()
    r = c.post("/v1/train", json={"task_id": "my_rows", "examples": ROWS, "test_size": 0}); assert r.status_code == 413, r.text
    monkeypatch.setattr(server, "_TRAIN_MAX_ROWS", 200); monkeypatch.setattr(server, "_TRAIN_MAX_BYTES", 10)
    r = c.post("/v1/train", json={"task_id": "my_bytes", "examples": ROWS, "test_size": 0}); assert r.status_code == 413, r.text
    monkeypatch.setattr(server, "_TRAIN_MAX_BYTES", 64 * 1024); monkeypatch.setattr(server, "_TRAIN_PER_IP_HOUR", 0)
    server._train_calls.clear()
    r = c.post("/v1/train", json={"task_id": "my_rate", "examples": ROWS, "test_size": 0}); assert r.status_code == 429, r.text


@pytest.mark.skipif(os.environ.get("JEFFY_SKIP_ENCODER_TESTS") == "1", reason="needs the encoder")
def test_flag_on_trains_four_rows_and_predicts(monkeypatch, tmp_path):
    monkeypatch.setenv("JEFFY_ENABLE_TRAINING", "1"); monkeypatch.setenv("JEFFY_TRAINED_DIR", str(tmp_path))
    monkeypatch.setenv("JEFFY_BACKEND", "torch")   # as tests/test_write_endpoint_security.py documents
    server._train_calls.clear()
    c = _client()
    r = c.post("/v1/train", json={"task_id": "my_teach_test", "examples": ROWS, "test_size": 0, "description": "taught in the playground"})
    assert r.status_code == 200, r.text
    p = c.post("/v1/predict", json={"text": "battery is great and it arrived fast", "task": "my_teach_test"})
    assert p.status_code == 200, p.text
    d = p.json(); assert d["label"] in ("ok", "bad") and 0 <= d["confidence"] <= 1 and "margin" in d
    caps = c.get("/v1/capabilities").json()["capabilities"]
    assert any(x["task_id"] == "my_teach_test" for x in caps)
