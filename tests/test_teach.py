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
    H = {"X-Jeffy-Session": "sessionTEACHTEST"}
    r = c.post("/v1/train", headers=H, json={"task_id": "my_teach_test", "examples": ROWS, "test_size": 0, "description": "taught in the playground"})
    assert r.status_code == 200, r.text
    p = c.post("/v1/predict", headers=H, json={"text": "battery is great and it arrived fast", "task": "my_teach_test"})
    assert p.status_code == 200, p.text
    d = p.json(); assert d["label"] in ("ok", "bad") and 0 <= d["confidence"] <= 1 and "margin" in d
    caps = c.get("/v1/capabilities", headers=H).json()["capabilities"]
    assert any(x["task_id"] == "my_teach_test" for x in caps)


# --- session scoping, TTL and concurrency (round 9) ---

def _train_as(c, session, task_id):
    return c.post("/v1/train", headers={"X-Jeffy-Session": session}, json={"task_id": task_id, "examples": ROWS, "test_size": 0})


def test_my_prefix_requires_session(monkeypatch, tmp_path):
    monkeypatch.setenv("JEFFY_ENABLE_TRAINING", "1"); monkeypatch.setenv("JEFFY_TRAINED_DIR", str(tmp_path)); monkeypatch.setenv("JEFFY_BACKEND", "torch")
    server._train_calls.clear()
    r = _client().post("/v1/train", json={"task_id": "my_nosession", "examples": ROWS, "test_size": 0})
    assert r.status_code == 400 and "X-Jeffy-Session" in r.text


def test_concurrency_cap_returns_503(monkeypatch, tmp_path):
    monkeypatch.setenv("JEFFY_ENABLE_TRAINING", "1"); monkeypatch.setenv("JEFFY_TRAINED_DIR", str(tmp_path))
    server._train_calls.clear()
    assert server._train_lock.acquire(blocking=False)
    try:
        r = _train_as(_client(), "sessionAAAAAAAA", "my_busy")
        assert r.status_code == 503, r.text
    finally:
        server._train_lock.release()


@pytest.mark.skipif(os.environ.get("JEFFY_SKIP_ENCODER_TESTS") == "1", reason="needs the encoder")
def test_session_heads_are_private_and_expire(monkeypatch, tmp_path):
    monkeypatch.setenv("JEFFY_ENABLE_TRAINING", "1"); monkeypatch.setenv("JEFFY_TRAINED_DIR", str(tmp_path)); monkeypatch.setenv("JEFFY_BACKEND", "torch")
    server._train_calls.clear()
    c = _client()
    public_before = c.get("/v1/capabilities").json()
    r = _train_as(c, "sessionAAAAAAAA", "my_private"); assert r.status_code == 200, r.text
    # the catalog everyone sees is unchanged
    public_after = c.get("/v1/capabilities").json()
    assert public_after["count"] == public_before["count"]
    assert not any(x["task_id"].startswith("my_") for x in public_after["capabilities"])
    # the owning session sees it and can predict on it
    mine = c.get("/v1/capabilities", headers={"X-Jeffy-Session": "sessionAAAAAAAA"}).json()
    assert any(x["task_id"] == "my_private" for x in mine["capabilities"])
    ok = c.post("/v1/predict", headers={"X-Jeffy-Session": "sessionAAAAAAAA"}, json={"text": "arrived early, battery is great", "task": "my_private"})
    assert ok.status_code == 200, ok.text
    # another session, or no session, gets 404 on it
    for headers in ({"X-Jeffy-Session": "sessionBBBBBBBB"}, {}):
        assert c.post("/v1/predict", headers=headers, json={"text": "hello", "task": "my_private"}).status_code == 404
        assert c.get("/v1/capabilities/my_private", headers=headers).status_code == 404
    # TTL: once expired, the sweeper unloads it and deletes its files
    assert server._session_heads["my_private"]["dir"].exists()
    server._session_heads["my_private"]["expires"] = 0
    server._sweep_session_heads()
    assert "my_private" not in server._session_heads
    assert c.post("/v1/predict", headers={"X-Jeffy-Session": "sessionAAAAAAAA"}, json={"text": "hello", "task": "my_private"}).status_code in (400, 404)
    assert not (tmp_path / "my_private").exists()
