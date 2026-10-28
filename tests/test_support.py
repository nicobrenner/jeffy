"""Support radius: a per-head description of the training data in encoder space, and the input's distance percentile
returned with every prediction as `support_percentile` (see src/jeffy/support.py)."""
import os

import numpy as np
import pytest

from jeffy.support import build_support, support_of


def test_percentile_is_monotone_in_distance():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 16)); y = ["a"] * 150 + ["b"] * 150
    X[:150, 0] += 3.0; X[150:, 1] += 3.0
    sp = build_support(X, y)
    assert sp["centroids"].shape == (2, 16) and len(sp["quantiles"]) == 1001
    near = support_of(sp, sp["centroids"][0].astype(np.float32))          # the centroid itself
    typical = support_of(sp, X[0])
    far = support_of(sp, rng.normal(size=16) * 10 - 10)
    assert near["support_percentile"] <= typical["support_percentile"] <= far["support_percentile"]
    assert far["support_percentile"] >= 95


@pytest.mark.skipif(os.environ.get("JEFFY_SKIP_ENCODER_TESTS") == "1", reason="needs the encoder")
def test_trained_head_reports_support_percentile(monkeypatch, tmp_path):
    monkeypatch.setenv("JEFFY_ENABLE_TRAINING", "1"); monkeypatch.setenv("JEFFY_TRAINED_DIR", str(tmp_path)); monkeypatch.setenv("JEFFY_BACKEND", "torch")
    from fastapi.testclient import TestClient
    import jeffy.server as server
    server._train_calls.clear()
    c = TestClient(server.app, raise_server_exceptions=False)
    rows = {"ok": ["the battery lasts all day", "arrived early and works", "great value"], "bad": ["screen cracked on day one", "stopped charging after a week", "never again"]}
    r = c.post("/v1/train", headers={"X-Jeffy-Session": "sessionSUPPORT01"}, json={"task_id": "my_support", "examples": rows, "test_size": 0})
    assert r.status_code == 200, r.text
    assert (tmp_path / "my_support" / "support.npz").exists()
    p = c.post("/v1/predict", headers={"X-Jeffy-Session": "sessionSUPPORT01"}, json={"text": "the battery is great", "task": "my_support"}).json()
    assert "support_percentile" in p and 0 <= p["support_percentile"] <= 100
    q = c.post("/v1/predict", headers={"X-Jeffy-Session": "sessionSUPPORT01"}, json={"text": "quarterly tax filing deadline for partnerships in Delaware", "task": "my_support"}).json()
    assert q["support_percentile"] >= p["support_percentile"]


def test_teach_box_guards_against_our_own_example():
    import jeffy.server as server
    html = server.PLAYGROUND_HTML
    assert "teachIsOurExample" in html and "teach_default_text" in html
    assert "That was our sample text" in html
