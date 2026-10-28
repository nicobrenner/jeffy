"""/v1/version must answer without loading the engine (no encoder download), and say what is running:
package version, default encoder, head count from the pack on disk, git sha when in a checkout. Nothing about the host."""
from fastapi.testclient import TestClient

from jeffy.server import app


def test_version_answers_without_engine():
    client = TestClient(app)            # no `with`: lifespan is not run, the engine is never loaded
    r = client.get("/v1/version")
    assert r.status_code == 200
    data = r.json()
    assert data["package"] == "jeffy-classify"
    assert isinstance(data["version"], str) and data["version"]
    assert data["default_encoder"].startswith("BAAI/") or data["default_encoder"]
    assert data["heads"] is None or data["heads"] >= 1
    assert "git_sha" in data
    for forbidden in ("ip", "host", "hostname", "path", "pack_dir"):
        assert forbidden not in data
