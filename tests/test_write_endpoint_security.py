"""Security tests for the /v1/train and /v1/feedback write endpoints.

These endpoints mutate server state and write files, so they must be
disabled unless explicitly enabled, and must never accept a task_id
that can escape the output directory.

Run:
    JEFFY_BACKEND=torch python -m pytest tests/test_write_endpoint_security.py -v
"""

import os

import pytest
from fastapi.testclient import TestClient

from jeffy.server import app, get_engine

client = TestClient(app, raise_server_exceptions=False)

EXAMPLES = {
    "positive": ["this is great", "love it", "amazing", "best ever", "wonderful"],
    "negative": ["terrible", "awful", "worst", "hate it", "complete waste"],
}

TRAVERSAL_IDS = [
    "../evil",
    "../../etc/cron.d/evil",
    "foo/../../bar",
    "a/b",
    "/absolute",
    "..",
    ".",
    "with space",
    "semi;colon",
    "null\x00byte",
    "",
    "x" * 65,
]


@pytest.fixture(scope="module", autouse=True)
def _load_engine():
    get_engine()


@pytest.fixture
def training_enabled(monkeypatch, tmp_path):
    monkeypatch.setenv("JEFFY_ENABLE_TRAINING", "1")
    monkeypatch.setenv("JEFFY_TRAINED_DIR", str(tmp_path / "trained"))
    monkeypatch.setenv("JEFFY_FEEDBACK_DIR", str(tmp_path / "feedback"))


class TestDisabledByDefault:
    """Without JEFFY_ENABLE_TRAINING the write endpoints must look absent."""

    def test_train_404_by_default(self, monkeypatch):
        monkeypatch.delenv("JEFFY_ENABLE_TRAINING", raising=False)
        r = client.post("/v1/train", json={"task_id": "x", "examples": EXAMPLES})
        assert r.status_code == 404

    def test_feedback_404_by_default(self, monkeypatch):
        monkeypatch.delenv("JEFFY_ENABLE_TRAINING", raising=False)
        r = client.post("/v1/feedback", json={
            "task_id": "banking77", "text": "hi", "correct_label": "card_arrival"})
        assert r.status_code == 404

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", "  "])
    def test_falsy_values_keep_it_disabled(self, monkeypatch, value):
        monkeypatch.setenv("JEFFY_ENABLE_TRAINING", value)
        r = client.post("/v1/train", json={"task_id": "x", "examples": EXAMPLES})
        assert r.status_code == 404

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values_enable_it(self, monkeypatch, tmp_path, value):
        monkeypatch.setenv("JEFFY_ENABLE_TRAINING", value)
        monkeypatch.setenv("JEFFY_TRAINED_DIR", str(tmp_path / "trained"))
        # Reaches validation rather than the 404 guard
        r = client.post("/v1/train", json={"task_id": "../bad", "examples": EXAMPLES})
        assert r.status_code == 400

    def test_read_endpoints_unaffected(self, monkeypatch):
        monkeypatch.delenv("JEFFY_ENABLE_TRAINING", raising=False)
        assert client.get("/health").status_code == 200
        assert client.get("/v1/capabilities").status_code == 200
        r = client.post("/v1/predict", json={"text": "my card was stolen", "task": "banking77"})
        assert r.status_code == 200


class TestTaskIdValidation:
    """task_id becomes a directory name, so traversal must be rejected."""

    @pytest.mark.parametrize("task_id", TRAVERSAL_IDS)
    def test_train_rejects_bad_task_id(self, training_enabled, task_id):
        r = client.post("/v1/train", json={"task_id": task_id, "examples": EXAMPLES})
        assert r.status_code in (400, 422), f"{task_id!r} was not rejected"

    @pytest.mark.parametrize("task_id", TRAVERSAL_IDS)
    def test_feedback_rejects_bad_task_id(self, training_enabled, task_id):
        r = client.post("/v1/feedback", json={
            "task_id": task_id, "text": "hi", "correct_label": "x"})
        assert r.status_code in (400, 422), f"{task_id!r} was not rejected"

    @pytest.mark.parametrize("task_id", ["ok", "ok_name", "ok-name", "OK123", "a" * 64])
    def test_valid_task_ids_pass_validation(self, training_enabled, task_id):
        r = client.post("/v1/train", json={"task_id": task_id, "examples": EXAMPLES})
        assert r.status_code == 200, r.text

    def test_binary_flag_names_validated(self, training_enabled):
        r = client.post("/v1/train", json={
            "task_id": "flagtest",
            "mode": "binary-flags",
            "texts": ["a", "b"],
            "flags": {"../escape": ["yes", "no"]},
        })
        assert r.status_code == 400

    def test_traversal_writes_nothing_outside_target(self, training_enabled, tmp_path):
        """The canonical exploit: task_id climbing out of the trained dir."""
        canary = tmp_path / "pwned"
        r = client.post("/v1/train", json={
            "task_id": f"../../../../../../{canary.relative_to(canary.anchor)}",
            "examples": EXAMPLES,
        })
        assert r.status_code in (400, 422)
        assert not canary.exists()


class TestTrainedDirIsolation:
    """Trained packs must never land in the installed package directory."""

    def test_writes_go_to_configured_dir(self, training_enabled, tmp_path):
        r = client.post("/v1/train", json={"task_id": "isolation_check", "examples": EXAMPLES})
        assert r.status_code == 200, r.text
        assert (tmp_path / "trained" / "isolation_check").exists()

    def test_nothing_written_into_package_dir(self, training_enabled, tmp_path):
        import jeffy
        pkg = os.path.dirname(jeffy.__file__)
        before = set(os.listdir(pkg))
        r = client.post("/v1/train", json={"task_id": "pkg_purity_check", "examples": EXAMPLES})
        assert r.status_code == 200, r.text
        assert set(os.listdir(pkg)) == before, "train endpoint polluted the installed package"
