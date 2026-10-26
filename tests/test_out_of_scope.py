"""`out_of_scope` must never claim an answer it did not compute.

The bug these lock in: `out_of_scope` was only evaluated when the caller passed a
`threshold`, but the field was initialised to False, so every request that omitted
the threshold came back `"out_of_scope": false` for any input at all. An external
tester fed four clearly unrelated prompts to a 27-class head, read four `false`
values, and reasonably reported out-of-scope detection as broken. False was a wrong
answer where None was the honest one.

Run:
    python -m pytest tests/test_out_of_scope.py -v
"""

import pytest
from fastapi.testclient import TestClient

from jeffy.server import app, get_engine

client = TestClient(app)

TASK = "banking77"
OFF_TOPIC = "my cat keeps knocking plants off the windowsill"


@pytest.fixture(scope="module", autouse=True)
def _load_engine():
    get_engine()


def _predict(text, **body):
    r = client.post("/v1/predict", json={"text": text, "task": TASK, **body})
    assert r.status_code == 200, r.text
    return r.json()


class TestNoThresholdIsNotFalse:
    def test_omitting_threshold_reports_null_not_false(self):
        """The whole point: no threshold anywhere means nothing was checked."""
        cap = client.get(f"/v1/capabilities/{TASK}").json()
        d = _predict(OFF_TOPIC)
        if cap.get("oos_threshold") is None:
            assert d["out_of_scope"] is None, (
                "uncalibrated pack with no caller threshold must report None, "
                "not False, or callers read it as working detection"
            )
            assert d["oos_threshold"] is None
        else:
            # calibrated pack: a verdict is fine, but it must be a real one
            assert isinstance(d["out_of_scope"], bool)
            assert d["oos_threshold"] == cap["oos_threshold"]

    def test_null_still_carries_a_usable_label(self):
        """Reporting None must not blank the prediction itself."""
        d = _predict(OFF_TOPIC)
        assert d["label"]
        assert d["label"] != "out_of_scope"
        assert d["label_id"]
        assert 0.0 <= d["confidence"] <= 1.0


class TestThresholdStillWorks:
    def test_high_threshold_flags_off_topic_input(self):
        d = _predict(OFF_TOPIC, threshold=0.9)
        assert d["out_of_scope"] is True
        assert d["label"] == "out_of_scope"
        assert d["label_id"] == ""
        assert d["oos_threshold"] == 0.9

    def test_threshold_zero_is_honoured_not_treated_as_absent(self):
        """0.0 is falsy; it must not be mistaken for 'no threshold given'."""
        d = _predict(OFF_TOPIC, threshold=0.0)
        assert d["out_of_scope"] is False, "0.0 means 'never abstain', not 'unchecked'"
        assert d["oos_threshold"] == 0.0

    def test_confident_in_scope_input_survives_a_threshold(self):
        d = _predict("I was charged twice for the same transaction", threshold=0.5)
        assert d["out_of_scope"] is False
        assert d["label"] != "out_of_scope"


class TestCalibratedPackDefault:
    def test_pack_threshold_applies_without_a_caller_threshold(self):
        """A calibrated pack gives the safe behaviour to callers who never read the docs."""
        engine = get_engine()
        cap = engine.capabilities[TASK]
        original = cap.oos_threshold
        try:
            cap.oos_threshold = 0.95
            d = _predict(OFF_TOPIC)
            assert d["out_of_scope"] is True
            assert d["oos_threshold"] == 0.95
        finally:
            cap.oos_threshold = original

    def test_caller_threshold_overrides_the_pack_default(self):
        engine = get_engine()
        cap = engine.capabilities[TASK]
        original = cap.oos_threshold
        try:
            cap.oos_threshold = 0.95
            d = _predict(OFF_TOPIC, threshold=0.0)
            assert d["out_of_scope"] is False
            assert d["oos_threshold"] == 0.0
        finally:
            cap.oos_threshold = original


class TestCapabilitiesAdvertisesIt:
    def test_threshold_is_discoverable_when_set(self):
        engine = get_engine()
        cap = engine.capabilities[TASK]
        original = cap.oos_threshold
        try:
            cap.oos_threshold = 0.42
            body = client.get("/v1/capabilities").json()
            row = next(c for c in body["capabilities"] if c["task_id"] == TASK)
            assert row["oos_threshold"] == 0.42
        finally:
            cap.oos_threshold = original
