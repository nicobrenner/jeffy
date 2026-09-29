"""API tests for the pretrained decision server.

Covers: identical labels across heads, reversed yes/no, arbitrary option IDs,
unsupported subsets, explicit capability binding, and label mapping.

Run:
    python -m pytest tests/test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient

from jeffy.server import app, get_engine

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def _load_engine():
    """Ensure engine is loaded before tests."""
    get_engine()


# --- /v1/predict tests ---

class TestPredict:
    def test_basic_prediction(self):
        r = client.post("/v1/predict", json={"text": "I was charged twice", "task": "banking77"})
        assert r.status_code == 200
        data = r.json()
        assert "label" in data
        assert "probabilities" in data
        assert "capability" in data
        assert data["capability"]["task_id"] == "banking77"

    def test_unknown_task_returns_400(self):
        r = client.post("/v1/predict", json={"text": "hello", "task": "nonexistent"})
        assert r.status_code == 400

    def test_identical_labels_different_heads(self):
        """sst2 and imdb both have labels 'positive'/'negative' but are different heads."""
        r1 = client.post("/v1/predict", json={"text": "Great movie!", "task": "sst2"})
        r2 = client.post("/v1/predict", json={"text": "Great movie!", "task": "imdb"})
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Both should return sentiment labels
        assert r1.json()["label"] in ("positive", "negative")
        assert r2.json()["label"] in ("positive", "negative")
        # But they come from different heads
        assert r1.json()["capability"]["task_id"] == "sst2"
        assert r2.json()["capability"]["task_id"] == "imdb"

    def test_binary_heads_distinct(self):
        """sms_spam and tweet_eval_offensive are both 2-class but different tasks."""
        r1 = client.post("/v1/predict", json={
            "text": "WINNER! Call now to claim your prize!", "task": "sms_spam"})
        r2 = client.post("/v1/predict", json={
            "text": "WINNER! Call now to claim your prize!", "task": "tweet_eval_offensive"})
        assert r1.status_code == 200
        assert r2.status_code == 200
        # Different label sets
        sms_labels = set(r1.json()["probabilities"].keys())
        off_labels = set(r2.json()["probabilities"].keys())
        assert "spam" in sms_labels or "ham" in sms_labels
        assert "offensive" in off_labels or "not_offensive" in off_labels

    def test_probability_sums_to_one(self):
        r = client.post("/v1/predict", json={"text": "test", "task": "ag_news"})
        probs = r.json()["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_label_id_matches_label(self):
        """Verify label_id maps to label via the capability's labels."""
        r = client.post("/v1/predict", json={"text": "test", "task": "emotion"})
        data = r.json()
        assert data["label"] == max(data["probabilities"], key=data["probabilities"].get)


# --- /v1/systemone tests ---

class TestSystemOne:
    def test_requires_capability_field(self):
        """Must specify which head to use."""
        r = client.post("/v1/systemone", json={
            "state": "hello",
            "questions": {"q": {"type": "choice", "criteria": {"a": None, "b": None}}},
        })
        assert r.status_code == 422  # missing required field

    def test_unknown_capability_returns_400(self):
        r = client.post("/v1/systemone", json={
            "state": "hello",
            "capability": "nonexistent",
            "questions": {"q": {"type": "choice", "criteria": {"a": None}}},
        })
        assert r.status_code == 400

    def test_choice_exact_labels(self):
        """Choice with criteria matching head labels exactly."""
        r = client.post("/v1/systemone", json={
            "state": "The stock market crashed today",
            "capability": "ag_news",
            "questions": {
                "topic": {
                    "type": "choice",
                    "criteria": {
                        "World": None, "Sports": None,
                        "Business": None, "Sci/Tech": None,
                    },
                },
            },
        })
        assert r.status_code == 200
        data = r.json()
        assert data["answers"]["topic"]["type"] == "choice"
        assert data["answers"]["topic"]["choice"] in ("World", "Sports", "Business", "Sci/Tech")

    def test_choice_mismatched_labels_rejected(self):
        """Criteria that don't match head labels should be rejected without label_map."""
        r = client.post("/v1/systemone", json={
            "state": "test",
            "capability": "ag_news",
            "questions": {
                "q": {
                    "type": "choice",
                    "criteria": {"politics": None, "entertainment": None},
                },
            },
        })
        assert r.status_code == 200
        assert r.json()["answers"]["q"]["type"] == "error"

    def test_choice_subset_rejected(self):
        """Subset of head labels should be rejected (no implicit subsetting)."""
        r = client.post("/v1/systemone", json={
            "state": "test",
            "capability": "ag_news",
            "questions": {
                "q": {
                    "type": "choice",
                    "criteria": {"World": None, "Sports": None},  # only 2 of 4
                },
            },
        })
        assert r.status_code == 200
        assert r.json()["answers"]["q"]["type"] == "error"

    def test_arbitrary_option_ids_via_label_map(self):
        """Caller uses arbitrary IDs; label_map translates to head labels."""
        r = client.post("/v1/systemone", json={
            "state": "The team won the championship",
            "capability": "ag_news",
            "label_map": {
                "opt_a": "World",
                "opt_b": "Sports",
                "opt_c": "Business",
                "opt_d": "Sci/Tech",
            },
            "questions": {
                "q": {
                    "type": "choice",
                    "criteria": {
                        "opt_a": "World news",
                        "opt_b": "Sports news",
                        "opt_c": "Business news",
                        "opt_d": "Science and technology",
                    },
                },
            },
        })
        assert r.status_code == 200
        ans = r.json()["answers"]["q"]
        assert ans["type"] == "choice"
        assert ans["choice"] in ("opt_a", "opt_b", "opt_c", "opt_d")
        assert set(ans["probabilities"].keys()) == {"opt_a", "opt_b", "opt_c", "opt_d"}

    def test_noul_requires_criteria(self):
        """noul must specify true/false mapping via criteria."""
        r = client.post("/v1/systemone", json={
            "state": "This is spam",
            "capability": "sms_spam",
            "questions": {"q": {"type": "noul"}},
        })
        assert r.status_code == 200
        assert r.json()["answers"]["q"]["type"] == "error"

    def test_noul_with_criteria(self):
        """noul with explicit true/false criteria."""
        r = client.post("/v1/systemone", json={
            "state": "WINNER! Call now!",
            "capability": "sms_spam",
            "questions": {
                "is_spam": {
                    "type": "noul",
                    "criteria": {"true": "spam", "false": "ham"},
                },
            },
        })
        assert r.status_code == 200
        ans = r.json()["answers"]["is_spam"]
        assert ans["type"] == "noul"
        assert 0.0 <= ans["noul"] <= 1.0

    def test_noul_reversed_polarity(self):
        """Reversed yes/no: 'true' = ham (not spam)."""
        r = client.post("/v1/systemone", json={
            "state": "URGENT! You have won a 1000 dollar prize. Call 09061234567 to claim NOW!",
            "capability": "sms_spam",
            "questions": {
                "is_legit": {
                    "type": "noul",
                    "criteria": {"true": "ham", "false": "spam"},
                },
            },
        })
        assert r.status_code == 200
        ans = r.json()["answers"]["is_legit"]
        assert ans["type"] == "noul"
        # P(ham) should be low for spam text
        assert ans["noul"] < 0.5

    def test_noul_on_multiclass_head_rejected(self):
        """noul on a head with >2 classes should be rejected."""
        r = client.post("/v1/systemone", json={
            "state": "test",
            "capability": "ag_news",
            "questions": {"q": {"type": "noul", "criteria": {"true": "World", "false": "Sports"}}},
        })
        assert r.status_code == 200
        assert r.json()["answers"]["q"]["type"] == "error"

    def test_score_type_rejected(self):
        r = client.post("/v1/systemone", json={
            "state": "test",
            "capability": "sst2",
            "questions": {"q": {"type": "score", "criteria": ["bad", "ok", "good"]}},
        })
        assert r.status_code == 200
        assert r.json()["answers"]["q"]["type"] == "error"


# --- /health and /v1/capabilities ---

class TestHealth:
    def test_health(self):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ready"
        assert r.json()["capabilities"] >= 1

    def test_capabilities_list(self):
        r = client.get("/v1/capabilities")
        assert r.status_code == 200
        caps = r.json()["capabilities"]
        assert len(caps) >= 1
        for cap in caps:
            assert "task_id" in cap
            assert "labels" in cap
            assert "test_accuracy" in cap

    def test_capability_detail(self):
        r = client.get("/v1/capabilities/banking77")
        assert r.status_code == 200
        data = r.json()
        assert data["n_classes"] == 77
        assert "example_request" in data
