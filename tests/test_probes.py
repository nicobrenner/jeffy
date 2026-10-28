"""Regression probes per head: phrasings with an expected label, or an expected abstain for texts that genuinely straddle
two classes of the taxonomy (issue #11). Runs only when JEFFY_RUN_PROBES=1 because it needs the encoder (~1.2 GB).

    JEFFY_RUN_PROBES=1 python -m pytest tests/test_probes.py -v
"""
import json
import os
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROBES = sorted((REPO / "data/probes").glob("*.jsonl"))

pytestmark = pytest.mark.skipif(os.environ.get("JEFFY_RUN_PROBES") != "1", reason="set JEFFY_RUN_PROBES=1 (needs the encoder)")


@pytest.fixture(scope="module")
def engine():
    from jeffy.engine import Engine
    e = Engine(str(REPO / "src/jeffy/pack"), device="cpu")
    e.load()
    return e


@pytest.mark.parametrize("probe_file", PROBES, ids=[p.stem for p in PROBES])
def test_probes(engine, probe_file):
    task_id = probe_file.stem
    probes = [json.loads(l) for l in probe_file.read_text().splitlines() if l.strip()]
    results = engine.predict_batch(task_id, [p["text"] for p in probes])
    failures = []
    for p, r in zip(probes, results):
        if p["expect"] == "abstain":
            ok = r["margin"] < 0.9 or r["confidence"] < 0.9
        else:
            ok = r["label"] == p["expect"]
        if not ok:
            failures.append(f"{p['text']!r}: expected {p['expect']}, got {r['label']} (conf {r['confidence']}, margin {r['margin']})")
    assert not failures, "\n".join(failures)
