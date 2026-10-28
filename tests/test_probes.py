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
    rows = [json.loads(l) for l in probe_file.read_text().splitlines() if l.strip()]
    gate = next((r for r in rows if "_gate" in r), {"_gate": "confidence", "_min": 0.9})
    probes = [r for r in rows if "_gate" not in r]
    results = engine.predict_batch(task_id, [p["text"] for p in probes])
    failures, confident_wrong, covered = [], [], 0
    for p, r in zip(probes, results):
        clears = r[gate["_gate"]] >= gate["_min"]
        covered += clears
        if p["expect"] == "abstain":
            ok = not clears                       # a straddling text must not clear the gate as a single answer
        else:
            ok = r["label"] == p["expect"]
            if not ok and clears:
                confident_wrong.append(p["text"])  # the case a router pays for
        if not ok:
            failures.append(f"{p['text']!r}: expected {p['expect']}, got {r['label']} (conf {r['confidence']}, margin {r['margin']}, gate {gate['_gate']}>={gate['_min']}: {'clears' if clears else 'abstains'})")
    print(f"{task_id}: gate {gate['_gate']}>={gate['_min']} covers {covered}/{len(probes)}; confident-wrong {len(confident_wrong)}")
    assert not failures, "\n".join(failures)
