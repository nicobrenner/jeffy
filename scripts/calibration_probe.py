"""Calibration probe: how often is a confident answer wrong, and how much traffic is confident?

For each head, score its held-out texts and report, per confidence threshold t:
  coverage(t)  = share of texts the head would answer on its own (confidence >= t)
  misroute(t)  = error rate among those answered texts (the number a router actually pays for)
The same table is produced for the top-2 margin as the gate, because a 0.97 answer whose runner-up is 0.02 and a
0.55 answer whose runner-up is 0.44 are different risks.

Heads and test sets (nothing here downloads except banking77's HF split, which the evaluator already uses):
  support_ticket     bitext 11 categories; test = the 1,350 texts of support_ticket_v2's held-out split (50 per intent),
                     labels mapped intent -> category with the map in v2's manifest. CAVEAT: v1 was trained on bitext
                     with its own split, so some of these texts may be in v1's training set; its numbers are optimistic.
  support_ticket_v2  bitext 27 intents on its own held-out split (disjoint from its training texts by construction).
  support_ticket_v2 -> category  v2's prediction mapped up to the parent category, scored against the same category labels.
  banking77          77 intents, the evaluator's test split (max 2000, seed 42).
  medication_name / dosage_form   the 25 OCR-style demo cases in data/eval_results/medication_classifier_demo.json (the only
                     held-out set in the repo for these heads; their training data is not in the repo).

Also runs the regression probe set data/probes/<task>.jsonl: phrasings with an expected label, or expected abstain
(`"expect": "abstain"`) for texts that genuinely straddle two classes of the taxonomy (issue #11).

    PYTHONPATH=src python scripts/calibration_probe.py                 # all heads, CPU
    PYTHONPATH=src python scripts/calibration_probe.py --tasks banking77 --max-test 500
Writes data/eval_results/calibration_probe.json and calibration_probe.csv.
"""
import argparse, csv, json, re, time
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
PACK = REPO / "src/jeffy/pack"
OUT = REPO / "data/eval_results"
THRESHOLDS = [0.5, 0.8, 0.9, 0.95, 0.99]
MARGINS = [0.2, 0.5, 0.8, 0.9, 0.95]


def v2_parent_map():
    m = json.loads((PACK / "support_ticket_v2/manifest.json").read_text())
    out = {}
    for intent, desc in m["label_descriptions"].items():
        mm = re.search(r"under category '([^']+)'", desc)
        out[intent] = mm.group(1).lower() if mm else None
    return out


def support_ticket_sets():
    split = json.loads((OUT / "support_ticket_v2/split.json").read_text())["test"]
    parent = v2_parent_map()
    texts, intents = [], []
    for intent in sorted(split):
        for t in split[intent]:
            texts.append(t); intents.append(intent)
    cats = [parent[i] for i in intents]
    return texts, intents, cats, parent


def banking77_set(max_test):
    from jeffy.build_pack import DATASETS, load_and_split
    _, _, test_texts, test_labels, _ = load_and_split("banking77", DATASETS["banking77"], max_test=max_test)
    return test_texts, test_labels   # raw label ids, compared with the head's label_id as the evaluator does


def table(conf, margin, correct):
    conf, margin, correct = np.asarray(conf), np.asarray(margin), np.asarray(correct, dtype=bool)
    rows = []
    for gate, values, cuts in (("confidence", conf, THRESHOLDS), ("margin", margin, MARGINS)):
        for t in cuts:
            m = values >= t
            rows.append({"gate": gate, "threshold": t, "coverage": float(m.mean()),
                         "misroute": float((~correct[m]).mean()) if m.any() else None, "n_covered": int(m.sum())})
    return rows


def run_probes(engine, task_id, label_of=lambda r: r["label"]):
    p = REPO / "data/probes" / f"{task_id}.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    gate = next((r for r in rows if "_gate" in r), {"_gate": "confidence", "_min": 0.9})   # per-file gate; medical heads use margin
    probes = [r for r in rows if "_gate" not in r]
    res = engine.predict_batch(task_id, [x["text"] for x in probes])
    out = []
    for x, r in zip(probes, res):
        got = label_of(r); clears = r[gate["_gate"]] >= gate["_min"]
        if x["expect"] == "abstain":
            ok = not clears   # a straddling text must not clear the gate as a single answer
        else:
            ok = got == x["expect"]
        out.append({**x, "got": got, "confidence": r["confidence"], "margin": r["margin"], "support_percentile": r.get("support_percentile"), "clears_gate": bool(clears), "ok": bool(ok)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["support_ticket", "support_ticket_v2", "banking77", "medication_name", "dosage_form"])
    ap.add_argument("--max-test", type=int, default=2000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from jeffy.engine import Engine
    eng = Engine(pack_dir=str(PACK), device=a.device); eng.load()
    results = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "thresholds": THRESHOLDS, "margins": MARGINS, "heads": {}}
    csv_rows = []

    def score(name, task_id, texts, labels, label_of=lambda r: r["label"]):
        t0 = time.time()
        res = eng.predict_batch(task_id, texts)
        preds = [label_of(r) for r in res]
        correct = [p == y for p, y in zip(preds, labels)]
        rows = table([r["confidence"] for r in res], [r["margin"] for r in res], correct)
        acc = float(np.mean(correct))
        results["heads"][name] = {"task_id": task_id, "n_test": len(texts), "accuracy": acc, "rows": rows,
                                  "seconds": round(time.time() - t0, 1)}
        for r in rows:
            csv_rows.append({"head": name, **r})
        print(f"\n== {name}: n={len(texts)} accuracy={acc:.4f} ({time.time()-t0:.0f}s)")
        print(f"{'gate':<11}{'t':>6}{'coverage':>10}{'misroute':>10}{'n':>7}")
        for r in rows:
            mr = "-" if r["misroute"] is None else f"{r['misroute']:.4f}"
            print(f"{r['gate']:<11}{r['threshold']:>6}{r['coverage']:>10.3f}{mr:>10}{r['n_covered']:>7}")
        return res

    if "support_ticket" in a.tasks or "support_ticket_v2" in a.tasks:
        texts, intents, cats, parent = support_ticket_sets()
        if "support_ticket" in a.tasks:
            score("support_ticket (11 categories)", "support_ticket", texts, cats)
        if "support_ticket_v2" in a.tasks:
            score("support_ticket_v2 (27 intents)", "support_ticket_v2", texts, intents)
            score("support_ticket_v2 -> category", "support_ticket_v2", texts, cats, label_of=lambda r: parent.get(r["label"]))
    if "medication_name" in a.tasks or "dosage_form" in a.tasks:
        cases = json.loads((OUT / "medication_classifier_demo.json").read_text())["test_results"]
        texts = [c["ocr_text"] for c in cases]
        if "medication_name" in a.tasks:
            score("medication_name (50 drugs, 25 demo cases)", "medication_name", texts, [c["expected_medication"] for c in cases])
        if "dosage_form" in a.tasks:
            score("dosage_form (10 forms, 25 demo cases)", "dosage_form", texts, [c["expected_form"] for c in cases])
    if "banking77" in a.tasks:
        texts, labels = banking77_set(a.max_test)
        score("banking77 (77 intents)", "banking77", texts, labels, label_of=lambda r: r["label_id"])

    results["probes"] = {}
    for task_id in a.tasks:
        pr = run_probes(eng, task_id)
        if pr is not None:
            results["probes"][task_id] = pr
            bad = [x for x in pr if not x["ok"]]
            print(f"\n== probes {task_id}: {len(pr)-len(bad)}/{len(pr)} pass")
            for x in bad:
                print(f"   FAIL {x['text']!r}: expected {x['expect']}, got {x['got']} conf={x['confidence']} margin={x['margin']} support_pct={x.get('support_percentile')}")
            if any(x.get("support_percentile") is not None for x in pr):
                print(f"   support_percentile: " + ", ".join(f"{x['text'][:28]!r}={x['support_percentile']}" for x in pr if x.get("support_percentile") is not None))

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "calibration_probe.json").write_text(json.dumps(results, indent=1))
    with (OUT / "calibration_probe.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["head", "gate", "threshold", "coverage", "misroute", "n_covered"]); w.writeheader(); w.writerows(csv_rows)
    print(f"\nwrote {OUT/'calibration_probe.json'} and .csv")


if __name__ == "__main__":
    main()
