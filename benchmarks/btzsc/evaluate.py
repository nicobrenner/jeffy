"""Run Jeffy through BTZSC eval, saving results incrementally."""

import sys
import json
import time
import os
from pathlib import Path

from btzsc.data import load_btzsc_dataset
from btzsc.metrics import compute_metrics

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adapter import JeffyBTZSCModel

ALL_DATASETS = [
    # Small/fast datasets first
    "rottentomatoes", "financialphrasebank", "trueteacher",
    "emotiondair", "empathetic",
    "biasframes_intent", "biasframes_offensive", "biasframes_sex",
    "wikitoxic_toxicaggregated", "wikitoxic_obscene", "wikitoxic_threat", "wikitoxic_insult",
    "banking77", "massive",
    "agnews", "capsotu", "manifesto",
    "yahootopics", "appreviews", "amazonpolarity",
    # Large/slow datasets last
    "yelpreviews", "imdb",
]

RESULTS_FILE = str(HERE / "results.json")

# Load existing partial results
if os.path.exists(RESULTS_FILE):
    with open(RESULTS_FILE) as f:
        all_results = json.load(f)
    print(f"Loaded {len(all_results)} existing results", flush=True)
else:
    all_results = {}

# Skip already-evaluated datasets
remaining = [d for d in ALL_DATASETS if d not in all_results or "error" in all_results.get(d, {})]
if not remaining:
    print("All datasets already evaluated!", flush=True)
    sys.exit(0)

print(f"Evaluating {len(remaining)} remaining datasets: {remaining}", flush=True)

print("Init adapter...", flush=True)
t0 = time.time()
model = JeffyBTZSCModel(device="cpu")
print(f"Init took {time.time()-t0:.1f}s\n", flush=True)

for ds_name in remaining:
    print(f"Loading {ds_name}...", flush=True)
    try:
        t0 = time.time()
        ds = load_btzsc_dataset(ds_name)
        print(f"  {ds.n_samples} samples, {ds.n_classes} classes (load: {time.time()-t0:.1f}s)", flush=True)

        print(f"  Predicting...", flush=True)
        t0 = time.time()
        preds = model.predict(ds.texts, ds.labels, batch_size=128)
        pred_time = time.time() - t0
        print(f"  Predicted in {pred_time:.1f}s ({ds.n_samples/pred_time:.0f} samples/s)", flush=True)

        metrics = compute_metrics(predictions=preds, references=ds.references)
        all_results[ds_name] = metrics
        print(f"  acc={metrics['accuracy']:.4f}  f1={metrics['macro_f1']:.4f}", flush=True)

        # Save incrementally
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
            f.write("\n")
        print(f"  Saved ({len(all_results)}/{len(ALL_DATASETS)} done)\n", flush=True)

    except Exception as e:
        print(f"  FAILED: {e}\n", flush=True)
        all_results[ds_name] = {"error": str(e)}
        with open(RESULTS_FILE, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
            f.write("\n")

print(f"\n{'='*60}", flush=True)
print(f"Completed: {len([v for v in all_results.values() if 'error' not in v])}/{len(ALL_DATASETS)}", flush=True)
valid = [m["macro_f1"] for m in all_results.values() if isinstance(m, dict) and "macro_f1" in m]
if valid:
    print(f"Average macro F1: {sum(valid)/len(valid):.4f}", flush=True)
