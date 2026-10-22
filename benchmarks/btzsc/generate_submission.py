"""Generate BTZSC leaderboard submission JSON from evaluation results."""

import json
import sys
from pathlib import Path

from btzsc import BTZSCResults
from btzsc.leaderboard import validate_result_payload

HERE = Path(__file__).resolve().parent
RESULTS_FILE = str(HERE / "results.json")

TASK_GROUPS = {
    "sentiment": ["imdb", "amazonpolarity", "appreviews", "yelpreviews", "rottentomatoes", "financialphrasebank"],
    "emotion": ["emotiondair", "empathetic"],
    "intent": ["banking77", "massive", "biasframes_intent"],
    "topic": ["agnews", "yahootopics", "trueteacher", "capsotu", "manifesto",
              "biasframes_offensive", "biasframes_sex",
              "wikitoxic_toxicaggregated", "wikitoxic_obscene",
              "wikitoxic_threat", "wikitoxic_insult"],
}

with open(RESULTS_FILE) as f:
    per_dataset = json.load(f)

missing = [d for group in TASK_GROUPS.values() for d in group if d not in per_dataset or "error" in per_dataset.get(d, {})]
if missing:
    print(f"WARNING: Missing datasets: {missing}")
    print("Cannot generate valid submission without all 22 datasets.")
    sys.exit(1)

task_summary = {}
for task_name, datasets in TASK_GROUPS.items():
    task_metrics = {}
    for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
        values = [per_dataset[d][metric] for d in datasets if metric in per_dataset[d]]
        task_metrics[metric] = sum(values) / len(values) if values else 0.0
    task_summary[task_name] = task_metrics

all_metrics = {}
for metric in ["accuracy", "macro_f1", "macro_precision", "macro_recall"]:
    values = [per_dataset[d][metric] for d in per_dataset if metric in per_dataset[d]]
    all_metrics[metric] = sum(values) / len(values) if values else 0.0
task_summary["overall"] = all_metrics

results = BTZSCResults(
    per_dataset_results=per_dataset,
    task_summary=task_summary,
    model_name="jeffy (BGE-large + logreg)",
    model_type="embedding",
    model_params=335141888,
    model_revision="v0.1.0a5",
    precision="float32",
    batch_size=128,
    max_samples=None,
    device="cpu",
)

out_path = str(HERE / "submission.json")
payload = results.to_json()

# Fix the auto-generated URL to point to the actual project
payload["model"]["url"] = "https://github.com/nicobrenner/jeffy"

# Validate
errors = validate_result_payload(payload)
if errors:
    print("VALIDATION ERRORS:")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
else:
    print("Submission JSON is VALID!")

# Write validated payload
with open(out_path, "w") as f:
    json.dump(payload, f, indent=2)
    f.write("\n")
print(f"Saved to {out_path}")

# Show summary
print(f"\n{'='*60}")
print("RESULTS SUMMARY")
print(f"{'='*60}")
print(f"Overall macro F1:        {all_metrics['macro_f1']:.4f}")
print(f"Overall accuracy:        {all_metrics['accuracy']:.4f}")
print(f"Overall macro precision: {all_metrics['macro_precision']:.4f}")
print(f"Overall macro recall:    {all_metrics['macro_recall']:.4f}")

print(f"\nBy task group:")
for task_name in ["sentiment", "emotion", "intent", "topic"]:
    tm = task_summary[task_name]
    print(f"  {task_name:<12} F1={tm['macro_f1']:.4f}  Acc={tm['accuracy']:.4f}")

print(f"\nBy dataset:")
for ds_name, metrics in sorted(per_dataset.items()):
    print(f"  {ds_name:<28} F1={metrics['macro_f1']:.4f}  Acc={metrics['accuracy']:.4f}")

# Show ranking comparison
print(f"\n{'='*60}")
print("RANKING vs BASELINES (by F1)")
print(f"{'='*60}")
ranking = results.compare_baselines(metric="f1")
print(ranking.to_string(index=True))
