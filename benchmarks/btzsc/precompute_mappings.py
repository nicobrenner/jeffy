"""Pre-compute BTZSC class->position mappings for ALL 22 datasets."""

import json
import time
from pathlib import Path
from datasets import load_dataset
from btzsc.data import load_btzsc_dataset

HERE = Path(__file__).resolve().parent

DATASETS = [
    # sentiment (6)
    "imdb", "amazonpolarity", "appreviews", "yelpreviews",
    "rottentomatoes", "financialphrasebank",
    # emotion (2)
    "emotiondair", "empathetic",
    # intent (3)
    "banking77", "massive", "biasframes_intent",
    # topic (11)
    "agnews", "yahootopics", "trueteacher", "capsotu", "manifesto",
    "biasframes_offensive", "biasframes_sex",
    "wikitoxic_toxicaggregated", "wikitoxic_obscene",
    "wikitoxic_threat", "wikitoxic_insult",
]

mappings = {}

for name in DATASETS:
    t0 = time.time()
    print(f"Loading {name}...", flush=True)

    try:
        ds = load_btzsc_dataset(name, max_samples=10)
        labels_list = ds.labels
        n_classes_btzsc = ds.n_classes

        raw = load_dataset("btzsc/btzsc", name=name, split="test")
        df = raw.to_pandas()

        first_text = df.iloc[0]["text"]
        n_hyp = (df["text"] == first_text).sum()

        class_to_pos = {}
        for row_idx in range(0, len(df), n_hyp):
            group = df.iloc[row_idx:row_idx + n_hyp]
            lt = group.iloc[0]["label_text"]
            if lt in class_to_pos:
                continue
            positive = group[group["labels"] == 1]
            if len(positive) > 0:
                pos = int(positive.index[0] - row_idx)
                class_to_pos[lt] = pos
            if len(class_to_pos) >= n_hyp:
                break

        mappings[name] = {
            "labels": labels_list,
            "n_classes": int(n_classes_btzsc),
            "n_hypotheses": int(n_hyp),
            "class_to_pos": {k: int(v) for k, v in class_to_pos.items()},
        }
        print(f"  {name}: {len(class_to_pos)} classes mapped in {time.time()-t0:.1f}s")

    except Exception as e:
        print(f"  FAILED {name}: {e}")

out = str(HERE / "mappings.json")
with open(out, "w") as f:
    json.dump(mappings, f, indent=2)
    f.write("\n")
print(f"\nSaved {len(mappings)} datasets to {out}")
