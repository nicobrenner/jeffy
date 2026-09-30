# Jeffy

Pretrained text classifiers: 13 ready-to-use heads, train your own in seconds.

```
pip install jeffy-classify-classify
```

## What you get

13 classifiers for common text tasks — intent detection, sentiment, topic routing, spam, emotion, and more. Each is a logistic regression head over a shared frozen encoder ([bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), 1024d).

```
text → encoder (1024d) → scaler → logistic regression → label + probabilities
```

You can also train your own classifier from a CSV or labeled examples in a few lines of code.

## Install

```bash
pip install jeffy-classify
```

The first prediction downloads the encoder (~1.2 GB, cached afterward). Inference requires only `sentence-transformers` and `scikit-learn`. Training from public datasets additionally requires `datasets` (`pip install jeffy-classify[build]`).

## Use a pretrained classifier

### Python SDK

```python
from jeffy.engine import Engine

engine = Engine()
engine.load()

# List available classifiers
for name, cap in engine.capabilities.items():
    print(f"{name}: {cap.description} ({cap.n_classes} classes)")

# Classify text
result = engine.predict("banking77", "I was charged twice for the same transaction")
print(result["label"])        # "transaction_charged_twice"
print(result["confidence"])   # 0.9986
print(result["probabilities"])  # {"transaction_charged_twice": 0.9986, ...}
```

### HTTP API

```bash
# Start server
jeffy-serve
# Open http://localhost:8400 for interactive playground

# Predict
curl -s -X POST http://localhost:8400/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice", "task": "banking77"}' | python3 -m json.tool
```

### CLI

```bash
# List capabilities
curl -s http://localhost:8400/v1/capabilities | python3 -c "
import json, sys
for c in json.load(sys.stdin)['capabilities']:
    print(f\"{c['task_id']:25s} {c['n_classes']:3d} classes  {c['test_accuracy']:.1%}  {c['name']}\")"
```

## Train your own classifier

### From Python

```python
from jeffy.train import train_classifier

clf = train_classifier(
    texts=["great product!", "terrible service", "fast shipping", "broken on arrival", ...],
    labels=["positive", "negative", "positive", "negative", ...],
    task_id="my_reviews",
)

# Use immediately
result = clf.predict("the quality exceeded my expectations")
print(result["label"])  # "positive"

# Save for later / deployment
clf.save("my_models")
```

### From a CSV file

```bash
jeffy-train --input reviews.csv --text-col review_text --label-col sentiment --task-id reviews
# Also supports .jsonl and .tsv
```

### Tuning options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | 0.01 | Regularization strength. Lower = more regularization, less overfitting. Try 0.001–1.0. |
| `test_size` | 0.2 | Fraction held out for evaluation. Set 0 to use all data for training. |
| `cv_folds` | 3 | Cross-validation folds for accuracy estimate. Set 0 to skip. |

**Validation guidance:** Start with the defaults. If training accuracy is much higher than CV accuracy, try lower C. If both are low, you may need more examples or better text preprocessing. With <50 examples per class, expect noisy estimates.

### Deploy a custom model

```bash
# Save during training
jeffy-train --input data.csv --text-col text --label-col label --task-id my_task --save-dir my_pack

# Serve with custom models alongside pretrained ones
JEFFY_PACK_DIR=my_pack jeffy-serve
```

## Pretrained capabilities

| Task | What it does | Classes | Test Acc | Test F1 | Head size |
|------|-------------|---------|----------|---------|-----------|
| banking77 | Banking customer intent | 77 | 94.3% | 94.3% | 334 KB |
| clinc_oos | Voice assistant intent + out-of-scope | 151 | 88.4% | 92.1% | 632 KB |
| massive_intent | Smart home voice commands | 60 | 88.1% | 86.4% | 271 KB |
| ag_news | News topic (world/sports/business/tech) | 4 | 90.5% | 90.5% | 41 KB |
| dbpedia | Wikipedia article category | 14 | 96.0% | 95.9% | 81 KB |
| sst2 | Movie review sentiment | 2 | 90.1% | 90.1% | 29 KB |
| imdb | Movie review sentiment (long text) | 2 | 94.8% | 94.8% | 29 KB |
| emotion | Text emotion (6 emotions) | 6 | 75.5% | 67.8% | 49 KB |
| sms_spam | SMS spam detection | 2 | 99.1% | 98.0% | 29 KB |
| snli | Natural language inference | 3 | 65.6% | 65.2% | 37 KB |
| tweet_eval_sentiment | Tweet sentiment (3-way) | 3 | 66.2% | 65.7% | 37 KB |
| tweet_eval_emotion | Tweet emotion | 4 | 78.1% | 74.7% | 41 KB |
| tweet_eval_offensive | Offensive language | 2 | 81.0% | 74.8% | 29 KB |

Test accuracy measured on held-out splits with 95% bootstrap CIs. Details in `data/eval_results/benchmark.json`.

**Weaknesses:** SNLI (65.6%) and tweet_eval_sentiment (66.2%) are below what task-specific models achieve. Emotion detection (75.5%) has limited class coverage. Probabilities are uncalibrated.

### Per-capability details

Each shipped classifier has a `manifest.json` with:
- Label names and their IDs
- Source dataset, HuggingFace path, and stated license
- Encoder identity and version
- Training and test example counts
- Integrity hashes (scaler and classifier coefficients)

View any capability's full metadata:
```bash
curl -s http://localhost:8400/v1/capabilities/banking77 | python3 -m json.tool
```

## Reproduce the evaluation

```bash
pip install jeffy-classify[build]

# Retrain all 13 heads from source datasets (~40 min, downloads ~5 GB)
jeffy-build --out data/model_pack

# Evaluate on held-out test sets with tuned baselines
jeffy-evaluate --baselines --latency --device cpu
```

## Evaluation notes

- **SST-2**: Evaluated on `validation` split (official test labels are not public).
- **SMS Spam**: Random split (test_size=0.2, seed=42); no standard benchmark split.
- **SNLI**: Input encoded as `premise [SEP] hypothesis`. Label -1 filtered.
- **CLINC-OOS**: 151 classes including out-of-scope. In-scope accuracy 96.5%, OOS detection 51.7%.
- **MASSIVE**: English only (config `en`).

## Deployment requirements

| Component | Size | Required for |
|-----------|------|-------------|
| Jeffy package | ~1.7 MB | Always |
| Encoder (bge-large-en-v1.5) | ~1.2 GB | Inference (downloaded on first use) |
| `datasets` package | ~100 MB | Retraining from HuggingFace only |

Runtime memory: ~2 GB (encoder loaded once, shared across all heads).

**Latency** (Apple M3 Max, CPU, batch 1):

| Stage | p50 | Notes |
|-------|-----|-------|
| Embedding | 80–250 ms | Dominates; varies with input length |
| Classifier | <1 ms | Negligible |
| Total | 88–252 ms | End-to-end, single example |

## Model security

Bundled pretrained artifacts use numpy `.npz` format (portable, no pickle). Custom-trained models also save a joblib pickle backup. **Only load custom pickle artifacts from trusted sources.** Each artifact's `manifest.json` includes integrity hashes verified on load.

## Provenance and licensing

Jeffy code is MIT-licensed. Head artifacts are derived from public datasets; redistribution permissions have **not been independently verified** for all sources. See `ATTRIBUTION.md` for per-dataset license status.

| Dataset | Stated license |
|---------|---------------|
| banking77, massive_intent, sms_spam | CC BY 4.0 |
| clinc_oos | CC BY 3.0 |
| dbpedia | CC BY-SA 3.0 |
| snli | CC BY-SA 4.0 |
| ag_news, imdb | Academic / non-commercial |
| sst2 | Stanford academic license |
| emotion | Academic |
| tweet_eval_* | Twitter TOS / academic |

The encoder ([bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)) is MIT-licensed.

## Tested

Verified with clean-environment wheel install:

| Component | Version |
|-----------|---------|
| Python | 3.12.11 |
| Platform | macOS 15.6.1, arm64 (Apple M3 Max) |
| scikit-learn | 1.9.1 |
| sentence-transformers | 6.1.0 |
| torch | 2.14.0 |
| numpy | 2.5.3 |

Pretrained artifacts use numpy `.npz` format, avoiding sklearn version coupling. Tested loading artifacts built with sklearn 1.7.2 on sklearn 1.9.1 without warnings.

## What's not included

- **No zero-shot / general classification.** Each task needs a trained head. Unknown tasks return an error.
- **No LLM fallback.** This release is pure embedding + classifier.
- **No automatic task routing.** You must specify which classifier to use.

## Roadmap

| Status | Milestone |
|--------|-----------|
| **Available** | Pretrained classifier library, SDK/API, playground, custom training from CSV/JSONL |
| **Next** | Public repository, downloadable release, landing page |
| **Planned** | Broader classifier catalog, released in verified batches |
| **Planned** | Automatic routing among supported classifiers |
| **Planned** | Optional local/API LLM fallback for unsupported tasks |
| **Exploring** | Assisted labeling, retraining from corrections, classifier sharing |

Suggestions for datasets, capabilities, or workflows are welcome as issues.

## Acknowledgments

Inspired by [Jeff](https://github.com/firelex/jeff) (Mathias Strasser).
