# Jeffy

A pretrained local decision engine: reusable embeddings + small classifiers, shipped with 13 pretrained capabilities and reproducible benchmarks.

**v0.1.0-alpha.1** — local alpha release.

## What this is

One shared frozen text encoder ([BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5), 1024 dimensions, ~1.2 GB) with 13 task-specific logistic regression heads. Each head is trained on up to 10,000 examples from a public dataset and stored as a ~30–630 KB artifact.

```
text → bge-large encoder (1024d) → StandardScaler → LogisticRegression → label + probabilities
```

No fine-tuning, no MLP, no text generation, no LLM fallback (in this release).

## What this is not

- **Not a universal classifier.** Requests for tasks without a pretrained head return an error. There is no zero-shot mode in v0.1.0.
- **Not a drop-in Jeff replacement.** The `/v1/systemone` endpoint requires explicit capability binding. It is a Jeff-style compatibility extension, not an unrestricted drop-in.
- **Probabilities are uncalibrated.** Raw logistic regression outputs.

## Quick start

```bash
# Create environment
python -m venv .venv && source .venv/bin/activate

# Install
pip install -e .

# Download encoder (first run only, ~1.2 GB)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-large-en-v1.5')"

# Start server (loads 13 pretrained heads from data/model_pack/)
jeffy-serve
# Open http://localhost:8400

# Or use directly
python -c "
from jeffy.engine import Engine
engine = Engine('data/model_pack', device='cpu')
engine.load()
r = engine.predict('banking77', 'I was charged twice')
print(r['label'], r['confidence'])
"
```

## API

### `POST /v1/predict`

```bash
curl -X POST http://localhost:8400/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for the same transaction", "task": "banking77"}'
```

Returns label, probabilities, confidence, latency breakdown (embedding + classifier separately), and capability metadata.

### `GET /v1/capabilities`

Lists all loaded pretrained heads with labels, test accuracy, and license.

### `POST /v1/systemone` (Jeff-style compatibility)

Requires explicit `capability` field. Rejects mismatched or partial criteria. Use `label_map` for arbitrary option IDs.

```bash
curl -X POST http://localhost:8400/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{"capability": "ag_news", "state": "The stock market crashed",
       "questions": {"topic": {"type": "choice",
         "criteria": {"World": null, "Sports": null, "Business": null, "Sci/Tech": null}}}}'
```

### `GET /`

Interactive playground with capability selector, probability bars, latency display, and raw JSON.

## Pretrained capabilities (v0.1.0-alpha.1)

| Task | Classes | Test Acc | 95% CI | Macro F1 | Head KB |
|------|---------|----------|--------|----------|---------|
| sms_spam | 2 | 99.1% | [98.6%, 99.6%] | 98.0% | 29 |
| dbpedia | 14 | 96.0% | [95.1%, 96.8%] | 95.9% | 81 |
| imdb | 2 | 94.8% | [93.9%, 95.7%] | 94.8% | 29 |
| banking77 | 77 | 94.3% | [93.4%, 95.3%] | 94.3% | 334 |
| ag_news | 4 | 90.5% | [89.3%, 91.8%] | 90.5% | 41 |
| sst2 | 2 | 90.1% | [88.2%, 92.1%] | 90.1% | 29 |
| clinc_oos | 151 | 88.4% | [87.1%, 89.9%] | 92.1% | 632 |
| massive_intent | 60 | 88.1% | [86.7%, 89.6%] | 86.4% | 271 |
| tweet_eval_offensive | 2 | 81.0% | [78.5%, 83.6%] | 74.8% | 29 |
| tweet_eval_emotion | 4 | 78.1% | [76.0%, 80.4%] | 74.7% | 41 |
| emotion | 6 | 75.5% | [73.6%, 77.5%] | 67.8% | 49 |
| tweet_eval_sentiment | 3 | 66.2% | [64.1%, 68.4%] | 65.7% | 37 |
| snli | 3 | 65.6% | [63.5%, 67.6%] | 65.2% | 37 |

Test accuracy on held-out splits. See `data/eval_results/benchmark.json` for full details.

### Comparison (200 identical examples per task, CPU)

Zero-shot Jeff (Qwen3.5-0.8B) vs task-trained Jeffy vs tuned TF-IDF+LR:

| Task | Jeff (0-shot) | Jeffy (trained) | TF-IDF+LR | QC−Jeff [95% CI] |
|------|-------------|-----------------|-----------|------------------|
| ag_news | 89.5% | 90.0% | 88.5% | +0.5% [−4.0%, +4.5%] |
| sst2 | 84.5% | 89.5% | 79.5% | +5.0% [−0.5%, +10.5%] |
| banking77 | 19.5% | 92.5% | 88.5% | +73.0% [+67.0%, +79.5%] |

10 additional tasks: not evaluated against Jeff in this release.

**Important context:**
- Jeff receives no task-specific training data. Jeffy heads are trained on up to 10,000 examples. This is a deployment comparison, not an equal-supervision experiment.
- TF-IDF+LR baseline is tuned via 3-fold CV on training data (C, analyzer, ngram_range, max_features).
- The tuned TF-IDF+LR baseline beats Jeffy on emotion (87.2% vs 75.5%) and dbpedia (96.6% vs 96.0%) in the full evaluation.

### Latency

Measurements on Apple M3 Max (macOS, arm64, 68.7 GB RAM). Batch size 1.

| Component | p50 | p95 | Notes |
|-----------|-----|-----|-------|
| Embedding | 80–250 ms | 90–260 ms | Varies with input length |
| Classifier | 0.15–0.20 ms | 0.20–0.25 ms | Negligible |
| Total (Jeffy) | 88–252 ms | 112–260 ms | CPU, single example |
| Jeff (0.8B) | 4,500–6,000 ms | 4,900–6,400 ms | CPU |

Thread settings were not perfectly matched between Jeffy and Jeff in the comparison run. These are measured latencies under the recorded configurations, not controlled speedup claims.

### Artifact sizes

| Component | Size |
|-----------|------|
| Model pack (13 heads) | 1.7 MB |
| Encoder (bge-large-en-v1.5) | ~1.2 GB (downloaded separately) |
| Per-head range | 29–632 KB |

## Building from source datasets

```bash
# Install build dependencies
pip install -e ".[build]"

# Train all 13 heads from HuggingFace datasets (~40 min, downloads ~5 GB)
jeffy-build --out data/model_pack

# Evaluate on held-out test sets with baselines
jeffy-evaluate --baselines --latency --device cpu
```

## Evaluation notes

- **SST-2**: Evaluated on the `validation` split. Official test labels are not public.
- **SMS Spam**: Random train/test split (test_size=0.2, seed=42). No standard benchmark split.
- **SNLI**: Input encoded as `premise [SEP] hypothesis`. Label -1 (unlabeled) filtered.
- **CLINC-OOS**: 151 classes including out-of-scope. In-scope accuracy 96.5%, OOS detection 51.7%.
- **MASSIVE**: English subset only (config `en`). String labels used directly.

## Model pack security

Model artifacts use Python pickle/joblib format. Only load artifacts from trusted sources. Each artifact includes integrity hashes in its `manifest.json` which are verified on load.

## Tested

- Python 3.12 on macOS arm64 (Apple M3 Max)
- Dependencies: sentence-transformers, scikit-learn, fastapi, uvicorn, scipy

## License

MIT (Jeffy code). Pretrained head artifacts are derived from public datasets under their respective licenses (documented in each manifest.json). The shared encoder (bge-large-en-v1.5) is MIT-licensed.

## Acknowledgments

Inspired by [Jeff](https://github.com/firelex/jeff) (Mathias Strasser, MIT/Apache 2.0). QuickClassify research at [OpenFam](https://github.com/nicobrenner).
