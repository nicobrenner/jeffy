# Jeffy

Pretrained classifiers you can run and retrain on CPU — text, game state, or any numeric features.

**[jeffyclassify.com](https://jeffyclassify.com)** · **[Live Playground](https://playground.jeffyclassify.com)** · **[GitHub](https://github.com/nicobrenner/jeffy)**

<p align="center">
  <a href="https://playground.jeffyclassify.com/#doom"><img src="examples/doom_battle.gif" width="49%" alt="Doom Battle" align="top"></a>
  <a href="https://playground.jeffyclassify.com/#poker"><img src="examples/poker_demo.gif" width="49%" alt="Poker AI" align="top"></a>
</p>
<p align="center">
  <a href="https://playground.jeffyclassify.com/#inbox"><img src="examples/inbox_demo.gif" width="49%" alt="Inbox Router" align="top"></a>
  <a href="https://playground.jeffyclassify.com/#fly"><img src="examples/fly_demo.gif" width="49%" alt="Fly Navigation" align="top"></a>
</p>

## Try it

```bash
pip install jeffy-classify
jeffy-serve
```

Or try without installing (requires [uv](https://docs.astral.sh/uv/getting-started/installation/)):

```bash
uvx --python 3.12 --from jeffy-classify jeffy-serve
```

Open http://localhost:8400, pick a classifier, and paste one of these:

| Classifier | Try this text |
|------------|---------------|
| banking77 | I was charged twice for the same transaction |
| sms_spam | WINNER! You have been selected for a free cruise. Reply YES to claim. |
| ag_news | The Federal Reserve raised interest rates by 25 basis points on Wednesday |

<p align="center">
  <img src="examples/playground.png" width="100%" alt="Jeffy Playground">
</p>

## Pretrained capabilities

68 classifiers ship with the package — 17 English/general-purpose and 51 multilingual intent classifiers covering languages from Afrikaans to Vietnamese. Weights are logistic regression coefficients (derived model parameters, not copies of training data). Source datasets and licenses are documented in `ATTRIBUTION.md`.

### English / general-purpose

| Task | What it does | Classes | Test Acc | Test F1 |
|------|-------------|---------|----------|---------|
| sms_spam | SMS spam detection | 2 | 99.1% | 98.0% |
| doom_fire | Doom game-state decisions (fire/turn) | 3 | 100.0% | 100.0% |
| fly_navigation | Drosophila connectome navigation | 3 | 98.0% | — |
| dbpedia | Wikipedia article category | 14 | 96.0% | 95.9% |
| imdb | Movie review sentiment (long text) | 2 | 94.8% | 94.8% |
| banking77 | Banking customer intent | 77 | 94.3% | 94.3% |
| ag_news | News topic (world/sports/business/tech) | 4 | 90.5% | 90.5% |
| sst2 | Movie review sentiment | 2 | 90.1% | 90.1% |
| poker_decision | Texas Hold'em poker decisions | 4 | 88.0% | — |
| clinc_oos | Voice assistant intent + out-of-scope | 151 | 88.4% | 92.1% |
| massive_intent | Smart home voice commands (English) | 60 | 88.1% | 86.4% |
| tweet_eval_offensive | Offensive language | 2 | 81.0% | 74.8% |
| tweet_eval_emotion | Tweet emotion | 4 | 78.1% | 74.7% |
| emotion | Text emotion (6 emotions) | 6 | 75.5% | 67.8% |
| inbox_router | Email inbox routing | 4 | 70.8% | 70.8% |
| tweet_eval_sentiment | Tweet sentiment (3-way) | 3 | 66.2% | 65.7% |
| snli | Natural language inference | 3 | 65.6% | 65.2% |

### Multilingual (51 languages)

51 intent classifiers trained on [Amazon MASSIVE](https://huggingface.co/datasets/mteb/amazon_massive_intent) — 60 voice-command intents per language, using frozen multilingual embeddings + per-language logistic regression. ~100 KB per classifier, no GPU needed.

With LaBSE embeddings (768-dim), mean accuracy is **83.3%** across all 51 languages — only 1.8pp below fine-tuned XLM-R (85.1%). All languages above 70%, 47/51 above 80%.

| Tier | Languages | Accuracy range |
|------|-----------|---------------|
| Tier 1 (≥85%) | fr, en, ru, pl, fa, hu, sv, ja, hi | 85.1–86.5% |
| Tier 2 (80–85%) | pt, id, az, nl, ms, tr, da, es, ko, ro, fi, sl, zh_cn, sq, nb, it, lv, mn, af, de, el, vi, te, ml, he, bn, tl, is, ur, hy, kn, th, ta, am, my, zh_tw, cy, sw | 80.5–85.0% |
| Tier 3 (<80%) | jv, ka, ar, km | 76.4–78.8% |

Our encoder comparison found that **mE5-small** (118M params, 82.1% mean on 6 languages) outperforms MiniLM-L12 (118M params, 76.9%) at identical parameter count — training objective matters more than model size. See [research](https://github.com/nicobrenner/jeffy-massive-multi-language-paper) for full results.

Test accuracy on held-out splits. Details in `data/eval_results/benchmark.json`. doom_fire and fly_navigation operate on numeric feature vectors instead of text — see `examples/doom/` and `examples/fly/`. inbox_router is a demo classifier trained on 24 examples — see the live demo in the playground.

**Weaknesses:** SNLI (65.6%) and tweet_eval_sentiment (66.2%) are below what task-specific models achieve. Emotion (75.5%) has limited class coverage. Probabilities are uncalibrated.

## Train a custom classifier

```bash
jeffy-train --example --save-dir my_models
```

```
Loaded 24 examples from reviews.csv
Training 'reviews': 24 examples, 2 classes
  Split: 19 train, 5 test
  Test accuracy: 100.0%
Saved to my_models/reviews/
```

`--example` uses a bundled 24-row product review CSV. To bring your own:

```bash
jeffy-train --input your_data.csv --text-col text --label-col label \
  --task-id your_task --save-dir my_models
```

Supports `.csv`, `.tsv`, and `.jsonl`.

## Serve a custom model

```bash
JEFFY_PACK_DIR=my_models jeffy-serve
```

```bash
curl -s -X POST http://localhost:8400/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The battery life is amazing", "task": "reviews"}'
# → {"label": "positive", "confidence": 0.87, ...}
```

## Install from source

```bash
git clone https://github.com/nicobrenner/jeffy.git
cd jeffy

# With uv (recommended)
uv venv && uv pip install -e .

# Or with pip
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

The first prediction downloads the required encoder(s) — [bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) (~1.2 GB) for English tasks, [paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (~470 MB) for multilingual — cached afterward.

## API

### Python

```python
from jeffy.engine import Engine

engine = Engine()
engine.load()

# List available classifiers
for name, cap in engine.capabilities.items():
    print(f"{name}: {cap.description} ({cap.n_classes} classes)")

# Classify text
result = engine.predict("banking77", "I was charged twice for the same transaction")
print(result["label"])          # "transaction_charged_twice"
print(result["confidence"])     # 0.999
print(result["probabilities"])  # {"transaction_charged_twice": 0.999, ...}
```

### SDK walkthrough

![SDK walkthrough](examples/demo.gif)

### HTTP

```bash
# Banking intent
curl -s -X POST http://localhost:8400/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I was charged twice for the same transaction", "task": "banking77"}'
# → {"label": "transaction_charged_twice", "confidence": 0.999, ...}

# Spam detection
curl -s -X POST http://localhost:8400/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "WINNER! You have been selected for a free cruise. Reply YES to claim.", "task": "sms_spam"}'
# → {"label": "spam", "confidence": 0.91, ...}

# News topic
curl -s -X POST http://localhost:8400/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The Federal Reserve raised interest rates by 25 basis points on Wednesday", "task": "ag_news"}'
# → {"label": "Business", "confidence": 0.86, ...}
```

### What is running

```bash
curl -s http://localhost:8400/v1/version
# → {"version": "0.1.0a21", "package": "jeffy-classify", "heads": 72, "default_encoder": "BAAI/bge-large-en-v1.5", "git_sha": null, ...}
```

Answers without loading the encoder, so it is the first thing to check when a deployed playground and the published
package disagree. It never reports anything about the host.

### List capabilities

```bash
curl -s http://localhost:8400/v1/capabilities | python3 -c "
import json, sys
for c in json.load(sys.stdin)['capabilities']:
    print(f\"{c['task_id']:25s} {c['n_classes']:3d} classes  {c['test_accuracy']:.1%}  {c['name']}\")"
```

### Per-capability metadata

Each shipped classifier has a `manifest.json` with label names, source dataset, HuggingFace path, stated license, encoder identity, training/test counts, and integrity hashes.

```bash
curl -s http://localhost:8400/v1/capabilities/banking77 | python3 -m json.tool
```

## Train from Python

```python
from jeffy.train import train_classifier

clf = train_classifier(
    texts=["great product!", "terrible service", "fast shipping", "broken on arrival"],
    labels=["positive", "negative", "positive", "negative"],
    task_id="my_reviews",
)

result = clf.predict("the quality exceeded my expectations")
print(result["label"])  # "positive"

clf.save("my_models")
```

### Tuning options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `C` | 0.01 | How aggressively the model fits your data. Low (0.001) = conservative, keeps predictions closer to "I'm not sure." High (1.0) = trusts individual training examples more. If the model is great on training data but bad on new data (overfitting), lower C. |
| `test_size` | 0.2 | What fraction of your data to hold back for testing. With 100 examples at 0.2, it trains on 80 and tests on 20. Set to 0 to train on everything (useful when you have very little data and will test manually). |
| `cv_folds` | 3 | Cross-validation: splits your training data into 3 parts, trains on 2 and tests on 1, rotates three times, averages the scores. Gives a more reliable accuracy estimate than a single split. Set to 0 to skip (faster, less reliable estimate). |

Start with the defaults. With <50 examples per class, expect noisy estimates.

### Per-flag binary classifiers

When several yes/no flags can fire on the same input, train one binary head per flag instead of one multi-class head. Each flag gets its own decision boundary, which works better than a single model when flags are rare and independent.

```python
from jeffy.train import train_binary_flags

flags = train_binary_flags(
    texts=emails,
    flags={
        "legal_threat":     ["no", "yes", "no", ...],
        "vulnerable_person": ["no", "no", "yes", ...],
    },
    task_id="intake",
    tune_C=True,
)

flags["legal_threat"].predict("I am contacting my solicitor")  # {"label": "yes", ...}
```

Over HTTP, pass `mode: "binary-flags"` with `texts` and `flags` to `POST /v1/train`.

Measured on a small private multilingual support-email set with unverified provenance (not in this repo, not publishable;
`scripts/care_intake_sweep.py` takes the data path as an argument): one balanced binary head per flag against one multi-class
head on the flag-set label. Binary-per-flag won or tied on 10 of 13 flags and was the only shape that fired at all on the
sparse flags (fewer than ten training positives); a flag with two training positives was unlearnable either way, which is a
labeling problem, not a head choice. Absolute scores from that set are not reported.

### Encoder choice and per-head C

On the same private set, the three multilingual encoders in reach (the default multilingual MiniLM, multilingual-e5-small,
LaBSE) and a per-head C grid (0.01, 0.1, 1 by 3-fold macro-F1): the C grid lifted two heads by one to five points at the
cost of three extra fits per head, so `tune_C=True` is worth turning on for small sets; LaBSE gained about one point over
the default at three times the encode time and four times the download. Keep the default multilingual encoder. The Arabic
slice was too small to rank encoders on it. Absolute scores are not reported, for the provenance reason above.

## Server configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `JEFFY_ENABLE_TRAINING` | unset (off) | Enables the `/v1/train` and `/v1/feedback` endpoints. They are **off by default** because they mutate server state and write files to disk. Leave disabled on any public or shared instance; both return 404 when off. |
| `JEFFY_PLAYGROUND_TEACH` | follows `JEFFY_ENABLE_TRAINING` | Shows the playground's "Not your labels? Teach it three of your own." box under every prediction: four rows pre-filled with the text just classified, submit calls `POST /v1/train`, the new head appears as `my_<id>` and the next prediction runs on it. Off whenever training is off, so a public deployment never shows it; set `on`/`off` to override. Path events `teach_open`, `teach_submit`, `teach_predict` land in the analytics file. |
| `JEFFY_TRAIN_MAX_ROWS` | 200 | `/v1/train` rejects more examples than this (HTTP 413). |
| `JEFFY_TRAIN_MAX_BYTES` | 65536 | `/v1/train` rejects more bytes of training text than this (HTTP 413). |
| `JEFFY_TRAIN_PER_IP_HOUR` | 20 | `/v1/train` calls allowed per client address per hour, in memory (HTTP 429). |
| `JEFFY_TRAINED_DIR` | `~/.jeffy/trained` | Where classifiers trained via `/v1/train` are written. Never the installed package directory. |
| `JEFFY_FEEDBACK_DIR` | `~/.jeffy-feedback` | Where `/v1/feedback` corrections are stored as JSONL. |
| `JEFFY_RETRAIN_THRESHOLD` | 50 | How many feedback corrections trigger an automatic retrain. |
| `JEFFY_PACK_DIR` | bundled pack | Directory of model pack artifacts to serve. |
| `JEFFY_DEVICE` | `cpu` | Torch device for the encoder. |
| `JEFFY_BACKEND` | `auto` | `onnx`, `torch`, or `auto`. |
| `JEFFY_NO_UPDATE_CHECK` | unset | Set to `1` to disable the update check described below. |

### Update check

On startup the server checks PyPI once a day for a newer release and logs a single line if one exists:

```
Update available: 0.1.0a17 → 0.1.0a18 (pip install -U jeffy-classify)
```

It runs on a background thread, never delays startup, and fails silently if offline. The result is cached in `~/.jeffy/update_check.json`. It sends no telemetry: the request is a plain GET of Jeffy's own public PyPI metadata, with nothing about you or your data attached. To turn it off entirely:

```bash
JEFFY_NO_UPDATE_CHECK=1 jeffy-serve
```

It is also skipped automatically when `CI` is set.

## Reproduce the evaluation

```bash
# Install with build dependencies
uv pip install -e ".[build]"
# or: pip install -e ".[build]"

# Retrain all heads from source datasets (~40 min, downloads ~5 GB)
jeffy-build --out data/model_pack

# Evaluate on held-out test sets with tuned baselines
jeffy-evaluate --baselines --latency --device cpu
```

## Confidence and abstaining

Every prediction returns `confidence` (top probability), `margin` (top-1 minus top-2) and `entropy`. The default routing
cutoff is **0.9 confidence**: act on the label when `confidence >= 0.9`, otherwise send the text to a review tier. The
number comes from `scripts/calibration_probe.py`, which scores each head's held-out texts and reports, per threshold, the
share of texts the head would answer on its own (coverage) and the error rate among those answers (misroute, the number a
router pays for). Measured on the shipped heads (CPU, `data/eval_results/calibration_probe.json`):

| head | held-out n | coverage at 0.9 | misroute at 0.9 | at 0.95 | at 0.99 |
|---|---|---|---|---|---|
| support_ticket (11 categories) | 1,350 | 77% | 0.0% | 50% / 0.0% | 4% / 0.0% |
| support_ticket_v2 (27 intents) | 1,350 | 79% | 0.0% | 57% / 0.0% | 4% / 0.0% |
| banking77 (77 intents) | 2,000 | 72% | 0.6% | 56% / 0.4% | 20% / 0.3% |

Two caveats the table cannot show. A held-out split from a templated dataset says nothing about phrasings the taxonomy does
not contain: issue #11's "change the address on my account" is a 0.97 `shipping` on `support_ticket` because the intent
genuinely lives in two classes of that taxonomy. So each head also carries a regression probe set in `data/probes/<task>.jsonl`:
phrasings with an expected label, or an expected abstain for texts that straddle classes (a straddling text must not come back
as a single confident answer). Run them with `JEFFY_RUN_PROBES=1 python -m pytest tests/test_probes.py`. And heads trained on
very little data are correct but under-confident: `medication_name` and `dosage_form` score 25/25 on their demo cases yet only
4% and 8% of those clear 0.9 confidence, while margin ≥ 0.5 covers 68% and 52% with no misroutes. For such heads gate on
`margin`, or lower the cutoff only after measuring misroute at that level.

## Evaluation notes

- **SST-2**: Evaluated on `validation` split (official test labels are not public).
- **SMS Spam**: Random split (test_size=0.2, seed=42); no standard benchmark split.
- **SNLI**: Input encoded as `premise [SEP] hypothesis`. Label -1 filtered.
- **CLINC-OOS**: 151 classes including out-of-scope. In-scope accuracy 96.5%, OOS detection 51.7%.
- **MASSIVE**: English head uses config `en`. Multilingual heads cover 51 languages — see the multilingual table above.

## Deployment

| Component | Size | Required for |
|-----------|------|-------------|
| Jeffy package (wheel) | ~6 MB | Always (includes all 68 classifiers) |
| Encoder (bge-large-en-v1.5) | ~1.2 GB | English tasks (downloaded on first use) |
| Encoder (paraphrase-multilingual-MiniLM-L12-v2) | ~470 MB | Multilingual tasks (downloaded on first use) |
| `datasets` package | ~100 MB | Retraining from HuggingFace only |

Runtime memory: ~2 GB with English encoder, ~2.5 GB with both loaded. Encoders shared across all heads.

**Latency** (CPU, single example, Linux aarch64):

| Stage | p50 | Notes |
|-------|-----|-------|
| Embedding | 50–80 ms | Dominates; varies with input length |
| Classifier | <1 ms | Negligible |
| Total | 50–80 ms | End-to-end |

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

Verified with clean-environment wheel and sdist install on Linux aarch64, Python 3.12, scikit-learn 1.9+, sentence-transformers 6.1+, numpy 2.5+. Pretrained artifacts use numpy `.npz` format, avoiding sklearn version coupling.

## What's not included

- **No zero-shot / general classification.** Each task needs a trained head. Unknown tasks return an error.
- **No LLM fallback.** This release is pure embedding + classifier.
- **No automatic task routing.** You must specify which classifier to use.
- **No hosted service.** Runs locally; [playground.jeffyclassify.com](https://playground.jeffyclassify.com) is a demo, not a production API.

## Roadmap

| Status | Milestone |
|--------|-----------|
| **Available** | Pretrained classifier library (68 classifiers across 51 languages), SDK/API, custom training from CSV/JSONL |
| **Available** | PyPI package (`pip install jeffy-classify`), non-text classifiers (numeric feature vectors) |
| **Available** | Classifier catalog UI with model detail pages, live try-it forms, and usage snippets |
| **Available** | Live Doom demo — real-time VizDoom classifier streaming in the playground |
| **Available** | Inbox classifier demo — animated training + classification workflow |
| **Available** | Poker AI demo — 4 AI players streaming Texas Hold'em decisions live |
| **Available** | Landing page at [jeffyclassify.com](https://jeffyclassify.com), live playground at [playground.jeffyclassify.com](https://playground.jeffyclassify.com) |
| **Available** | Multilingual classifiers — 51 languages via frozen multilingual embeddings + per-language logistic regression ([research](https://github.com/nicobrenner/jeffy-massive-multi-language-paper)) |
| **Next** | [Live demos for more classifiers, broader catalog released in verified batches](https://github.com/nicobrenner/jeffy/issues/2) |
| **Planned** | [Automatic routing among supported classifiers](https://github.com/nicobrenner/jeffy/issues/3) |
| **Planned** | [Optional local/API LLM fallback for unsupported tasks](https://github.com/nicobrenner/jeffy/issues/4) |
| **Planned** | [Hosted classifier catalog and decision-routing service](https://github.com/nicobrenner/jeffy/issues/5) |
| **Planned** | [Embeddable preference recorder](https://github.com/nicobrenner/jeffy/issues/6) — a drop-in integration for web platforms that trains classifiers from live human decisions and contextual data, capturing expertise as reusable models |
| **Exploring** | [Assisted labeling, retraining from corrections, classifier sharing](https://github.com/nicobrenner/jeffy/issues/7) |

[Vote on what you want to see next →](https://github.com/nicobrenner/jeffy/issues) Comment on an issue or open a new one with your use case.
