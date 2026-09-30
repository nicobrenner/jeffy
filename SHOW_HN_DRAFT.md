# Show HN: Jeffy – pretrained classifiers you can run and retrain on CPU

Jeffy ships 13 text classifiers (intent detection, sentiment, spam, topic routing, etc.) that work out of the box and can be retrained with your own data in seconds on CPU.

It uses one shared frozen encoder (bge-large-en-v1.5) with tiny logistic regression heads (~30-600 KB each). Install, pick a classifier, and run predictions. Or train your own from a CSV:

```bash
pip install jeffy-classify

python -c "
from jeffy.engine import Engine
engine = Engine()
engine.load()
r = engine.predict('banking77', 'I was charged twice')
print(r['label'], r['confidence'])  # transaction_charged_twice 0.999
"
```

Train a custom classifier:
```python
from jeffy.train import train_classifier

clf = train_classifier(
    texts=['great product', 'terrible service', ...],
    labels=['positive', 'negative', ...],
    task_id='my_reviews',
)
clf.predict('the quality exceeded expectations')  # positive (91%)
clf.save('my_models')
```

Benchmarks: https://github.com/[REPO]/blob/main/data/eval_results/benchmark.json

- 94.3% on Banking77 (77 intents), 99.1% on SMS spam, 90.5% on AG News
- ~100ms per prediction on CPU (embedding dominates, classifier <1ms)
- 1.7 MB for all 13 heads; encoder is ~1.2 GB (downloaded once)
- MIT licensed code; dataset licenses documented per head

What Jeffy doesn't do:
- No zero-shot classification (you need a trained head per task)
- No LLM fallback (planned)
- No automatic routing between classifiers (planned)
- Probabilities are uncalibrated

GitHub: [REPO URL]

---

## Introductory comment (draft)

We built Jeffy because we wanted classifiers that (a) work immediately without API keys, (b) run on CPU, and (c) can be retrained from a handful of examples.

The architecture is simple: a frozen sentence-transformer encoder produces 1024-dimensional embeddings, and each task gets its own logistic regression head. The encoder is shared across all tasks (~1.2 GB, downloaded once). Each head is a few hundred KB.

Training a new classifier takes seconds — you provide texts and labels, Jeffy encodes them and fits logistic regression. For the 13 shipped heads, we trained on up to 10,000 examples per task from public datasets and evaluated on held-out splits.

Current limitations:
- Each task needs its own head. You can't ask an arbitrary question — you pick a classifier.
- Embedding quality limits accuracy. We're at 90%+ on intent/topic/spam tasks, weaker on NLI (65.6%) and nuanced sentiment.
- The ~100ms latency is dominated by the encoder; the classifier itself is <1ms.
- No calibration yet — the probabilities are raw logistic regression outputs.

We're exploring a general scorer that handles arbitrary questions with changing options, but haven't solved question conditioning yet (the scorer can't distinguish "what is the sentiment about delivery?" from "what is the sentiment about product quality?" on the same text). That's the research direction; this release is the working library.

Feedback we'd especially appreciate:
- What tasks would be useful to add?
- Is the train-from-CSV workflow clear enough?
- Would you use this for anything your current setup handles differently?
