# JeffyBench

> ### Correction, 2026-10-26: the SetFit and FastText rows below are withdrawn
>
> We re-ran our own classifier baselines and they did not reproduce. They were
> too weak, in our favour.
>
> - **FastText** was run on library defaults (lr 0.1, 5 epochs) on a 77-class
>   problem. It scored 1.1%, *below* the 1.3% chance rate, with macro F1 0.0003.
>   Correctly tuned it gets **85.9%** on banking77
> - **SetFit** was run with an unidentified smaller model. With the documented
>   `paraphrase-mpnet-base-v2` at 20 shots per class it gets **85.0%**
> - **qwen3-32B at 3.0%** is unexplained and also withdrawn. It scores below a
>   0.5B model on all three tasks, which is a parsing failure, not a capability
> - **Jeffy's own numbers are unchanged** and reproduce exactly
>
> So the honest headline is **94.3% vs 85.9%, an 8.4 point lead on banking77**,
> not the 40 points the table below implies. Jeffy still wins on every dataset,
> at 1.2 GB shared across all 70 heads against FastText's 800 MB for one.
>
> The cause: these numbers were published with **results but no runner script**,
> so for two weeks nobody could re-run them, including us. Full detail,
> corrected tables and the rules adopted since are in
> **[CHANGELOG.md](CHANGELOG.md)**. Corrected results:
> [`setfit_fasttext_benchmark_rerun.json`](../data/eval_results/setfit_fasttext_benchmark_rerun.json).
> Runner: [`classifier_baselines.py`](classifier_baselines.py).
>
> The tables below are left as published, with the withdrawn rows marked, so
> the correction can be checked rather than taken on trust.

Jeffy vs 16 models on text classification: 10 local LLMs, 2 decision models, 2 traditional classifiers, and the leading hosted decision model.

**Current benchmark version: v2 (classifiers), v1 (LLMs and decision models).**
See [CHANGELOG.md](CHANGELOG.md).

**[Interactive results](https://jeffyclassify.com/benchmark/)** · **[Blog post](https://jeffyclassify.com/blog/when-llms-fail-at-classification/)**

## The key chart

As class count grows, LLMs and decision models collapse while Jeffy stays accurate. No model other than Jeffy reaches 70% on banking77 (77 classes). The crossover happens between 2 and 4 classes.

![Accuracy vs number of classes](scaling_chart.png)

## banking77 — 77 classes (customer intent)

This is where the gap is widest. Jeffy at 94.3%, the next best model (Jev) at 80%, and the best LLM (phi4 14B) at 68%.

![banking77 benchmark results](banking77_chart.png)

| Model | Type | Accuracy | Latency p50 |
|-------|------|----------|-------------|
| **Jeffy** | **classifier** | **94.3%** | **118ms** |
| Jev (hosted) | decision model | 80.0% | 134ms |
| phi4 (14B) | LLM | 68.0% | 15.4s |
| gemma4 (9.6B) | LLM | 68.0% | 696ms |
| gemma3 (4B) | LLM | 64.0% | 956ms |
| llama3.1 (8B) | LLM | 62.0% | 8.9s |
| GLiNER2.5 | decision model | 56.0% | 1.7s |
| ~~SetFit~~ | classifier | ~~54.3%~~ **85.0%** [^c] | 6ms |
| phi4-mini (3.8B) | LLM | 52.0% | 181ms |
| llama3.2 (3B) | LLM | 46.0% | 143ms |
| mistral (7B) | LLM | 39.0% | 17s |
| Laya | decision model | 38.0% | 815ms |
| phi (2.7B) | LLM | 30.0% | 231ms |
| qwen2.5 (0.5B) | LLM | 15.0% | 55ms |
| ~~qwen3 (32B)~~ | LLM | ~~3.0%~~ withdrawn [^c] | 9.5s |
| ~~FastText~~ | classifier | ~~1.1%~~ **85.9%** [^c] | <1ms |

### ag_news — 4 classes (news topic)

| Model | Type | Accuracy | Latency p50 |
|-------|------|----------|-------------|
| Laya | decision model | 94.0% | 273ms |
| **Jeffy** | **classifier** | **90.5%** | **80ms** |
| ~~SetFit~~ | classifier | ~~85.9%~~ **86.8%** [^c] | 9ms |
| phi4 (14B) | LLM | 83.0% | 493ms |
| Jev (hosted) | decision model | 83.0% | 127ms |
| gemma4 (9.6B) | LLM | 82.0% | 969ms |
| phi4-mini (3.8B) | LLM | 81.0% | 176ms |
| mistral (7B) | LLM | 81.0% | 9s |

### sst2 — 2 classes (binary sentiment)

| Model | Type | Accuracy | Latency p50 |
|-------|------|----------|-------------|
| Jev (hosted) | decision model | 96.0% | 127ms |
| phi4 (14B) | LLM | 95.0% | 286ms |
| GLiNER2.5 | decision model | 93.0% | 336ms |
| **Jeffy** | **classifier** | **90.1%** | **104ms** |
| llama3.1 (8B) | LLM | 90.0% | 15.5s |
| gemma3 (4B) | LLM | 89.0% | 500ms |
| llama3.2 (3B) | LLM | 89.0% | 115ms |

## Run it yourself

```bash
pip install jeffy-classify datasets

# Pull an LLM to compare
ollama pull phi4

# Run the benchmark
python benchmarks/jeffybench.py --models phi4,mistral,llama3.1

# Single task
python benchmarks/jeffybench.py --tasks banking77 --models phi4
```

## Setup

- 100 held-out samples per task
- All LLMs via Ollama, temperature 0, prompted with "reply with ONLY the category name"
- Decision models get the label set at call time
- Jeffy and FastText trained on standard training splits; SetFit on 20 examples per class,
  because its contrastive stage is quadratic in training-set size (a protocol limit, not a choice)
- All results in [`data/eval_results/`](../data/eval_results/)

[^c]: Withdrawn in the 2026-10-26 correction at the top of this file. The replacement number comes from [`setfit_fasttext_benchmark_rerun.json`](../data/eval_results/setfit_fasttext_benchmark_rerun.json), measured on the full evaluation split; see [CHANGELOG.md](CHANGELOG.md) for the configuration.
