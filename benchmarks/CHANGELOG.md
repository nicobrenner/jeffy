# JeffyBench changelog

Results are comparable **within a version only**, the way MLPerf, HELM and
lm-eval-harness do it. Every published number cites a version, a results file
and the runner commit that produced it.

---

## v2 — 2026-10-26

**The classifier baselines were wrong and are withdrawn. Jeffy's numbers are
unchanged.**

Runner: [`benchmarks/classifier_baselines.py`](classifier_baselines.py)
Results: [`data/eval_results/setfit_fasttext_benchmark_rerun.json`](../data/eval_results/setfit_fasttext_benchmark_rerun.json)
Log: `experiments/scratch_classifier/logs/classifier_baselines_rerun.log`

### banking77, the headline dataset

| model | v1 published | v2 | examples used |
|---|---|---|---|
| Jeffy | 94.3% | 94.3% (unchanged) | 10,003 |
| SetFit, 20 shots/class, paraphrase-mpnet-base-v2 | 54.3% | **85.0%** | 1,540 |
| FastText, library defaults | 1.1% | **17.9%** | 10,003 |
| FastText, defaults + bigrams | — | 3.3% | 10,003 |
| FastText, tuned (lr 1.0, 25 epochs, bigrams) | — | **85.9%** | 10,003 |

On the full evaluation splits rather than the 100-example subset: SetFit 85.0%,
FastText tuned 85.9%, Jeffy 94.3%.

sst2 and ag_news moved too: SetFit 87% / 87% (v1: 81% / 86%), FastText with
defaults 82% / 92% (v1: 57% / 37%).

### What was wrong

- **FastText** was run on library defaults (learning rate 0.1, 5 epochs) on a
  77-class problem. Those defaults are tuned for binary sentiment. At 1.1% it
  scored *below* the 1.3% chance rate for 77 classes, with macro F1 0.0003,
  which means it was predicting a single class. That is a broken run, not a
  result, and it should never have been published
- **SetFit** was run with an unidentified smaller model. The v1 file records
  347 MB, which matches no standard SetFit checkpoint; the v2 run uses the
  documented `paraphrase-mpnet-base-v2` at 438 MB. The v1 procedure has not
  been recovered (see `data/eval_results/setfit_v1_run_forensics.json` for the
  one surviving candidate artifact and why it does not match)

### How it went wrong

The v1 numbers were committed with **results but no runner**. `benchmarks/jeffybench.py`
never contained SetFit, FastText or the decision models at all. Nobody could
re-run them, so nobody could catch the defaults problem, for two weeks.

### What changes downstream

The headline claim survives and gets weaker: **Jeffy 94.3% vs the best baseline
85.9% is an 8.4 point lead, not the 40 points v1 implied.** Jeffy still wins on
every dataset. The size and latency story is intact and is now the stronger
half of the argument: FastText tuned is under 0.1 ms but ships 800 MB, SetFit is
438 MB and 40 ms on CPU, Jeffy is 118 ms against a shared 1.2 GB encoder that
serves all 70 heads at once.

### Rules adopted with this version

1. **No number publishes without its runner.** A results file whose script does
   not exist in the repo is withdrawn, not explained
2. **No number publishes without its denominator**: examples per class, epochs,
   base model, seeds, dataset revision, hardware. All of it is in `config` in
   the results JSON, and on the page
3. **Sanity gates**, any of which failing means a configuration bug:
   - nothing may score at or below chance (1.3% on banking77)
   - macro F1 near zero means one class is being predicted
   - no model may score below a model an order of magnitude smaller on every task
4. **Matched data budget.** Report every model at the same examples-per-class
   where the method allows it, and say so where it does not. SetFit's
   contrastive stage is quadratic in training-set size, so its 1,540 is a
   protocol limit, not a choice, and the table says so

### Still open in v2

- **qwen3-32B at 3.0% on banking77** fails gate 3: it scores below qwen2.5-0.5B
  on all three tasks. Almost certainly a parsing failure on reasoning tokens.
  Not yet re-run. Treat it as withdrawn, not as a result
- The decision models (Jev, GLiNER2.5) and the 10 LLMs were not re-run and
  remain v1. They are not yet traceable to a runner either

---

## v1 — 2026-10-11 — WITHDRAWN for the classifier baselines

Commit a92b81f. 16 models across LLMs, decision models and classifiers.
Results: `data/eval_results/setfit_fasttext_benchmark.json` (kept in place so
the correction can be checked against it, not because it is usable).

Jeffy's own numbers in v1 reproduce and carry over to v2 unchanged. The SetFit
and FastText rows do not, and the qwen3-32B row is unexplained.
