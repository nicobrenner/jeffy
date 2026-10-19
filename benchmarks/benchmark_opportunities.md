# Jeffy Benchmark Opportunities

Sorted by opportunity score (positive gap + audience reach).

## Scoring

- **Gap**: Jeffy accuracy vs published SOTA (positive = Jeffy beats SOTA)
- **Audience**: estimated reach of the leaderboard (high/medium/low/none)
- **Opportunity**: combined score factoring gap closeness and audience size

## Tier 1: High opportunity (post these first)

| Dataset | Jeffy Acc | SOTA Acc | Gap | SOTA Model | Audience | Leaderboard |
|---|---|---|---|---|---|---|
| wikitoxic_toxicaggregated | 91.7% | ~90.4% | **+1.3%** | CapsNet+BERT | Medium | https://vercel.hyper.ai/en/sota/tasks/toxic-comment-classification/benchmark/toxic-comment-classification-on-jigsaw-toxic |
| amazonpolarity | 95.6% | ~97.4% | -1.8% | BERT-large+UDA | Medium | https://vercel.hyper.ai/en/sota/tasks/sentiment-analysis/benchmark/sentiment-analysis-on-amazon-review-polarity |
| yelpreviews | 96.8% | ~98.3% | ALBERT | -1.5% | Medium | Limited |
| financialphrasebank | 93.2% | ~97.2% | -4.0% | FinBERT | Medium | https://benchlm.ai/benchmarks/financialphrasebank |
| imdb | 94.1% | ~97.4% | -3.3% | XLNet/BERT-large | High | Papers With Code |

Why tier 1: wikitoxic may beat SOTA outright. amazonpolarity/yelpreviews are within 2% of much larger models. financialphrasebank has the Jev comparison angle (Jeffy 93.2% vs Jev 76.98%). IMDB is the most-watched text classification benchmark.

## Tier 2: Good opportunity (competitive, decent audience)

| Dataset | Jeffy Acc | SOTA Acc | Gap | SOTA Model | Audience | Leaderboard |
|---|---|---|---|---|---|---|
| massive (intent) | 85.5% | ~89.5% | -4.0% | XLM-R/mT5 | Medium | https://vercel.hyper.ai/en/sota/tasks/intent-classification/benchmark/intent-classification-on-massive |
| agnews | 90.6% | ~95.7% | -5.1% | BERT+NS-LF | High | https://hyper.ai/en/sota/tasks/text-classification/benchmark/text-classification-on-ag-news |
| banking77 | 88.3% | ~94.8% | -6.5% | RoBERTa-L+ICDA | Medium | https://hyper.ai/en/sota/tasks/intent-detection/benchmark/intent-detection-on-banking77 |
| wikitoxic_insult | 92.3% | unclear | ? | Part of Jigsaw | Low | Jigsaw competition |
| wikitoxic_obscene | 92.8% | unclear | ? | Part of Jigsaw | Low | Jigsaw competition |
| wikitoxic_threat | 99.3% | unclear | ? | Part of Jigsaw | Low | Jigsaw competition |

Why tier 2: agnews and banking77 are well-known benchmarks. massive is growing. wikitoxic subtasks lack individual leaderboards but the scores are strong.

## Tier 3: Lower opportunity (big gap or no audience)

| Dataset | Jeffy Acc | SOTA Acc | Gap | SOTA Model | Audience |
|---|---|---|---|---|---|
| rottentomatoes | 88.0% | ~93.4% | -5.4% | BERT fine-tuned | Low |
| capsotu | 79.9% | unclear | ? | - | None |
| yahootopics | 69.3% | ~66%? | +3%? | unclear | Low |
| biasframes_intent | 90.6% | unclear | ? | - | None |
| biasframes_offensive | 90.1% | unclear | ? | - | None |
| biasframes_sex | 98.2% | unclear | ? | - | None |
| empathetic | 79.7% | unclear | ? | - | None |
| appreviews | 89.1% | unclear | ? | - | None |
| manifesto | 73.2% | unclear | ? | - | None |

## Tier 4: Skip (not competitive)

| Dataset | Jeffy Acc | SOTA Acc | Gap | Notes |
|---|---|---|---|---|
| emotiondair | 75.5% | ~94.6% | -19.1% | DeBERTa-XL dominates, 6-class emotion is hard for logreg |
| trueteacher | 53.4% | unclear | - | Semantic consistency task, not a classification problem for embeddings |

## Other benchmark opportunities (not BTZSC datasets)

### FastFit head-to-head (HIGH PRIORITY)
- IBM Research, NAACL 2024, 222 GitHub stars
- Same positioning as Jeffy ("when LLMs are unfit")
- Benchmarks on Banking77 and other few-shot tasks
- Direct competitor comparison would tap into their audience
- https://github.com/IBM/fastfit

### JevBench classification subset
- 182 ranked systems, active community
- Jeffy already scored: Banking77 #1 (93.2%), AG News #4 (89.0%), SST-2 #6 (90.2%)
- Cherry-pick classification tasks for head-to-head with decision models
- https://benchmarkheaven.com/jev-models

### BTZSC leaderboard (DONE - pending submission)
- ICLR 2026 paper, small audience (8 GitHub stars)
- Jeffy #1 overall (F1=0.854 vs #2 Qwen3-Reranker-8B at 0.714)
- Submission JSON validated, ready to push to HuggingFace
- Good credential, low traffic driver

## Notes

- SOTA models are all fine-tuned (BERT-large, DeBERTa-XL, RoBERTa) so this is apples-to-apples with Jeffy
- Jeffy's angle isn't "beats BERT-large" but "gets within 1-5% with a 1.7MB model pack on CPU"
- The efficiency story (model size, training time, no GPU) matters more than raw accuracy for positioning
