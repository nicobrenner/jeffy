# BTZSC Benchmark Results

Jeffy evaluated on all 22 datasets from [BTZSC](https://github.com/IliasAarab/btzsc) (Benchmark for Zero-Shot Text Classification, ICLR 2026).

## Results

**Overall: #1 on the leaderboard** with macro F1 = 0.854, ahead of Qwen3-Reranker-8B (0.714).

| Task Group | Macro F1 | Accuracy |
|---|---|---|
| Sentiment (6 datasets) | 0.926 | 0.928 |
| Intent (3 datasets) | 0.886 | 0.881 |
| Topic (11 datasets) | 0.827 | 0.846 |
| Emotion (2 datasets) | 0.737 | 0.776 |
| **Overall (22 datasets)** | **0.854** | **0.867** |

## Per-dataset scores

| Dataset | Task | F1 | Acc |
|---|---|---|---|
| yelpreviews | sentiment | 0.968 | 0.968 |
| amazonpolarity | sentiment | 0.956 | 0.956 |
| imdb | sentiment | 0.941 | 0.941 |
| wikitoxic_threat | topic | 0.955 | 0.993 |
| financialphrasebank | sentiment | 0.919 | 0.932 |
| wikitoxic_obscene | topic | 0.926 | 0.928 |
| wikitoxic_insult | topic | 0.920 | 0.923 |
| wikitoxic_toxicagg | topic | 0.917 | 0.917 |
| biasframes_sex | topic | 0.916 | 0.982 |
| agnews | topic | 0.906 | 0.906 |
| banking77 | intent | 0.906 | 0.883 |
| biasframes_intent | intent | 0.906 | 0.906 |
| biasframes_offensive | topic | 0.899 | 0.901 |
| appreviews | sentiment | 0.890 | 0.891 |
| rottentomatoes | sentiment | 0.880 | 0.880 |
| massive | intent | 0.847 | 0.855 |
| capsotu | topic | 0.806 | 0.799 |
| empathetic | emotion | 0.796 | 0.797 |
| yahootopics | topic | 0.690 | 0.693 |
| emotiondair | emotion | 0.678 | 0.755 |
| manifesto | topic | 0.643 | 0.732 |
| trueteacher | topic | 0.521 | 0.534 |

## Files

- `adapter.py` - BTZSC adapter (maps Jeffy predictions to BTZSC format)
- `evaluate.py` - incremental evaluation script (saves after each dataset)
- `generate_submission.py` - generates leaderboard submission JSON
- `results.json` - raw results for all 22 datasets
- `submission.json` - validated leaderboard submission
- `mappings.json` - precomputed BTZSC label mappings
- `precompute_mappings.py` - mapping generator

## Reproducing

```bash
pip install btzsc
cd benchmarks/btzsc
python evaluate.py
python generate_submission.py
```
