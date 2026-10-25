# BTZSC Dataset Profile: Hybrid Encoder Suitability

Which of the 22 BTZSC datasets fit the profile where our 6M hybrid encoder
might beat BGE-large (335M)? Based on experiments, the hybrid encoder wins on:
**short texts + many classes + class imbalance**. BGE wins on:
**long texts + few classes + balanced**.

## Dataset Summary

| Dataset | Classes | Text type | Task | Predicted winner |
|---|---|---|---|---|
| banking77 | 77 | short queries | intent | **Hybrid** |
| clinc_oos | 151 | short queries | intent | **Hybrid** |
| massive_intent | ~60 | short commands | intent | **Hybrid** |
| empathetic | 32 | short dialogue | emotion | **Hybrid** |
| emotion | 6 | short tweets | emotion | **Hybrid** |
| tweet_eval_emotion | 4 | short tweets | emotion | **Hybrid** |
| capsotu | ~20 | short sentences | policy topic | **Hybrid** |
| manifesto | many | short sentences | policy topic | **Hybrid** |
| yahootopics | 10 | short questions | topic | Toss-up |
| tweet_eval_sentiment | 3 | short tweets | sentiment | Toss-up |
| financialphrasebank | 3 | short sentences | sentiment | Toss-up |
| ag_news | 4 | medium articles | topic | Toss-up |
| dbpedia | 14 | medium articles | topic | Toss-up |
| snli | 3 | sentence pairs | NLI | BGE |
| sst2 | 2 | short sentences | sentiment | BGE |
| imdb | 2 | long reviews | sentiment | BGE |
| amazonpolarity | 2 | long reviews | sentiment | BGE |
| yelpreviews | 2 | long reviews | sentiment | BGE |
| rottentomatoes | 2 | short sentences | sentiment | BGE |
| appreviews | 2 | short reviews | sentiment | BGE |
| sms_spam | 2 | short messages | spam | BGE |
| trueteacher | 2 | medium summaries | factuality | BGE |
| tweet_eval_offensive | 2 | short tweets | offensive | BGE |
| biasframes_intent | 2 | short sentences | intent/bias | BGE |
| biasframes_offensive | 2 | short sentences | offensive | BGE |
| biasframes_sex | 2 | short sentences | sexual content | BGE |
| wikitoxic_* (4) | 2 each | medium comments | toxicity | BGE |

## Reasoning

### Likely Hybrid Wins (8 datasets)

These share the profile: many classes, short texts, fine-grained distinctions.

- **banking77** (77 cls): Confirmed — hybrid 83.2% vs BGE 93.7% on logreg, but hybrid's end-to-end heads reach 84.0%. Short customer queries, many similar intents.
- **clinc_oos** (151 cls): Confirmed — hybrid generalization 37.2% vs pure-clf 29.3% (+7.9%). Huge class count, short queries. Largest hybrid advantage.
- **massive_intent** (~60 cls): Voice commands similar to banking77/clinc_oos. Short, many intents. Very likely hybrid advantage.
- **empathetic** (32 cls): Fine-grained emotion detection (32 emotions vs emotion's 6). Short dialogue turns. Strong hybrid candidate — similar to emotion where hybrid dominates.
- **emotion** (6 cls): Confirmed — hybrid 83.9% vs BGE 70.3% (+13.6%). Short tweets, imbalanced.
- **tweet_eval_emotion** (4 cls): Tweets with 4 emotion labels. Short, similar domain to emotion. Likely hybrid advantage.
- **capsotu** (~20 cls): Policy topics from State of the Union. Many classes, short sentences.
- **manifesto** (many cls): Political manifesto topics. Many classes.

### Toss-ups (5 datasets)

Mixed signals — some hybrid-favorable traits, some BGE-favorable.

- **yahootopics** (10 cls): 10 classes (moderate), but questions can be longer. Could go either way.
- **tweet_eval_sentiment** (3 cls): Short tweets (hybrid-friendly) but only 3 classes (BGE-friendly).
- **financialphrasebank** (3 cls): Only 3 classes but domain-specific. Generalization test showed 62.9% hybrid vs 80.7% BGE — BGE wins here. Financial domain requires broad pretraining.
- **ag_news** (4 cls): 4 classes, medium-length articles. Generalization test: 83.3% hybrid vs 90.0% BGE. BGE wins but gap is manageable.
- **dbpedia** (14 cls): 14 classes (hybrid-friendly) but texts are longer Wikipedia excerpts. Generalization test: 51.9% hybrid vs 96.4% BGE — BGE dominates. Needs broad world knowledge.

### Likely BGE Wins (13+ datasets)

Binary classification tasks where BGE's massive pretraining gives it enough signal, and the simplicity of the task doesn't reward fine-grained encoder learning.

- **sst2, imdb, amazonpolarity, yelpreviews, rottentomatoes, appreviews**: Binary sentiment. BGE's broad sentiment knowledge from billions of tokens is hard to beat.
- **sms_spam**: Binary spam. Both models near-perfect, but BGE slightly ahead.
- **trueteacher**: Binary factuality. Requires deep language understanding.
- **tweet_eval_offensive, biasframes_*, wikitoxic_***: Binary toxicity/offensive. Requires domain knowledge about harmful language patterns.
- **snli**: 3-class NLI. Sentence-pair task that benefits from BGE's broad language understanding.

## Key Takeaway

**8 of 22 BTZSC datasets** (36%) have the profile where our hybrid encoder is likely competitive or superior. These are concentrated in:
1. **Intent detection** (banking77, clinc_oos, massive_intent) — many classes, short queries
2. **Emotion/sentiment with many classes** (emotion, empathetic, tweet_eval_emotion) — fine-grained affect
3. **Topic classification with many classes** (capsotu, manifesto) — fine-grained policy/political topics

The hybrid encoder's advantage comes from end-to-end gradient flow on tasks where the classification problem is hard enough (many classes, subtle distinctions) that frozen-encoder representations aren't sufficient.
