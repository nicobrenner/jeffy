# Attribution and Upstream Licenses

## Jeffy Code

MIT License. See LICENSE.

## Shared Encoder

**BAAI/bge-large-en-v1.5**
- Source: https://huggingface.co/BAAI/bge-large-en-v1.5
- License: MIT
- Downloaded at runtime by sentence-transformers; not included in this repository.

## Inspiration

**Jeff** by Mathias Strasser
- Source: https://github.com/firelex/jeff
- License: MIT (code), Apache 2.0 (model weights)
- Jeffy's /v1/systemone endpoint and playground design are inspired by Jeff.
  No Jeff code is included in this repository.

## Pretrained Head Artifacts

The model pack contains logistic regression coefficients and scaler statistics
trained on public datasets. These are derived model parameters — numerical
arrays that do not contain any training text.

No dataset in our inventory explicitly prohibits distributing trained model weights.

Each artifact's `manifest.json` records the source dataset and license.

| Dataset | Source License | Source | Weight Distribution | Note |
|---------|---------------|--------|---------------------|------|
| banking77 | CC BY 4.0 | [HF](https://huggingface.co/datasets/legacy-datasets/banking77) | Permitted (derivative work) | |
| clinc_oos | CC BY 3.0 | [HF](https://huggingface.co/datasets/clinc_oos) | Permitted (derivative work) | |
| massive_intent | CC BY 4.0 | [HF](https://huggingface.co/datasets/mteb/amazon_massive_intent) | Permitted (derivative work) | |
| sms_spam | CC BY 4.0 | [HF](https://huggingface.co/datasets/ucirvine/sms_spam) | Permitted (derivative work) | |
| snli | CC BY-SA 4.0 | [HF](https://huggingface.co/datasets/stanfordnlp/snli) | Permitted (share-alike applies) | |
| dbpedia | CC BY-SA 3.0 | [HF](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | Permitted (share-alike applies) | |
| ag_news | Not specified on HF | [HF](https://huggingface.co/datasets/fancyzhx/ag_news) | No explicit restriction found | Academic origin; widely used |
| imdb | Not specified on HF | [HF](https://huggingface.co/datasets/stanfordnlp/imdb) | No explicit restriction found | Academic origin; widely used |
| sst2 | Not specified on HF | [HF](https://huggingface.co/datasets/stanfordnlp/sst2) | No explicit restriction found | Stanford NLP |
| emotion | Not specified on HF | [HF](https://huggingface.co/datasets/dair-ai/emotion) | No explicit restriction found | |
| tweet_eval_* (3) | Not specified on HF | [HF](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | Uncertain | Source data collected via Twitter API; model weights contain no tweet text, but users should review source terms |

**What we distribute:** Numerical classifier parameters (coefficients, scaler means/scales).
**What we do NOT distribute:** Training text, dataset copies, or dataset downloads.
**Reproduction:** Each head can be retrained from its source dataset via `jeffy-build`.

For the three tweet_eval heads, the source data was collected under Twitter API terms.
Our weights do not contain tweet text, but the relationship between API terms and
derived-model distribution has not been independently reviewed.

## scikit-learn

Bundled pretrained artifacts use numpy's .npz format for portability.
Custom-trained models also save a joblib/pickle backup.
scikit-learn (BSD-3-Clause) is required at runtime to reconstruct classifiers.

## sentence-transformers

The encoder is loaded via sentence-transformers, which is Apache 2.0 licensed.
