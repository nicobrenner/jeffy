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

Each artifact's `manifest.json` records the source dataset. Source terms were
checked on HuggingFace and original project sites (Sep 2026).

**What we distribute:** Numerical classifier parameters (coefficients, scaler means/scales).
**What we do NOT distribute:** Training text, dataset copies, or dataset downloads.
**Reproduction:** Each head can be retrained from its source dataset via `jeffy-build`.

### Verified: explicit license permitting derivative works

| Dataset | License | Source | Checked |
|---------|---------|--------|---------|
| banking77 | CC BY 4.0 | [HF](https://huggingface.co/datasets/legacy-datasets/banking77) | HF metadata |
| clinc_oos | CC BY 3.0 | [HF](https://huggingface.co/datasets/clinc_oos) | HF metadata |
| massive_intent | CC BY 4.0 | [HF](https://huggingface.co/datasets/mteb/amazon_massive_intent) | HF metadata |
| sms_spam | CC BY 4.0 | [HF](https://huggingface.co/datasets/ucirvine/sms_spam) | HF metadata |
| snli | CC BY-SA 4.0 | [HF](https://huggingface.co/datasets/stanfordnlp/snli) | HF metadata |
| dbpedia | CC BY-SA 3.0 | [HF](https://huggingface.co/datasets/fancyzhx/dbpedia_14) | HF metadata |
| tweet_eval_sentiment | CC BY 3.0 | [HF](https://huggingface.co/datasets/cardiffnlp/tweet_eval) | Listed per-subset on HF card |

### No explicit license on HF; no restriction on trained weights found

| Dataset | HF License Field | Original Source | Finding |
|---------|-----------------|-----------------|---------|
| ag_news | "unknown" | AG corpus (original site unreachable) | No terms found. Academic paper origin, widely redistributed. |
| imdb | "other" | [Stanford](https://ai.stanford.edu/~amaas/data/sentiment/) | Original site requests citation only. No use restrictions stated. |
| sst2 | "unknown" | Stanford NLP | No terms on HF or original page beyond citation. |
| emotion | "other" | [HF](https://huggingface.co/datasets/dair-ai/emotion) | HF card states "educational and research purposes only." |

### Unresolved: source terms require further review

| Dataset | Issue | Detail |
|---------|-------|--------|
| emotion | HF card says "educational and research purposes only" | This may restrict commercial use of the dataset. Whether it applies to derived model weights (which contain no text) is not established. |
| tweet_eval_emotion | Twitter API TOS required; per-subset license undefined | All tweet_eval subsets require Twitter TOS compliance. Our weights contain no tweet text. The sentiment subset is CC BY 3.0; the emotion and offensive subsets have undefined per-subset licenses. |
| tweet_eval_offensive | Twitter API TOS required; per-subset license undefined | Same as above. |

No dataset in this inventory explicitly prohibits distributing trained model
weights. The three unresolved cases involve ambiguous or restrictive dataset-use
terms whose applicability to derived numerical parameters has not been reviewed.

## scikit-learn

Bundled pretrained artifacts use numpy's .npz format for portability.
Custom-trained models also save a joblib/pickle backup.
scikit-learn (BSD-3-Clause) is required at runtime to reconstruct classifiers.

## sentence-transformers

The encoder is loaded via sentence-transformers, which is Apache 2.0 licensed.
