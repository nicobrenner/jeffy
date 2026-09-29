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

The model pack contains logistic regression weights trained on public datasets.
These are derived model parameters (coefficients and scaler statistics), not
copies of the training data. Redistribution permissions for derived artifacts
have **not been independently verified** for all source datasets.

Each artifact's `manifest.json` records the source dataset and its stated license.
Users should verify that their use of these artifacts complies with the original
dataset terms.

| Dataset | Stated License | Redistribution Verified |
|---------|---------------|------------------------|
| banking77 | CC BY 4.0 | No |
| clinc_oos | CC BY 3.0 | No |
| massive_intent | CC BY 4.0 | No |
| ag_news | Academic / non-commercial | No |
| dbpedia | CC BY-SA 3.0 | No |
| sst2 | Stanford academic license | No |
| emotion | Academic | No |
| imdb | Academic / non-commercial | No |
| sms_spam | CC BY 4.0 | No |
| snli | CC BY-SA 4.0 | No |
| tweet_eval_sentiment | Twitter TOS / academic | No |
| tweet_eval_emotion | Twitter TOS / academic | No |
| tweet_eval_offensive | Twitter TOS / academic | No |

"No" means the license terms have been noted from the dataset's HuggingFace page
but have not been reviewed by a lawyer for redistribution of derived model weights.

## scikit-learn

Model artifacts are saved using joblib/pickle and require scikit-learn to load.
scikit-learn is BSD-3-Clause licensed.

## sentence-transformers

The encoder is loaded via sentence-transformers, which is Apache 2.0 licensed.
