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

| Dataset | Stated License | Artifact Type | Distribution Basis |
|---------|---------------|---------------|-------------------|
| banking77 | CC BY 4.0 | derived weights | CC BY permits derivative works |
| clinc_oos | CC BY 3.0 | derived weights | CC BY permits derivative works |
| massive_intent | CC BY 4.0 | derived weights | CC BY permits derivative works |
| sms_spam | CC BY 4.0 | derived weights | CC BY permits derivative works |
| snli | CC BY-SA 4.0 | derived weights | CC BY-SA permits derivative works (share-alike) |
| dbpedia | CC BY-SA 3.0 | derived weights | CC BY-SA permits derivative works (share-alike) |
| ag_news | Academic / non-commercial | derived weights | **Unresolved**: license may restrict commercial redistribution |
| imdb | Academic / non-commercial | derived weights | **Unresolved**: license may restrict commercial redistribution |
| sst2 | Stanford academic license | derived weights | **Unresolved**: terms not reviewed for derived works |
| emotion | Academic | derived weights | **Unresolved**: specific terms not documented on HF page |
| tweet_eval_sentiment | Twitter TOS / academic | derived weights | **Unresolved**: Twitter-derived data may have redistribution limits |
| tweet_eval_emotion | Twitter TOS / academic | derived weights | **Unresolved**: Twitter-derived data may have redistribution limits |
| tweet_eval_offensive | Twitter TOS / academic | derived weights | **Unresolved**: Twitter-derived data may have redistribution limits |

Artifacts are logistic regression coefficients and scaler parameters trained on the
datasets. They do not contain copies of training text. "Derived weights" means the
artifacts are a mathematical transformation of the training data, not a subset of it.

For CC BY and CC BY-SA datasets, derivative works are explicitly permitted.
For datasets marked **Unresolved**, the license terms have not been independently
verified to permit redistribution of derived model weights. Users should review
the source terms before commercial deployment.

## scikit-learn

Bundled pretrained artifacts use numpy's .npz format for portability.
Custom-trained models also save a joblib/pickle backup.
scikit-learn (BSD-3-Clause) is required at runtime to reconstruct classifiers.

## sentence-transformers

The encoder is loaded via sentence-transformers, which is Apache 2.0 licensed.
