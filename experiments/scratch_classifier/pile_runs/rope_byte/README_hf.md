# Jeffy byte-LM image priors

Shared lossless image priors: a plain causal byte LM over 28x28 tiles (2D RoPE), trained on CLIC-2020, evaluated on Kodak-24.
PNG (optimized) = 13.448 bpp, WebP-lossless = 9.540, JPEG XL (e9) = 8.688 on the same 756x504 crops.

| tag | params | best Kodak bpp | epochs |
|---|---:|---:|---:|
| full_d128_20ep | 824,448 | 9.156 | 20 |
| full_d128_20ep | 824,448 | 9.156 | 20 |
| full_d64_20ep | 215,616 | 9.299 | 9 |
| full_h4_d64_20ep | 215,616 | 9.300 | 20 |
| full_d64 | 215,616 | 11.220 | 3 |
| chan_d64 | 215,616 | 11.245 | 3 |
| ctx_d64 | 215,424 | 12.248 | 3 |
| base_d64 | 215,424 | 12.427 | 3 |

Code: `experiments/scratch_classifier/` in the jeffy repo (`shared_prior_model.py`, `train_shared_prior.py`, `bytelm_codec.py`).
Pushed 2026-10-24 20:19.