# Vast.ai GPU Options for Pythia Comparison Training

Searched 2026-10-21. Budget: $10 credit. Need ~4 hours for full suite (20M+50M+70M+100M).

## Best Budget Options (under $0.20/hr)

| GPU | VRAM | $/hr | Reliability | DL Mbps | RAM | ID |
|-----|------|------|-------------|---------|-----|----|
| Tesla P40 | 24GB | $0.107 | 0.990 | 573 | 64GB | 47185198 |
| Tesla V100 | 32GB | $0.108 | 0.993 | 424 | 48GB | 49022078 |
| Tesla V100 | 32GB | $0.123 | 0.997 | 930 | 40GB | 55562196 |
| RTX 3090 | 24GB | $0.188 | 0.997 | 940 | 64GB | 54912995 |
| RTX A5000 | 24GB | $0.189 | 0.966 | 918 | 32GB | 55997992 |

## Mid-tier (A6000 class, under $0.50/hr)

| GPU | VRAM | $/hr | Reliability | DL Mbps | RAM | ID |
|-----|------|------|-------------|---------|-----|----|
| RTX A6000 | 48GB | $0.401 | 0.986 | 863 | 63GB | 21004756 |
| RTX A6000 | 48GB | $0.402 | 0.993 | 795 | 54GB | 52702108 |
| A100 PCIE | 40GB | $0.455 | 0.802 | 2921 | 79GB | 56227339 |

## Recommendation

**V100 at $0.108-0.123/hr is the sweet spot.** For our models (20M-100M params), a V100 32GB is more than enough VRAM. Full training suite would cost:
- ~4 hrs × $0.12/hr = **~$0.50 total** (well within $10 budget)
- Even if it takes 8 hours: still under $1

The V100 (ID 55562196) at $0.123/hr with 0.997 reliability and 930 Mbps download is the best pick: reliable, fast internet for uploading code/downloading data, and cheap.

**Skip the A100** — our models are small enough that V100 is plenty, and at 4x the price it's not worth it for this experiment.

## Notes
- Tesla P40 is cheapest but older (Pascal), may have slower FP16
- V100 has excellent FP16 (Tensor Cores), great for our PyTorch training
- All models fit comfortably in 24GB VRAM (100M model needs ~2-3GB)
- Budget allows 80+ hours of V100 time at $0.12/hr
