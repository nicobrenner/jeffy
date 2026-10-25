# Pythia Model Suite — Reference for Jeffy Scaling Comparison

## Architecture Configs

All Pythia models use GPT-NeoX architecture: rotary PE (25%), parallel residual, GELU, untied embeddings, Flash Attention.

| Model | Params (total) | Non-embed params | d_model | n_layers | n_heads | d_ff | Learning Rate |
|-------|---------------|-----------------|---------|----------|---------|------|---------------|
| Pythia-14M | 14,067,712 | 1,189,888 | 128 | 6 | 4 | 512 | 1.0e-3 |
| Pythia-31M | ~31,000,000 | — | 256 | 6 | 8 | 1024 | 1.0e-3 |
| Pythia-70M | 70,426,624 | 18,915,328 | 512 | 6 | 8 | 2048 | 1.0e-3 |
| Pythia-160M | 162,322,944 | 85,056,000 | 768 | 12 | 12 | 3072 | 6.0e-4 |
| Pythia-410M | 405,334,016 | 302,311,424 | 1024 | 24 | 16 | 4096 | 3.0e-4 |

Common: vocab_size=50304, max_seq_len=2048, rotary_pct=0.25, hidden_act=gelu, tie_word_embeddings=false, use_parallel_residual=true

## Training Setup

- **Dataset**: The Pile (825 GiB, English, 22 sources) — non-deduplicated version
- **Tokenizer**: GPT-NeoX-20B tokenizer, vocab_size = 50,304
- **Batch size**: 2,097,152 tokens (2M) per step — 1024 sequences × 2048 tokens
- **Total steps**: 143,000
- **Total tokens**: 299,892,736,000 (~300B) — just under 1 epoch of The Pile
- **Optimizer**: Adam (presumably with standard betas)
- **Precision**: float16
- **Key detail**: ALL models trained on exact same data in exact same order

## Checkpoint Schedule

- **Regular**: every 1,000 steps from step 1,000 to step 143,000 (143 checkpoints)
- **Early**: steps 0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 (11 checkpoints)
- **Total**: 154 checkpoints per model
- Hosted on HuggingFace: `EleutherAI/pythia-{size}` with `revision="step{N}"`

## Step-to-Token Mapping

tokens_seen = step × 2,097,152

| Step | Tokens | Note |
|------|--------|------|
| 512 | 1.07B | Early checkpoint, closest to 1B |
| 1,000 | 2.10B | First regular checkpoint |
| 3,000 | 6.29B | |
| 5,000 | 10.49B | |
| 10,000 | 20.97B | |
| 13,000 | 27.26B | |
| 23,000 | 48.24B | |
| 43,000 | 90.18B | |
| 143,000 | 299.89B | Final (full training) |

## Tok/Param Ratios

| Model | At full training (300B) | At step 1000 (2.1B) | At step 3000 (6.3B) |
|-------|------------------------|---------------------|---------------------|
| Pythia-14M | 21,318 | 149 | 447 |
| Pythia-31M | 9,677 | 68 | 203 |
| Pythia-70M | 4,258 | 30 | 89 |
| Pythia-160M | 1,848 | 13 | 39 |
| Pythia-410M | 740 | 5.2 | 15.5 |

## Published Eval Results (LAMBADA OpenAI — Zero-Shot)

LAMBADA perplexity (lower = better) and accuracy (higher = better):

### Pythia-70M (70.4M params)
| Step | Tokens | tok/param | LAMBADA ppl | LAMBADA acc | PIQA acc |
|------|--------|-----------|-------------|-------------|----------|
| 512 | 1.07B | 15.2 | 116,756 | 0.0000 | 0.5386 |
| 1000 | 2.10B | 29.8 | 4,465 | 0.0501 | 0.5604 |
| 3000 | 6.29B | 89.3 | 411.7 | 0.1213 | 0.5794 |
| 13000 | 27.26B | 387 | 136.2 | 0.2104 | 0.5865 |
| 23000 | 48.24B | 685 | 121.3 | 0.2226 | 0.6077 |
| 43000 | 90.18B | 1,280 | 112.4 | 0.2282 | 0.5903 |
| 143000 | 299.89B | 4,258 | 142.4 | 0.1846 | 0.5947 |

### Pythia-160M (162.3M params)
| Step | Tokens | tok/param | LAMBADA ppl | LAMBADA acc | PIQA acc |
|------|--------|-----------|-------------|-------------|----------|
| 512 | 1.07B | 6.6 | 100,126 | 0.0000 | 0.5326 |
| 1000 | 2.10B | 12.9 | 2,591 | 0.0646 | 0.5555 |
| 3000 | 6.29B | 38.7 | 135.0 | 0.2276 | 0.5930 |
| 13000 | 27.26B | 168 | 49.7 | 0.2998 | 0.6132 |
| 23000 | 48.24B | 297 | 36.3 | 0.3394 | 0.6208 |
| 43000 | 90.18B | 556 | 32.1 | 0.3511 | 0.6355 |
| 143000 | 299.89B | 1,848 | 38.1 | 0.3284 | 0.6273 |

### Pythia-410M (405.3M params)
| Step | Tokens | tok/param | LAMBADA ppl | LAMBADA acc | PIQA acc |
|------|--------|-----------|-------------|-------------|----------|
| 512 | 1.07B | 2.6 | 210,165 | 0.0000 | 0.5462 |
| 1000 | 2.10B | 5.2 | 4,770 | 0.0382 | 0.5501 |
| 3000 | 6.29B | 15.5 | 87.3 | 0.2630 | 0.5996 |
| 13000 | 27.26B | 67.3 | 20.9 | 0.4161 | 0.6289 |
| 23000 | 48.24B | 119 | 17.0 | 0.4324 | 0.6322 |
| 43000 | 90.18B | 223 | 15.0 | 0.4624 | 0.6496 |
| 143000 | 299.89B | 740 | 10.8 | 0.5162 | 0.6676 |

### Summary at step 1000 (2.1B tokens) — our target comparison point
| Model | Params | tok/param | LAMBADA ppl | PIQA acc | ARC-easy acc |
|-------|--------|-----------|-------------|----------|--------------|
| Pythia-70M | 70.4M | 29.8 | 4,465 | 0.5604 | 0.3085 |
| Pythia-160M | 162.3M | 12.9 | 2,591 | 0.5555 | 0.3064 |
| Pythia-410M | 405.3M | 5.2 | 4,770 | 0.5501 | 0.2934 |

### Summary at step 3000 (6.3B tokens) — alternative comparison
| Model | Params | tok/param | LAMBADA ppl | PIQA acc | ARC-easy acc |
|-------|--------|-----------|-------------|----------|--------------|
| Pythia-70M | 70.4M | 89.3 | 411.7 | 0.5794 | — |
| Pythia-160M | 162.3M | 38.7 | 135.0 | 0.5930 | — |
| Pythia-410M | 405.3M | 15.5 | 87.3 | 0.5996 | — |

## Key Observations for Our Comparison

1. **At step 1000 (2.1B tokens)**: Pythia-70M (30 tok/param) and Pythia-410M (5.2 tok/param) have nearly identical LAMBADA ppl (~4500 vs ~4800). The 410M is 5.8x larger but not better — it's massively undertrained at 5.2 tok/param.

2. **At step 3000 (6.3B tokens)**: Bigger models pull ahead more. Pythia-410M (15.5 tok/param) has ppl 87 vs 70M's 412 (89 tok/param). At 15+ tok/param, even the bigger model starts learning.

3. **Pythia-70M peaks around step 23000-43000 on LAMBADA** (ppl ~112-121), then *gets worse* at 143000 (ppl 142). Possible overfitting to The Pile distribution or LAMBADA metric noise.

4. **No 14M or 31M eval data in the published eval suite** — only 70M and above. We'd need to run our own evals on those checkpoints.

5. **Our comparison strategy**: Train a 20M model on The Pile for ~2B tokens (step 1000 equivalent). At 100 tok/param, compare against Pythia-70M at same token count (30 tok/param). If our 20M model achieves similar LAMBADA/PIQA scores to Pythia-70M, we've shown a 3.5x parameter reduction.

## Architecture Differences (Pythia vs Our Models)

| Feature | Pythia (GPT-NeoX) | Our Models (vanilla Transformer) |
|---------|-------------------|----------------------------------|
| Position encoding | Rotary (RoPE, 25%) | Learned absolute |
| Residual | Parallel | Sequential (post-norm) |
| Activation | GELU | GELU |
| Embeddings | Untied | Tied (weight sharing) |
| Attention bias | Yes | Default |
| Vocab size | 50,304 | 4,096 (custom BPE) |
| Sequence length | 2,048 | 256 |

**Important**: To make a fair comparison, we should either:
- (a) Match Pythia's architecture exactly and use their tokenizer, or
- (b) Accept architecture differences and compare on the same eval benchmarks with same data

Option (a) is cleaner for the paper. Option (b) is easier to implement.

## Hugging Face Access

Models: `EleutherAI/pythia-{14m,31m,70m,160m,410m}`
Checkpoints: `revision="step{N}"` (e.g., `revision="step1000"`)
Tokenizer: `EleutherAI/gpt-neox-20b` (vocab_size=50304)
Dataset: `EleutherAI/the_pile` or pre-tokenized `EleutherAI/pile-standard-pythia-preshuffled`
