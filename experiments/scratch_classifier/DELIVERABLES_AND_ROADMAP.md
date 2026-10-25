# Jeffy Research: Deliverables & Roadmap

Written 2026-10-24. A review of where the research record stands, what is ready to ship, what is one experiment away, and what comes next.

## A. State of the record

Two research logs. The internal log was kept current (written 500+ times) through 2026-10-23 09:41, then lagged ~36 hours while results went only to memory notes and JSONs. **Both were backfilled on 2026-10-24** with a dated entry; the standing rule is now that results are logged the turn they land.

| Log | Audience | Current through |
|-----|----------|-----------------|
| `jeffy/RESEARCH_LOG.md` | Public, dated entries | 2026-10-24 (compression, depth, v5) |
| `experiments/scratch_classifier/RESEARCH_LOG.md` | Internal lab notebook, all figures | 2026-10-24 |

The 2026-10-23/24 entry covers:

- 2D RoPE on MNIST (causal and bidirectional, d=64, 215K params)
- The bidirectional PPL 1.001 trap (information leak, not learning) — `exploit_ppl_results.json`
- 8.7x lossless MNIST compression via arithmetic coding — `compression_results.json`
- MNIST classification from loss patterns / hidden states — `mnist_classification_results.json`
- Color photo compression: model-overhead analysis, tiny shared model at PNG parity (622 KB vs 610 KB)
- BlurSort 1D pixel ordering: raster wins by 40% (negative result; 2D decomposition untested)
- Depth vs width at 2.6M params, full curve — `oracle_depth_test_full.json`
- Oracle v5 (2L x 2 iterations) — training 2026-10-24

**Repo hygiene:** `experiments/` (13 GB, 105 `.pt` files, 54 scripts) is entirely untracked. `.gitignore` does not cover `*.pt`, `pile_runs/`, `checkpoints/`, `*.mlpackage`, or `oracle_v4_checkpoints/`. Nothing from the scratch work is in version control.

**First tidy-up task:** backfill both logs with a single dated 2026-10-23/24 entry covering the list above (sources: memory notes + JSONs; ~1 page), then commit `experiments/` with a proper `.gitignore`.

## B. Deliverable packages

Each package = one post + the code that reproduces it + a results file. Ranked by readiness.

### 1. Post: "Prediction is compression" — READY, and the centrepiece changed (2026-10-24 evening)

**New headline:** a shared byte-LM prior (215K params, **216 KB int8 decoder**, no image-specific architecture, trained 34 min on one A100 for $0.60 on 667 CLIC images) losslessly compresses **Kodak-24 to 9.33 bpp — 31% below PNG, 2% below WebP-lossless, on every image**; JPEG XL is 7% smaller. d128 (1.6 MB f16) reaches 9.16. A working codec (`bytelm_codec.py`) round-trips real files **bit-exact** (700×700 photo: 380,724 B vs PNG 603,774). Honest limits in the post: JXL still wins; at 12 MP the 28-px context loses to WebP/JXL (context is the bottleneck); decode is ~10 s per tile-wavefront on MPS. Figures: codec table on identical crops, training curves with fits (and their fragility), bits heatmaps, bpp-vs-decoder-size (33K / 215K / 824K). The MNIST 8.7× result becomes the opening act.

*(original plan below)*

**Claim:** A 215K-parameter causal byte model + arithmetic coding losslessly compresses MNIST to 0.92 bpp (8.7x), beating PNG (2.73 bpp) 3x and gzip (2.00 bpp) 2.2x. PixelCNN (0.79) and Flow++ (0.69) do better with 100-1000x more parameters. Coder output verified against cross-entropy within ~4 bits/image.

**Coda (same architecture, zero changes):** on a 700x700 color photo a 33K-param shared model codes to 548 KB (real arithmetic coder, decode-verified, within 0.8 KB of the CE estimate). COIN-style (ship the 65 KB model with the image) totals 613 KB vs 590 KB optimized PNG — loses by 4%; model bytes are a fixed cost. **But the same frozen weights beat optimized PNG on images they never saw:** NASA Pluto 0.98x, generic RGB sample 0.97x (`eval_shared_prior.py`). A prior trained on one photo already transfers — the shared-prior direction is now evidence-backed, not speculative.

**Sidebar:** the bidirectional PPL 1.001 trap — near-perfect perplexity through attention routing, useless for compression or classification. A cautionary result on reading perplexity.

**Have:** `train_rope_byte_2d.py`, `compression_demo.py`, `exploit_ppl.py`, `eval_color_compression.py`, `reconstruct_color_compression.py`, all result JSONs, per-digit table.
**Need:** three figures (bpp bar chart vs PNG/gzip/PixelCNN; per-digit bpp; model-overhead vs image-size curve), one CC0 demo photo (the family photo must not be published), a 1-paragraph arithmetic-coding explainer. ~1 day of writing.

### 2. Post: "Chinchilla BBQ: byte-level models don't overfit" — MOSTLY READY

**Claim:** Byte-level (vocab 256) RoPE transformers at d=32/64/128 keep improving to 7,200 tok/param (360x Chinchilla) on text, and the same architecture on MNIST pixels sets its all-time best at 4,860 t/p after 30 epochs. Final PPL equals best PPL; no overfitting. Staircase descent pattern in both modalities.

**Framing from the scratch log:** the vocabulary tax (96.5% of a small BPE model is lookup tables) — bytes convert embedding parameters into compute parameters.

**Have:** `train_rope_byte.py`, runs `run_d{32,64,128}_tp{1800,7200}`, `mnist_d32_tp{1800,7200}`, scratch-log sections 3355-3713.
**Need:** one figure (PPL vs tok/param, text and MNIST on shared axes). **Gap to disclose:** no matched BPE control at the same t/p. Oracle v4 (BPE-4096, 100 epochs on 5.4K conversations) overfitting is the anecdotal contrast, not a controlled one. State it plainly or run the control (~$1 A100).

### 3. Post: "Depth recurrence: free depth for tiny models" — SEEDS DONE (2026-10-24), claim calibrated

4 seeds each: baseline 1.6700±0.0034, 4L_d192 1.6342±0.0013, 2L×2iter 1.6291±0.0048. Recurrence ahead of real 4L on 3/4 seeds (mean −0.005, t≈1.7 — suggestive, not significant); baseline vs deeper >10σ. Publishable claim: **"two layers run twice match or slightly beat four real layers at the same budget — free depth."** Oracle v5 side-by-side is the honest coda (val_loss gain ≠ quality gain on a memorization-bound set).

*(original status below)*

**Claim:** At a fixed ~2.6M budget, 20 epochs on A100: 2L baseline 1.6686 → 4L_d192 1.6328 → 6L_d160 1.6373 → **2L x 2 iterations 1.6295**. Same parameters as baseline, iterate the layers, beat every real-depth variant. Backed by the earlier classifier result (1x 83.0% / 2x 83.9% / 4x 82.7%) and the inference-time depth-scaling experiment already in the scratch log.

**Payoff demo:** Oracle v5 vs v2 side-by-side on 20 novel prompts (`eval_oracle_side_by_side.py`), judged by a human — the v4 lesson that val_loss does not predict quality.

**Have:** `test_oracle_depth.py`, `oracle_depth_test_full.json`, `depth_vs_width_findings.md`.
**Need:** 3 seeds x {baseline, 4L_d192, 2L_x2iter} at 20 epochs (~60 min A100, ~$1.20). The top three are within 0.008 on one seed; without seeds the honest claim shrinks to "2 layers is too few." With seeds it becomes the headline. Plus the v5 human eval.

### 4. White paper: "The Minimum Viable Agent" — SYNTHESIS (after 1-3)

The spine is the MVA thesis (small orchestrator + tools/RAG, not a knowledge store). The evidence chain:

1. Vocabulary tax → go byte-level (scratch log 271-390, 3355-3410)
2. Byte-level overtraining resistance → train small, train long (Post 2)
3. Depth recurrence → compute depth without parameter depth; adaptive compute for an orchestrator (Post 3)
4. One byte model, many capabilities: compression 8.7x, OOD detection 99.96% AUROC, classification 92-95% from loss patterns (Post 1 + `mnist_classification_results.json`)
5. Scaling evidence at 20M/100M vs Pythia @ 10B tokens (root log 2026-10-22)
6. Tiny Oracle as the end-to-end demo (browser, ONNX, 2.7M params)

Format: GitHub README + long-form markdown + PDF, same shape as the frozen-embeddings paper in Jeffy Hub. Target: mid-November, after Posts 1-3 exist as sections.

### 5. Code package: `experiments/rope_byte/` — NEEDS CURATION

Move the keepers into one clean directory with a README and `reproduce.sh`:
`train_rope_byte.py`, `train_rope_byte_2d.py`, `compression_demo.py`, `exploit_ppl.py`, `test_oracle_depth.py`, `eval_oracle_side_by_side.py`, `eval_color_compression.py`, `reconstruct_color_compression.py`, `test_blursort_ordering.py` + result JSONs + `depth_vs_width_findings.md`.
Weights to Hugging Face (`nicobrenner/...`), not git. Add `.gitignore` entries: `*.pt`, `*.mlpackage/`, `pile_runs/`, `checkpoints/`, `oracle_v4_checkpoints/`, `data/*.jsonl`.

### Already shipped (for the record)

Frozen Multilingual Embeddings paper (`Jeffy Hub/paper.pdf`), Jeffy classifier library + playground, BTZSC benchmark (#1 result), Tiny Oracle v2 browser demo, n8n community node.

## C. What to run in parallel right now

| Where | What | Time | Why |
|-------|------|------|-----|
| **A100** (second instance, or chain after v5) | Seeds: 3 x {baseline, 4L_d192, 2L_x2iter}, 20 ep | ~60 min, ~$1.20 | Turns Post 3 from anecdote into claim. Highest value per dollar on the list. |
| **MPS** (idle now) | Retrain shared model D *with checkpoint saved*, then (a) arithmetic-code the real bitstream to confirm 622 KB, (b) evaluate D's cross-entropy on a *different* photo with no retraining | 22 min + minutes | (a) closes the "estimated vs actual" gap for Post 1; (b) is the first test of the shared-prior idea — if a 33K prior transfers, per-image model overhead disappears |
| **CPU** | Figures for Posts 1 and 2 from existing JSONs | ~1 h | Zero compute, unblocks writing |
| **A100** (later, $1 each) | BPE-4096 control at matched t/p for Post 2 | ~30 min | Closes the disclosed gap in Post 2 |

Not now: BlurSort 2D decomposition (needs a design pass first — 1D reordering was the wrong test), FFT-RoPE, adaptive teacher, classifier-pretraining regularization, FineWeb-Edu vs Pile, 20M → 50B. All real, all after the three posts.

## D. Roadmap

**Week of 2026-10-27 — tidy and lock results**
- Backfill both research logs; `.gitignore` + commit `experiments/`; weights to HF
- Seeds run; v5 vs v2 human eval; deploy v5 to the browser demo if it wins
- Post 1 drafted (figures + text)

**Week of 2026-11-03 — posts**
- Post 2 figure + draft (with the BPE-control gap stated or closed)
- Post 3 draft with seeds
- Shared-prior compression result folded into Post 1 if positive

**Mid-November — white paper**
- Assemble Posts 1-3 + Pythia scaling + MVA architecture figure (exists in Jeffy Hub) into the long-form paper + PDF

**Backlog (ordered)**
1. Shared-prior compression at scale (e.g. ImageNet-64 subset): one 200K model, many images
2. BlurSort 2D decomposition into self-similar frames ("pixel movies")
3. Classifier pretraining as overtraining regularizer (20M, $1-2)
4. FineWeb-Edu vs raw Pile at 20M
5. Extreme overtraining 20M → 50B tokens
6. Adaptive teacher-student training (novel; no exact prior art found)
7. FFT coefficients as extra RoPE dimensions
8. PyTorch upstream: RoPE in `nn.TransformerEncoderLayer` (after paper)
9. Byte-level EN/AR encoder for the care-intake classifier

## E. Standing rules now baked into the scripts

Checkpoint on every best-loss improvement (fixed today in `test_color_compression_variants.py`, `reconstruct_color_compression.py`, `test_blursort_ordering.py`); unbuffered output; ETA table (CPU/MPS/A100/H100) before any launch; pull checkpoints during training, not after; destroy the GPU instance in a trap, not at the end of the happy path.
