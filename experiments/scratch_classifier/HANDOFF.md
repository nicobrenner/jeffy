# Session handoff (read this first in a new session)

Purpose: let a fresh Claude Code session pick up this research program with as little loss as possible. Keep it short;
the durable record is `RESEARCH_LOG.md` (lab notebook, append-only, every result the turn it lands). Update this file at
the end of every working block (not just at the end of a session). Last update: **2026-10-25, ~14:10 PDT**.

## How to resume (5 minutes)
1. Read this file, then the last ~300 lines of `RESEARCH_LOG.md` (`tail -300 RESEARCH_LOG.md`).
2. `git -C /Users/nico/Documents/Projects/jeffy status -sb` — branch `btzsc-benchmark`, must not be `[ahead N]`.
   Commit as `2296756+nicobrenner@users.noreply.github.com` (repo-local `user.email` is set). Repo is PUBLIC: never commit
   `posts/*` drafts before Nico has reviewed them; never commit `IMG_4428.jpeg` or any derivative.
3. `vastai show instances --raw` — if anything is live that this file does not list as running, ask before touching it.
4. Memory (`~/.claude/projects/-Users-nico-Documents-Projects/memory/MEMORY.md`) holds the standing rules; the ones that bite:
   ETA table (CPU/MPS/A100/H100) before any launch; <30 min local = just run it; <1 h GPU = schedule it; >1 h GPU = explain
   first; checkpoint + `python3 -u` + resume before launching; pull checkpoints during training; destroy the instance in a
   trap; results in tag-specific files, merge never overwrite; WebSearch 200/session cap → Chrome tools / playwright.

## Running right now (2026-10-25 ~21:45 PDT)
- **GPU: two-scale prior `ts_d64` on vast.ai A100 instance 57766003 ($0.67/h, New York)** — `gpu_two_scale.sh` (trap destroys on exit;
  best ckpt pulled every 60 s to `checkpoints/prior_ts_d64_best.pt`; log `logs/train_ts_d64.log`, pipeline log `logs/gpu_two_scale.log`).
  6 epochs × 60K pairs ≈ 35 min → ETA ≈ 22:20 PDT. Target: Kodak phase (1,1) conditioned bpp < frozen unconditioned (≈11.4).
- Grader sub-agent: `probe_persona_blind4.jsonl` (1,052 rows) → `grades_persona_blind4.jsonl` (conv_persona / conv / v2_recipe / v2_t08,
  122-prompt set + 40 first-impression openers). Key `probe_persona_blind4_key.json`.
- Demo server on 127.0.0.1:8765 (dropdown: conv int8 default, conv_persona int8, conv_fix int8, conv f32, v2).

## Done this block (all logged in RESEARCH_LOG.md, same date)
- Nico's two video experiments run as asked: existing AV1/HEVC video vs extracted frames as JXL (image sets 18–29× the video lossless,
  8–15× lossy-transparent); global similarity grouping vs AV1 lossless (beats single-mode image methods on mixed content by 17%,
  still 9–19% behind AV1; giving AV1 the similarity order is worth −5% on returning content).
- `[name]` placeholder bug fixed (`fix_name_placeholders.py`, source: Synthetic-Persona-Chat; 791 human-role-play convs dropped);
  `oracle_v7a_conv_fix` trained on A100 (best ep 32, val 1.2089 vs conv 1.2019 on the same val). **Next: probe conv_fix at the
  deployed recipe → one blind pass conv vs conv_fix → swap the demo ONNX if it holds.** GPU pipeline end-detection bug fixed.
- Conv blind pass unblinded: conv@T0.5 beats v7a 177/45/22; salad 30→15%, off-persona 32→4%; repetition 24% is the new defect.
- **Native-YUV video control (`blursort_video_yuv.py`)**: in yuv420p the inter codecs are 2.1 bpp on static akiyo (half of per-frame/tile JXL),
  interleave+JXL ties them (2.18); under motion AV1 4.19 vs tile+JXL 4.27 vs per-frame 4.46 — the RGB-mode "JXL beats AV1 2×" was an artefact.
- **Lossy video control**: tile+JXL needs 3–17× the bits of AV1 at equal PSNR → video→image is lossless-only; no streaming claim. Closed.
- **Demo updated**: `demo/index.html` runs `oracle_v7a_conv_int8.onnx` (10.8 MB, +0.005 nats vs f32) with the graded recipe; verified headless.
  Serve with `python3 -m http.server 8765 --directory demo`. `.onnx` is gitignored; push to HF still pending. Data bug: 1,372 unfilled `[name]`.
- Sampler blind pass unblinded: no-repeat-3gram + rep-penalty 1.3 at T0.5 kills repetition (29.5% → 0.4%) and salad (6.6 → 1.6%) at zero
  rubric cost (paired 60/113/71); T0.6 loses 0.39. **Deployed recipe: v7a_conv, T0.5, top-k 40, nr3, rp1.3.** coh≥2 unchanged → model is the ceiling.
- `oracle_v7a_conv`: best val 1.2019 @ epoch 29/40; held-out oracle NLL 2.2952 (v7a 2.6049, +4 control 2.4484). Probes at
  T=0.8/0.5: persona-marker 33%/38% (v7a 7%).
- CT v2 (prompt-act token): classifier 93%, consumption null (label derivable from prompt). **CT v3 (response-act plan token):
  consumption real (Δ wrong−true +0.055; +0.12 when the plan is "ask back"), self-prediction 74.5% vs 66.7% majority.**
  Next: blind pass probe_v7ct3 vs plus4 vs forced-tell; equal-compute control plus8; then CT v3 on the converged trunk.
- Real video (Xiph akiyo/foreman/bus) video→image: tile+JXL ≥ per-frame always; interleave wins only static; **AV1 lossless
  ties interleave+JXL on static (4.99 vs 4.87)**; lossless inter codecs in RGB mode collapse under motion. Proper comparison
  must be in native YUV 4:2:0 and (for streaming claims) lossy at matched quality — not done.
- Prior art verified (arXiv/Crossref/Scholar): SReC, iVPF, L3C; image-set compression line incl. Gergel/Cheng/Li DCC'06 and
  Sakurai & Inoue 2021 (animated FLIF for image sets = closest to our test). In `notes/blursort_polyphase.md` §4.
- Web search when WebSearch is capped: see memory `feedback_browser_search.md` (real Chrome binary + copied profile, visible
  window, Scholar/DDG only; never navigate away while a captcha is on screen).

## Where each thread stands
| thread | state | next |
|---|---|---|
| Post 1 "prediction is compression" | draft `posts/01_prediction_is_compression.md` (not committed) | Nico reads → revise |
| Post 2 "Chinchilla BBQ" | figure done, BPE control missing (~$1 A100) | run BPE control, draft |
| Post 3 "free depth" | draft `posts/03_free_depth.md` (not committed) | Nico reads → revise |
| BlurSort | video→image closed: lossless tile ≈ AV1 within 2–10%; lossy 3–17× behind AV1 | two-scale AR coding experiment on stills (ETA table first); 4th post = image-set framing |
| Tiny Oracle | converged 8.5M base @T0.5 is the new reference (3.57 total vs v7a 1.67); ATL negative ×2; CT v3 plan token consumed | sampler done (recipe fixed) → CT v3 blind pass (v7ct3 / plus4 / forced-tell) → CT v3 on converged trunk → bigger base only if still salad |
| Tiered classifier | results done (T3 ordered segments +25 pts) | write-up |
| White paper (MVA) | after posts 1–3 | — |
| HF | checkpoints v7a / ATL arms / v7ct / plus4 / conv + demo ONNX (f32, int8) NOT yet pushed | push private |

## Open questions for Nico (do not decide alone)
- Which Post to publish first; whether BlurSort becomes post 4 or folds into post 1.
- Oracle: go bigger (11M/18M) only if the converged 8.5M is still mostly salad at T=0.5.

## Known traps (this session)
- `git push` "declined due to email privacy" → the earlier pushes had silently failed; always verify with `git status -sb`.
- Background commands piped through `grep`/`python` buffer until exit → read the log file directly; use `python3 -u`.
- Planet frames are `.bmp`, not `.jpg`. Kodak/Pluto are the only publishable images.
- Rubric grades from different grader instances are not comparable; one blind pass per comparison, equal-compute control inside it.
- v1 control tokens were 96% `[answer]` → nothing to learn; any control-token label must carry entropy (v2: 1.5 bits).
