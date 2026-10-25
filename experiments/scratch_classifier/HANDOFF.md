# Session handoff (read this first in a new session)

Purpose: let a fresh Claude Code session pick up this research program with as little loss as possible. Keep it short;
the durable record is `RESEARCH_LOG.md` (lab notebook, append-only, every result the turn it lands). Update this file at
the end of every working block (not just at the end of a session). Last update: **2026-10-25, ~12:30 PDT**.

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

## Running right now (2026-10-25 ~12:30 PDT)
- **Blind grading pass, sampler levers** (sub-agent, one grader): `pile_runs/atl/probe_sampler_blind4.jsonl` (976 rows) →
  `grades_sampler_blind4.jsonl`; key `probe_sampler_blind4_key.json` = 4 arms of the converged base {T0.5; T0.5+no-repeat-3gram;
  T0.5+nr3+rep-penalty 1.3; T0.6+nr3+rp1.3}. If incomplete, re-launch the grader (same rubric as RESEARCH_LOG "sampler levers").
- Nothing on GPU. Nothing on MPS.

## Done this block (all logged in RESEARCH_LOG.md, same date)
- Conv blind pass unblinded: conv@T0.5 beats v7a 177/45/22; salad 30→15%, off-persona 32→4%; repetition 24% is the new defect.
- **Native-YUV video control (`blursort_video_yuv.py`)**: in yuv420p the inter codecs are 2.1 bpp on static akiyo (half of per-frame/tile JXL),
  interleave+JXL ties them (2.18); under motion AV1 4.19 vs tile+JXL 4.27 vs per-frame 4.46 — the RGB-mode "JXL beats AV1 2×" was an artefact.
- Sampler: `atl_probe.py --rep_penalty/--no_repeat_ngram/--top_k`; auto-repetition 46% → 4% with nr3+rp1.3 at T0.5; blind pass running.
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
| BlurSort | video→image done in RGB and native YUV (tile ≈ AV1 lossless within 2–10% under motion; interleave ties only static) | lossy RD curve vs AV1 (only if Nico wants the video claim); two-scale coding experiment (ETA table first); 4th post |
| Tiny Oracle | converged 8.5M base @T0.5 is the new reference (3.57 total vs v7a 1.67); ATL negative ×2; CT v3 plan token consumed | sampler blind pass (running) → CT v3 blind pass (v7ct3 / plus4 / forced-tell) → CT v3 on converged trunk → bigger base only if still salad |
| Tiered classifier | results done (T3 ordered segments +25 pts) | write-up |
| White paper (MVA) | after posts 1–3 | — |
| HF | checkpoints v7a / ATL arms / v7ct / plus4 / conv NOT yet pushed | push private |

## Open questions for Nico (do not decide alone)
- Which Post to publish first; whether BlurSort becomes post 4 or folds into post 1.
- Oracle: go bigger (11M/18M) only if the converged 8.5M is still mostly salad at T=0.5.

## Known traps (this session)
- `git push` "declined due to email privacy" → the earlier pushes had silently failed; always verify with `git status -sb`.
- Background commands piped through `grep`/`python` buffer until exit → read the log file directly; use `python3 -u`.
- Planet frames are `.bmp`, not `.jpg`. Kodak/Pluto are the only publishable images.
- Rubric grades from different grader instances are not comparable; one blind pass per comparison, equal-compute control inside it.
- v1 control tokens were 96% `[answer]` → nothing to learn; any control-token label must carry entropy (v2: 1.5 bits).
