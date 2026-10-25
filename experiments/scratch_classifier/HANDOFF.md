# Session handoff (read this first in a new session)

Purpose: let a fresh Claude Code session pick up this research program with as little loss as possible. Keep it short;
the durable record is `RESEARCH_LOG.md` (lab notebook, append-only, every result the turn it lands). Update this file at
the end of every working block (not just at the end of a session). Last update: **2026-10-25, ~11:00 PDT**.

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

## Running right now (2026-10-25 11:00)
- **A100 (vast.ai instance 57701871, $0.58/h): `oracle_v7a_conv`** — Tiny Oracle v7a recipe trained to convergence
  (TinyStories 8.5M init → chat_v6plain, 40 epochs, one cosine at lr 3e-4). Pipeline `gpu_oracle_v7a_conv.sh`
  (log in scratchpad `gpu_oracle_v7a_conv.log`; pulls `checkpoints/best_lm_oracle_v7a_conv.pt` every minute; destroys the
  instance on exit). 44 s/epoch → ~30 min. Epoch 3 val 1.3358 (v7a after 8 epochs: 1.3815; +4 control: 1.2923).
- **MPS: `oracle_v7ct2`** — CT v2 fine-tune (4 epochs from `best_lm_oracle_v7a_plus4.pt` on `data/chat_v6ct2_train.jsonl`),
  log `logs/train_oracle_v7ct2.log`. Next: `python3 atl_control2.py eval checkpoints/best_lm_oracle_v7ct2.pt --n 400`
  and `probe`. Baseline (un-tuned base) saved: `pile_runs/atl/ct2_eval_oracle_v7a_plus4.json` (Δ wrong−true = −0.003).
- **CPU: BlurSort video→image on real video** (`blursort_video_to_image.py` on Xiph akiyo/bus/foreman, 256² crops,
  K 4/16/64, stride 1 and 4) → `pile_runs/blursort/video_to_image_{akiyo,bus,foreman}_frames_c256_s{1,4}.json`.

## Where each thread stands
| thread | state | next |
|---|---|---|
| Post 1 "prediction is compression" | draft `posts/01_prediction_is_compression.md` (not committed) | Nico reads → revise |
| Post 2 "Chinchilla BBQ" | figure done, BPE control missing (~$1 A100) | run BPE control, draft |
| Post 3 "free depth" | draft `posts/03_free_depth.md` (not committed) | Nico reads → revise |
| BlurSort | `notes/blursort_polyphase.md` (analysis + prior art + two-scale design §5); frame-order n=4 and video→image done | real-video result → log; two-scale coding experiment (ETA table first); 4th post |
| Tiny Oracle | v7a 8.5M base; ATL negative ×2 (control beats both arms); CT v1 null by design | converged base → T=0.5 probe → one blind pass (base, converged, converged@T0.5) → CT v2 on converged trunk |
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
