# Session handoff (read this first in a new session)

Purpose: let a fresh Claude Code session pick up this research program with as little loss as possible. Keep it short;
the durable record is `RESEARCH_LOG.md` (lab notebook, append-only, every result the turn it lands). Update this file at
the end of every working block (not just at the end of a session). Last update: **2026-10-27, 13:50 PDT**.

## How to resume (5 minutes)
1. Read this file, then the last ~400 lines of `RESEARCH_LOG.md` (`tail -400 RESEARCH_LOG.md`).
2. `git -C /Users/nico/Documents/Projects/jeffy status -sb` — branch `btzsc-benchmark`, must not be `[ahead N]`.
   Commit as `2296756+nicobrenner@users.noreply.github.com` (repo-local `user.email` is set). Repo is PUBLIC: never commit
   `posts/*` drafts before Nico has reviewed them; never commit `IMG_4428.jpeg` or any derivative.
3. `vastai show instances --raw` — if anything is live that this file does not list as running, ask before touching it.
4. Memory (`~/.claude/projects/-Users-nico-Documents-Projects-jeffy/memory/MEMORY.md`) holds the standing rules; the ones that bite:
   v2 lineage only (8.5M TinyStories line is dead); <30 min local signal before any GPU; run `date` before writing a time;
   no regex intent rules in the harness (small classifiers instead); Oracle lines never end with a period; blind picks are
   the evaluation Nico trusts; "Needs you" block at the end of messages that need him.

## Needs Nico (2026-10-27 13:50 PDT)
1. **`demo/review_followthrough.html`** (http://localhost:8765/review_followthrough.html): 18 continuations, score the Oracle's
   second turn 0–5 + notes, export `review_followthrough_scores.json`. His new review format (`review_page.py`): 0–5 per reply,
   notes, continuations first, ≤20 items. Never send 30 first-reply pairs again (he stopped `blind_v5b` at 6: "same as always").
2. FYI decided: v5b vs ct2 is a feel-tie on first replies; the next data pass runs on the pre-LN recipe (reads context, 3× smaller),
   ct2 recipe is the fallback if the pass does not clear "at least as funny" in his scores.
3. Still open: crawler/room verdicts, teach-mode export, README/benchmark correction, posts.

## Running right now (2026-10-27, 13:50 PDT)
- **Nothing running** (MPS idle, no GPU). v5c done (identity 31/32, drift 5/96, recall 26/80; not better than v5b; leakage = recap-reply template shape, not dose). Paused until Nico's picks on `blind_v5b.html` / `blind_tone.html` or his chat verdict on v5b. Demo on 8765 has `v5b` as a labelled candidate; default still ct2.
- **Models that matter:** `best_lm_oracle_v2_grown_ct2.pt` = Nico's pick (demo default; cannot copy from context); `best_lm_oracle_v5b.pt` = first
  model passing identity that reads recaps (demo candidate, `oracle_v5b_int8.onnx`); `ct3` = grader's pick. Trainer flags that matter: `--pre-ln`, `--oracle-only-loss`.

## Done this block (2026-10-27 night; all in RESEARCH_LOG.md)
- **Demo:** knobs (prime off/1st/full, cross-turn toggle, "crawler now"); crawler v5 after Nico's video (long thin legs that
  step onto scanned words, rings/strikethrough/keyword boxes, status block, ramping rate), tripod gait, no traces, no page
  widening; pick-up-and-escape; **The Room** (Oracle remarks on crawler events: seeded turn + model continuation); default
  model = Nico's pick. Nico: "amazing", "love it".
- **"Less funny, more forceful":** blind round 4 (ct2 under the 18:17 harness vs current) → 6/7 both-bad, harness cleared;
  memory line cleared; admit bank recited in ~8% of replies but Nico: "the lines are great" — the issue is delivery
  (witty line fired at friendly/vulnerable turns). ct4 (wide bank) removed recital but went flat → not the fix.
- **Tone labels + judge v1** (`tone_clf.py`, `tone_score.py`); **multi-session recap data** (576 agent-written rows,
  `ms_build.py`, `ms_probe.py`, `copy_probe.py`, `induction_probe{,2,3}.py`).
- **Root cause found:** post-LN blocks induction at d256 → `--pre-ln`; `--rope` implemented (custom block, exporter aware).
- Corpus bug noted: 3,412 training rows carry a training-time system prompt that the model recites ("make up something
  entertaining instead of admitting ignorance") → strip in the next fixer pass. Context roadmap written (rolling recap tier,
  `*recap` act, KV-cache ordering, 512 later).

## Pipeline / roadmap (2026-10-27)
| # | step | status |
|---|---|---|
| 1 | data pass v6 (three-headed): inquiry rows (ask → answer → USE the answer), frame rewrite (warm/neutral openers+kickers, witty bodies kept), recap-reply diversity + strip recited system prompts; then one pre-LN 2L run | in progress |
| 1b | grounding: a tiny queryable world (quantities/units/comparisons/time/Room state) read at inference; Room = seed | idea, logged 13:44 |
| 1c | real-time loop in the demo (perceive → classify → act between prompts; online teach updates) | idea, logged 13:44 |
| 2 | rolling recap tier + `*recap` act (needs a model that copies) | after 1 |
| 3 | KV cache in the ONNX export; then 512 window (grow table or RoPE) | after 2 |
| 4 | tone judge refined on Nico's picks (shares/reactions) → best-of-N scoring; intent gate vetoes dunno on reactions | labels needed |
| 5 | fixer pass: strip recited system prompts; add room-event + visit-memory lines | ready to run |
| 6 | verifier in the browser; classifiers in-browser (same forward pass) | parked |
| 7 | teach-mode corrections into retrains; engagement as reward | waiting on exports |
Parked: posts; benchmark README correction; HF push of new checkpoints; BlurSort.

## Known traps (this session)
- `git push` "declined due to email privacy" → always verify with `git status -sb`.
- Background commands piped through `grep`/`python` buffer until exit → read the log file; use `python3 -u`.
- **Parallel Bash calls share one cwd**: a `cd` in one breaks relative paths in the others → use absolute paths / `git -C`.
- **`str.replace` on a heading that does not exist is a silent no-op** → assert the anchor exists (this file went stale for 7 h).
- Random-word induction probes are far out of distribution for a chat model → use `induction_probe3.py` (real turn repeated).
- A fixed-period copy task is solvable by a positional head; only variable periods test induction (`induct_unit2.py`).
- Rubric grades from different grader instances are not comparable; one blind pass per comparison.
