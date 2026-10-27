# Session handoff (read this first in a new session)

Purpose: let a fresh Claude Code session pick up this research program with as little loss as possible. Keep it short;
the durable record is `RESEARCH_LOG.md` (lab notebook, append-only, every result the turn it lands). Update this file at
the end of every working block (not just at the end of a session). Last update: **2026-10-27, 15:30 PDT**.

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

## Needs Nico (2026-10-27 15:30 PDT)
1. FYI the demo garble ("ring kay Todina…"): not reproduced on ct2/v5b int8 with the default sampler. Which dropdown entry and sampler was selected?
2. FYI decided: pre-LN data passes stopped (v6, v6b both lose to v5b; low-diversity row families leak as templates). Next
   model-side work = best-of-N with a judge trained on his 0–5 scores, or retrieval for the recap tier. No run pending.
3. Still open: crawler/room verdicts, teach-mode export, README/benchmark correction, posts.

## Running right now (2026-10-27 15:30 PDT)
- **Nothing running.** Candidates in the demo: ct2 (default, ★), v5b (☆ TEST THIS). v6/v6b exported but NOT in the dropdown.
- Nico's labels: tone picks (28 rows, `tone_labels_nico1.jsonl`) + follow-through scores (18, `review_followthrough_scores.json`).
- Review format: `review_page.py` (0–5 per reply + notes, continuations first, ≤20 items). Pairs on shared history: `cont_pairs.py`.

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
| 1 | data pass v6/v6b: done, both lose to v5b (template leakage from low-diversity rows). Next: judge on Nico's 0–5 scores → best-of-N; recap tier by retrieval | decided 15:30 |
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
