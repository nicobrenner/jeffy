# Session handoff (read this first in a new session)

Purpose: let a fresh Claude Code session pick up this research program with as little loss as possible. Keep it short;
the durable record is `RESEARCH_LOG.md` (lab notebook, append-only, every result the turn it lands). Update this file at
the end of every working block (not just at the end of a session). Last update: **2026-10-28, 08:55 PDT**.

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

## Needs Nico (2026-10-27 15:19 PDT)
0a. **Talk to the lab from the demo:** ✎ lab button (bottom right) opens the sidebar; Cmd+Enter sends, 📎 lab under a turn attaches the note to it. We answer by appending to `pile_runs/lab/replies.jsonl` (`{t, who, text, re}`); the page shows the thread. Seeded turns (weather / gift / visit / room) now have dots too and a "not now" button — score them, that curates the banks.
0. **Chat and score instead of pages:** every Oracle reply in the demo now has faint 0–5 dots + a note; nothing leaves the browser until you press teach-export (adds `chat_scores.json`). Scored chats are the judge's next training set. Decision for you when you feel like it: a per-conversation budget on the two overused lines ("Pretend I said something wise…", "beyond 2 million parameters" — each at most once per conversation, then the family's other 16 lines)? No bank changes made.
0b. **Shipped from your best-of-4 scores:** best-of-4 won 9-1-8 (Wilcoxon p 0.009), so the demo's ☆ entry is now ct2 + best-of-4 by the judge (~4x slower per reply, each reply shows the judge's pick). Try it; the bank-line fatigue you noted from item 12 on is the next thing measured (judge retrain with your 36 new labels + a "seen before in this conversation" feature).
1. FYI the demo garble ("ring kay Todina…"): not reproduced on ct2/v5b int8 with the default sampler. Which dropdown entry and sampler was selected?
2. FYI decided: pre-LN data passes stopped (v6, v6b both lose to v5b; low-diversity row families leak as templates). Next
   model-side work = best-of-N with a judge trained on his 0–5 scores, or retrieval for the recap tier. No run pending.
3. Still open: crawler/room verdicts, teach-mode export, README/benchmark correction, posts.

## Running right now (2026-10-28, 06:50 PDT)
- **Nothing running.** Copy-circuit scout done: five configs ≤ 2.8M, 8 epochs each, 0/16 copies everywhere, no induction above chance — **copy ceiling at this size is the finding**; grounding / recap-by-reading / weather-reading parked. No retrain.
- Nico's labels: tone picks (28 rows, `tone_labels_nico1.jsonl`) + follow-through scores (18, `review_followthrough_scores.json`).
- Review format: `review_page.py` (0–5 per reply + notes, continuations first, ≤20 items). Pairs on shared history: `cont_pairs.py`.

## Branch 2 (Jeffy product) — pointer 2026-10-27, 20:15 PDT
Separate worktree `/Users/nico/Documents/Projects/jeffy-main`, branch `branch2/calibration` (commits 9167e96, 5bdc4e5: calibration table,
probe sets, README "Confidence and abstaining" with 0.9 default). Record = `JEFFY_BRANCH.md` in the paper repo clone, local, NOT pushed
(push needs Nico). Evidence = session scratchpad `jeffy_evidence/01–20`. Open: reconcile item 1 with the reviewer's production-host
figures (our file = earlier host only); medical heads are accurate but under-confident → margin gate. Rule: read the log, not the
counter; print window boundaries next to every count. Branch 2 never touches this directory or MPS.

## Creatures — the world as the Oracle's own environment (design, 2026-10-28, 08:27)
**Shipped:** Crawly (strider; fetches words, now returns a RELATED word via `demo/assoc_v1.json` and a gift Room event), the tokenizer
(lives at the input box, chops sent words into the real token pieces, hands pieces to Crawly, "tokens" Room event), presence (visit record,
"…" then a first-visit / welcome-back turn), weather events. Banks: `weather_bank.py` (weather, presence), `room_build.py` (Room events).
**Third creature, parked:** the sampler as a moody roommate — mood = temperature; visible as a small figure near the model badge whose
posture follows the sampler preset; its Room events ("the roommate turned the heat up") when the preset changes. Build after Crawly's policy.
**Crawly as a pet model (adopted design, prior art in RESEARCH_LOG 2026-10-28, 08:27):**
- State: drives d = (hunger-for-words, curiosity, fatigue, affection) ∈ [0,1]^4, decaying/accumulating per tick; plus what it sees
  (visible words with the detector's labels, user activity, Oracle turn events, tab visibility).
- Policy: ~10k-param MLP/GRU over 6 options — wander, fetch-word, bring-to-user, bring-to-Oracle, nap, nudge (the current gait code is the
  option implementation; options framework = the formal home). FeUdal split: the Oracle emits a one-token goal per turn (e.g. *fetch /
  *rest / *show), the policy decides the primitive sequence; deliberate non-compliance (Trico) is allowed by the policy, not scripted.
- Reward (offline training from logs): drive satisfaction + engagement proxies the page already measures — hover/click on Crawly, the user
  mentioning Crawly in chat, the Oracle acknowledging it (Room turn), turns per conversation.
- Measurement: Godspeed Animacy subscale + Chesney & Lawson companionship items (a 6-item page for Nico, once), and behaviourally
  unprompted mentions of Crawly per session from the chat export.
- **First < 30 min signal:** logged-policy baseline — add the drive vector and log (drives, action, engagement proxies) per tick in the demo
  with the current hand policy; no learning yet; the log is the first dataset. Then a behaviour-cloning + reward-weighted policy offline.

## Branch 1 (Tiny Oracle) — plan 2026-10-27 16:45 PDT, updated 20:25
Nico split the work into two branches (memory `project_two_branches_manager_mode`); this is Branch 1's block plan.
**Status 20:25:** item 1 done and shipped (judge AUC 0.76 on his rows; best-of-4 won his blind scores 9-1-8, p 0.009; demo ☆ default).
**Status 20:31:** judge v2 done (LOO AUC 0.83 vs v1 0.79; demo uses it). Diversity pick: no-op — exact recital is 0/30 at turns 2 and 3 in
every arm because the cross-turn 3-gram ban already blocks it; what reads as canned is bank-line *variants* and the same line across
conversations (13/30 turn-2 picks). `review_fatigue.html` built (single vs best-of-4 at turn 3; +diversity was identical 30/30).
**Status 20:38:** family classifier done (`family_clf.py`; 0.66 acc on unseen lines, knows its own lines 41/41, weak on day/how);
cross-family pick in the harness (cfg `family`), fires 1/30 — the judge's 0.99 bank lines rarely have a rival within the margin.
Turn-3 collapse diagnosed: the judge marks down anything read behind a 2-turn history (same replies 0.40 → 0.62 with no history),
and the window is the rest (ct2 +0.11 from last-turn-only history); prime no help, v5b no better. Not the model → no DAgger.
**Status 20:54:** history knob shipped (default full). ct5 fixer pass: training-prompt recital 8/90 → 0, recap 10% → 12%, but judge
mean below ct2 at every turn (single 0.33/0.32/0.17 vs 0.14/0.18/0.09) — same apologetic drift as ct4; not exported.
**Status 21:11:** ct6 = ct5 with prompts varied: t2/t3 and apology count recover, t1 still loses (0.17 vs 0.33), bank recital worse (6 vs 3) → not
exported; **data passes stopped until the judge has a target that is not ct2's own habits** (his chat scores). Judge v3 (history-
matched rows) loses to v2 everywhere; both below chance on his 22 turn-3 rows. In-chat scoring shipped (`chat_scores.json` via teach export).
**Status 21:18:** line budget shipped (0→0 with the cross-turn ban on, 1→0 with it off). `world.py` grounding tier: (b) facts work for v5b,
(c) comparisons table does not (retrieval picks topical-not-semantic lines; models do not read them) → no knob. World knob shipped 21:38 (facts, pre-LN only). v5g (3-epoch continuation with 2,040 grounding rows, probe facts held out): grounding 4 → 5/26,
gold-recap recall 26 → 34 (reads harder) but identity 32 → 28, human-life 0 → 10/96, extracted recap 13 → 8 → not exported.
v5h ran (22:22, Nico's call): from scratch, 20 epochs, grounding rows in, probe facts held out → grounding 2/26 (v5b 4), other-slot 1/16,
distractor admits 0/10, identity 13/32. copy_probe: v5b 1/16 names, 0 pets, 0 cities — **no copy circuit**. **Grounding parked.** Copy scout (2026-10-28, 06:50): base / RoPE / deep d192 L3 / 8 heads / attention-only block 0, all 0/16 on copy_probe, repeat loss
never below chance on `induct_eval.py` — the copy ceiling at ≤ 2.8M, 2–3 layers, 8 epochs is the finding; the ct lineage stops here on
reading tasks. **Curriculum scout (2026-10-28, 08:22):** repeated real-token spans + echo rows at 30% → 5% over 8 epochs on RoPE pre-LN gives repeat loss 4.72 / 6.42 / 7.56
at p3 / p5 / p7 (chance 8.32) — the first induction signature in this class; four plain epochs erode it. copy_probe unchanged (0–1/16).
**Next model block (when chat scores arrive):** full v5b recipe with the curriculum kept at ~10% all the way + grounding rows; gate = copy_probe ≥ 8/16. Environment tier shipped 22:28: weather line from browser signals + idle check-in through the Room machinery; `weather_bank.py` is the file
for Nico to curate. v5w (700 weather rows, continuation) fails the probe (5/30, accidental) and loosens identity → weather default off.
**For Nico:** try world = weather on v5b with the idle knob at 1 min; the lines come from `weather_bank.py` — edit freely, the demo reads
`demo/weather_bank.json` (re-export with `python3 -c "import json;from weather_bank import BANK;json.dump(BANK,open('demo/weather_bank.json','w'))"`).
**Next block (signal from chat, no pages):** (a) `nico_judge.py` reads `chat_scores.json` as a set (`chat`) and retrains when ≥ 30 rows arrive;
(b) propose the per-conversation family budget (above) and, if he says yes, implement it as a harness/demo pick rule on the harness's own
record (no regex) and measure overused-line picks before/after; (c) engagement export: turns per conversation on the ☆ entry vs ct2 plain.
**Superseded:** (a) judge v3 with his fatigue-page scores (first 2–3-turn rows he has labelled) + history-length-matched training
rows so the judge is honest at depth; (b) demo knob: history window last-turn-only for ct2, measured on a fresh ≤12-item 3-turn page;
(c) widen the family margin to 0.3 only if his fatigue notes say canned — otherwise leave it; (d) ct6 = ct5 with the 3,294 recited
prompts VARIED instead of dropped (one line in `prep_ct5.py`, 8-min signal, same gates). Then items 2 and 3 below.
1. **Top task (now): Nico-judge v1** — a lands/quality judge on ct2's hidden states trained on the 659 LLM "lands" labels plus
   Nico's own 46 rows (18 follow-through scores, 28 tone picks), evaluated leave-one-out on Nico's rows (AUC vs his ≥3 / "lands").
   Why first: <30-min signal, uses the only ground truth we trust, and it is the gate for best-of-N (roadmap #4). If AUC ≥ 0.7:
   wire it as the best-of-N scorer in `harness.py` (`judge` cfg) and build `review_bestof.html` (≤18 continuations, A = single
   sample, B = best-of-4 by the judge) for Nico. If < 0.7: report, and the next labels page targets the judge's confusions.
2. Next: intent gate from the `move` labels (share/reaction/greeting/ask) on hidden states → veto `dunno`/assertive lines on
   reactions and shares inside best-of-N scoring (no regex).
3. Then: recap tier by retrieval — the harness keeps user-stated facts and writes the one relevant line into the system
   prompt (design + probe with v5b, the only model that reads it); needs a fact-extractor classifier, not a regex.

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
| 1 | data pass v6/v6b: done, both lose to v5b (template leakage from low-diversity rows). Next: judge on Nico's 0–5 scores → best-of-N; recap tier by retrieval | decided 15:19 |
| 1b | grounding: a tiny queryable world (quantities/units/comparisons/time/Room state) read at inference; Room = seed | built as `world.py` (21:17): fact extractor + recap line helps v5b (recap 10→13/80, t3 0.15→0.24); comparisons table by retrieval grounds nothing (0 real hits) — needs training rows with table lines, not retrieval alone |
| 1c (done 22:30: idle check-in once per absence, never hidden; absence weather on return) | real-time loop in the demo (perceive → classify → act between prompts; online teach updates) | idea, logged 13:44 |
| 2 | rolling recap tier + `*recap` act (needs a model that copies) | after 1 |
| 3 | KV cache in the ONNX export; then 512 window (grow table or RoPE) | after 2 |
| 4 | tone judge refined on Nico's picks (shares/reactions) → best-of-N scoring; intent gate vetoes dunno on reactions | labels needed |
| 5 | fixer pass: strip recited system prompts; add room-event + visit-memory lines | ct5 (strip) and ct6 (vary) both fix recital, both lose t1 to ct2 on the judge → **ct data passes exhausted** until chat scores retarget the judge |
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
