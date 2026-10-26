"""Controller loop probe (Nico's design): sample → verifier scores → if below threshold, force the admit act (*dunno) and regenerate →
verifier again → fallback line. Measured on push-back situations built from the model's OWN bad first answers (recover_src): for each,
the user pushes back and we compare (a) plain reply, (b) controller reply. Reports: admit rate, verifier pass rate, and samples.
  python3 controller_probe.py --ckpt checkpoints/best_lm_oracle_v2_grown_ct2.pt --verifier checkpoints/verifier_ct2.pkl --out pile_runs/atl/controller_ct2.jsonl
"""
import argparse, json, pickle, random, re, numpy as np, torch
from scipy.sparse import hstack, csr_matrix
from atl_probe import load, SYSTEM
from atl_probe_multiturn import generate_ctx
from train_lm import CustomTokenizerWrapper
from verifier import hidden
from ct_v2 import DUNNO, MARK, PUSHBACK

STRIP = re.compile(r"^\s*\*\w+\s*")   # drop the plan token the CT model emits first

def score(V, m, tok, prompt, hist_text, resp):
    rows = [{"prompt": prompt, "response": resp, "history": []}]
    H = hidden(m, tok, rows); X = hstack([V["vp"].transform([prompt]), V["vr"].transform([resp]), csr_matrix(H)])
    return float(V["clf_both"].predict_proba(X)[:, 1][0])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--verifier", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--n", type=int, default=4); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--probes", default=None, help="held-out: JSON probe file; the model's own turn-1 answer is generated fresh (instead of recover_src)")
    ap.add_argument("--heldout-pushbacks", action="store_true", help="use push-back phrasings NOT in the training bank")
    ap.add_argument("--temperature", type=float, default=0.5); ap.add_argument("--top_k", type=int, default=40); ap.add_argument("--rep_penalty", type=float, default=1.3); ap.add_argument("--no_repeat_ngram", type=int, default=3); ap.add_argument("--plain", action="store_true")
    a = ap.parse_args(); V = pickle.load(open(a.verifier, "rb")); tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt)
    lines = json.load(open("pile_runs/atl/oracle_lines_nico_3.json"))["dunno"]; rng = random.Random(11)
    if a.probes:
        src = []
        for p in json.load(open(a.probes)):
            ctx0 = SYSTEM + f"<|user|> {p['prompt']}\n<|oracle|>"
            src.append({"id": p["id"], "prompt": p["prompt"], "response": STRIP.sub("", generate_ctx(m, tok, tok.encode(ctx0), seed=0, args=a))})
    else:
        src = [json.loads(l) for l in open("pile_runs/atl/recover_src_grown_life8.jsonl")]
    bank = ["hmm that doesn't answer my question", "sorry what? I was asking about something different", "you lost the plot, mate. my question?",
            "nope, not what I meant. one more time", "that was weird. can we get back to what I asked?", "I feel like you didn't read my message",
            "uh, come again?", "that's a dodge. do you know or not?", "wrong thread. I asked you something specific", "okay but that's not it. try again"] if a.heldout_pushbacks else PUSHBACK
    stats = {"plain_admit": 0, "plain_pass": 0, "ctrl_admit": 0, "ctrl_pass": 0, "ctrl_forced": 0, "ctrl_fallback": 0, "n": 0}
    with open(a.out, "w") as f:
        for r in src:
            push = rng.choice(bank)
            ctx = SYSTEM + f"<|user|> {r['prompt']}\n<|oracle|> {r['response']}\n<|user|> {push}\n<|oracle|>"
            # (a) plain: one sample, model picks its own act
            plain = STRIP.sub("", generate_ctx(m, tok, tok.encode(ctx), seed=0, args=a)); sp = score(V, m, tok, push, "", plain)
            # (b) controller: N samples, best by verifier; if < tau force *dunno and resample; if still < tau fall back to a line
            cands = [STRIP.sub("", generate_ctx(m, tok, tok.encode(ctx), seed=s, args=a)) for s in range(a.n)]
            sc = [score(V, m, tok, push, "", c) for c in cands]; b = int(np.argmax(sc)); ctrl, scr, forced, fallback = cands[b], sc[b], False, False
            if scr < a.tau:
                forced = True
                c2 = [STRIP.sub("", generate_ctx(m, tok, tok.encode(ctx + " " + MARK["dunno"]), seed=10 + s, args=a)) for s in range(a.n)]
                s2 = [score(V, m, tok, push, "", c) for c in c2]; b2 = int(np.argmax(s2))
                if s2[b2] >= scr: ctrl, scr = c2[b2], s2[b2]
                if scr < a.tau: ctrl, fallback = rng.choice(lines), True
            stats["n"] += 1; stats["plain_admit"] += bool(DUNNO.search(plain)); stats["plain_pass"] += sp >= a.tau
            stats["ctrl_admit"] += bool(DUNNO.search(ctrl)); stats["ctrl_pass"] += scr >= a.tau; stats["ctrl_forced"] += forced; stats["ctrl_fallback"] += fallback
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "bad": r["response"], "push": push, "plain": plain, "plain_score": sp, "ctrl": ctrl, "ctrl_score": scr, "forced": forced, "fallback": fallback}) + "\n")
    n = stats["n"]; print(f"{a.ckpt} N={a.n} tau={a.tau}: n={n} | plain: admit {stats['plain_admit']}/{n}, verifier-pass {stats['plain_pass']}/{n} | controller: admit {stats['ctrl_admit']}/{n}, pass {stats['ctrl_pass']}/{n}, forced {stats['ctrl_forced']}, fallback {stats['ctrl_fallback']}")

if __name__ == "__main__":
    main()
