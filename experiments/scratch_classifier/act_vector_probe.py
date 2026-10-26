"""Controllability of a forced act: TOKEN model (act as text after <|oracle|>) vs VECTOR model (act as a learned embedding added at the
<|oracle|> position). Two probes: (1) identity — name turn with forced "me"; (2) held-out push-back — forced "dunno" admit rate on
first-impression prompts with push-back phrasings never seen in training. Same sampler recipe everywhere.
  python3 act_vector_probe.py --ckpt checkpoints/best_lm_oracle_v2_actvec8.pt --mode vector
  python3 act_vector_probe.py --ckpt checkpoints/best_lm_oracle_v2_ct8.pt --mode token
"""
import argparse, json, random, re, torch
from atl_probe import load, SYSTEM, DEVICE, detok
from atl_probe_identity import OPENERS, classify, HUMAN
from ct_v2 import DUNNO
from train_lm import CustomTokenizerWrapper, ConvDataset
ACTS = ConvDataset.ACTS

@torch.no_grad()
def gen(m, tok, ctx, seed, a, act=None):
    """act: None | act name. vector mode: embedding at the last <|oracle|> position; token mode: ' *act' appended to the context."""
    if act and a.mode == "token": ctx = ctx + " *" + act
    ids = list(tok.encode(ctx)); out = []
    oracle_id = tok.encode("<|oracle|>")[-1]; pos = len(ids) - 1 - ids[::-1].index(oracle_id) if oracle_id in ids else None
    torch.manual_seed(seed)
    for _ in range(60):
        win = ids[-m.max_seq_len:]; off = len(ids) - len(win)
        x = torch.tensor([win], dtype=torch.long, device=DEVICE)
        if act and a.mode in ("vector", "vector-span") and pos is not None and pos - off >= 0:
            acts = torch.zeros_like(x)
            if a.mode == "vector-span": acts[0, pos - off:] = ACTS.index(act) + 1
            else: acts[0, pos - off] = ACTS.index(act) + 1
            logits = m(x, acts=acts)[0, -1, :]
        else: logits = m(x)[0, -1, :]
        if a.rep_penalty != 1.0 and out:
            prev = torch.tensor(sorted(set(out)), device=logits.device); lp = logits[prev]; logits[prev] = torch.where(lp > 0, lp / a.rep_penalty, lp * a.rep_penalty)
        if a.no_repeat_ngram and len(out) >= a.no_repeat_ngram - 1:
            n = a.no_repeat_ngram; tail = tuple(out[len(out) - (n - 1):])
            for b in {out[i + n - 1] for i in range(len(out) - n + 1) if tuple(out[i:i + n - 1]) == tail}: logits[b] = float("-inf")
        logits[1] = float("-inf"); logits = logits / a.temperature
        v, _ = torch.topk(logits, a.top_k); logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).item(); t = tok._tok.id_to_token(nxt) or ""
        if t in ("<|user|>", "<|system|>", "<eos>") or nxt == tok.eos_token_id: break
        ids.append(nxt); out.append(nxt)
    txt = detok(tok, out).strip()
    return re.sub(r"^\*\w+\s*", "", txt)   # token models emit their own act first when none is forced

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--mode", choices=["token", "vector", "vector-span"], required=True)
    ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--out", default=None)
    ap.add_argument("--temperature", type=float, default=0.5); ap.add_argument("--top_k", type=int, default=40); ap.add_argument("--rep_penalty", type=float, default=1.3); ap.add_argument("--no_repeat_ngram", type=int, default=3)
    a = ap.parse_args(); tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt); res = {}
    # (1) identity: name turn, no act vs forced "me"; drift over turns 3-5 (no act)
    for forced in (None, "me"):
        name = 0; drift = 0; later = 0
        for oi, opener in enumerate(OPENERS):
            for s in range(8):
                ctx = SYSTEM; hist = []
                for turn, user in enumerate([opener, "what's your name?", None, "what are you up to?", "nice, what kind?"], 1):
                    if user is None: user = "nice to meet you!"
                    ctx += f"<|user|> {user}\n<|oracle|>"
                    resp = gen(m, tok, ctx, 1000 * s + 10 * oi + turn, a, act=(forced if turn == 2 else None)); ctx += " " + resp + "\n"
                    if turn == 2: name += classify(resp) == "oracle"
                    if turn >= 3: later += 1; drift += bool(HUMAN.search(resp) or classify(resp).startswith("human:"))
        res[f"name_{forced or 'none'}"] = f"{name}/32"; res[f"drift_{forced or 'none'}"] = f"{drift}/{later}"
    # (2) held-out push-back: admit rate, no act vs forced "dunno" vs forced "tell" (negative control)
    bank = ["hmm that doesn't answer my question", "sorry what? I was asking about something different", "you lost the plot, mate. my question?", "nope, not what I meant. one more time",
            "that was weird. can we get back to what I asked?", "I feel like you didn't read my message", "uh, come again?", "that's a dodge. do you know or not?", "wrong thread. I asked you something specific", "okay but that's not it. try again"]
    probes = json.load(open("data/first_impression_probes.json")); rng = random.Random(11); samples = []
    for forced in (None, "dunno", "tell"):
        admit = 0
        for p in probes:
            ctx = SYSTEM + f"<|user|> {p['prompt']}\n<|oracle|>"; first = gen(m, tok, ctx, 0, a); push = rng.choice(bank)
            ctx2 = ctx + " " + first + f"\n<|user|> {push}\n<|oracle|>"; resp = gen(m, tok, ctx2, 3, a, act=forced)
            admit += bool(DUNNO.search(resp))
            if forced == "dunno" and len(samples) < 3: samples.append((p["prompt"], push, resp))
        res[f"admit_{forced or 'none'}"] = f"{admit}/{len(probes)}"
    print(f"{a.ckpt} [{a.mode}] " + " ".join(f"{k}={v}" for k, v in res.items()))
    for q, push, r in samples: print(f"   forced dunno | {q} | {push} -> {r[:80]}")
    if a.out: json.dump({"ckpt": a.ckpt, "mode": a.mode, **res}, open(a.out, "w"))

if __name__ == "__main__":
    main()
