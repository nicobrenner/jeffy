"""THE harness: everything between the bare model and the user, in one place, so probes evaluate what the demo serves.
Mirrors demo/index.html: system prompt, the two-turn PRIME, sampler presets (recipe03 = the demo default), context window handling;
plus the inference-time components that exist so far: best-of-N with the reply verifier, the controller (force the admit act and
resample when nothing passes, fall back to one of Nico's lines), and a memory line (facts the page knows, e.g. the user's name).
  respond(m, tok, history, user, cfg) -> (reply, info)
cfg keys: prime (bool), sampler (name), best_of (int), verifier (pkl path or None), controller (bool), tau (float), act_mode
("token"|"vector"|None), memory (dict or None), seed (int).
"""
import json, pickle, random, re, numpy as np, torch
from scipy.sparse import hstack, csr_matrix
from atl_probe import SYSTEM, DEVICE, detok
from train_lm import ConvDataset

PRIME = [("Hey, what are you?", "I'm Tiny Oracle. 2 million parameters of questionable wisdom, running right here in your browser. I fit in 4 megabytes. A fruit fly has more neurons than I have parameters."),
         ("That's not much", "I know. And yet here I am, boiling with opinions.")]
SAMPLERS = {"recipe03": dict(temperature=0.3, top_k=40, rep_penalty=1.3, no_repeat_ngram=3),
            "recipe": dict(temperature=0.5, top_k=40, rep_penalty=1.3, no_repeat_ngram=3),
            "old": dict(temperature=0.8, top_k=40, rep_penalty=1.0, no_repeat_ngram=0),
            "sched": dict(temperature=0.3, top_k=40, rep_penalty=1.3, no_repeat_ngram=3, later_temperature=0.5)}   # T0.3 on the opener, T0.5 once there is history
MAX_CONTEXT = 256
ACTS = ConvDataset.ACTS
YESNO_Q = re.compile(r"^\s*(?:is|are|do|does|did|can|could|would|will|have|has|was|were|should|am|isn't|aren't|don't|doesn't|anyone|anything)\b.*\?\s*$", re.I)
_YES_IDS = {}
def yes_ids(tok):   # token ids that start a reply with Yes/Yeah/Yep (with or without the leading space)
    if id(tok) not in _YES_IDS:
        ids = set()
        for w in ("Yes", "Yeah", "Yep", " Yes", " Yeah", " Yep", "yes", " yes", "No", " No", "Nope", " Nope", "no", " no"):
            e = tok.encode(w)
            if e: ids.add(e[0])
        _YES_IDS[id(tok)] = ids
    return _YES_IDS[id(tok)]
DUNNO = re.compile(r"\b(i (don'?t|do not) know|no idea|no clue|not sure (what|how|if|about|i)|i have no|can'?t help|beyond me|not in my training|never heard of|i made that up|i invented|came from nowhere|not (a|an) \w+ expert|outside my weights|not my department)\b", re.I)
DEFAULT = dict(prime="first", sampler="sched", best_of=1, verifier=None, controller=False, tau=0.5, act_mode=None, memory=None, seed=0, ban_yesno=False, cross_turn=True)   # = the demo defaults
_LINES = None

def lines():
    global _LINES
    if _LINES is None: _LINES = json.load(open("pile_runs/atl/oracle_lines_nico_3.json"))["dunno"]
    return _LINES

def build_context(history, user, prime=True, memory=None):
    if prime == "first": prime = not history   # the primed example only for the first reply, then it leaves the context
    t = SYSTEM
    if memory:
        t = t.rstrip("\n") + " " + " ".join(f"{k}: {v}." for k, v in memory.items()) + "\n"
    if prime:
        for u, o in PRIME: t += f"<|user|> {u}\n<|oracle|> {o}\n"
    for h in history: t += f"<|user|> {h['user']}\n<|oracle|> {h['oracle']}\n"
    return t + f"<|user|> {user}\n<|oracle|>"

def trim(tok, ctx_ids, prime_len):
    """demo rule: keep system+prime, then the most recent turns that fit"""
    if len(ctx_ids) <= MAX_CONTEXT: return ctx_ids
    head = ctx_ids[:prime_len]; tail = ctx_ids[prime_len:]
    return head + tail[-(MAX_CONTEXT - prime_len):]

@torch.no_grad()
def generate(m, tok, ctx, S, seed, act=None, act_mode=None, prime_len=0, prior=None, user=None, ban_yesno=False):
    """prior: the Oracle's earlier replies (cross-turn repetition control); user + ban_yesno: no Yes/No opener unless asked yes/no (off by default:
    banning tokens only moves the reflex to the next stock opener — "No" 13/30, then "Sure"/"Thank you"; prime-first is the real fix)"""
    ban_yes = False
    if ban_yesno and user is not None:
        try: ban_yes = classify_intent(m, tok, [], user)[0] != "yesno_question"
        except Exception: ban_yes = False
    if act and act_mode == "token": ctx = ctx + " *" + act
    ids = trim(tok, list(tok.encode(ctx)), prime_len); out = []
    prior_ids = [t for r in (prior or []) for t in tok.encode(" " + r)]; prior_set = set(prior_ids)
    prior_grams = {tuple(prior_ids[j:j + 3]) for j in range(len(prior_ids) - 2)}; cross = 1 + (S["rep_penalty"] - 1) * 0.5
    oracle_id = tok.encode("<|oracle|>")[-1]; pos = len(ids) - 1 - ids[::-1].index(oracle_id) if oracle_id in ids else None
    torch.manual_seed(seed)
    for _ in range(60):
        win = ids[-m.max_seq_len:]; off = len(ids) - len(win)
        x = torch.tensor([win], dtype=torch.long, device=DEVICE)
        if act and act_mode == "vector" and getattr(m, "act_emb", None) is not None and pos is not None and pos - off >= 0:
            acts = torch.zeros_like(x); acts[0, pos - off] = ACTS.index(act) + 1; logits = m(x, acts=acts)[0, -1, :]
        else: logits = m(x)[0, -1, :]
        if S["rep_penalty"] != 1.0 and out:
            prev = torch.tensor(sorted(set(out)), device=logits.device); lp = logits[prev]; logits[prev] = torch.where(lp > 0, lp / S["rep_penalty"], lp * S["rep_penalty"])
        if cross != 1.0 and prior_set:
            pv = torch.tensor(sorted(prior_set - set(out)), device=logits.device)
            if len(pv): lp = logits[pv]; logits[pv] = torch.where(lp > 0, lp / cross, lp * cross)
        if len(out) >= 2:
            for g in prior_grams:
                if g[0] == out[-2] and g[1] == out[-1]: logits[g[2]] = float("-inf")
        n = S["no_repeat_ngram"]
        if n and len(out) >= n - 1:
            tail = tuple(out[len(out) - (n - 1):])
            for b in {out[i + n - 1] for i in range(len(out) - n + 1) if tuple(out[i:i + n - 1]) == tail}: logits[b] = float("-inf")
        if ban_yes and re.match(r"^\s*(\*\w*\s*)?$", detok(tok, out)):   # first word of the reply proper (after a plan token, if any)
            for b in yes_ids(tok): logits[b] = float("-inf")
        logits[1] = float("-inf"); logits = logits / S["temperature"]
        v, _ = torch.topk(logits, S["top_k"]); logits[logits < v[-1]] = float("-inf")
        nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1).item(); t = tok._tok.id_to_token(nxt) or ""
        if t in ("<|user|>", "<|system|>", "<eos>") or nxt == tok.eos_token_id: break
        ids.append(nxt); out.append(nxt)
    return re.sub(r"^\*\w+\s*", "", detok(tok, out).strip())

_INTENT = {}
def classify_intent(m, tok, history, user, path="checkpoints/intent_v1.pkl"):
    """User-turn intent (greeting / yesno_question / open_question / task / pushback / share / reaction) from the Oracle's own hidden states —
    the learned replacement for every regex that guessed intent. Returns (label, {label: prob}). ~71% held-out accuracy on v1 (1,993 LLM-labelled turns)."""
    if path not in _INTENT: _INTENT[path] = pickle.load(open(path, "rb"))
    C = _INTENT[path]
    from intent_clf import hidden
    prev = history[-1]["oracle"] if history else ""
    H = hidden(m, tok, [{"text": user, "prev_oracle": prev}]); X = hstack([C["vec"].transform([user]), csr_matrix(H)])
    probs = C["clf_both"].predict_proba(X)[0]; labels = list(C["clf_both"].classes_)
    return labels[int(np.argmax(probs))], dict(zip(labels, [float(x) for x in probs]))

_VER = {}
def verifier_score(m, tok, V, prompt, resp):
    from verifier import hidden
    H = hidden(m, tok, [{"prompt": prompt, "response": resp, "history": []}])
    X = hstack([V["vp"].transform([prompt]), V["vr"].transform([resp]), csr_matrix(H)])
    return float(V["clf_both"].predict_proba(X)[:, 1][0])

def respond(m, tok, history, user, cfg=None):
    c = {**DEFAULT, **(cfg or {})}; S = dict(SAMPLERS[c["sampler"]])
    if history and "later_temperature" in S: S["temperature"] = S["later_temperature"]
    ctx = build_context(history, user, c["prime"], c["memory"]); prime_len = len(tok.encode(build_context([] if c["prime"] != "first" or not history else [{"user": "", "oracle": ""}], "", c["prime"], c["memory"]))) - 3
    info = {"n": c["best_of"], "forced": False, "fallback": False, "score": None}
    prior = [h["oracle"] for h in history] if c.get("cross_turn", True) else []   # cross-turn repetition control (demo toggle)
    if c["best_of"] <= 1 or not c["verifier"]:
        return generate(m, tok, ctx, S, c["seed"], prime_len=prime_len, prior=prior, user=user, ban_yesno=c["ban_yesno"]), info
    V = _VER.setdefault(c["verifier"], pickle.load(open(c["verifier"], "rb")))
    cands = [generate(m, tok, ctx, S, c["seed"] * 100 + s, prime_len=prime_len, prior=prior, user=user, ban_yesno=c["ban_yesno"]) for s in range(c["best_of"])]
    sc = [verifier_score(m, tok, V, user, r) for r in cands]; b = int(np.argmax(sc)); best, score = cands[b], sc[b]
    if c["controller"] and score < c["tau"] and c["act_mode"]:
        intent, probs = classify_intent(m, tok, history, user); info["intent"] = intent
        if intent != "pushback" and probs.get("pushback", 0) < 0.25: return best, {**info, "score": score}   # no push-back → no forced admission
        info["forced"] = True
        c2 = [generate(m, tok, ctx, S, c["seed"] * 100 + 50 + s, act="dunno", act_mode=c["act_mode"], prime_len=prime_len, prior=prior) for s in range(c["best_of"])]
        s2 = [verifier_score(m, tok, V, user, r) for r in c2]; b2 = int(np.argmax(s2))
        if s2[b2] >= score: best, score = c2[b2], s2[b2]
        if score < c["tau"]: best = random.Random(c["seed"]).choice(lines()); info["fallback"] = True
    info["score"] = score
    return best, info
