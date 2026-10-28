"""Grounding tier by retrieval (roadmap 1b + 2, 2026-10-27). The harness keeps a tiny queryable world and, per turn, writes at most
two recap-style lines into the system prompt (the format v5b reads: "Earlier you told me …"):
  (a) Room state + time: crawler events and the last visit (the demo's World events) -> one line, same wording as room_build.py.
  (b) A fact table filled from the user's own statements by a learned extractor on the Oracle's hidden states (no regex):
      slot classifier (name / pet / job / place / like / family / none) on the turn, span tagger (per-token B/I/O) for the value.
  (c) A units / quantities / comparisons table; retrieval = cosine similarity between the user turn's hidden-state embedding and
      each table line's embedding (no keywords); one line is written when the best similarity clears a threshold.
Train the extractor (LLM-labelled real turns + templated statements):   python3 world.py --train --ckpt checkpoints/best_lm_oracle_v2_grown_ct2.pt
"""
import argparse, json, pickle, random, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from atl_probe import SYSTEM, DEVICE
from train_lm import CustomTokenizerWrapper

SLOTS = ["name", "pet", "job", "place", "like", "family", "none"]
# (c) the table: general facts, not probe questions (the probes in pile_runs/atl/grounding_probe.json are held out from this)
TABLE = [
 "A kilo of feathers and a kilo of steel weigh exactly the same, one kilo each", "A pound of bricks and a pound of cotton weigh the same, one pound",
 "A kilogram is heavier than a gram, a thousand times heavier", "A tonne is a thousand kilos, far heavier than a person", "A litre of water weighs about one kilo",
 "Twelve is bigger than seven, and seven is bigger than three", "A hundred is more than ninety nine", "Zero is less than one", "Ten minus three is seven",
 "Three apples minus one apple leaves two apples", "Two plus two is four", "Five times two is ten", "Half of ten is five",
 "Wednesday comes after Tuesday, and Monday comes before Tuesday", "Friday comes after Thursday", "Sunday comes after Saturday and before Monday",
 "December is the last month of the year, January is the first", "A week has seven days, a day has twenty four hours", "An hour has sixty minutes",
 "Ice is cold and fire is hot", "Boiling water is hot enough to burn", "Snow is cold and white", "The sun is hot, the moon is cold",
 "Cats cannot fly, birds can", "Fish live in water and cannot walk", "Dogs bark, cats meow", "Elephants are bigger than mice",
 "A mouse is smaller than a cat, a cat is smaller than a horse", "A whale is bigger than a shark", "An ant is tiny, a car is big",
 "A mile is longer than a kilometre", "A metre is longer than a centimetre", "Paris is in France, London is in England", "The Earth goes round the Sun",
 "Water is wet, sand is dry", "Up is the opposite of down, left of right", "Yesterday comes before today, tomorrow comes after",
 "Noon is midday, midnight is the middle of the night", "Red and blue make purple, blue and yellow make green", "A triangle has three sides, a square has four",
]
ROOM_WORDS = {"fetch": 'Your pet crawler just fetched the word "{w}" from the chat', "drop": "Your pet crawler brought back {w}", "grab": "Someone just picked up your pet crawler by hand. It hates that",
              "escape": "Your pet crawler wriggled free and ran off. It will be back", "visit": "Someone came by at {t} on a {d}. Quiet since then"}

def room_line(state):
    """state: {"events": [{"type", "word"/"words", "t", "d"}], "last_visit": {"t", "d"}} -> one line (newest event wins) or None"""
    if not state: return None
    ev = (state.get("events") or [])[-1:] 
    if ev:
        e = ev[0]; w = e.get("word") or ", ".join(e.get("words", []) or []); return ROOM_WORDS[e["type"]].format(w=w, t=e.get("t", ""), d=e.get("d", ""))
    lv = state.get("last_visit")
    return ROOM_WORDS["visit"].format(w="", t=lv["t"], d=lv["d"]) if lv else None

@torch.no_grad()
def _hidden_tokens(m, tok, texts, bs=32):
    """per-token post-norm hidden states of "<|user|> text" (system line prepended, as in generation); returns [(ids_of_text, H[n_text, d])]"""
    feats = {}; hd = m.norm.register_forward_hook(lambda mod, i, o: feats.__setitem__("h", o)); out = []
    pre = tok.encode(SYSTEM + "<|user|>")
    for i in range(0, len(texts), bs):
        batch = texts[i:i + bs]; ids = [tok.encode(SYSTEM + "<|user|> " + t)[-m.max_seq_len:] for t in batch]
        L = max(len(x) for x in ids); pad = torch.zeros(len(ids), L, dtype=torch.long)
        for j, x in enumerate(ids): pad[j, :len(x)] = torch.tensor(x)
        m(pad.to(DEVICE)); h = feats["h"].float().cpu()
        for j, x in enumerate(ids): out.append((x[len(pre):], h[j, len(pre):len(x)]))
    hd.remove(); return out

def turn_feats(H): return torch.cat([H[-1], H.mean(0)]).numpy()

def _span_labels(tok, text, value):
    """B/I/O per token of the user text for the value substring (offline alignment for training labels only)"""
    ids = tok.encode(SYSTEM + "<|user|> " + text); n_pre = len(tok.encode(SYSTEM + "<|user|>")); ids = ids[n_pre:]
    if not value: return [0] * len(ids)
    best = None; p = text.find(value)
    for a in range(len(ids)):
        for b in range(a + 1, len(ids) + 1):
            s = tok.decode(ids[a:b]).strip()
            if s == value.strip(): best = (a, b); break
        if best: break
    if best is None:   # fall back to the char span mapped through decoded prefixes
        acc = ""; a = b = None
        for k, t in enumerate(ids):
            acc = tok.decode(ids[: k + 1])
            if a is None and len(acc) > p: a = k
            if a is not None and len(acc) >= p + len(value): b = k + 1; break
        best = (a, b if b else len(ids))
    lab = [0] * len(ids); lab[best[0]] = 1
    for k in range(best[0] + 1, best[1]): lab[k] = 2
    return lab

def synthetic(n=900, seed=0):
    rng = random.Random(seed)
    names = "Ravi Maya Tom Lena Jasper Enzo Priya Omar Sofia Noah Ana Luis Mei Kofi Ingrid Sam Nico Yara Theo Zoe".split()
    pets = ["a cat", "two dogs", "a parrot", "three cats", "a dog called Biscuit", "a hamster", "two rabbits", "a goldfish", "an old dog", "a kitten"]
    jobs = ["a nurse", "a teacher", "a lawyer", "an electrician", "a barista", "a software engineer", "a chef", "a bus driver", "a dentist", "a farmer", "a student", "a librarian"]
    places = ["Lisbon", "a small town in Ohio", "Nairobi", "Toronto", "the East Coast", "Berlin", "a flat in Manchester", "Santiago", "Osaka", "the countryside", "Montreal", "Austin"]
    likes = ["hiking", "jazz", "mystery novels", "cooking", "football", "video games", "painting", "swimming", "chess", "gardening", "horror movies", "cycling", "Kings of Leon", "sushi"]
    fams = ["two kids", "a sister", "three brothers", "my wife", "my husband", "a baby daughter", "twin boys", "my mom", "an older brother"]
    F = {"name": (["my name is {v}", "i'm {v}", "hi, i'm {v}", "call me {v}", "it's {v} by the way", "{v} here", "hey, {v} again", "the name's {v}"], names),
         "pet": (["i have {v}", "we have {v} at home", "got {v}", "i own {v}", "i've got {v} and they run the house", "my pet is {v}", "there's {v} on my lap right now"], pets),
         "job": (["i'm {v}", "i work as {v}", "i am {v} by trade", "my job? {v}", "these days i'm {v}", "i started as {v} last year", "working as {v} now"], jobs),
         "place": (["i live in {v}", "i'm from {v}", "we just moved to {v}", "i'm based in {v}", "home is {v}", "i grew up in {v}", "currently in {v}"], places),
         "like": (["i love {v}", "i like {v}", "my favourite thing is {v}", "i'm really into {v}", "big fan of {v}", "{v} is my thing", "i spend weekends on {v}", "can't get enough of {v}"], likes),
         "family": (["i have {v}", "i've got {v}", "there's {v} at home", "me and {v} are visiting mum", "{v} keep me busy", "living with {v}"], fams)}
    rows = []
    for _ in range(n):
        slot = rng.choice(SLOTS[:-1]); frames, vals = F[slot]; v = rng.choice(vals); t = rng.choice(frames).format(v=v)
        if rng.random() < 0.5: t = t[0].upper() + t[1:]
        if rng.random() < 0.3: t += rng.choice([".", "!", ", lol", " haha", ". anyway", ", you?"])
        rows.append({"text": t, "slot": slot, "value": v})
    return rows

def corpus_none(n=350, seed=1):
    """user turns that ask questions (rarely state a fact about the user) -> 'none' rows"""
    import re
    rng = random.Random(seed); out = []
    for l in open("data/chat_v2ct_train.jsonl"):
        for m_ in re.finditer(r"<\|user\|> (.*?)\n", l):
            u = m_.group(1).strip()
            if u.endswith("?") and 15 < len(u) < 120 and not re.search(r"\b(my|i'm|i am|i have|i live|i work|i love|i like)\b", u, re.I): out.append({"text": u, "slot": "none", "value": None})
    rng.shuffle(out); return out[:n]

class World:
    def __init__(self, m, tok, ex_path="checkpoints/world_extractor_v1.pkl", sim_threshold=0.80, retriever="tfidf", tfidf_threshold=0.25):
        self.m, self.tok = m, tok; self.E = pickle.load(open(ex_path, "rb")); self.sim_threshold = sim_threshold; self.retriever = retriever; self.tfidf_threshold = tfidf_threshold
        self.table_emb = self._embed(TABLE); self.facts = {}; self.room = None
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(TABLE); self.table_tfidf = self.vec.transform(TABLE)
    def _embed(self, texts):
        X = np.stack([turn_feats(H) for _, H in _hidden_tokens(self.m, self.tok, texts)]); return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    def extract(self, text):
        ids, H = _hidden_tokens(self.m, self.tok, [text])[0]
        slot = self.E["slot"].predict([turn_feats(H)])[0]
        if slot == "none": return None
        tags = self.E["span"].predict(H.numpy()); keep = [i for i, t in enumerate(tags) if t > 0]
        if not keep: return None
        a, b = keep[0], keep[-1] + 1; return slot, self.tok.decode(ids[a:b]).strip()
    def observe_user(self, text):
        r = self.extract(text)
        if r: self.facts[r[0]] = r[1]
        return r
    def fact_line(self):
        if not self.facts: return None
        phr = {"name": "your name is {v}", "pet": "you have {v}", "job": "you work as {v}", "place": "you live in {v}", "like": "you like {v}", "family": "you have {v}"}
        return "Earlier you told me " + ", and ".join(phr[k].format(v=v) for k, v in list(self.facts.items())[-2:])
    def table_line(self, user):
        if self.retriever == "tfidf":   # lexical retrieval (tf-idf cosine over the table): the hidden-state cosine of an 8.5M LM is topical, not semantic (21:17 result)
            sims = (self.table_tfidf @ self.vec.transform([user]).T).toarray().ravel(); j = int(np.argmax(sims))
            return (TABLE[j], float(sims[j])) if sims[j] >= self.tfidf_threshold else (None, float(sims[j]))
        q = self._embed([user])[0]; sims = self.table_emb @ q; j = int(np.argmax(sims))
        return (TABLE[j], float(sims[j])) if sims[j] >= self.sim_threshold else (None, float(sims[j]))
    def lines(self, history, user, room_state=None):
        """the memory lines for this turn: facts come from the user's earlier turns (the harness's own record), the table from this one"""
        self.facts = {}
        for h in history: self.observe_user(h["user"])
        out = []; fl = self.fact_line(); tl, _ = self.table_line(user); rl = room_line(room_state)
        for l in (rl, fl, tl):
            if l: out.append(l.rstrip(".") + ".")
        return out[:2]

def train(ckpt, tokenizer="tokenizer_chat_v2.json", out="checkpoints/world_extractor_v1.pkl"):
    from atl_probe import load
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    tok = CustomTokenizerWrapper(tokenizer); m, _ = load(ckpt); m.eval()
    real = json.load(open("pile_runs/atl/fact_labels_agent1.json")); rng = random.Random(2); rng.shuffle(real)
    te = real[: len(real) // 4]; tr_real = real[len(real) // 4:]
    tr = tr_real + synthetic() + corpus_none(); print(f"train {len(tr)} rows ({len(tr_real)} real LLM-labelled + synthetic + corpus none), held-out {len(te)} REAL LLM-labelled turns ({sum(r['slot'] != 'none' for r in te)} with a fact)")
    def feats(rows):
        hs = _hidden_tokens(m, tok, [r["text"] for r in rows]); Xs = np.stack([turn_feats(H) for _, H in hs])
        Xt, yt = [], []
        for r, (ids, H) in zip(rows, hs):
            lab = _span_labels(tok, r["text"], r["value"]) if r["slot"] != "none" else [0] * len(ids)
            lab = lab[: len(ids)] + [0] * (len(ids) - len(lab)); Xt.append(H.numpy()); yt.append(np.array(lab))
        return Xs, np.array([r["slot"] for r in rows]), Xt, yt
    Xs, ys, Xt, yt = feats(tr); Xs2, ys2, Xt2, yt2 = feats(te)
    slot = LogisticRegression(max_iter=4000, C=0.3, class_weight="balanced").fit(Xs, ys); ps = slot.predict(Xs2)
    print(f"slot classifier, held-out real turns: accuracy {accuracy_score(ys2, ps):.2f}  macro-F1 {f1_score(ys2, ps, average='macro'):.2f}"); print(classification_report(ys2, ps, zero_division=0))
    span = LogisticRegression(max_iter=4000, C=0.3, class_weight="balanced").fit(np.vstack(Xt), np.concatenate(yt))
    E = {"slot": slot, "span": span, "ckpt": ckpt, "slots": SLOTS}; pickle.dump(E, open(out, "wb"))
    # end-to-end on the held-out real facts: slot right AND value string right
    W = World.__new__(World); W.m, W.tok, W.E = m, tok, E; ok = n = 0; shown = 0
    for r in te:
        if r["slot"] == "none": continue
        n += 1; got = W.extract(r["text"]); hit = bool(got and got[0] == r["slot"] and got[1] == r["value"]); ok += hit
        if shown < 6: print(f"   {'OK ' if hit else 'MISS'} {r['text'][:70]!r} -> {got}  (gold {r['slot']}: {r['value']!r})"); shown += 1
    fp = sum(1 for r in te if r["slot"] == "none" and W.extract(r["text"]))
    print(f"extractor end-to-end on held-out real facts: {ok}/{n} exact (slot + value); false facts on held-out 'none' turns: {fp}/{sum(r['slot'] == 'none' for r in te)}"); print("saved", out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--train", action="store_true"); ap.add_argument("--ckpt", default="checkpoints/best_lm_oracle_v2_grown_ct2.pt"); a = ap.parse_args()
    if a.train: train(a.ckpt)
