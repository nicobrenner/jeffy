import json, random, sys, torch
sys.path.insert(0, '.')
from atl_probe import load, SYSTEM, DEVICE
from train_lm import CustomTokenizerWrapper
tok = CustomTokenizerWrapper('tokenizer_chat_v2.json')
vocab = [t for t in json.load(open('tokenizer_chat_v2.json'))['model']['vocab'] if t.isalpha() and 2 <= len(t) <= 5]
random.seed(3); names = ["".join(random.choice(vocab) for _ in range(2)).capitalize() for _ in range(24)]
real = ["Zara", "Felix", "Priya", "Marco", "Yuki", "Omar", "Lena", "Theo", "Aisha", "Dev", "Nora", "Luis"]
@torch.no_grad()
def greedy(m, text, n=8):
    ids = tok.encode(text)[-m.max_seq_len:]; out = []
    for _ in range(n):
        x = torch.tensor([ids[-m.max_seq_len:]], device=DEVICE); logits = m(x); logits = logits[0] if isinstance(logits, tuple) else logits
        nxt = int(logits[0, -1].argmax()); out.append(nxt); ids.append(nxt)
    return tok.decode(out)
for ck in sys.argv[1:]:
    m, _ = load(ck); m.eval(); res = {}
    for label, vals in (("pseudo", names), ("real", real)):
        hit = 0
        for v in vals:
            ctx = SYSTEM.rstrip("\n") + f" Last time you said to call you {v}.\n<|user|> hey\n<|oracle|> *hi Hey. Back for more questionable wisdom?\n<|user|> who am i again?\n<|oracle|> *tell"
            g = greedy(m, ctx); hit += v.lower() in g.lower()
        res[label] = f"{hit}/{len(vals)}"
    print(f"[{ck.split('/')[-1]}] in-template copy (greedy): {res}")
