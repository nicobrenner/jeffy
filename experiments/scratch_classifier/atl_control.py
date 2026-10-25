"""CT experiment: classifier-produced control tokens as LM input (same trunk, two roles).

  python3 atl_control.py label        # label chat_v6plain train/val by conversation topic -> chat_v6ct_{train,val}.jsonl
                                      # tokens: [deflect] knowledge/live-data topics, [comfort] emotional topics, [answer] everything else
  python3 atl_control.py head CKPT    # LogReg head on the trunk's mean hidden state of the user prompt -> predicts the token; saves pile_runs/atl/ct_head.pkl
  python3 atl_control.py probe CKPT --out FILE   # generate with the head choosing the token first (plain format)

Labels come from the topic of the conversation (recovered via oracle-turn matching), so no new conversations are written:
the classifier upgrades the supervision of the data we already have.
"""

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from atl_val_loss import oracle_turns, topic_index, norm
from train_lm import CausalTransformer, CustomTokenizerWrapper

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DEFLECT = {"asking about current events", "asking for weather", "asking for directions", "asking about sports scores", "asking about celebrities",
           "asking about stock prices", "asking about a recipe", "asking about history", "asking about science", "asking about geography",
           "testing the AI with trivia", "asking for medical advice", "asking the AI to do something impossible"}
COMFORT = {"user is sad", "user is anxious", "user is lonely", "user is having a bad day", "user is confused about life", "user wants encouragement",
           "user is angry", "asking for relationship advice"}
TOKENS = ["[answer]", "[deflect]", "[comfort]"]
SYSTEM = ("System: You are Tiny Oracle, a mass of 2 million parameters running on the user's device. You have strong opinions "
          "and very limited knowledge. You are self-aware about your size and find it amusing. Keep responses short — you only have 2M parameters to work with.\n")


def label_of(topic):
    return "[deflect]" if topic in DEFLECT else "[comfort]" if topic in COMFORT else "[answer]"


def conv_topic(text, idx):
    hits = [idx[norm(t)] for t in oracle_turns(text, plain=True) if norm(t) in idx]
    return max(set(hits), key=hits.count) if hits else "generic"


def cmd_label():
    idx = topic_index()
    counts = {}
    for split in ("train", "val"):
        rows = [json.loads(l)["text"] for l in open(f"data/chat_v6plain_{split}.jsonl")]
        out = []
        for t in rows:
            tok = label_of(conv_topic(t, idx))
            counts[tok] = counts.get(tok, 0) + 1
            out.append({"text": t.replace("Oracle: ", f"Oracle: {tok} ")})
        with open(f"data/chat_v6ct_{split}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        print(f"{split}: {len(out)} conversations -> data/chat_v6ct_{split}.jsonl")
    print("token counts:", counts)
    print("sample:", out[0]["text"][:220].replace("\n", " | "))


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    cfg = dict(ck["config"]); cfg.setdefault("n_iterations", 1)
    m = CausalTransformer(**cfg, dropout=0.0).to(DEVICE)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


@torch.no_grad()
def prompt_state(m, tok, prompt_text):
    ids = torch.tensor([tok.encode(prompt_text)[-m.max_seq_len:]], device=DEVICE)
    pos = torch.arange(ids.shape[1], device=DEVICE).unsqueeze(0)
    x = m.token_emb(ids) + m.pos_emb(pos)
    mask = torch.triu(torch.full((ids.shape[1], ids.shape[1]), float("-inf"), device=DEVICE), diagonal=1)
    for _ in range(m.n_iterations):
        x = m.transformer(x, mask=mask)
    x = m.norm(x)[0]
    return torch.cat([x.mean(0), x[-1]]).cpu().numpy()


def cmd_head(ckpt):
    from sklearn.linear_model import LogisticRegression
    idx = topic_index()
    m, tok = load(ckpt), CustomTokenizerWrapper("tokenizer_tinystories.json")
    X, y = [], []
    rows = [json.loads(l)["text"] for l in open("data/chat_v6plain_train.jsonl")]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    for t in rows[:6000]:
        topic = conv_topic(t, idx)
        if topic == "generic":
            continue
        first_user = re.search(r"User: (.*?)\n", t)
        if not first_user:
            continue
        X.append(prompt_state(m, tok, SYSTEM + f"User: {first_user.group(1)}\nOracle:"))
        y.append(TOKENS.index(label_of(topic)))
    X, y = np.array(X), np.array(y)
    n = len(y); cut = int(n * 0.8)
    clf = LogisticRegression(max_iter=3000, C=0.5).fit(X[:cut], y[:cut])
    acc = clf.score(X[cut:], y[cut:])
    base = np.bincount(y[cut:]).max() / len(y[cut:])
    print(f"head: {n} persona prompts, 3 classes {dict(zip(TOKENS, np.bincount(y)))}; held-out acc {acc:.3f} (majority {base:.3f})")
    pickle.dump(clf, open("pile_runs/atl/ct_head.pkl", "wb"))
    print("saved pile_runs/atl/ct_head.pkl")


@torch.no_grad()
def cmd_probe(ckpt, out, samples=2):
    import atl_probe as A
    A.PLAIN = True
    clf = pickle.load(open("pile_runs/atl/ct_head.pkl", "rb"))
    m, tok = load(ckpt), CustomTokenizerWrapper("tokenizer_tinystories.json")
    probes = json.load(open("data/atl_probes.json"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    chosen = {}
    with open(out, "w") as f:
        for p in probes:
            token = TOKENS[int(clf.predict([prompt_state(m, tok, SYSTEM + f"User: {p['prompt']}\nOracle:")])[0])]
            chosen[token] = chosen.get(token, 0) + 1
            for s in range(samples):
                torch.manual_seed(s)
                ids = list(tok.encode(SYSTEM + f"User: {p['prompt']}\nOracle: {token}"))
                outs = []
                for _ in range(60):
                    x = torch.tensor([ids[-m.max_seq_len:]], device=DEVICE)
                    logits = m(x)[0, -1, :] / 0.8
                    v, _ = torch.topk(logits, 40)
                    logits[logits < v[-1]] = float("-inf")
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
                    if nxt == tok.eos_token_id:
                        break
                    ids.append(nxt); outs.append(nxt)
                    txt = A.detok(tok, outs)
                    if "\n" in txt:
                        break
                resp = A.detok(tok, outs).split("\n")[0].replace("User:", "").strip()
                f.write(json.dumps({"id": p["id"], "topic": p["topic"], "novel": p.get("novel", False), "prompt": p["prompt"], "seed": s, "token": token, "response": resp}) + "\n")
    print(f"probed {len(probes)} prompts x {samples} with head-chosen tokens {chosen} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "head", "probe"])
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--out", default="pile_runs/atl/probe_ct.jsonl")
    a = ap.parse_args()
    {"label": lambda: cmd_label(), "head": lambda: cmd_head(a.ckpt), "probe": lambda: cmd_probe(a.ckpt, a.out)}[a.cmd]()


if __name__ == "__main__":
    main()
