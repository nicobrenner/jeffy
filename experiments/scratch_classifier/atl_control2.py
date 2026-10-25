"""CT v2: the LM is its own classifier. Each Oracle turn starts with a control token the model must PREDICT (classifier
role: next-token prediction right after "Oracle:") and then CONSUME (generator role: the response is conditioned on it).

v1 (atl_control.py) failed by design: 96% of tokens were [answer] (0.25 bits), so the LM learned that a token sits there,
not which one (NLL true 2.4798 vs wrong 2.4846). v2 fixes the label: a per-turn dialogue act of the preceding user turn
(wh-question / yes-no question / greeting-closing / statement) with the persona override (fact = knowledge or live-data
topics the Oracle must deflect, hug = emotional topics) -> ~1.6 bits, majority class ~59%.

  python3 atl_control2.py label                 # chat_v6plain -> data/chat_v6ct2_{train,val}.jsonl (+ class counts, entropy)
  python3 atl_control2.py eval CKPT [--n 400]   # consumption: response NLL with true / wrong / no token; classifier: LM's
                                                #   own token prediction accuracy vs the label; per-class breakdown
  python3 atl_control2.py probe CKPT --out F    # generate with the model choosing its own token first (self-conditioning)

Markers are `*word` because `[`/`]`/`<`/`>` are <unk> in the 5,500-token TinyStories BPE and `*` is a real token (id 11).
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from atl_control import DEFLECT, COMFORT, SYSTEM
from atl_val_loss import oracle_turns, topic_index, norm
from train_lm import CausalTransformer, CustomTokenizerWrapper

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
CLASSES = ["wh", "yn", "hi", "tell", "fact", "hug"]
MARK = {c: f"*{c}" for c in CLASSES}
GREET = r"^(hi|hello|hey|good (morning|evening|afternoon|night)|yo|sup|thanks|thank you|thx|bye|goodbye|see you|ok|okay|cool|nice|great|awesome|lol|haha|sure|yes|no|yeah|nope)\b"
WH = r"\b(what|who|whom|whose|where|when|why|how|which)\b"


def act_of(user_turn):
    s = user_turn.strip().lower()
    if re.match(GREET, s) and len(s.split()) <= 6:
        return "hi"
    if "?" in s:
        return "wh" if re.search(WH + r"[^?]*\?", s) else "yn"
    return "tell"


def turn_pairs(text):
    """[(user_turn, oracle_turn_text, span_start_of_oracle_content)] for a plain-format conversation."""
    out = []
    for m in re.finditer(r"User: (.*?)\nOracle: (.*?)(?=\nUser:|\nSystem:|$)", text, flags=re.S):
        out.append((m.group(1), m.group(2), m.start(2)))
    return out


def label_turn(user_turn, oracle_turn, idx):
    topic = idx.get(norm(oracle_turn.strip()))
    if topic in DEFLECT:
        return "fact"
    if topic in COMFORT:
        return "hug"
    return act_of(user_turn)


def cmd_label():
    idx = topic_index()
    for split in ("train", "val"):
        rows = [json.loads(l)["text"] for l in open(f"data/chat_v6plain_{split}.jsonl")]
        counts = Counter()
        out = []
        for t in rows:
            pieces, pos = [], 0
            for user, orc, s in turn_pairs(t):
                lab = label_turn(user, orc, idx)
                counts[lab] += 1
                pieces.append(t[pos:s] + MARK[lab] + " ")
                pos = s
            pieces.append(t[pos:])
            out.append({"text": "".join(pieces)})
        with open(f"data/chat_v6ct2_{split}.jsonl", "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        n = sum(counts.values())
        H = -sum(v / n * math.log2(v / n) for v in counts.values())
        print(f"{split}: {len(out)} conversations, {n} oracle turns -> data/chat_v6ct2_{split}.jsonl")
        print(f"  classes {dict(counts)}  majority {max(counts.values())/n:.3f}  entropy {H:.2f} bits")
    print("sample:", out[0]["text"][:300].replace("\n", " | "))


def load(ckpt):
    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    cfg = dict(ck["config"]); cfg.setdefault("n_iterations", 1)
    m = CausalTransformer(**cfg, dropout=0.0).to(DEVICE)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


@torch.no_grad()
def resp_nll(m, tok, prefix, response):
    """Mean NLL per token of `response` given `prefix` (prefix tokens are context only)."""
    a = len(tok.encode(prefix))
    ids = tok.encode(prefix + response)[: m.max_seq_len]
    if a >= len(ids):
        return None
    x = torch.tensor([ids], device=DEVICE)
    logp = F.log_softmax(m(x)[0, :-1].float(), dim=-1)
    nll = -logp.gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)
    return nll[a - 1:].mean().item(), len(ids) - a


@torch.no_grad()
def marker_logprobs(m, tok, prefix):
    """Sequence log-prob of each class marker right after `prefix` (= the LM acting as classifier)."""
    a = len(tok.encode(prefix))
    out = {}
    for c in CLASSES:
        ids = tok.encode(prefix + " " + MARK[c])[: m.max_seq_len]
        x = torch.tensor([ids], device=DEVICE)
        logp = F.log_softmax(m(x)[0, :-1].float(), dim=-1)
        out[c] = logp.gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)[a - 1:].sum().item()
    return out


def cmd_eval(ckpt, n_conv, out):
    idx = topic_index()
    m, tok = load(ckpt), CustomTokenizerWrapper("tokenizer_tinystories.json")
    rows = [json.loads(l)["text"] for l in open("data/chat_v6plain_val.jsonl")]
    rng = np.random.default_rng(0)
    rng.shuffle(rows)
    per = defaultdict(lambda: {"true": [], "wrong": [], "none": [], "n_tok": 0, "cls_hit": 0, "cls_n": 0})
    conf = Counter()
    done = 0
    for t in rows:
        pairs = turn_pairs(t)
        if not pairs:
            continue
        done += 1
        if done > n_conv:
            break
        # score only the FIRST oracle turn (the rest would need the earlier turns' markers in context too; keep it clean)
        user, orc, s = pairs[0]
        lab = label_turn(user, orc, idx)
        wrong = CLASSES[(CLASSES.index(lab) + 1 + int(rng.integers(len(CLASSES) - 1))) % len(CLASSES)]
        prefix = t[:s]  # "...User: ...\nOracle: "
        assert prefix.endswith("Oracle: ")
        resp = orc.strip()
        r_true = resp_nll(m, tok, prefix + MARK[lab] + " ", resp)
        r_wrong = resp_nll(m, tok, prefix + MARK[wrong] + " ", resp)
        r_none = resp_nll(m, tok, prefix, resp)
        if not (r_true and r_wrong and r_none):
            continue
        d = per[lab]
        d["true"].append(r_true[0]); d["wrong"].append(r_wrong[0]); d["none"].append(r_none[0]); d["n_tok"] += r_true[1]
        lp = marker_logprobs(m, tok, prefix.rstrip())
        pred = max(lp, key=lp.get)
        d["cls_n"] += 1; d["cls_hit"] += int(pred == lab); conf[(lab, pred)] += 1
    tot = {"true": [], "wrong": [], "none": [], "cls_hit": 0, "cls_n": 0}
    print(f"\n{ckpt}: first oracle turn of {sum(d['cls_n'] for d in per.values())} val conversations")
    print(f"{'class':<6}{'n':>5}{'NLL true':>10}{'wrong':>8}{'none':>8}{'Δ wrong-true':>14}{'cls acc':>9}")
    for c in CLASSES:
        d = per[c]
        if not d["true"]:
            continue
        for k in ("true", "wrong", "none"):
            tot[k] += d[k]
        tot["cls_hit"] += d["cls_hit"]; tot["cls_n"] += d["cls_n"]
        print(f"{c:<6}{len(d['true']):>5}{np.mean(d['true']):>10.4f}{np.mean(d['wrong']):>8.4f}{np.mean(d['none']):>8.4f}"
              f"{np.mean(d['wrong']) - np.mean(d['true']):>14.4f}{d['cls_hit']/d['cls_n']:>9.3f}")
    maj = max(len(per[c]["true"]) for c in CLASSES) / len(tot["true"])
    print(f"{'all':<6}{len(tot['true']):>5}{np.mean(tot['true']):>10.4f}{np.mean(tot['wrong']):>8.4f}{np.mean(tot['none']):>8.4f}"
          f"{np.mean(tot['wrong']) - np.mean(tot['true']):>14.4f}{tot['cls_hit']/tot['cls_n']:>9.3f}  (majority {maj:.3f})")
    print("confusion (label -> predicted):", {f"{a}->{b}": v for (a, b), v in sorted(conf.items()) if a != b and v})
    res = {"ckpt": ckpt, "n": len(tot["true"]), "nll_true": float(np.mean(tot["true"])), "nll_wrong": float(np.mean(tot["wrong"])),
           "nll_none": float(np.mean(tot["none"])), "cls_acc": tot["cls_hit"] / tot["cls_n"], "majority": maj,
           "per_class": {c: {"n": len(per[c]["true"]), "true": float(np.mean(per[c]["true"])), "wrong": float(np.mean(per[c]["wrong"])),
                             "none": float(np.mean(per[c]["none"])), "cls_acc": per[c]["cls_hit"] / max(per[c]["cls_n"], 1)}
                         for c in CLASSES if per[c]["true"]},
           "confusion": {f"{a}->{b}": v for (a, b), v in conf.items()}}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print("saved", out)


@torch.no_grad()
def cmd_probe(ckpt, out, samples=2, temperature=0.8):
    import atl_probe as A
    A.PLAIN = True
    m, tok = load(ckpt), CustomTokenizerWrapper("tokenizer_tinystories.json")
    probes = json.load(open("data/atl_probes.json"))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    chosen = Counter()
    with open(out, "w") as f:
        for p in probes:
            prefix = SYSTEM + f"User: {p['prompt']}\nOracle:"
            lp = marker_logprobs(m, tok, prefix)
            c = max(lp, key=lp.get)
            chosen[c] += 1
            for s in range(samples):
                torch.manual_seed(s)
                ids = list(tok.encode(prefix + " " + MARK[c]))
                outs = []
                for _ in range(60):
                    x = torch.tensor([ids[-m.max_seq_len:]], device=DEVICE)
                    logits = m(x)[0, -1, :] / temperature
                    v, _ = torch.topk(logits, 40)
                    logits[logits < v[-1]] = float("-inf")
                    nxt = torch.multinomial(F.softmax(logits, dim=-1), 1).item()
                    if nxt == tok.eos_token_id:
                        break
                    ids.append(nxt); outs.append(nxt)
                    if "\n" in A.detok(tok, outs):
                        break
                resp = A.detok(tok, outs).split("\n")[0].replace("User:", "").strip()
                f.write(json.dumps({"id": p["id"], "topic": p["topic"], "novel": p.get("novel", False), "prompt": p["prompt"], "seed": s,
                                    "token": MARK[c], "heuristic": act_of(p["prompt"]), "response": resp}) + "\n")
    print(f"probed {len(probes)} prompts x {samples}; self-chosen tokens {dict(chosen)} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["label", "eval", "probe"])
    ap.add_argument("ckpt", nargs="?")
    ap.add_argument("--n", type=int, default=400, help="eval: number of val conversations")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.cmd == "label":
        cmd_label()
    elif a.cmd == "eval":
        cmd_eval(a.ckpt, a.n, a.out or f"pile_runs/atl/ct2_eval_{Path(a.ckpt).stem.replace('best_lm_', '')}.json")
    else:
        cmd_probe(a.ckpt, a.out or "pile_runs/atl/probe_ct2.jsonl")


if __name__ == "__main__":
    main()
