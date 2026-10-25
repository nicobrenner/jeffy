"""Grader-free ATL metric: per-topic held-out loss (nats/token over oracle turns) for one or more checkpoints.

Topics are recovered by matching each validation conversation's oracle turns to the generated/hand-written
records that carry a 'topic' field; generic filler conversations are reported under 'generic'.

Usage: python3 atl_val_loss.py --val data/chat_v6plain_val.jsonl --tokenizer tokenizer_tinystories.json --plain \
           checkpoints/best_lm_oracle_v7a.pt checkpoints/best_lm_atl_static_r1.pt checkpoints/best_lm_atl_adaptive_r1.pt
"""

import argparse
import glob
import json
import re
import statistics as st
from pathlib import Path

import torch
import torch.nn.functional as F

from train_lm import CausalTransformer, CustomTokenizerWrapper

CT = False
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def oracle_turns(text, plain):
    pat = r"Oracle: (.*?)(?=\nUser:|\nSystem:|$)" if plain else r"<\|oracle\|>(.*?)(?=<\|user\||<\|system\||<eos>|$)"
    return [re.sub(r"^(\[(answer|deflect|comfort)\]|\*(wh|yn|hi|tell|fact|hug)) ?", "", t.strip()) for t in re.findall(pat, text, flags=re.S)]


def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower())[:80]


def topic_index():
    idx = {}
    for p in glob.glob("data/oracle_gen/*.jsonl") + glob.glob("data/oracle_v3_batch*.jsonl"):
        for l in open(p):
            r = json.loads(l)
            for t in oracle_turns(r["text"], plain=False):
                idx[norm(t)] = r.get("topic", "handwritten")
    return idx


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = dict(ck["config"]); cfg.setdefault("n_iterations", 1)
    m = CausalTransformer(**cfg, dropout=0.0).to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


@torch.no_grad()
def conv_loss(m, tok, text, plain):
    """Mean NLL per token over oracle-response tokens only (prompt tokens are context, not scored)."""
    ids = tok.encode(text)[: m.max_seq_len]
    x = torch.tensor([ids], device=DEVICE)
    logp = F.log_softmax(m(x)[0, :-1].float(), dim=-1)
    nll = -logp.gather(-1, x[0, 1:].unsqueeze(-1)).squeeze(-1)
    # score positions inside oracle turns: find the token spans by re-encoding prefixes
    marker = "Oracle:" if plain else "<|oracle|>"
    ender = "\nUser:" if plain else "<|user|>"
    mask = torch.zeros(len(ids) - 1, dtype=torch.bool, device=DEVICE)
    pos = 0
    while True:
        s = text.find(marker, pos)
        if s < 0:
            break
        e = text.find(ender, s)
        e = len(text) if e < 0 else e
        start = s + len(marker)
        if CT:  # control token ("[answer]" v1 family or "*wh" v2 family) is context, not a scored target
            mt = re.match(r" (\[(answer|deflect|comfort)\]|\*(wh|yn|hi|tell|fact|hug))", text[start:])
            if mt:
                start += mt.end()
        a = len(tok.encode(text[:start]))
        b = len(tok.encode(text[:e]))
        mask[max(a - 1, 0): min(b - 1, len(mask))] = True
        pos = e
    if mask.sum() == 0:
        return None
    return nll[mask].mean().item(), int(mask.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--val", default="data/chat_v6plain_val.jsonl")
    ap.add_argument("--tokenizer", default="tokenizer_tinystories.json")
    ap.add_argument("--plain", action="store_true")
    ap.add_argument("--ct", action="store_true", help="val set carries control tokens after Oracle:; exclude them from scoring")
    ap.add_argument("--out", default="pile_runs/atl/val_loss_by_topic.json")
    args = ap.parse_args()
    global CT
    CT = args.ct
    tok = CustomTokenizerWrapper(args.tokenizer)
    idx = topic_index()
    rows = [json.loads(l)["text"] for l in open(args.val)]
    topics = []
    for t in rows:
        turns = [norm(x) for x in oracle_turns(t, args.plain)]
        hits = [idx[x] for x in turns if x in idx]
        topics.append(max(set(hits), key=hits.count) if hits else "generic")
    print(f"{len(rows)} val conversations; topics recovered for {sum(t != 'generic' for t in topics)}, generic {sum(t == 'generic' for t in topics)}")
    results = {}
    for ck in args.ckpts:
        m = load(ck, DEVICE)
        per_topic, tot_n, tot_l = {}, 0, 0.0
        for text, topic in zip(rows, topics):
            r = conv_loss(m, tok, text, args.plain)
            if r is None:
                continue
            l, n = r
            per_topic.setdefault(topic, []).append(l)
            tot_l += l * n; tot_n += n
        name = Path(ck).stem.replace("best_lm_", "")
        results[name] = {"overall_oracle_nll": tot_l / tot_n, "per_topic": {k: st.mean(v) for k, v in per_topic.items()}, "n_per_topic": {k: len(v) for k, v in per_topic.items()}}
        print(f"{name:<22} oracle-token NLL overall {tot_l / tot_n:.4f} | persona topics {st.mean(st.mean(v) for k, v in per_topic.items() if k != 'generic'):.4f} | generic {st.mean(per_topic.get('generic', [float('nan')])):.4f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(args.out, "w"), indent=1)
    if len(results) > 1:
        names = list(results)
        base = results[names[0]]["per_topic"]
        print(f"\nper-topic delta vs {names[0]} (negative = better), persona topics only:")
        for nm in names[1:]:
            d = {k: results[nm]["per_topic"][k] - base[k] for k in base if k != "generic" and k in results[nm]["per_topic"]}
            best = sorted(d.items(), key=lambda kv: kv[1])[:5]; worst = sorted(d.items(), key=lambda kv: -kv[1])[:3]
            print(f"  {nm}: mean {st.mean(d.values()):+.4f} | improved most: {[(k[:28], round(v, 3)) for k, v in best]} | worsened most: {[(k[:28], round(v, 3)) for k, v in worst]}")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
