"""In-distribution induction signature: a real user turn from the val corpus repeated twice inside a chat context; loss on the
repeat vs the first pass. Random-word probes are far out of distribution for a chat model; this one is not.
  python3 induction_probe3.py --ckpt A.pt [B.pt ...]"""
import argparse, json, re, random, torch, torch.nn.functional as F
from atl_probe import load, SYSTEM, DEVICE
from train_lm import CustomTokenizerWrapper
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--n", type=int, default=64); a = ap.parse_args()
    tok = CustomTokenizerWrapper("tokenizer_chat_v2.json"); rng = random.Random(0)
    turns = [u for l in open("data/chat_v2_fix_val.jsonl") for u in re.findall(r"<\|user\|> ([^\n]{60,200})", json.loads(l)["text"])]
    rng.shuffle(turns); turns = turns[: a.n]
    for ck in a.ckpt:
        m, _ = load(ck); m.eval(); A = B = 0.0
        with torch.no_grad():
            for u in turns:
                pre = tok.encode(SYSTEM + "<|user|> "); seg = tok.encode(u + " " + u); ids = (pre + seg)[:256]; half = len(tok.encode(u)); off = len(pre)
                x = torch.tensor([ids], device=DEVICE); lg = m(x); lg = lg[0] if isinstance(lg, tuple) else lg
                loss = F.cross_entropy(lg[0, :-1], x[0, 1:], reduction="none")
                A += loss[off : off + half - 1].mean().item(); B += loss[off + half + 1 : len(ids) - 1].mean().item()
        print(f"[{ck.split('/')[-1]}] real user turn repeated: first pass {A/len(turns):.2f}  repeat {B/len(turns):.2f}  (induction → repeat far below first)")
