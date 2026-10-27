"""In-distribution induction signature: random SINGLE-TOKEN words repeated; per-token loss first pass vs repeat.
  python3 induction_probe2.py --ckpt A.pt [B.pt ...]"""
import argparse, json, random, torch, torch.nn.functional as F
from atl_probe import load, DEVICE
from train_lm import CustomTokenizerWrapper
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--n", type=int, default=48); a = ap.parse_args()
    tok = CustomTokenizerWrapper("tokenizer_chat_v2.json"); W = json.load(open("pile_runs/atl/single_token_words.json")); rng = random.Random(1)
    for ck in a.ckpt:
        m, _ = load(ck); m.eval(); A = B = 0.0
        with torch.no_grad():
            for _ in range(a.n):
                s = " ".join(rng.choice(W) for _ in range(50)); ids = tok.encode(s + " " + s)[:256]; half = len(tok.encode(s))
                x = torch.tensor([ids], device=DEVICE); lg = m(x); lg = lg[0] if isinstance(lg, tuple) else lg
                loss = F.cross_entropy(lg[0, :-1], x[0, 1:], reduction="none"); A += loss[: half - 1].mean().item(); B += loss[half + 1:].mean().item()
        print(f"[{ck.split('/')[-1]}] single-token repeat: first pass {A/a.n:.2f}  repeat {B/a.n:.2f}  (chance ≈ ln(len(W)) = {torch.log(torch.tensor(float(len(W)))).item():.2f}; induction → repeat near 0)")
