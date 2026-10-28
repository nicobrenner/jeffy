"""Induction signature of a TRAINED checkpoint at fixed periods: random in-vocab token sequences repeated with period p; loss on the
first pass vs on the repeats. A copy/induction circuit drives repeat loss far below first-pass loss (chance = ln(vocab)).
  python3 induct_eval.py --ckpt A.pt B.pt [--periods 3 5 7]"""
import argparse, torch, torch.nn.functional as F
from atl_probe import load, DEVICE
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--periods", nargs="+", type=int, default=[3, 5, 7]); ap.add_argument("--n", type=int, default=64); a = ap.parse_args()
    torch.manual_seed(0)
    for ck in a.ckpt:
        m, _ = load(ck); m.eval(); V = m.vocab_size; T = 120; out = []
        for p in a.periods:
            x = torch.randint(5, V, (a.n, p)).repeat(1, T // p + 1)[:, :T].to(DEVICE); first = torch.zeros(a.n, T, dtype=torch.bool); first[:, :p] = True
            with torch.no_grad(): lg = m(x); l = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction="none").view(a.n, -1).cpu(); f = first[:, 1:]
            out.append(f"p{p}: first {l[f].mean():.2f} repeat {l[~f].mean():.2f}")
        print(f"[{ck.split('/')[-1]}] induction (chance {torch.log(torch.tensor(float(V))):.2f}): " + " | ".join(out))
