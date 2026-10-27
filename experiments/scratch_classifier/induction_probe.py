"""Induction-head signature (Olsson et al. 2022, arXiv 2209.11895): feed a random token sequence repeated twice; a model with
induction circuits predicts the second copy far better than the first (loss drops on the repeat). Reports mean token loss on
first vs second half, 64 random sequences of length 2x60.   python3 induction_probe.py --ckpt A.pt B.pt"""
import argparse, random, torch, torch.nn.functional as F
from atl_probe import load, DEVICE
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", nargs="+", required=True); ap.add_argument("--len", type=int, default=60); ap.add_argument("--n", type=int, default=64); a = ap.parse_args()
    rng = random.Random(0); seqs = [[rng.randrange(10, 4000) for _ in range(a.len)] for _ in range(a.n)]
    for ck in a.ckpt:
        m, _ = load(ck); m.eval(); l1 = l2 = 0.0
        with torch.no_grad():
            for s in seqs:
                x = torch.tensor([s + s], device=DEVICE); logits = m(x); logits = logits[0] if isinstance(logits, tuple) else logits
                loss = F.cross_entropy(logits[0, :-1], x[0, 1:], reduction="none")
                l1 += loss[: a.len - 1].mean().item(); l2 += loss[a.len:].mean().item()
        print(f"[{ck.split('/')[-1]}] random-repeat loss: first copy {l1/a.n:.2f}  second copy {l2/a.n:.2f}  (induction present if the second is much lower; chance ≈ {torch.log(torch.tensor(3990.)).item():.2f})")
