# Does a 4096-way tied output (only 64 ids used) block induction? Synthetic ids, variable period, CPU, d128. tied vs untied.
import sys, torch, torch.nn.functional as F, time
sys.path.insert(0, '/Users/nico/Documents/Projects/jeffy/experiments/scratch_classifier')
from train_lm import CausalTransformer
torch.manual_seed(0); V, T, B = 4096, 128, 64
def batch():
    x = torch.empty(B, T, dtype=torch.long); first = torch.zeros(B, T, dtype=torch.bool)
    for i in range(B):
        L = int(torch.randint(16, 60, (1,))); s = torch.randint(0, 64, (L,)) * 37 % V; reps = (T + L - 1) // L
        x[i] = s.repeat(reps)[:T]; first[i, :L] = True
    return x, first
for tie in (True, False):
    m = CausalTransformer(vocab_size=V, max_seq_len=T, d_model=128, n_heads=4, n_layers=2, d_ff=512, dropout=0.1, tie_weights=tie); opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.01); t0 = time.time()
    for step in range(1501):
        x, first = batch(); lg = m(x); loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            m.eval()
            with torch.no_grad():
                x, first = batch(); lg = m(x); l = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction='none').view(B, -1); f = first[:, 1:]
                print(f"tied={tie} step {step:4d}: first pass {l[f].mean():.2f}  repeats {l[~f].mean():.2f}  (chance 4.16)  {time.time()-t0:.0f}s", flush=True)
            m.train()
