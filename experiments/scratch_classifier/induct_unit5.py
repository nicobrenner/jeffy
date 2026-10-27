# The Oracle's own width (d256 L2 H4 dff1024, tied, V=4096 all ids used): post-LN vs pre-LN, lr 3e-4 vs 1e-3, fair task, MPS.
import sys, torch, torch.nn.functional as F, time
sys.path.insert(0, '/Users/nico/Documents/Projects/jeffy/experiments/scratch_classifier')
from train_lm import CausalTransformer
DEV = torch.device('mps'); V, T, B = 4096, 128, 64
def batch():
    x = torch.empty(B, T, dtype=torch.long); first = torch.zeros(B, T, dtype=torch.bool)
    for i in range(B):
        L = int(torch.randint(16, 60, (1,))); s = torch.randint(0, V, (L,)); reps = (T + L - 1) // L
        x[i] = s.repeat(reps)[:T]; first[i, :L] = True
    return x.to(DEV), first.to(DEV)
for pre_ln, lr in ((False, 3e-4), (True, 3e-4), (True, 1e-3), (False, 1e-3)):
    torch.manual_seed(0); m = CausalTransformer(vocab_size=V, max_seq_len=T, d_model=256, n_heads=4, n_layers=2, d_ff=1024, dropout=0.1)
    for layer in m.transformer.layers: layer.norm_first = pre_ln
    m = m.to(DEV); opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=0.01); t0 = time.time()
    for step in range(2001):
        x, first = batch(); lg = m(x); loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 500 == 0:
            m.eval()
            with torch.no_grad():
                x, first = batch(); lg = m(x); l = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction='none').view(B, -1); f = first[:, 1:]
                print(f"pre_ln={pre_ln} lr={lr} step {step:4d}: first pass {l[f].mean():.2f}  repeats {l[~f].mean():.2f}  (chance 8.32)  {time.time()-t0:.0f}s", flush=True)
            m.train()
