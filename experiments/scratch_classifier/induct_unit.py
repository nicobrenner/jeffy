# Standalone induction unit test: the SAME CausalTransformer class, tiny config, pure token-id copy task, CPU, plain AdamW loop.
import sys, torch, torch.nn.functional as F, time
sys.path.insert(0, '/Users/nico/Documents/Projects/jeffy/experiments/scratch_classifier')
from train_lm import CausalTransformer
torch.manual_seed(0); V, L, B = 64, 40, 64
def batch():
    s = torch.randint(0, V, (B, L)); return torch.cat([s, s], 1)
for cfg in [dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, dropout=0.0), dict(d_model=128, n_heads=4, n_layers=2, d_ff=512, dropout=0.1)]:
    m = CausalTransformer(vocab_size=V, max_seq_len=2 * L, **cfg); opt = torch.optim.AdamW(m.parameters(), lr=1e-3); t0 = time.time()
    for step in range(1501):
        x = batch(); lg = m(x); loss = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        if step % 300 == 0:
            m.eval()
            with torch.no_grad():
                x = batch(); lg = m(x); l = F.cross_entropy(lg[:, :-1].reshape(-1, V), x[:, 1:].reshape(-1), reduction='none').view(B, -1)
                print(f"dropout {cfg['dropout']} step {step:4d}: first pass {l[:, :L-1].mean():.2f}  repeat {l[:, L:].mean():.2f}  (chance {torch.log(torch.tensor(64.)):.2f})  {time.time()-t0:.0f}s", flush=True)
            m.train()
