"""Function-preserving growth of a CausalTransformer (Net2Net, Chen/Goodfellow/Shlens 2015, adapted to post-norm
TransformerEncoderLayer blocks with tied embeddings). Nico's programme (2026-10-26): grow a small trained model into a
bigger one in stages, verifying at each step that the function is unchanged, then keep training.

Widening (d_model, n_heads, d_ff): every new hidden unit is a copy of an existing one (random choice when the new width is
not a multiple of the old); outgoing weights of a copied unit are divided by its copy count so sums are unchanged.
Bookkeeping for post-norm blocks: the residual stream before an LN may be at any consistent scale (LN is scale invariant),
so weights that READ the stream are duplicated (and divided by the copy count), weights that WRITE to it are duplicated
with the same scale as the residual they add to. The tied embedding is read by the input (pre-LN, scale 1/count) and by the
head (post-LN, needs 1/count) — consistent, so one tensor serves both.
Deepening: insert blocks whose two branches output zero (out_proj = 0, linear2 = 0); in a post-norm stack such a block maps
x -> LN2(LN1(x)) = x when x is already an LN output (true for every block but the first, and we only insert after block 1).

  python3 grow_model.py checkpoints/best_lm_2m_scaling.pt --d-model 384 --n-heads 8 --d-ff 1280 --n-layers 4 --out checkpoints/grown_2m_to_8m.pt
"""
import argparse
import copy

import numpy as np
import torch
import torch.nn as nn

from train_lm import CausalTransformer


def widen_map(old, new, rng):
    """index map g: new index -> old index (identity for the first `old`, random copies after), and copy counts."""
    g = np.concatenate([np.arange(old), rng.integers(0, old, size=new - old)])
    counts = np.bincount(g, minlength=old)[g].astype(np.float32)
    return g, counts


def widen(model, d_new, h_new, ff_new, seed=0):
    rng = np.random.default_rng(seed)
    d_old, V = model.d_model, model.vocab_size
    L = len(model.transformer.layers); ff_old = model.transformer.layers[0].linear1.out_features; h_old = model.transformer.layers[0].self_attn.num_heads
    assert d_new % h_new == 0 and h_new % h_old == 0 and d_new // h_new == d_old // h_old, "grow heads by duplication: head_dim must stay the same"
    # residual-stream map: LayerNorm is only invariant under a UNIFORM scale, so the stream must be widened by an exact
    # integer factor with every column copied the same number of times (random extra copies break the LN statistics).
    assert d_new % d_old == 0, "d_model must grow by an integer factor (LayerNorm invariance)"
    g = np.tile(np.arange(d_old), d_new // d_old); cnt = np.full(d_new, d_new // d_old, np.float32)
    gh = np.concatenate([np.arange(h_old)] * (h_new // h_old))  # head map: duplicate heads
    hd = d_old // h_old
    # per-head index map over the q/k/v projection outputs: head j of the new model = head gh[j] of the old
    gq = np.concatenate([np.arange(gh[j] * hd, (gh[j] + 1) * hd) for j in range(h_new)]); cq = np.bincount(gq, minlength=d_old)[gq].astype(np.float32)
    new = CausalTransformer(vocab_size=V, d_model=d_new, n_heads=h_new, n_layers=L, d_ff=ff_new, max_seq_len=model.max_seq_len,
                            dropout=0.0, tie_weights=True, n_iterations=model.n_iterations)
    sd = model.state_dict(); nd = {}
    t = lambda k: sd[k].detach().clone()
    g_t, cnt_t, gq_t, cq_t = map(torch.as_tensor, (g, cnt, gq, cq))
    # embeddings: read pre-LN by block 1 (scale 1/cnt) and used post-LN by the tied head (also 1/cnt)
    nd["token_emb.weight"] = t("token_emb.weight")[:, g_t] / cnt_t
    nd["pos_emb.weight"] = t("pos_emb.weight")[:, g_t] / cnt_t
    for i in range(L):
        p = f"transformer.layers.{i}."
        first = i == 0
        # attention in_proj: (3d, d). Reads the stream: block 1 reads the halved-dup embedding (no extra scaling), later blocks read LN outputs (dup full -> divide by cnt)
        W = t(p + "self_attn.in_proj_weight"); b = t(p + "self_attn.in_proj_bias")
        Wq, Wk, Wv = W[:d_old], W[d_old:2 * d_old], W[2 * d_old:]; bq, bk, bv = b[:d_old], b[d_old:2 * d_old], b[2 * d_old:]
        rd = (1.0 if first else 1.0 / cnt_t)  # read scaling per input column
        Wq2 = Wq[gq_t][:, g_t] * rd; Wk2 = Wk[gq_t][:, g_t] * rd; Wv2 = Wv[gq_t][:, g_t] * rd
        nd[p + "self_attn.in_proj_weight"] = torch.cat([Wq2, Wk2, Wv2]); nd[p + "self_attn.in_proj_bias"] = torch.cat([bq[gq_t], bk[gq_t], bv[gq_t]])
        # out_proj: (d, d) reads concatenated heads (duplicated heads -> divide by cq), writes to the residual: block 1 residual is at scale 1/cnt, later ones at scale 1 (LN output)
        Wo = t(p + "self_attn.out_proj.weight"); bo = t(p + "self_attn.out_proj.bias")
        ws = (1.0 / cnt_t if first else 1.0)
        nd[p + "self_attn.out_proj.weight"] = (Wo[g_t][:, gq_t] / cq_t) * ws[:, None] if first else Wo[g_t][:, gq_t] / cq_t
        nd[p + "self_attn.out_proj.bias"] = bo[g_t] * (ws if first else 1.0)
        # LN1 reads the (scaled) stream and outputs full-scale dup: gamma/beta duplicated
        nd[p + "norm1.weight"] = t(p + "norm1.weight")[g_t]; nd[p + "norm1.bias"] = t(p + "norm1.bias")[g_t]
        # FF: linear1 reads LN1 output (dup full -> /cnt on inputs; widen hidden by copying units), linear2 writes at scale 1 (residual = LN1 output), divide by hidden copy count
        gf, cf = widen_map(ff_old, ff_new, rng); gf_t, cf_t = torch.as_tensor(gf), torch.as_tensor(cf)
        W1 = t(p + "linear1.weight"); b1 = t(p + "linear1.bias"); W2 = t(p + "linear2.weight"); b2 = t(p + "linear2.bias")
        nd[p + "linear1.weight"] = W1[gf_t][:, g_t] / cnt_t; nd[p + "linear1.bias"] = b1[gf_t]
        nd[p + "linear2.weight"] = W2[g_t][:, gf_t] / cf_t; nd[p + "linear2.bias"] = b2[g_t]
        nd[p + "norm2.weight"] = t(p + "norm2.weight")[g_t]; nd[p + "norm2.bias"] = t(p + "norm2.bias")[g_t]
    nd["norm.weight"] = t("norm.weight")[g_t]; nd["norm.bias"] = t("norm.bias")[g_t]
    missing, unexpected = new.load_state_dict(nd, strict=False)
    assert not [m for m in missing if m != "lm_head.weight"], missing
    return new


def deepen(model, n_layers_new):
    """insert zero-branch (identity) blocks after the first block."""
    L = len(model.transformer.layers); assert n_layers_new > L
    new = CausalTransformer(vocab_size=model.vocab_size, d_model=model.d_model, n_heads=model.transformer.layers[0].self_attn.num_heads,
                            n_layers=n_layers_new, d_ff=model.transformer.layers[0].linear1.out_features, max_seq_len=model.max_seq_len,
                            dropout=0.0, tie_weights=True, n_iterations=model.n_iterations)
    new.token_emb.weight.data.copy_(model.token_emb.weight.data); new.pos_emb.weight.data.copy_(model.pos_emb.weight.data)
    new.norm.load_state_dict(model.norm.state_dict())
    old_layers = list(model.transformer.layers); k = n_layers_new - L
    order = [old_layers[0]] + [None] * k + old_layers[1:]  # identity blocks right after block 1
    for dst, src in zip(new.transformer.layers, order):
        if src is None:
            with torch.no_grad():
                dst.self_attn.out_proj.weight.zero_(); dst.self_attn.out_proj.bias.zero_(); dst.linear2.weight.zero_(); dst.linear2.bias.zero_()
                dst.norm1.weight.fill_(1.0); dst.norm1.bias.zero_(); dst.norm2.weight.fill_(1.0); dst.norm2.bias.zero_()
        else:
            dst.load_state_dict(src.state_dict())
    return new


@torch.no_grad()
def check(a, b, V, n=4, T=64, seed=0):
    g = torch.Generator().manual_seed(seed); x = torch.randint(0, V, (n, T), generator=g)
    a.eval(); b.eval(); la, lb = a(x), b(x)
    return (la - lb).abs().max().item(), (la.log_softmax(-1) - lb.log_softmax(-1)).abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("--d-model", type=int, default=384); ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-ff", type=int, default=1280); ap.add_argument("--n-layers", type=int, default=4); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False); cfg = dict(ck["config"]); cfg.setdefault("n_iterations", 1)
    m = CausalTransformer(**cfg, dropout=0.0); m.load_state_dict(ck["model_state"]); m.eval()
    V = cfg["vocab_size"]
    print(f"source: {cfg} ({sum(p.numel() for n, p in m.named_parameters() if not n.startswith('lm_head'))/1e6:.2f}M)")
    w = widen(m, a.d_model, a.n_heads, a.d_ff); w.eval()
    dl, dlp = check(m, w, V); print(f"after widening to d{a.d_model} h{a.n_heads} ff{a.d_ff}: max |Δlogit| {dl:.2e}, max |Δlogprob| {dlp:.2e}")
    d = deepen(w, a.n_layers) if a.n_layers > len(w.transformer.layers) else w; d.eval()
    dl, dlp = check(m, d, V); print(f"after deepening to L{a.n_layers}: max |Δlogit| {dl:.2e}, max |Δlogprob| {dlp:.2e}")
    nparams = sum(p.numel() for n, p in d.named_parameters() if not n.startswith("lm_head"))
    out_cfg = dict(vocab_size=V, d_model=a.d_model, n_heads=a.n_heads, n_layers=a.n_layers, d_ff=a.d_ff, max_seq_len=cfg["max_seq_len"], tie_weights=True, n_iterations=cfg["n_iterations"])
    torch.save({"config": out_cfg, "model_state": d.state_dict(), "tokenizer_path": ck.get("tokenizer_path"), "epoch": 0, "val_loss": ck.get("best_val_loss"),
                "best_val_loss": float("inf"), "grown_from": a.ckpt}, a.out)
    print(f"saved {a.out}: {nparams/1e6:.2f}M params")


if __name__ == "__main__":
    main()
