"""Shared image prior: a causal byte LM over 28x28 windows with two optional context extensions.

interleave=True : tokens are R,G,B per pixel (PixelCNN-style channel conditioning), 3x sequence length.
l_context=True  : the window is 29x29 whose row 0 and column 0 are the already-decoded neighbours
                  (pixel row above, pixel column to the left, corner); loss is only on the inner 28x28.
                  Decoder-valid in raster patch order; image borders use pad value 128.

Attention uses F.scaled_dot_product_attention(is_causal=True) so no mask buffer is stored.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from train_rope_byte_2d import apply_rope_2d, precompute_freqs_2d

PAD = 128
CONFIGS = {
    "base": dict(interleave=False, l_context=False),
    "chan": dict(interleave=True, l_context=False),
    "ctx": dict(interleave=False, l_context=True),
    "full": dict(interleave=True, l_context=True),
    "full_h4": dict(interleave=True, l_context=True, halo=4),   # 4-px decoded-neighbour strip (32x32 window, 3072 tokens)
    "full_h8": dict(interleave=True, l_context=True, halo=8),   # 36x36 window, 3888 tokens
}


class Block(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d_model // n_heads
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.ff = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))
        self.dropout = dropout

    def forward(self, x, freqs):
        B, S, D = x.shape
        h = self.ln1(x)
        q, k, v = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4).unbind(0)
        q, k = apply_rope_2d(q, freqs), apply_rope_2d(k, freqs)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.out(a.transpose(1, 2).reshape(B, S, D))
        return x + self.ff(self.ln2(x))


class PriorLM(nn.Module):
    def __init__(self, d_model=64, n_layers=4, n_heads=4, patch=28, interleave=True, l_context=True, dropout=0.0, halo=1):
        super().__init__()
        self.patch, self.interleave, self.l_context, self.halo = patch, interleave, l_context, (halo if l_context else 0)
        self.grid = patch + self.halo
        self.n_ch = 3 if interleave else 1
        self.seq_len = self.grid * self.grid * self.n_ch
        self.token_emb = nn.Embedding(256, d_model)
        self.channel_emb = nn.Embedding(3, d_model) if interleave else None
        self.layers = nn.ModuleList(Block(d_model, n_heads, dropout) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, 256, bias=False)
        self.lm_head.weight = self.token_emb.weight

        freqs = precompute_freqs_2d(d_model // n_heads, self.grid, self.grid)
        if interleave:
            freqs = freqs.repeat_interleave(3, dim=0)
        self.register_buffer("freqs", freqs, persistent=False)

        pos = torch.arange(self.grid * self.grid)
        r, c = pos // self.grid, pos % self.grid
        is_patch = (r >= self.halo) & (c >= self.halo)
        self.register_buffer("is_patch", is_patch.repeat_interleave(self.n_ch), persistent=False)
        self.register_buffer("chan_idx", torch.arange(self.seq_len) % 3, persistent=False)

        nn.init.normal_(self.token_emb.weight, std=0.02)
        for blk in self.layers:
            nn.init.normal_(blk.qkv.weight, std=0.02)
            nn.init.normal_(blk.out.weight, std=0.02)

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def to_sequence(self, windows):
        """windows: (B, grid, grid, C) uint8/long with C = 3 if interleave else 1 -> (B, S) long."""
        return windows.reshape(windows.shape[0], -1).long()

    def forward(self, ids):
        x = self.token_emb(ids)
        if self.interleave:
            x = x + self.channel_emb(self.chan_idx[: ids.shape[1]])
        for blk in self.layers:
            x = blk(x, self.freqs)
        return self.lm_head(self.norm(x))

    def bits(self, ids):
        """Total coding cost in bits of the patch tokens, per sample. Shape (B,)."""
        logp = F.log_softmax(self(ids)[:, :-1].float(), dim=-1)
        nll = -logp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1) / 0.6931471805599453
        tgt_mask = self.is_patch[1:].to(nll.dtype)
        b = (nll * tgt_mask).sum(dim=1)
        if self.is_patch[0]:
            b = b + 8.0  # first token has no context: uniform 8 bits
        return b

    def loss(self, ids):
        """Mean NLL (nats) over patch tokens — the training objective."""
        logits = self(ids)[:, :-1].float()
        tgt = ids[:, 1:]
        nll = F.cross_entropy(logits.reshape(-1, 256), tgt.reshape(-1), reduction="none").reshape(tgt.shape)
        m = self.is_patch[1:].to(nll.dtype)
        return (nll * m).sum() / (m.sum() * tgt.shape[0])
