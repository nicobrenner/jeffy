"""2D RoPE byte model for images.

Extends the 1D RoPE byte model with:
- 2D rotary position embeddings (row + column frequencies)
- Bidirectional attention (no causal mask)
- Image-aware training (next-pixel prediction with full context)

For MNIST: 28x28 = 784 pixels, each a byte [0-255].
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset


def precompute_freqs_2d(head_dim, height, width, theta=10000.0):
    """Precompute 2D rotary frequencies.

    Splits head_dim in half: first half encodes row position,
    second half encodes column position. This way pixels that are
    spatially adjacent (in 2D) are positionally adjacent.
    """
    half = head_dim // 2
    row_freqs = 1.0 / (theta ** (torch.arange(0, half, 2).float() / half))
    col_freqs = 1.0 / (theta ** (torch.arange(0, half, 2).float() / half))

    rows = torch.arange(height)
    cols = torch.arange(width)

    row_angles = torch.outer(rows, row_freqs)  # (H, half//2)
    col_angles = torch.outer(cols, col_freqs)  # (W, half//2)

    # For each position (r, c), concat row and col frequencies
    # Shape: (H*W, head_dim//2) as complex numbers
    freqs_list = []
    for r in range(height):
        for c in range(width):
            row_part = torch.polar(torch.ones_like(row_angles[r]), row_angles[r])
            col_part = torch.polar(torch.ones_like(col_angles[c]), col_angles[c])
            freqs_list.append(torch.cat([row_part, col_part]))

    return torch.stack(freqs_list)  # (H*W, head_dim//2)


def apply_rope_2d(x, freqs):
    """Apply 2D rotary embeddings. x shape: (B, n_heads, S, head_dim)."""
    B, H, S, D = x.shape
    x_complex = torch.view_as_complex(x.float().reshape(B, H, S, D // 2, 2))
    freqs = freqs[:S].unsqueeze(0).unsqueeze(0)
    x_rotated = torch.view_as_real(x_complex * freqs).reshape(B, H, S, D)
    return x_rotated.type_as(x)


class RoPE2DAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.0):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout

    def forward(self, x, freqs, mask=None):
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)

        q = apply_rope_2d(q, freqs)
        k = apply_rope_2d(k, freqs)

        scale = self.head_dim ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn + mask[:S, :S]
        attn = F.softmax(attn, dim=-1)
        if self.training and self.dropout > 0:
            attn = F.dropout(attn, p=self.dropout)

        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out)


class RoPE2DTransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = RoPE2DAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x, freqs, mask=None):
        x = x + self.attn(self.ln1(x), freqs, mask)
        x = x + self.ff(self.ln2(x))
        return x


class RoPE2DByteLM(nn.Module):
    """Byte-level model with 2D RoPE for images."""

    def __init__(self, vocab_size=256, d_model=64, n_layers=4, n_heads=4,
                 height=28, width=28, dropout=0.0, tie_weights=True,
                 bidirectional=True):
        super().__init__()
        self.d_model = d_model
        self.height = height
        self.width = width
        self.seq_len = height * width
        self.vocab_size = vocab_size
        self.bidirectional = bidirectional

        self.token_emb = nn.Embedding(vocab_size, d_model)

        d_ff = 4 * d_model
        self.layers = nn.ModuleList([
            RoPE2DTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_weights:
            self.lm_head.weight = self.token_emb.weight

        head_dim = d_model // n_heads
        self.register_buffer('freqs', precompute_freqs_2d(head_dim, height, width))

        if not bidirectional:
            self.register_buffer('causal_mask', torch.triu(
                torch.full((self.seq_len, self.seq_len), float("-inf")), diagonal=1))
        else:
            self.causal_mask = None

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        for layer in self.layers:
            nn.init.normal_(layer.attn.qkv.weight, std=0.02)
            nn.init.normal_(layer.attn.out_proj.weight, std=0.02)

    def forward(self, input_ids):
        x = self.token_emb(input_ids)
        mask = self.causal_mask
        for layer in self.layers:
            x = layer(x, self.freqs, mask)
        x = self.norm(x)
        return self.lm_head(x)

    def get_features(self, input_ids, pool='mean'):
        """Extract features for classification."""
        x = self.token_emb(input_ids)
        mask = self.causal_mask
        for layer in self.layers:
            x = layer(x, self.freqs, mask)
        x = self.norm(x)
        if pool == 'mean':
            return x.mean(dim=1)
        elif pool == 'max':
            return x.max(dim=1).values
        elif pool == 'cls':
            return x[:, 0, :]
        elif pool == 'both':
            return torch.cat([x.mean(dim=1), x.max(dim=1).values], dim=-1)
        return x


class MNISTByteDataset(Dataset):
    """MNIST images as byte sequences with labels."""
    def __init__(self, bytes_path, labels=None, seq_len=784):
        self.data = np.fromfile(bytes_path, dtype=np.uint8)
        self.labels = labels
        self.seq_len = seq_len
        self.n_images = len(self.data) // seq_len

    def __len__(self):
        return min(self.n_images, len(self.labels)) if self.labels is not None else self.n_images

    def __getitem__(self, idx):
        start = idx * self.seq_len
        img = torch.from_numpy(self.data[start:start + self.seq_len].astype(np.int64))
        if self.labels is not None:
            return img, self.labels[idx]
        return img


class MNISTByteLMDataset(IterableDataset):
    """MNIST bytes for language modeling (next-byte prediction)."""
    def __init__(self, path, seq_len, max_tokens=None):
        self.path = path
        self.seq_len = seq_len
        self.max_tokens = max_tokens

    def __iter__(self):
        data = np.memmap(self.path, dtype=np.uint8, mode='r')
        n_chunks = len(data) // self.seq_len
        tokens_yielded = 0
        import random
        while True:
            indices = list(range(n_chunks))
            random.shuffle(indices)
            for idx in indices:
                start = idx * self.seq_len
                chunk = torch.from_numpy(data[start:start + self.seq_len].astype(np.int64))
                tokens_yielded += self.seq_len
                if self.max_tokens and tokens_yielded > self.max_tokens:
                    return
                yield chunk


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=50):
    model.eval()
    total_loss = 0
    n = 0
    for batch in val_loader:
        if n >= max_batches:
            break
        if isinstance(batch, (list, tuple)):
            ids = batch[0].to(device)
        else:
            ids = batch.to(device)
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        logits = model(ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
            ids[:, 1:].contiguous().view(-1),
        )
        total_loss += loss.item()
        n += 1
    model.train()
    avg = total_loss / max(n, 1)
    return avg, math.exp(min(avg, 20))


@torch.no_grad()
def evaluate_classification(model, dataloader, device, pool='mean'):
    """Evaluate frozen features + logistic regression."""
    from sklearn.linear_model import LogisticRegression
    model.eval()
    all_feats, all_labels = [], []
    for imgs, labels in dataloader:
        feats = model.get_features(imgs.to(device), pool=pool)
        all_feats.append(feats.cpu().numpy())
        all_labels.append(labels.numpy())
    feats = np.concatenate(all_feats)
    labels = np.concatenate(all_labels)
    # Split: use first 80% for train, last 20% for val
    split = int(len(feats) * 0.8)
    clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
    clf.fit(feats[:split], labels[:split])
    return clf.score(feats[split:], labels[split:])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--d-model', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=4)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--tok-param', type=float, default=1800)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--height', type=int, default=28)
    parser.add_argument('--width', type=int, default=28)
    parser.add_argument('--bidirectional', action='store_true', default=True)
    parser.add_argument('--causal', action='store_true')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else
                        ('mps' if torch.backends.mps.is_available() else 'cpu'))
    parser.add_argument('--data-dir', type=str, default='pile_runs/custom_vocab/byte_mnist')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--eval-classification', action='store_true',
                        help='Run classification eval after training')
    args = parser.parse_args()

    if args.causal:
        args.bidirectional = False

    seq_len = args.height * args.width
    vocab_size = 256
    device = args.device

    model = RoPE2DByteLM(
        vocab_size=vocab_size, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, height=args.height, width=args.width,
        tie_weights=True, bidirectional=args.bidirectional,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    total_tokens = int(args.tok_param * n_params)
    n_steps = total_tokens // (args.batch_size * seq_len)
    mode = "bidirectional" if args.bidirectional else "causal"

    run_name = f'mnist_2d_d{args.d_model}_{mode}_tp{int(args.tok_param)}'
    run_dir = Path(f'pile_runs/rope_byte/{run_name}')
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*70}")
    print(f"  2D RoPE Byte Model — MNIST")
    print(f"{'='*70}")
    print(f"  Mode: {mode}")
    print(f"  d_model={args.d_model}, {args.n_layers}L, {args.n_heads}H")
    print(f"  Image: {args.height}x{args.width} = {seq_len} positions")
    print(f"  Params: {n_params:,} ({n_params/1000:.1f}K)")
    print(f"  Training: {total_tokens/1e6:.1f}M tokens, {n_steps:,} steps, {args.tok_param:.0f} tok/param")
    print(f"  Device: {device}")
    print(f"  Run dir: {run_dir}")
    print()

    # Data
    train_ds = MNISTByteLMDataset(
        str(Path(args.data_dir) / 'train.bytes'), seq_len, max_tokens=total_tokens)
    val_ds = MNISTByteLMDataset(
        str(Path(args.data_dir) / 'val.bytes'), seq_len, max_tokens=seq_len * 200)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # Resume
    ckpt_path = run_dir / 'checkpoint.pt'
    best_path = run_dir / 'best_model.pt'
    start_step = 0
    best_ppl = float('inf')
    best_state = None

    if args.resume and ckpt_path.exists():
        print(f"  Resuming from checkpoint...")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        optimizer.load_state_dict(ckpt['optimizer_state'])
        scheduler.load_state_dict(ckpt['scheduler_state'])
        start_step = ckpt['step']
        best_ppl = ckpt['best_ppl']
        best_state = ckpt.get('best_model_state')
        print(f"  Resumed at step {start_step:,}, best_ppl={best_ppl:.3f}")
        print()

    # Training
    model.train()
    step = 0
    running_loss = 0
    t0 = time.time()
    eval_every = max(50, n_steps // 20)
    log_every = max(100, n_steps // 10)
    save_every = max(200, n_steps // 5)
    evals = []

    for batch in train_loader:
        step += 1
        if step <= start_step:
            scheduler.step()
            continue

        ids = batch.to(device)
        logits = model(ids)

        if args.bidirectional:
            # Masked LM style: predict each pixel from all others
            # Use all positions as targets (BERT-style, not shifted)
            # But we still use next-token format for consistency with eval
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                ids[:, 1:].contiguous().view(-1),
            )
        else:
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                ids[:, 1:].contiguous().view(-1),
            )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()

        tp = step * args.batch_size * seq_len / n_params

        if step % eval_every == 0:
            val_loss, val_ppl = evaluate(model, val_loader, device)
            tag = ""
            if val_ppl < best_ppl:
                best_ppl = val_ppl
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                tag = " *BEST*"
            evals.append({'step': step, 'tok_param': round(tp, 1), 'val_ppl': round(val_ppl, 4)})
            print(f"  eval step {step:>6,}: ppl={val_ppl:.3f} tp={tp:.0f}{tag}", flush=True)
            model.train()

        if step % log_every == 0:
            avg = running_loss / log_every
            elapsed = time.time() - t0
            tok_s = step * args.batch_size * seq_len / elapsed
            print(f"  step {step:>6,}/{n_steps:,} loss={avg:.4f} ppl={math.exp(min(avg,20)):.2f} "
                  f"{tok_s:,.0f} tok/s", flush=True)
            running_loss = 0

        if step % save_every == 0:
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'step': step,
                'best_ppl': best_ppl,
                'best_model_state': best_state,
            }, ckpt_path)

        if step >= n_steps:
            break

    # Final eval
    val_loss, val_ppl = evaluate(model, val_loader, device)
    if val_ppl < best_ppl:
        best_ppl = val_ppl
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    elapsed = time.time() - t0
    tok_s = n_steps * args.batch_size * seq_len / elapsed if elapsed > 0 else 0
    print(f"\n  Final ppl={val_ppl:.3f}, best={best_ppl:.3f}")
    print(f"  Time: {elapsed:.1f}s ({tok_s:,.0f} tok/s)")

    # Save best model
    if best_state:
        model.load_state_dict(best_state)
        torch.save({
            'model_state': best_state,
            'config': {
                'd_model': args.d_model, 'n_layers': args.n_layers,
                'n_heads': args.n_heads, 'height': args.height, 'width': args.width,
                'bidirectional': args.bidirectional,
            },
            'val_ppl': best_ppl,
        }, best_path)

    results = {
        'model': '2d_rope_byte',
        'mode': mode,
        'd_model': args.d_model, 'n_layers': args.n_layers, 'n_heads': args.n_heads,
        'height': args.height, 'width': args.width,
        'n_params': n_params, 'tok_param': args.tok_param,
        'best_val_ppl': round(best_ppl, 4),
        'final_val_ppl': round(val_ppl, 4),
        'training_time_s': round(elapsed, 1),
        'evals': evals,
    }

    # Classification eval
    if args.eval_classification:
        print(f"\n{'='*70}")
        print(f"  CLASSIFICATION EVALUATION")
        print(f"{'='*70}")

        import torchvision
        mnist_train = torchvision.datasets.MNIST(root='/tmp/mnist', train=True, download=True)
        mnist_test = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True)
        train_labels = np.array([label for _, label in mnist_train])
        test_labels = np.array([label for _, label in mnist_test])

        train_cls = MNISTByteDataset(
            str(Path(args.data_dir) / 'train.bytes'), train_labels, seq_len)
        test_cls = MNISTByteDataset(
            str(Path(args.data_dir) / 'val.bytes'), test_labels, seq_len)

        train_loader_cls = DataLoader(train_cls, batch_size=32, num_workers=0)
        test_loader_cls = DataLoader(test_cls, batch_size=32, num_workers=0)

        from sklearn.linear_model import LogisticRegression

        model.eval()
        cls_results = {}
        for pool in ['mean', 'max', 'both']:
            print(f"\n  Pool={pool}:")
            train_feats, train_lbls = [], []
            with torch.no_grad():
                for imgs, labels in train_loader_cls:
                    feats = model.get_features(imgs.to(device), pool=pool)
                    train_feats.append(feats.cpu().numpy())
                    train_lbls.append(labels.numpy())
                    if len(train_lbls) * 32 >= 10000:
                        break
            train_feats = np.concatenate(train_feats)[:10000]
            train_lbls = np.concatenate(train_lbls)[:10000]

            test_feats, test_lbls = [], []
            with torch.no_grad():
                for imgs, labels in test_loader_cls:
                    feats = model.get_features(imgs.to(device), pool=pool)
                    test_feats.append(feats.cpu().numpy())
                    test_lbls.append(labels.numpy())
            test_feats = np.concatenate(test_feats)
            test_lbls = np.concatenate(test_lbls)

            clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
            clf.fit(train_feats, train_lbls)
            train_acc = clf.score(train_feats, train_lbls)
            test_acc = clf.score(test_feats, test_lbls)
            print(f"    Features: {train_feats.shape[1]}-dim")
            print(f"    Train: {train_acc*100:.1f}%  Test: {test_acc*100:.1f}%")
            cls_results[pool] = {'train_acc': train_acc, 'test_acc': test_acc,
                                 'feat_dim': train_feats.shape[1]}

        # Raw pixel baseline
        print(f"\n  Raw pixels baseline:")
        train_raw = np.fromfile(str(Path(args.data_dir) / 'train.bytes'), dtype=np.uint8)
        test_raw = np.fromfile(str(Path(args.data_dir) / 'val.bytes'), dtype=np.uint8)
        train_pix = train_raw[:10000*seq_len].reshape(10000, seq_len).astype(float) / 255.0
        test_pix = test_raw[:10000*seq_len].reshape(10000, seq_len).astype(float) / 255.0
        clf = LogisticRegression(max_iter=2000, C=1.0, solver='lbfgs')
        clf.fit(train_pix, train_labels[:10000])
        pix_train = clf.score(train_pix, train_labels[:10000])
        pix_test = clf.score(test_pix, test_labels[:10000])
        print(f"    Train: {pix_train*100:.1f}%  Test: {pix_test*100:.1f}%")

        results['classification'] = cls_results
        results['raw_pixel_baseline'] = {'train_acc': pix_train, 'test_acc': pix_test}

    with open(run_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to {run_dir / 'results.json'}")


if __name__ == '__main__':
    main()
