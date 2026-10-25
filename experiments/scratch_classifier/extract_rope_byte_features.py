"""Extract features from RoPE byte models for 30 classification datasets.

Uses d=64 and d=128 byte-level language models as feature extractors:
encode text as raw UTF-8 bytes, run through the frozen transformer,
pool hidden states, train logistic regression on top.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
from jeffy.build_pack import DATASETS, load_and_split
from train_rope_byte import RoPEByteLM

device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


def load_rope_model(run_dir, d_model, n_layers=4, max_seq_len=256):
    n_heads = d_model // 16
    model = RoPEByteLM(vocab_size=256, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, max_seq_len=max_seq_len, tie_weights=True).to(device)
    best_path = run_dir / 'best_model.pt'
    ckpt_path = run_dir / 'checkpoint.pt'
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
    elif ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('best_model_state', ckpt.get('model_state'))
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"No checkpoint in {run_dir}")
    model.eval()
    return model


def texts_to_byte_ids(texts, max_len=256):
    """Convert texts to byte-level input tensors."""
    batch = []
    lengths = []
    for text in texts:
        b = list(text.encode('utf-8'))[:max_len]
        lengths.append(len(b))
        if len(b) < max_len:
            b = b + [0] * (max_len - len(b))
        batch.append(b)
    return torch.tensor(batch, dtype=torch.long), lengths


@torch.no_grad()
def extract_features(model, texts, max_seq_len=256, batch_size=64, pool='both'):
    """Extract features from model hidden states.

    pool='mean': mean of all non-padding positions
    pool='last': last non-padding position
    pool='both': concatenate mean and last
    """
    model.eval()
    all_mean = []
    all_last = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        ids, lengths = texts_to_byte_ids(batch_texts, max_seq_len)
        ids = ids.to(device)

        x = model.token_emb(ids)
        mask = model.causal_mask
        for layer in model.layers:
            x = layer(x, model.freqs, mask)
        x = model.norm(x)

        for j, length in enumerate(lengths):
            if length == 0:
                length = 1
            mean_feat = x[j, :length].mean(dim=0).cpu().numpy()
            last_feat = x[j, length-1].cpu().numpy()
            all_mean.append(mean_feat)
            all_last.append(last_feat)

        if device == 'mps' and i % (batch_size * 10) == 0 and i > 0:
            torch.mps.empty_cache()

    mean_feats = np.array(all_mean)
    last_feats = np.array(all_last)

    if pool == 'mean':
        return mean_feats
    elif pool == 'last':
        return last_feats
    else:
        return np.concatenate([mean_feats, last_feats], axis=1)


def evaluate_dataset(model, ds_name, config, max_seq_len=256, pool='both'):
    """Load dataset, extract features, train+eval logistic regression."""
    train_texts, train_labels, test_texts, test_labels, labels = \
        load_and_split(ds_name, config)

    n_classes = len(set(train_labels))

    le = LabelEncoder()
    train_y = le.fit_transform(train_labels)
    test_y = le.transform(test_labels)

    t0 = time.time()
    train_feats = extract_features(model, train_texts, max_seq_len, pool=pool)
    test_feats = extract_features(model, test_texts, max_seq_len, pool=pool)
    extract_time = time.time() - t0

    scaler = StandardScaler()
    train_feats = scaler.fit_transform(train_feats)
    test_feats = scaler.transform(test_feats)

    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(train_feats, train_y)

    test_pred = clf.predict(test_feats)
    test_acc = accuracy_score(test_y, test_pred)
    test_f1 = f1_score(test_y, test_pred, average='weighted')

    return {
        'dataset': ds_name,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'n_classes': n_classes,
        'n_train': len(train_texts),
        'n_test': len(test_texts),
        'feature_dim': train_feats.shape[1],
        'extract_time_s': extract_time,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, required=True)
    parser.add_argument('--d-model', type=int, required=True)
    parser.add_argument('--pool', type=str, default='both', choices=['mean', 'last', 'both'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--datasets', nargs='+', default=None)
    args = parser.parse_args()

    run_dir = Path(args.model_dir)
    model = load_rope_model(run_dir, args.d_model)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: d={args.d_model}, {total_params:,} params, pool={args.pool}")
    print(f"Device: {device}")
    print(f"Max seq len: {model.max_seq_len}")
    print()

    dataset_names = args.datasets or list(DATASETS.keys())

    results = []
    errors = []
    t_start = time.time()

    for i, ds_name in enumerate(dataset_names):
        if ds_name not in DATASETS:
            print(f"  Skipping {ds_name} (not in DATASETS)")
            continue

        config = DATASETS[ds_name]
        print(f"[{i+1}/{len(dataset_names)}] {ds_name} ({config['description']})...")

        try:
            result = evaluate_dataset(model, ds_name, config,
                                      max_seq_len=model.max_seq_len, pool=args.pool)
            results.append(result)
            print(f"  acc={result['test_acc']:.3f}  f1={result['test_f1']:.3f}  "
                  f"({result['n_train']} train, {result['n_test']} test, "
                  f"{result['n_classes']} classes, {result['extract_time_s']:.1f}s)")
        except Exception as e:
            print(f"  ERROR: {e}")
            errors.append({'dataset': ds_name, 'error': str(e)})

    elapsed = time.time() - t_start

    avg_acc = np.mean([r['test_acc'] for r in results]) if results else 0
    avg_f1 = np.mean([r['test_f1'] for r in results]) if results else 0

    print(f"\n{'='*70}")
    print(f"SUMMARY: d={args.d_model}, pool={args.pool}")
    print(f"  Datasets: {len(results)} ok, {len(errors)} errors")
    print(f"  Avg accuracy: {avg_acc:.3f}")
    print(f"  Avg F1: {avg_f1:.3f}")
    print(f"  Total time: {elapsed:.0f}s")

    output_data = {
        'results': results,
        'errors': errors,
        'elapsed_s': elapsed,
        'model': f'rope_byte_d{args.d_model}',
        'd_model': args.d_model,
        'pool': args.pool,
        'avg_acc': avg_acc,
        'avg_f1': avg_f1,
        'n_params': total_params,
    }

    output_path = args.output or str(run_dir / f'feature_extractor_results_{args.pool}.json')
    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
