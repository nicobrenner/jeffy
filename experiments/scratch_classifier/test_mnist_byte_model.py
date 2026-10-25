"""Test MNIST byte models beyond PPL: image generation + classification."""
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from train_rope_byte import RoPEByteLM

device = 'mps' if torch.backends.mps.is_available() else 'cpu'

def load_model(run_dir, d_model, n_layers=4, n_heads=4):
    model = RoPEByteLM(vocab_size=256, d_model=d_model, n_layers=n_layers,
                       n_heads=n_heads, max_seq_len=784, tie_weights=True).to(device)
    ckpt_path = run_dir / 'checkpoint.pt'
    best_path = run_dir / 'best_model.pt'
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
    elif ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = ckpt.get('best_model_state', ckpt.get('model_state'))
        model.load_state_dict(state)
    model.eval()
    return model


def generate_image(model, seed_bytes=16, temperature=0.8):
    """Generate a 784-byte image (28x28) from the model."""
    seed = torch.randint(0, 256, (1, seed_bytes), device=device)
    generated = seed.clone()
    with torch.no_grad():
        for _ in range(784 - seed_bytes):
            logits = model(generated)
            next_logits = logits[0, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_byte = torch.multinomial(probs, 1).unsqueeze(0)
            generated = torch.cat([generated, next_byte], dim=1)
    return generated[0, -784:].cpu().numpy().astype(np.uint8)


@torch.no_grad()
def extract_features(model, data_bytes, labels, seq_len=784):
    """Extract features from model's last hidden layer for classification."""
    model.eval()
    n_images = len(data_bytes) // seq_len
    features = []
    for i in range(min(n_images, 10000)):
        chunk = data_bytes[i*seq_len:(i+1)*seq_len]
        ids = torch.tensor(chunk, dtype=torch.long, device=device).unsqueeze(0)
        x = model.token_emb(ids)
        mask = model.causal_mask
        for layer in model.layers:
            x = layer(x, model.freqs, mask)
        x = model.norm(x)
        feat = x[0].mean(dim=0).cpu().numpy()
        features.append(feat)
    return np.array(features), labels[:len(features)]


def linear_probe(train_feats, train_labels, val_feats, val_labels):
    """Train a simple linear classifier on extracted features."""
    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(train_feats, train_labels)
    train_acc = clf.score(train_feats, train_labels)
    val_acc = clf.score(val_feats, val_labels)
    return train_acc, val_acc


def image_completion(model, image_bytes, prefix_len=392):
    """Given the first half of an image, predict the second half."""
    prefix = torch.tensor(image_bytes[:prefix_len], dtype=torch.long, device=device).unsqueeze(0)
    generated = prefix.clone()
    with torch.no_grad():
        for _ in range(784 - prefix_len):
            logits = model(generated)
            next_byte = logits[0, -1, :].argmax().unsqueeze(0).unsqueeze(0)
            generated = torch.cat([generated, next_byte], dim=1)
    predicted = generated[0, prefix_len:].cpu().numpy().astype(np.uint8)
    actual = image_bytes[prefix_len:784]
    mae = np.mean(np.abs(predicted.astype(float) - actual.astype(float)))
    return predicted, actual, mae


def main():
    import torchvision

    print("=" * 70)
    print("  MNIST BYTE MODEL EVALUATION — BEYOND PPL")
    print("=" * 70)

    # Load MNIST labels
    train_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=True, download=True)
    val_ds = torchvision.datasets.MNIST(root='/tmp/mnist', train=False, download=True)
    train_labels = np.array([label for _, label in train_ds])
    val_labels = np.array([label for _, label in val_ds])

    # Load byte data
    train_bytes = np.fromfile('pile_runs/custom_vocab/byte_mnist/train.bytes', dtype=np.uint8)
    val_bytes = np.fromfile('pile_runs/custom_vocab/byte_mnist/val.bytes', dtype=np.uint8)

    # Load models
    models = {}
    for name, run_dir in [
        ('d32_mnist_1800', Path('pile_runs/rope_byte/run_d32_mnist_tp1800')),
        ('d32_mnist_7200', Path('pile_runs/rope_byte/run_d32_mnist_tp7200')),
    ]:
        if run_dir.exists():
            print(f"\nLoading {name}...")
            models[name] = load_model(run_dir, d_model=32, n_layers=4, n_heads=4)
            results_path = run_dir / 'results.json'
            if results_path.exists():
                r = json.load(open(results_path))
                print(f"  PPL: {r['best_val_ppl']:.2f}")

    if not models:
        print("No MNIST models found!")
        return

    # === TEST 1: Image Generation ===
    print(f"\n{'='*70}")
    print(f"  TEST 1: IMAGE GENERATION")
    print(f"{'='*70}")
    print("  Generating 28x28 images from each model...")

    for name, model in models.items():
        print(f"\n  [{name}]")
        for i in range(3):
            img_bytes = generate_image(model, seed_bytes=16, temperature=0.8)
            img = img_bytes.reshape(28, 28)
            # Print ASCII visualization
            print(f"    Sample {i+1}:")
            for row in range(0, 28, 2):
                line = "      "
                for col in range(28):
                    val = img[row, col]
                    if val > 200: line += "##"
                    elif val > 150: line += "@@"
                    elif val > 100: line += "**"
                    elif val > 50: line += ".."
                    else: line += "  "
                print(line)
            mean_val = img.mean()
            std_val = img.std()
            print(f"    Stats: mean={mean_val:.1f}, std={std_val:.1f}")
            # Compare to real MNIST stats (~33 mean, ~79 std)

    print(f"\n  Real MNIST stats: mean={train_bytes[:784*100].reshape(-1,784).mean():.1f}, "
          f"std={train_bytes[:784*100].reshape(-1,784).astype(float).std():.1f}")

    # === TEST 2: Image Completion ===
    print(f"\n{'='*70}")
    print(f"  TEST 2: IMAGE COMPLETION (top half → predict bottom half)")
    print(f"{'='*70}")

    for name, model in models.items():
        maes = []
        for i in range(100):
            img_start = i * 784
            img_bytes = val_bytes[img_start:img_start+784]
            _, _, mae = image_completion(model, img_bytes, prefix_len=392)
            maes.append(mae)
        mean_mae = np.mean(maes)
        print(f"  [{name}] Mean pixel MAE (completion): {mean_mae:.1f} / 255")
        # Random baseline
        random_mae = np.mean([np.mean(np.abs(np.random.randint(0,256,392).astype(float) -
                     val_bytes[i*784+392:i*784+784].astype(float))) for i in range(100)])
        print(f"  [random baseline] Mean pixel MAE: {random_mae:.1f} / 255")

    # === TEST 3: Feature Extraction + Linear Probe ===
    print(f"\n{'='*70}")
    print(f"  TEST 3: DIGIT CLASSIFICATION (feature extraction + linear probe)")
    print(f"{'='*70}")

    for name, model in models.items():
        print(f"\n  [{name}] Extracting features...")
        train_feats, train_lbls = extract_features(model, train_bytes, train_labels)
        val_feats, val_lbls = extract_features(model, val_bytes, val_labels)
        print(f"    Features: {train_feats.shape}")
        train_acc, val_acc = linear_probe(train_feats, train_lbls, val_feats, val_lbls)
        print(f"    Train accuracy: {train_acc*100:.1f}%")
        print(f"    Val accuracy:   {val_acc*100:.1f}%")

    # Raw pixel baseline
    print(f"\n  [raw pixels baseline] Linear probe on raw 784-dim pixels...")
    train_raw = train_bytes[:10000*784].reshape(10000, 784).astype(float) / 255.0
    val_raw = val_bytes[:10000*784].reshape(10000, 784).astype(float) / 255.0
    train_acc, val_acc = linear_probe(train_raw, train_labels[:10000], val_raw, val_labels[:10000])
    print(f"    Train accuracy: {train_acc*100:.1f}%")
    print(f"    Val accuracy:   {val_acc*100:.1f}%")


if __name__ == '__main__':
    main()
