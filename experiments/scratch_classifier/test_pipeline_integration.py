"""End-to-end integration test: hybrid encoder through Jeffy's Engine pipeline.

Builds a mini model pack with hybrid-encoder artifacts, loads it through
Engine, and verifies predictions work end-to-end.

Usage:
    python -m experiments.scratch_classifier.test_pipeline_integration
"""

import hashlib
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from jeffy.catalog import HYBRID_ENCODER, HYBRID_ENCODER_DIM
from jeffy.engine import Engine
from jeffy.model_pack import ArtifactManifest, save_artifact

HERE = Path(__file__).resolve().parent
CKPT_PATH = HERE / "checkpoints" / "best_hybrid.pt"

TEST_DATASETS = ["emotion", "sms_spam"]


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_hybrid_pack(pack_dir: Path, device: str):
    """Build a mini model pack using the hybrid encoder."""
    import torch
    from transformers import AutoTokenizer
    sys.path.insert(0, str(HERE))
    from train_hybrid import HybridTransformer

    from jeffy.build_pack import DATASETS, load_and_split

    print("Loading hybrid encoder checkpoint...", flush=True)
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab_map = ckpt["vocab_map"]

    model = HybridTransformer(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        d_ff=config["d_ff"], max_seq_len=config["max_seq_len"],
        task_classes=config["task_classes"],
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode(texts):
        unk_id = 0
        max_len = config["max_seq_len"]
        all_emb = []
        for i in range(0, len(texts), 128):
            batch = texts[i:i+128]
            enc = tokenizer(batch, truncation=True, padding="max_length",
                            max_length=max_len, return_tensors="pt")
            remapped = torch.tensor([
                [vocab_map.get(t.item(), unk_id) for t in row]
                for row in enc["input_ids"]
            ], dtype=torch.long)
            with torch.no_grad():
                emb = model.encode(remapped.to(device), enc["attention_mask"].to(device))
            all_emb.append(emb.cpu().numpy())
            if device == "mps" and i > 0 and (i // 128) % 20 == 0:
                torch.mps.empty_cache()
                torch.mps.synchronize()
        return np.concatenate(all_emb, axis=0)

    # Copy hybrid encoder checkpoint into pack
    hybrid_dir = pack_dir / "hybrid_encoder"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CKPT_PATH, hybrid_dir / "best_hybrid.pt")
    print(f"  Copied checkpoint to {hybrid_dir / 'best_hybrid.pt'}")

    results = {}

    for ds_name in TEST_DATASETS:
        ds_config = DATASETS[ds_name]
        print(f"\nBuilding artifact for {ds_name}...", flush=True)

        train_texts, train_labels, test_texts, test_labels, label_map = \
            load_and_split(ds_name, ds_config, max_train=5000, max_test=1000)

        unique = sorted(set(train_labels) | set(test_labels))
        l2i = {l: i for i, l in enumerate(unique)}
        train_y = np.array([l2i[l] for l in train_labels])
        test_y = np.array([l2i[l] for l in test_labels])

        print(f"  Encoding {len(train_texts)} train + {len(test_texts)} test...", flush=True)
        train_X = encode(train_texts)
        test_X = encode(test_texts)

        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        test_X_scaled = scaler.transform(test_X)

        clf = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs",
                                 multi_class="multinomial")
        clf.fit(train_X_scaled, train_y)

        train_acc = accuracy_score(train_y, clf.predict(train_X_scaled))
        test_acc = accuracy_score(test_y, clf.predict(test_X_scaled))
        print(f"  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}")

        # Build label map as strings
        labels_dict = {}
        for label_str, label_name in label_map.items():
            if label_str in l2i:
                labels_dict[str(l2i[label_str])] = label_name

        manifest = ArtifactManifest(
            dataset=ds_name,
            description=ds_config["description"],
            n_classes=len(unique),
            task_type=ds_config["task_type"],
            encoder=HYBRID_ENCODER,
            encoder_dim=HYBRID_ENCODER_DIM,
            labels=labels_dict,
            class_order=[str(c) for c in clf.classes_],
            hf_source=ds_config["hf"],
            hf_config=ds_config.get("hf_config"),
            hf_revision=None,
            license="test",
            train_examples=len(train_texts),
            train_accuracy=round(train_acc, 4),
            test_examples=len(test_texts),
            test_accuracy=round(test_acc, 4),
            scaler_mean_hash=hashlib.sha256(scaler.mean_.tobytes()).hexdigest()[:8],
            classifier_coef_hash=hashlib.sha256(clf.coef_.tobytes()).hexdigest()[:8],
        )

        save_artifact(pack_dir, ds_name, clf, scaler, manifest)
        print(f"  Saved artifact to {pack_dir / ds_name}")

        results[ds_name] = {"train_acc": train_acc, "test_acc": test_acc,
                            "n_classes": len(unique), "labels": labels_dict}

    return results


def test_engine(pack_dir: Path, device: str, build_results: dict):
    """Load model pack through Engine and test predictions."""
    print(f"\n{'='*60}")
    print("Testing Engine pipeline")
    print(f"{'='*60}", flush=True)

    engine = Engine(pack_dir=str(pack_dir), device=device)
    engine.load()

    print(f"\nLoaded {len(engine.capabilities)} capabilities:")
    for task_id, cap in engine.capabilities.items():
        print(f"  {task_id}: {cap.description} ({cap.n_classes} classes, encoder={cap.encoder})")

    test_inputs = {
        "emotion": [
            ("I am so happy today!", "joy"),
            ("This makes me really angry", "anger"),
            ("I'm terrified of what might happen", "fear"),
            ("What a wonderful surprise!", "surprise"),
        ],
        "sms_spam": [
            ("Hey, are you free for lunch today?", "ham"),
            ("WINNER! You have won a free prize! Call now!", "spam"),
            ("Meeting rescheduled to 3pm", "ham"),
        ],
    }

    all_correct = 0
    all_total = 0

    for task_id, cases in test_inputs.items():
        if task_id not in engine.capabilities:
            print(f"\n{task_id}: NOT in engine capabilities, skipping")
            continue

        print(f"\n{task_id} predictions:")
        for text, expected_category in cases:
            result = engine.predict(task_id, text)
            if "error" in result:
                print(f"  ERROR: {result['error']}")
                continue

            label = result["label"]
            conf = result["confidence"]
            latency = result["latency_ms"]
            encoder_used = result["capability"]["encoder"]

            match = expected_category.lower() in label.lower()
            all_correct += int(match)
            all_total += 1

            status = "OK" if match else "MISS"
            print(f"  [{status}] \"{text[:50]}...\"")
            print(f"       -> {label} ({conf:.3f}) [{latency:.0f}ms] encoder={encoder_used}")

    print(f"\n{'='*60}")
    print(f"Pipeline test: {all_correct}/{all_total} correct")
    print(f"{'='*60}")

    return all_correct, all_total


def main():
    device = get_device()
    print(f"Device: {device}")

    if not CKPT_PATH.exists():
        print(f"ERROR: Checkpoint not found at {CKPT_PATH}")
        print("Run train_hybrid.py first.")
        sys.exit(1)

    with tempfile.TemporaryDirectory(prefix="jeffy_hybrid_pack_") as tmpdir:
        pack_dir = Path(tmpdir)
        print(f"Building test pack in {pack_dir}\n")

        build_results = build_hybrid_pack(pack_dir, device)
        correct, total = test_engine(pack_dir, device, build_results)

        print(f"\nBuild results:")
        for ds, r in build_results.items():
            print(f"  {ds}: test_acc={r['test_acc']:.3f} ({r['n_classes']} classes)")

        if correct == total:
            print("\nAll predictions matched expected categories!")
        else:
            print(f"\n{total - correct} predictions did not match expected categories")


if __name__ == "__main__":
    main()
