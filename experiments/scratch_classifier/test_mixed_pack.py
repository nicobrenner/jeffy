"""Test mixed model pack: some artifacts use hybrid encoder, others use BGE.

This validates the task-adaptive encoder selection architecture where
Jeffy picks the best encoder per task.

Usage:
    python -m experiments.scratch_classifier.test_mixed_pack
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

from jeffy.catalog import ENCODER, ENCODER_DIM, HYBRID_ENCODER, HYBRID_ENCODER_DIM
from jeffy.engine import Engine
from jeffy.model_pack import ArtifactManifest, save_artifact

HERE = Path(__file__).resolve().parent
HYBRID_CKPT = HERE / "checkpoints" / "best_hybrid.pt"

HYBRID_TASKS = ["emotion"]
BGE_TASKS = ["sst2"]


def get_device():
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_mixed_pack(pack_dir: Path, device: str):
    """Build a pack with both hybrid and BGE artifacts."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    sys.path.insert(0, str(HERE))
    from train_hybrid import HybridTransformer
    from jeffy.build_pack import DATASETS, load_and_split

    # Load hybrid encoder
    ckpt = torch.load(HYBRID_CKPT, map_location=device, weights_only=False)
    config = ckpt["config"]
    vocab_map = ckpt["vocab_map"]
    hybrid_model = HybridTransformer(
        vocab_size=config["vocab_size"], d_model=config["d_model"],
        n_heads=config["n_heads"], n_layers=config["n_layers"],
        d_ff=config["d_ff"], max_seq_len=config["max_seq_len"],
        task_classes=config["task_classes"],
    ).to(device)
    hybrid_model.load_state_dict(ckpt["model_state"])
    hybrid_model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def encode_hybrid(texts):
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
                emb = hybrid_model.encode(remapped.to(device), enc["attention_mask"].to(device))
            all_emb.append(emb.cpu().numpy())
        return np.concatenate(all_emb, axis=0)

    # Load BGE encoder
    bge = SentenceTransformer(ENCODER, device=device)
    def encode_bge(texts):
        return bge.encode(texts, batch_size=128, show_progress_bar=False)

    # Copy hybrid checkpoint
    hybrid_dir = pack_dir / "hybrid_encoder"
    hybrid_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(HYBRID_CKPT, hybrid_dir / "best_hybrid.pt")

    results = {}

    all_tasks = [(t, HYBRID_ENCODER, HYBRID_ENCODER_DIM, encode_hybrid) for t in HYBRID_TASKS] + \
                [(t, ENCODER, ENCODER_DIM, encode_bge) for t in BGE_TASKS]

    for ds_name, enc_name, enc_dim, encode_fn in all_tasks:
        ds_config = DATASETS[ds_name]
        print(f"\nBuilding {ds_name} (encoder={enc_name})...", flush=True)

        train_texts, train_labels, test_texts, test_labels, label_map = \
            load_and_split(ds_name, ds_config, max_train=3000, max_test=500)

        unique = sorted(set(train_labels) | set(test_labels))
        l2i = {l: i for i, l in enumerate(unique)}
        train_y = np.array([l2i[l] for l in train_labels])
        test_y = np.array([l2i[l] for l in test_labels])

        train_X = encode_fn(train_texts)
        test_X = encode_fn(test_texts)

        scaler = StandardScaler()
        train_X_scaled = scaler.fit_transform(train_X)
        test_X_scaled = scaler.transform(test_X)

        clf = LogisticRegression(C=0.01, max_iter=1000, solver="lbfgs")
        clf.fit(train_X_scaled, train_y)
        test_acc = accuracy_score(test_y, clf.predict(test_X_scaled))
        train_acc = accuracy_score(train_y, clf.predict(train_X_scaled))
        print(f"  test_acc={test_acc:.3f} ({enc_name})")

        labels_dict = {}
        for label_str, label_name in label_map.items():
            if label_str in l2i:
                labels_dict[str(l2i[label_str])] = label_name

        manifest = ArtifactManifest(
            dataset=ds_name, description=ds_config["description"],
            n_classes=len(unique), task_type=ds_config["task_type"],
            encoder=enc_name, encoder_dim=enc_dim,
            labels=labels_dict, class_order=[str(c) for c in clf.classes_],
            hf_source=ds_config["hf"], hf_config=ds_config.get("hf_config"),
            hf_revision=None, license="test",
            train_examples=len(train_texts), train_accuracy=round(train_acc, 4),
            test_examples=len(test_texts), test_accuracy=round(test_acc, 4),
            scaler_mean_hash=hashlib.sha256(scaler.mean_.tobytes()).hexdigest()[:8],
            classifier_coef_hash=hashlib.sha256(clf.coef_.tobytes()).hexdigest()[:8],
        )
        save_artifact(pack_dir, ds_name, clf, scaler, manifest)
        results[ds_name] = {"test_acc": test_acc, "encoder": enc_name}

    return results


def test_mixed_engine(pack_dir: Path, device: str):
    """Test Engine with mixed encoders."""
    print(f"\n{'='*60}")
    print("Testing mixed-encoder Engine")
    print(f"{'='*60}", flush=True)

    engine = Engine(pack_dir=str(pack_dir), device=device)
    engine.load()

    print(f"Loaded {len(engine.capabilities)} capabilities:")
    encoders_loaded = set()
    for task_id, cap in engine.capabilities.items():
        print(f"  {task_id}: encoder={cap.encoder}")
        encoders_loaded.add(cap.encoder)

    print(f"\nEncoders in use: {encoders_loaded}")
    assert len(encoders_loaded) == 2, f"Expected 2 encoders, got {encoders_loaded}"

    # Test emotion (hybrid)
    r = engine.predict("emotion", "I feel so happy and excited!")
    print(f"\nemotion: '{r['label']}' ({r['confidence']:.3f}) encoder={r['capability']['encoder']}")
    assert r["capability"]["encoder"] == HYBRID_ENCODER

    # Test sst2 (BGE)
    r = engine.predict("sst2", "This movie was absolutely fantastic")
    print(f"sst2: '{r['label']}' ({r['confidence']:.3f}) encoder={r['capability']['encoder']}")
    assert r["capability"]["encoder"] == ENCODER

    print(f"\nMixed-encoder pipeline: PASSED")


def main():
    device = get_device()
    print(f"Device: {device}")

    with tempfile.TemporaryDirectory(prefix="jeffy_mixed_pack_") as tmpdir:
        pack_dir = Path(tmpdir)
        build_results = build_mixed_pack(pack_dir, device)
        test_mixed_engine(pack_dir, device)

        print(f"\nSummary:")
        for ds, r in build_results.items():
            print(f"  {ds}: test_acc={r['test_acc']:.3f} (encoder={r['encoder']})")


if __name__ == "__main__":
    main()
