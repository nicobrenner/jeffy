"""Rebuild the bundled support_ticket pack with the issue #11 account fix.

fix_support_ticket.py showed more data does not help: at 24,722 examples per
class "change the address on my account" is predicted shipping at 0.9997.
fix_support_ticket_2.py showed 24 hand-written account examples do help, at
500 examples per class.

The shipped pack is trained on 50 per class, so this re-runs the comparison at
the shipped pack's own size and recipe, measures both against the untouched
550-example test split, and only then writes the pack.

    python scripts/rebuild_support_ticket.py           # measure only
    python scripts/rebuild_support_ticket.py --write   # also replace the pack

Writes data/eval_results/support_ticket_rebuild.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

TRAIN = REPO / "data/eval_results/bitext_support_train.json"
TEST = REPO / "data/eval_results/bitext_support_test.json"
PACK = REPO / "src/jeffy/pack"
OUT = REPO / "data/eval_results/support_ticket_rebuild.json"

# The phrasing Bitext's account class is missing: changing a contact field on an
# existing account, as opposed to changing the account's plan or type. Bitext
# files most of those under shipping, which is correct for a delivery address
# and wrong for everything else.
ACCOUNT_PATCH = [
    "change the address on my account",
    "how do I change my email address",
    "update my contact details",
    "change my email",
    "I need to update the email on my account",
    "how can I edit my phone number",
    "change my contact email please",
    "update the address on my profile",
    "I want to change my registered address",
    "edit my personal details",
    "update my personal information",
    "how do I change the phone number on my account",
    "change my mailing address on file",
    "I need to correct my name on the account",
    "update my email preferences and address",
    "can I change the email I signed up with",
    "my contact details are out of date",
    "please update my account address",
    "how to edit account contact information",
    "change the phone number linked to my account",
    "I moved, update my address on the account",
    "fix the email address on my account",
    "amend my contact information",
    "I need my account details changed",
]

# Held back from ACCOUNT_PATCH, so a flip here is generalization and not recall
# of a patched example.
UNSEEN = {
    "how do I update the email on file",
    "I need to change my home address",
    "where do I edit my contact number",
    "I need to update my billing address and phone number on my account",
}

PROBES = [
    ("change the address on my account", "account"),
    ("how do I change my email address", "account"),
    ("update my contact details", "account"),
    ("I need to edit my profile", "account"),
    ("change my email", "account"),
    ("I cannot log into my account", "account"),
    ("reset my password", "account"),
    ("I want to delete my account", "account"),
    ("how do I update the email on file", "account"),
    ("I need to change my home address", "account"),
    ("where do I edit my contact number", "account"),
    # From the Chatwoot end-to-end run, where it routed to shipping.
    ("I need to update my billing address and phone number on my account", "account"),
    # Must stay put: these really are shipping and delivery.
    ("where is my package", "delivery"),
    ("I have to change the delivery address", "shipping"),
    ("I need to edit my shipping address", "shipping"),
    ("my order has not arrived yet", "delivery"),
]


def _flatten(blob):
    texts, labels = [], []
    for label in sorted(blob):
        for t in blob[label]:
            texts.append(t)
            labels.append(label)
    return texts, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="replace src/jeffy/pack/support_ticket with the patched model")
    args = ap.parse_args()

    from sklearn.metrics import accuracy_score

    from jeffy.model_pack import ArtifactManifest, coef_hash, save_artifact, scaler_hash
    from jeffy.train import train_classifier

    train_texts, train_labels = _flatten(json.loads(TRAIN.read_text()))
    test_texts, test_labels = _flatten(json.loads(TEST.read_text()))
    print(f"train {len(train_texts)}  test {len(test_texts)}")

    overlap = set(train_texts) & set(test_texts)
    if overlap:
        raise SystemExit(f"train/test overlap on {len(overlap)} texts, aborting")

    probe_texts = [p[0] for p in PROBES]
    results = {"n_train_base": len(train_texts), "n_test": len(test_texts),
               "n_patch": len(ACCOUNT_PATCH), "runs": []}
    keep = {}

    for patched in (False, True):
        texts = list(train_texts)
        labels = list(train_labels)
        if patched:
            texts += ACCOUNT_PATCH
            labels += ["account"] * len(ACCOUNT_PATCH)

        tag = f"+{len(ACCOUNT_PATCH)} account examples" if patched else "baseline"
        print(f"\n=== {tag}: {len(texts)} examples")
        # Same recipe as the shipped pack: default C, 20% split, 3-fold CV.
        tc = train_classifier(texts, labels, task_id="support_ticket", verbose=True)

        enc = tc.encoder
        X_test = tc.scaler.transform(enc.encode(test_texts, batch_size=64))
        preds = tc.clf.predict(X_test)
        held_out = float(accuracy_score(test_labels, preds))

        X_probe = tc.scaler.transform(enc.encode(probe_texts, batch_size=64))
        probs = tc.clf.predict_proba(X_probe)
        classes = list(tc.clf.classes_)

        X_fit = tc.scaler.transform(enc.encode(texts, batch_size=64))
        fit_acc = float(accuracy_score(labels, tc.clf.predict(X_fit)))

        run = {"patched": patched, "n_train": len(texts),
               "train_accuracy": round(fit_acc, 4),
               "held_out_accuracy": round(held_out, 4), "probes": []}
        print(f"  held-out accuracy on the 550 test examples: {held_out:.4f}")
        for (text, expected), row in zip(PROBES, probs):
            top = int(np.argmax(row))
            ok = classes[top] == expected
            run["probes"].append({
                "text": text, "expected": expected, "predicted": classes[top],
                "confidence": round(float(row[top]), 4), "correct": ok,
                "held_back": text in UNSEEN,
            })
            print(f"  {'ok   ' if ok else 'WRONG'} {classes[top]:12s} {row[top]:.4f}  "
                  f"{text}{' [unseen]' if text in UNSEEN else ''}")

        n_ok = sum(p["correct"] for p in run["probes"])
        n_unseen_ok = sum(p["correct"] for p in run["probes"] if p["held_back"])
        run["probes_correct"] = f"{n_ok}/{len(PROBES)}"
        run["held_back_correct"] = f"{n_unseen_ok}/{len(UNSEEN)}"
        print(f"  -> {n_ok}/{len(PROBES)} probes, {n_unseen_ok}/{len(UNSEEN)} of them held back")
        results["runs"].append(run)
        keep[patched] = tc

    OUT.write_text(json.dumps(results, indent=2))
    print(f"\nsaved {OUT}")

    if not args.write:
        print("measure-only run, pack untouched (pass --write to replace it)")
        return

    base, new = results["runs"][0], results["runs"][1]
    if new["held_out_accuracy"] < base["held_out_accuracy"] - 0.01:
        raise SystemExit(
            f"refusing to write: held-out accuracy fell from "
            f"{base['held_out_accuracy']} to {new['held_out_accuracy']}")

    tc = keep[True]
    # Keep the shipped manifest's provenance fields; only the model, the counts
    # and the hashes change. train_examples counts everything fed in, matching
    # how the existing manifest counts its 550.
    old = json.loads((PACK / "support_ticket/manifest.json").read_text())
    manifest = ArtifactManifest(
        dataset="support_ticket",
        description=old["description"],
        n_classes=len(tc.labels),
        task_type=old["task_type"],
        encoder=tc.encoder_name,
        encoder_dim=int(tc.scaler.mean_.shape[0]),
        labels=tc.labels,
        class_order=list(tc.clf.classes_),
        hf_source=old["hf_source"] + " + 24 hand-written account examples (issue #11)",
        hf_config=old["hf_config"],
        hf_revision=old["hf_revision"],
        license=old["license"],
        train_examples=new["n_train"],
        train_accuracy=new["train_accuracy"],
        test_examples=results["n_test"],
        test_accuracy=new["held_out_accuracy"],
        scaler_mean_hash=scaler_hash(tc.scaler),
        classifier_coef_hash=coef_hash(tc.clf),
    )
    save_artifact(PACK, "support_ticket", tc.clf, tc.scaler, manifest)
    print(f"wrote {PACK / 'support_ticket'}")


if __name__ == "__main__":
    main()
