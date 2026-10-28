"""Fixer pass ct5 (roadmap item 5, 2026-10-27): the ct2 corpus (data/chat_v2ct_*.jsonl) minus every row whose training-time system
prompt the model was reciting ("make up something entertaining instead of admitting ignorance" family, 3,294 train / 183 val rows),
plus the 700 room-event + visit-memory rows from room_build.py (data/chat_v2room.jsonl). Train with the ct2 recipe:
  python3 prep_ct5.py && python3 train_lm.py --chat-data data/chat_v2ct5_train.jsonl --tokenizer tokenizer_chat_v2.json \\
      --resume checkpoints/grown_v2_8m.pt --fresh-opt --epochs 8 --d-model 512 --n-heads 8 --n-layers 2 --d-ff 2048 --sample-every 1 --tag oracle_v2_grown_ct5"""
import random
PH = "make up something entertaining instead of admitting ignorance"   # offline corpus cleaning (regex-free string match), not a harness rule
for split in ("train", "val"):
    rows = list(open(f"data/chat_v2ct_{split}.jsonl")); keep = [l for l in rows if PH not in l]
    extra = list(open("data/chat_v2room.jsonl")) if split == "train" else []
    out = keep + extra; random.Random(5).shuffle(out); open(f"data/chat_v2ct5_{split}.jsonl", "w").writelines(out)
    print(f"{split}: {len(rows)} -> {len(keep)} after stripping {len(rows) - len(keep)} recited-prompt rows, + {len(extra)} room/visit rows = {len(out)}")
