"""Induction pre-pretraining data: random sequences of vocabulary pieces repeated twice (optionally 3x), so the only way to
predict the repeat is an induction circuit (previous-token head + induction head). Unlimited, cheap, local.
  python3 induction_data.py --n 40000 --len 100 --out data/induction_train.jsonl"""
import argparse, json, random
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40000); ap.add_argument("--len", type=int, default=100); ap.add_argument("--out", default="data/induction_train.jsonl"); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    random.seed(a.seed); vocab = [t for t in json.load(open("tokenizer_chat_v2.json"))["model"]["vocab"] if t.isalpha()]
    with open(a.out, "w") as f:
        for _ in range(a.n):
            L = random.randint(a.len // 3, a.len); s = " ".join(random.choice(vocab) for _ in range(L)); reps = random.choice([2, 2, 3])
            f.write(json.dumps({"text": " ".join([s] * reps)}) + "\n")
    print(f"{a.n} repeat sequences -> {a.out}")
