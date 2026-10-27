"""Continuation pairs for a review page: shared history (model A's turn 1 + the reactive follow-up written for it), both
models answer the SAME follow-up. A's turn 2 comes from its existing simuser run; B's is generated here with the demo harness.
  python3 cont_pairs.py --a-tag ct2_full --a-ckpt checkpoints/best_lm_oracle_v2_grown_ct2.pt --b-ckpt checkpoints/best_lm_oracle_v6.pt --b-tag v6 --fu pile_runs/atl/hsim_ct2_full_fu2.json --out items.json [--regen-a]
Writes review items (review_page.py format) with src labels A/B for the key."""
import argparse, json
from atl_probe import load
from train_lm import CustomTokenizerWrapper
import harness

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--a-tag", required=True); ap.add_argument("--a-ckpt", required=True); ap.add_argument("--b-ckpt", required=True); ap.add_argument("--b-tag", required=True)
    ap.add_argument("--fu", required=True); ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=18); ap.add_argument("--regen-a", action="store_true"); ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json")
    a = ap.parse_args(); tok = CustomTokenizerWrapper(a.tokenizer)
    rows = [json.loads(l) for l in open(f"pile_runs/atl/hsim_{a.a_tag}.jsonl")]; by = {}
    for r in rows: by.setdefault(r["id"], {})[r["turn"]] = r
    fu = json.load(open(a.fu)); cfg = dict(harness.DEFAULT, prime="first", sampler="sched")
    mb, _ = load(a.b_ckpt); ma = load(a.a_ckpt)[0] if a.regen_a else None
    items = []
    for i, t in by.items():
        if 1 not in t or i not in fu or len(items) >= a.n: continue
        hist = [{"user": t[1]["prompt"], "oracle": t[1]["response"]}]; u2 = fu[i]
        rb, _ = harness.respond(mb, tok, hist, u2, {**cfg, "seed": 2})
        ra = harness.respond(ma, tok, hist, u2, {**cfg, "seed": 2})[0] if ma else t[2]["response"]
        items.append({"history": hist, "user": u2, "replies": [{"text": ra, "src": f"{a.a_tag}/{i}"}, {"text": rb, "src": f"{a.b_tag}/{i}"}], "meta": {"id": i}})
        print(f"[{i}] {u2!r}\n   A: {ra!r}\n   B: {rb!r}")
    json.dump(items, open(a.out, "w"), indent=0); print(len(items), "items ->", a.out)
