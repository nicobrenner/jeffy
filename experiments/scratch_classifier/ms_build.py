"""Multi-session memory rows (the recap tier). From pile_runs/atl/multisession_src.jsonl (chained corpus sessions) and the
agent-written multisession_out_part*.jsonl (recap / followup / reply), build training rows:
  <|system|> SYSTEM + " " + recap \n <|user|> B-open user \n <|oracle|> *act B-open oracle \n <|user|> followup \n <|oracle|> *act reply
Held-out ids (every 5th) go to pile_runs/atl/ms_heldout.jsonl for the recall probe (ms_probe.py).
  python3 ms_build.py --copies 6 --out data/chat_v2ms_train.jsonl
"""
import argparse, glob, json, re
from atl_probe import SYSTEM
from ct_v2 import act_of, MARK

def load():
    src = {json.loads(l)["id"]: json.loads(l) for l in open("pile_runs/atl/multisession_src.jsonl")}
    out = []
    for f in sorted(glob.glob("pile_runs/atl/multisession_out_part*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r["id"] in src and r.get("recap") and r.get("followup") and r.get("reply"): out.append({**src[r["id"]], **r})
    return out

def row(r):
    b = re.findall(r"^(user|oracle): (.*)$", r["conv_b_open"], flags=re.M)
    t = SYSTEM.rstrip("\n") + " " + r["recap"].rstrip(".") + ".\n"
    for role, txt in b[:2]:
        t += f"<|user|> {txt}\n" if role == "user" else f"<|oracle|> {MARK[act_of(txt)]} {txt}\n"
    t += f"<|user|> {r['followup']}\n<|oracle|> {MARK[act_of(r['reply'])]} {r['reply']}"
    return t

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--copies", type=int, default=6); ap.add_argument("--out", default="data/chat_v2ms_train.jsonl"); a = ap.parse_args()
    rows = load(); held = [r for i, r in enumerate(rows) if i % 5 == 0]; train = [r for i, r in enumerate(rows) if i % 5 != 0]
    with open(a.out, "w") as f:
        for _ in range(a.copies):
            for r in train: f.write(json.dumps({"text": row(r)}) + "\n")
    with open("pile_runs/atl/ms_heldout.jsonl", "w") as f:
        for r in held: f.write(json.dumps(r) + "\n")
    print(f"{len(rows)} recap rows -> {len(train)} train x {a.copies} = {len(train)*a.copies} rows in {a.out}; {len(held)} held out"); print(row(train[0]).replace("\n", " | ")[:500])
