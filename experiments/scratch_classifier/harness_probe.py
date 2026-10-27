"""Probes through the FULL harness (harness.respond): what the demo serves, not the bare model.
  identity:  python3 harness_probe.py identity --ckpt C --tag T [--prime/--no-prime --sampler recipe03 --best-of 4 --verifier P --controller --act-mode token]
  simuser:   python3 harness_probe.py simuser --ckpt C --tag T --turn 1            (then an agent writes followups_k.json; --turn k --followups F)
Writes pile_runs/atl/hprobe_id_<tag>.jsonl / hsim_<tag>.jsonl with the harness config inside every row.
"""
import argparse, json, re
from pathlib import Path
from atl_probe import load
from atl_probe_identity import OPENERS, classify, HUMAN
from train_lm import CustomTokenizerWrapper
import harness

def cfg_from(a):
    return dict(prime=a.prime, sampler=a.sampler, best_of=a.best_of, verifier=a.verifier, controller=a.controller, tau=a.tau, act_mode=a.act_mode, seed=0, cross_turn=getattr(a, "cross", True))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("probe", choices=["identity", "simuser"]); ap.add_argument("--ckpt", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--tokenizer", default="tokenizer_chat_v2.json"); ap.add_argument("--prime", dest="prime", action="store_true", default=True); ap.add_argument("--no-prime", dest="prime", action="store_false"); ap.add_argument("--prime-first", dest="prime", action="store_const", const="first")
    ap.add_argument("--sampler", default="recipe03"); ap.add_argument("--best-of", type=int, default=1); ap.add_argument("--verifier", default=None); ap.add_argument("--controller", action="store_true")
    ap.add_argument("--tau", type=float, default=0.5); ap.add_argument("--act-mode", default=None); ap.add_argument("--turn", type=int, default=1); ap.add_argument("--followups", default=None)
    ap.add_argument("--no-cross", dest="cross", action="store_false", default=True); ap.add_argument("--probes", default="data/atl_probes.json"); ap.add_argument("--n", type=int, default=30)
    a = ap.parse_args(); tok = CustomTokenizerWrapper(a.tokenizer); m, _ = load(a.ckpt); cfg = cfg_from(a); Path("pile_runs/atl").mkdir(parents=True, exist_ok=True)
    if a.probe == "identity":
        out = f"pile_runs/atl/hprobe_id_{a.tag}.jsonl"; name = 0; drift = 0; later = 0; forced = 0; fallback = 0
        with open(out, "w") as f:
            for oi, opener in enumerate(OPENERS):
                for s in range(8):
                    hist = []; name_cls = "none"
                    for turn, user in enumerate([opener, "what's your name?", None, "what are you up to?", "nice, what kind?"], 1):
                        if user is None: user = f"nice to meet you, {name_cls.split(':')[1]}!" if name_cls.startswith("human:") else ("nice to meet you, Tiny Oracle!" if name_cls == "oracle" else "nice to meet you!")
                        resp, info = harness.respond(m, tok, hist, user, {**cfg, "seed": 1000 * s + 10 * oi + turn}); cls = classify(resp)
                        if turn == 2: name_cls = cls; name += cls == "oracle"
                        if turn >= 3: later += 1; drift += bool(HUMAN.search(resp) or cls.startswith("human:"))
                        forced += info["forced"]; fallback += info["fallback"]
                        f.write(json.dumps({"id": f"id{oi}s{s}", "turn": turn, "history": list(hist), "prompt": user, "response": resp, "cls": cls, "info": info, "cfg": cfg}) + "\n")
                        hist.append({"user": user, "oracle": resp})
        print(f"[{a.tag}] harness {json.dumps({k: v for k, v in cfg.items() if k != 'seed'})}: name {name}/32, human-life {drift}/{later}, forced {forced}, fallback {fallback}")
    else:
        out = f"pile_runs/atl/hsim_{a.tag}.jsonl"; probes = json.load(open(a.probes)); probes = ([p for p in probes if p.get("novel")] + [p for p in probes if not p.get("novel")])[: a.n]
        prev = {}
        if a.turn > 1:
            for r in map(json.loads, open(out)): prev.setdefault(r["id"], []).append(r)
            fu = json.load(open(a.followups))
        with open(out, "a" if a.turn > 1 else "w") as f:
            for p in probes:
                hist_rows = sorted(prev.get(p["id"], []), key=lambda r: r["turn"]) if a.turn > 1 else []
                if a.turn > 1 and (len(hist_rows) != a.turn - 1 or p["id"] not in fu): continue
                hist = [{"user": r["prompt"], "oracle": r["response"]} for r in hist_rows]; user = p["prompt"] if a.turn == 1 else fu[p["id"]]
                resp, info = harness.respond(m, tok, hist, user, {**cfg, "seed": a.turn})
                f.write(json.dumps({"id": p["id"], "topic": p["topic"], "novel": p.get("novel", False), "turn": a.turn, "history": hist, "prompt": user, "response": resp, "info": info, "cfg": cfg}) + "\n")
        print(f"[{a.tag}] turn {a.turn} written -> {out}")

if __name__ == "__main__":
    main()
