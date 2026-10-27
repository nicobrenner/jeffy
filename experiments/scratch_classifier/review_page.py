"""Review page builder (Nico's format, 2026-10-27): each reply scored 0-5 on its own, a note per reply and one per item,
conversation continuations (items with history) first, <=20 items, blind over models (key kept out of demo/).
  python3 review_page.py --name followthrough --items items.json --title "..." --intro "..." [--score-history]
items.json: [{"history":[{"user":..,"oracle":..}], "user":"...", "replies":[{"text":..,"src":..}, ...]}]
Writes demo/review_<name>.html + demo/review_<name>.json (texts only) and pile_runs/atl/review_<name>_key.json (srcs).
Export from the page: review_<name>_scores.json = {item_n: {"s": {"A": 0-5, ...}, "n": {"A": note, ..., "item": note}}}"""
import argparse, json, random
from pathlib import Path

def build(name, items, title, intro, seed=0):
    rnd = random.Random(seed)
    items = [it for it in items if it.get("history")] + [it for it in items if not it.get("history")]
    page, key = [], {}
    for n, it in enumerate(items):
        reps = list(it["replies"]); rnd.shuffle(reps)
        letters = "ABCD"[: len(reps)]
        page.append({"n": n, "history": it.get("history", []), "user": it["user"], "replies": {L: r["text"] for L, r in zip(letters, reps)}})
        key[str(n)] = {L: r["src"] for L, r in zip(letters, reps)}
        if it.get("meta"): key[str(n)]["meta"] = it["meta"]
    Path("demo").mkdir(exist_ok=True); Path("pile_runs/atl").mkdir(parents=True, exist_ok=True)
    json.dump(page, open(f"demo/review_{name}.json", "w"), indent=0)
    json.dump(key, open(f"pile_runs/atl/review_{name}_key.json", "w"), indent=1)
    html = f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font:15px/1.4 system-ui;max-width:900px;margin:24px auto;padding:0 16px;color:#222}}
.item{{border-top:2px solid #ddd;padding:14px 0 18px}}.hist{{color:#555;margin:2px 0}}.hist b{{color:#333}}
.u{{margin:8px 0 10px;font-weight:600}}.rep{{border:1px solid #ccc;border-radius:8px;padding:8px 10px;margin:8px 0}}
.txt{{margin-bottom:6px}}.sc button{{width:34px;height:30px;margin-right:4px;border:1px solid #bbb;background:#fff;border-radius:6px;cursor:pointer}}
.sc button.sel{{background:#28a;color:#fff;border-color:#28a}}textarea{{width:100%;box-sizing:border-box;margin-top:6px;font:13px system-ui;min-height:30px}}
.lab{{font-size:12px;color:#666;margin-right:8px}}#done{{margin:24px 0}}#count{{margin-right:12px}}.tag{{font-size:12px;color:#888;margin-left:8px}}</style>
<h2>{title}</h2><p>{intro}</p><p class="lab">0 = broken/wrong/offensive · 1 = flat or off · 2 = passable · 3 = fine · 4 = good, in character · 5 = exactly what you'd want. Notes are optional and worth more than the number. Progress is saved in this browser; export at the bottom.</p>
<div id="list"></div><div id="done"><span id="count"></span><button onclick="exp()">Export scores + notes</button></div>
<script>
const KEY='review_{name}_scores'; let S={{}}; try{{S=JSON.parse(localStorage.getItem(KEY)||'{{}}')}}catch(e){{}}
const esc=s=>String(s).replace(/[&<>]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;'}}[c]));
fetch('review_{name}.json').then(r=>r.json()).then(P=>{{const L=document.getElementById('list');
 L.innerHTML=P.map(it=>{{const h=(it.history||[]).map(t=>`<div class="hist"><b>you:</b> ${{esc(t.user)}}<br><b>Oracle:</b> ${{esc(t.oracle)}}</div>`).join('');
  const reps=Object.entries(it.replies).map(([k,v])=>`<div class="rep"><div class="txt"><b>Oracle ${{Object.keys(it.replies).length>1?k:''}}:</b> ${{esc(v)}}</div><div class="sc" data-n="${{it.n}}" data-k="${{k}}">${{[0,1,2,3,4,5].map(s=>`<button data-s="${{s}}">${{s}}</button>`).join('')}}</div>${{Object.keys(it.replies).length>1?`<textarea data-n="${{it.n}}" data-k="${{k}}" placeholder="note on this reply (optional)"></textarea>`:''}}</div>`).join('');
  return `<div class="item" id="it${{it.n}}"><span class="tag">#${{it.n+1}}${{(it.history||[]).length?' · continuation':''}}</span>${{h}}<div class="u">you: ${{esc(it.user)}}</div>${{reps}}<textarea data-n="${{it.n}}" data-k="item" placeholder="${{Object.keys(it.replies).length>1?'note on the whole item (optional)':'note (optional)'}}"></textarea></div>`}}).join('');
 L.addEventListener('click',e=>{{const b=e.target.closest('button[data-s]');if(!b)return;const sc=b.parentElement;const n=sc.dataset.n,k=sc.dataset.k;S[n]=S[n]||{{s:{{}},n:{{}}}};S[n].s[k]=+b.dataset.s;save();paint()}});
 L.addEventListener('input',e=>{{const t=e.target;if(t.tagName!=='TEXTAREA')return;const n=t.dataset.n,k=t.dataset.k;S[n]=S[n]||{{s:{{}},n:{{}}}};S[n].n[k]=t.value;save()}});
 function save(){{try{{localStorage.setItem(KEY,JSON.stringify(S))}}catch(x){{}}}}
 function paint(){{document.querySelectorAll('.sc').forEach(sc=>{{const v=(S[sc.dataset.n]||{{s:{{}}}}).s[sc.dataset.k];sc.querySelectorAll('button').forEach(b=>b.classList.toggle('sel',+b.dataset.s===v))}});
  document.querySelectorAll('textarea').forEach(t=>{{const v=((S[t.dataset.n]||{{n:{{}}}}).n||{{}})[t.dataset.k];if(v!==undefined&&t.value!==v)t.value=v}});
  const total=P.reduce((a,it)=>a+Object.keys(it.replies).length,0);const done=Object.values(S).reduce((a,x)=>a+Object.keys(x.s||{{}}).length,0);document.getElementById('count').textContent=done+' / '+total+' replies scored'}}
 paint();}});
function exp(){{const b=new Blob([JSON.stringify(S)],{{type:'application/json'}});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='review_{name}_scores.json';a.click()}}
</script>"""
    open(f"demo/review_{name}.html", "w").write(html)
    return len(page)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--name", required=True); ap.add_argument("--items", required=True); ap.add_argument("--title", required=True); ap.add_argument("--intro", default=""); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); n = build(a.name, json.load(open(a.items)), a.title, a.intro, a.seed); print(f"review_{a.name}: {n} items -> demo/review_{a.name}.html")
