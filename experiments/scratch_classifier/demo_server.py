"""Demo server with cross-origin isolation headers (COOP/COEP) so onnxruntime-web can use multi-threaded WASM (SharedArrayBuffer).
python3 -m http.server sends no such headers -> ort runs single-threaded (the page sets numThreads = crossOriginIsolated ? hw : 1).
  python3 demo_server.py 8766
"""
import sys, http.server, functools
import json, os, time, threading
LAB = os.environ.get("ARENA_LAB") or os.path.join("pile_runs", "lab")   # tests point this at a scratch folder
_enc = None; _elock = threading.Lock()
def _warm():   # load the encoder in the background at start, so the Crawler's first visit is scored by meaning
    global _enc
    try:
        with _elock:
            if _enc is None:
                from sentence_transformers import SentenceTransformer; _enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    except Exception: pass
WORLDS = os.path.join(LAB, "worlds.json"); _wlock = threading.Lock()   # connected arenas: registry, link requests, inboxes, lite views (local relay)
def _wload():
    try: return json.load(open(WORLDS))
    except Exception: return {"worlds": {}}
def _wsave(d):
    os.makedirs(LAB, exist_ok=True); tmp = WORLDS + ".tmp"; json.dump(d, open(tmp, "w")); os.replace(tmp, WORLDS)
def _w(d, code): return d["worlds"].setdefault(code, {"name": code, "links": [], "requests": [], "inbox": [], "view": None, "seen": 0})   # local only (.gitignore): the researcher thread — notes from Nico, replies from the lab sessions
class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/lab/recover"):   # one-off recovery bundle written by the manager (pile_runs/lab is local only)
            f = os.path.join(LAB, "recover.json"); body = open(f, "rb").read() if os.path.exists(f) else b"null"
            self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/world"):   # the arena's disk copy: latest autosave + every snapshot ever synced (used when the browser's storage is empty)
            out = {}
            for k in ("arena_autosave", "arena_snapshots"):
                f = os.path.join(LAB, f"sync_{k}.json"); out[k] = json.load(open(f)) if os.path.exists(f) else None
            body = json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/replies"):
            os.makedirs(LAB, exist_ok=True); f = os.path.join(LAB, "replies.jsonl"); rows = [l for l in open(f)] if os.path.exists(f) else []
            body = ("[" + ",".join(l.strip() for l in rows if l.strip()) + "]").encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/img"):   # same-origin image proxy so the arena can read pixels (pixel sorting needs that)
            import urllib.parse, urllib.request; u = (urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("url") or [""])[0]
            try:
                if not u.lower().startswith(("http://", "https://")): raise ValueError("bad url")
                r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (tiny-oracle arena)"}), timeout=8); ct = r.headers.get("Content-Type", "")
                if not ct.startswith("image/"): raise ValueError("not an image")
                data = r.read(4_000_000); self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(data))); self.end_headers(); self.wfile.write(data); return
            except Exception as e: self.send_response(502); self.end_headers(); return
        if self.path.startswith("/jeffy/capabilities"):
            import urllib.request
            try: raw = urllib.request.urlopen("http://127.0.0.1:8411/v1/capabilities", timeout=10).read(); code = 200
            except Exception as e: raw = json.dumps({"error": str(e)[:200]}).encode(); code = 502
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return
        if self.path.startswith("/worlds/state"):
            import urllib.parse; code = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("code", [""])[0]
            with _wlock:
                d = _wload(); me = _w(d, code); me["seen"] = time.time(); _wsave(d)
                out = {"me": {k: me[k] for k in ("name", "links", "requests", "inbox")}, "peers": {c: {"name": d["worlds"].get(c, {}).get("name", c), "seen": d["worlds"].get(c, {}).get("seen", 0), "view": d["worlds"].get(c, {}).get("view")} for c in me["links"]}, "now": time.time()}
            body = json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/notes"):
            os.makedirs(LAB, exist_ok=True); f = os.path.join(LAB, "notes.jsonl"); rows = [l for l in open(f)] if os.path.exists(f) else []
            body = ("[" + ",".join(l.strip() for l in rows if l.strip()) + "]").encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/asset"):   # traveling mode: the page's stylesheets, fonts and images, same-origin; CSS url()/@import are rewritten to come back through here
            import urllib.parse, urllib.request, re as R
            u = (urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("url") or [""])[0]
            try:
                if not u.lower().startswith(("http://", "https://")): raise ValueError("bad url")
                r = urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0 (tiny-oracle arena)"}), timeout=10); ct = r.headers.get("Content-Type", "application/octet-stream"); data = r.read(6_000_000)
                if not (ct.startswith(("text/css", "image/", "font/", "application/font", "application/x-font", "application/vnd.ms-fontobject")) or u.split("?")[0].lower().endswith((".css", ".woff", ".woff2", ".ttf", ".otf"))): raise ValueError("not a style, font or image")
                if "css" in ct or u.split("?")[0].lower().endswith(".css"):
                    css = data.decode("utf-8", "replace")
                    fix = lambda m: 'url("' + ("/lab/asset?url=" + urllib.parse.quote(urllib.parse.urljoin(r.geturl(), m.group(2).strip()), safe="") if not m.group(2).strip().startswith("data:") else m.group(2).strip()) + '")'
                    css = R.sub(r"""url\(\s*(['"]?)([^'")]+)\1\s*\)""", fix, css)
                    css = R.sub(r"""@import\s+(['"])([^'"]+)\1""", lambda m: '@import url("/lab/asset?url=' + urllib.parse.quote(urllib.parse.urljoin(r.geturl(), m.group(2)), safe="") + '")', css)
                    data = css.encode(); ct = "text/css; charset=utf-8"
                self.send_response(200); self.send_header("Content-Type", ct); self.send_header("Content-Length", str(len(data))); self.send_header("Cache-Control", "max-age=3600"); self.end_headers(); self.wfile.write(data); return
            except Exception as e: self.send_response(502); self.end_headers(); return
        if self.path.startswith("/lab/page"):   # traveling mode: the raw page HTML for the arena to sanitize and render (localhost only, on request; the browser keeps only an allowlist of tags)
            import urllib.parse, urllib.request
            url = (urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query).get("url") or [""])[0]; out = {}
            try:
                if not url.lower().startswith(("http://", "https://")): raise ValueError("not a web address")
                r = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tiny-oracle arena)"}), timeout=10)
                if "html" not in r.headers.get("Content-Type", "html"): raise ValueError("not a web page")
                out = {"url": r.geturl(), "html": r.read(2_500_000).decode("utf-8", "replace")}
            except Exception as e: out = {"error": str(e)[:200]}
            body = json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/fetch"):   # arena terrain: fetch a page the user typed, return its visible text (localhost only, on request)
            import urllib.parse, urllib.request, html as H, re as R
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query); url = (q.get("url") or [""])[0]; out = {}
            try:
                if not url.lower().startswith(("http://", "https://")): raise ValueError("not a web address")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tiny-oracle arena)"}); raw = urllib.request.urlopen(req, timeout=8).read(600000).decode("utf-8", "replace")
                raw0 = raw; title = R.search(r"<title[^>]*>(.*?)</title>", raw, R.S | R.I); raw = R.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
                paras = R.findall(r"(?is)<p[^>]*>(.*?)</p>", raw); body_html = " ".join(paras) if sum(len(x) for x in paras) > 400 else raw
                text = H.unescape(R.sub(r"<[^>]+>", " ", body_html)); text = R.sub(r"\[\d+\]", " ", text); text = R.sub(r"\s+", " ", text).strip()[:20000]
                imgs = []
                for src in R.findall(r'(?is)<img[^>]+?src="([^"]+)"', raw0):
                    u = urllib.parse.urljoin(url, H.unescape(src))
                    if u.startswith("data:") or R.search(r"\.svg|icon|logo|sprite|pixel|1x1|static/images", u, R.I) or not R.search(r"\.(jpe?g|png|webp|gif)", u, R.I): continue
                    if u not in imgs: imgs.append(u)
                    if len(imgs) >= 4: break
                ui = []   # interface elements as plain data (tag + visible text): the web-debugger creature's material
                for tag, inner in R.findall(r"(?is)<(button|h1|h2|h3|label|a)\b[^>]*>(.*?)</\1>", raw0):
                    t = R.sub(r"\s+", " ", H.unescape(R.sub(r"<[^>]+>", " ", inner))).strip()
                    if 2 <= len(t) <= 40 and not any(x["text"] == t for x in ui): ui.append({"tag": tag.lower(), "text": t})
                    if len(ui) >= 40: break
                for ph in R.findall(r'(?is)<input[^>]+placeholder="([^"]{2,40})"', raw0)[:4]: ui.append({"tag": "input", "text": H.unescape(ph)})
                import random as _r; _r.shuffle(ui); ui = ui[:14]
                links, seen = [], set()   # the crawler creature's next steps: links with their visible text
                for href, inner in R.findall(r'(?is)<a\b[^>]*?href="([^"#][^"]*)"[^>]*>(.*?)</a>', raw):
                    u = urllib.parse.urljoin(url, H.unescape(href)).split("#")[0]; t = R.sub(r"\s+", " ", H.unescape(R.sub(r"<[^>]+>", " ", inner))).strip()
                    if not u.startswith(("http://", "https://")) or u in seen or u == url or not (2 <= len(t) <= 80) or t.lower().startswith(("http", "www.")) or R.search(r"\.(pdf|zip|jpe?g|png|gif|svg|mp4|mp3)$|[?&]action=|Special:|Help:|Talk:|File:|Category:|Wikipedia:|Portal:|Template", u, R.I): continue
                    seen.add(u); links.append({"href": u, "text": t})
                    if len(links) >= 200: break
                out = {"title": H.unescape(title.group(1).strip())[:120] if title else url, "text": text, "images": imgs, "ui": ui, "links": links, "url": url}
            except Exception as e: out = {"error": str(e)[:200]}
            body = json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        return super().do_GET()
    def do_POST(self):
        if self.path.startswith("/lab/embed"):   # the Crawler's sense of 'close to the floor': all-MiniLM-L6-v2 sentence vectors on CPU, loaded on first use
            n = int(self.headers.get("Content-Length", 0)); q = json.loads(self.rfile.read(n) or b"{}"); texts = [str(t)[:300] for t in (q.get("texts") or [])][:1200]
            try:
                global _enc
                with _elock:
                    if _enc is None:
                        from sentence_transformers import SentenceTransformer; _enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
                    v = _enc.encode(texts, batch_size=64, normalize_embeddings=True) if texts else []
                out = {"vecs": [[round(float(x), 4) for x in row] for row in v]}
            except Exception as e: out = {"error": str(e)[:200]}
            body = json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/jeffy/"):   # data-lab: same-origin proxy to the local Jeffy server (the arena page cannot call another port under COEP)
            import urllib.request; n = int(self.headers.get("Content-Length", 0)); data = self.rfile.read(n)
            try: raw = urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8411/v1/" + self.path[len("/jeffy/"):], data=data, headers={"Content-Type": "application/json"}), timeout=10).read(); code = 200
            except Exception as e: raw = json.dumps({"error": str(e)[:200]}).encode(); code = 502
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(raw))); self.end_headers(); self.wfile.write(raw); return
        if self.path.startswith("/worlds/"):
            n = int(self.headers.get("Content-Length", 0)); q = json.loads(self.rfile.read(n) or b"{}"); act = self.path.split("/")[2].split("?")[0]; res = {"ok": True}
            with _wlock:
                d = _wload(); me = _w(d, q.get("code", ""))
                if act == "register": me["name"] = str(q.get("name", me["name"]))[:40]; me["seen"] = time.time()
                elif act == "link":
                    other = str(q.get("other", "")).strip().lower()
                    if other == q.get("code") or not other: res = {"ok": False, "error": "that is your own code"}
                    elif other not in d["worlds"]: res = {"ok": False, "error": "no world with that code has been open on this server"}
                    elif q["code"] in d["worlds"][other]["links"]: res = {"ok": True, "already": True}
                    elif q["code"] not in d["worlds"][other]["requests"]: d["worlds"][other]["requests"].append(q["code"])
                elif act == "approve":
                    other = q.get("other"); me["requests"] = [r for r in me["requests"] if r != other]
                    if q.get("yes") and other in d["worlds"]:
                        if other not in me["links"]: me["links"].append(other)
                        o = d["worlds"][other]
                        if q["code"] not in o["links"]: o["links"].append(q["code"])
                elif act == "unlink":
                    other = q.get("other"); me["links"] = [x for x in me["links"] if x != other]
                    if other in d["worlds"]: d["worlds"][other]["links"] = [x for x in d["worlds"][other]["links"] if x != q["code"]]
                elif act == "send":
                    to = q.get("to")
                    if to in me["links"] and to in d["worlds"]: item = dict(q.get("item") or {}); item["from"] = q["code"]; item["t"] = time.time(); d["worlds"][to]["inbox"].append(item); d["worlds"][to]["inbox"] = d["worlds"][to]["inbox"][-50:]
                    else: res = {"ok": False, "error": "not linked"}
                elif act == "ack": ids = set(q.get("ids") or []); me["inbox"] = [i for i in me["inbox"] if i.get("id") not in ids]
                elif act == "view": me["view"] = q.get("view"); me["seen"] = time.time()
                _wsave(d)
            body = json.dumps(res).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/sync"):   # the page pushes its browser-local data here so nothing depends on pressing export
            n = int(self.headers.get("Content-Length", 0)); data = json.loads(self.rfile.read(n) or b"{}"); os.makedirs(LAB, exist_ok=True)
            if isinstance(data.get("arena_snapshots"), list):   # merge, never shrink: a fresh browser must not erase the snapshots on disk
                f = os.path.join(LAB, "sync_arena_snapshots.json"); old = json.load(open(f)) if os.path.exists(f) else []
                seen = {(x.get("t"), x.get("name")) for x in data["arena_snapshots"]}
                data["arena_snapshots"] = sorted([x for x in old if (x.get("t"), x.get("name")) not in seen] + data["arena_snapshots"], key=lambda x: x.get("t") or 0)[-60:]
            if isinstance(data.get("arena_autosave"), dict):   # rolling history: keep one autosave per 5 minutes, the last 96 (8 hours of play)
                f = os.path.join(LAB, "sync_arena_autosave.json"); hist = os.path.join(LAB, "arena_autosaves"); os.makedirs(hist, exist_ok=True)
                size = lambda x: len(x.get("words") or []) + len(x.get("residents") or []) + len(x.get("things") or [])
                shrinks = os.path.exists(f) and size(data["arena_autosave"]) < 0.5 * size(json.load(open(f)))   # a much smaller world never replaces a bigger one without a copy
                if os.path.exists(f) and (shrinks or time.time() - max([os.path.getmtime(os.path.join(hist, x)) for x in os.listdir(hist)] or [0]) > 300):
                    os.replace(f, os.path.join(hist, time.strftime("%Y%m%d-%H%M%S") + ".json"))
                    for x in sorted(os.listdir(hist))[:-96]: os.remove(os.path.join(hist, x))
            for k in ("engagement", "chat_scores", "corrections", "visits", "arena_snapshots", "arena_autosave"):
                if k in data:
                    tmp = os.path.join(LAB, f"sync_{k}.json.tmp"); json.dump(data[k], open(tmp, "w")); os.replace(tmp, os.path.join(LAB, f"sync_{k}.json"))
            body = b'{"ok":true}'; self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/notes"):
            n = int(self.headers.get("Content-Length", 0)); row = json.loads(self.rfile.read(n) or b"{}"); row.setdefault("t", time.time()); row.setdefault("who", "nico")
            os.makedirs(LAB, exist_ok=True)
            with open(os.path.join(LAB, "notes.jsonl"), "a") as f: f.write(json.dumps(row) + "\n")
            body = b'{"ok":true}'; self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        self.send_response(404); self.end_headers()
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin"); self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin"); self.send_header("Cache-Control", "no-cache"); super().end_headers()
    def log_message(self, *a): pass
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
threading.Thread(target=_warm, daemon=True).start()
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory="demo")).serve_forever()
