"""Demo server with cross-origin isolation headers (COOP/COEP) so onnxruntime-web can use multi-threaded WASM (SharedArrayBuffer).
python3 -m http.server sends no such headers -> ort runs single-threaded (the page sets numThreads = crossOriginIsolated ? hw : 1).
  python3 demo_server.py 8766
"""
import sys, http.server, functools
import json, os, time
LAB = os.path.join("pile_runs", "lab")   # local only (.gitignore): the researcher thread — notes from Nico, replies from the lab sessions
class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/lab/replies"):
            os.makedirs(LAB, exist_ok=True); f = os.path.join(LAB, "replies.jsonl"); rows = [l for l in open(f)] if os.path.exists(f) else []
            body = ("[" + ",".join(l.strip() for l in rows if l.strip()) + "]").encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/notes"):
            os.makedirs(LAB, exist_ok=True); f = os.path.join(LAB, "notes.jsonl"); rows = [l for l in open(f)] if os.path.exists(f) else []
            body = ("[" + ",".join(l.strip() for l in rows if l.strip()) + "]").encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        if self.path.startswith("/lab/fetch"):   # arena terrain: fetch a page the user typed, return its visible text (localhost only, on request)
            import urllib.parse, urllib.request, html as H, re as R
            q = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query); url = (q.get("url") or [""])[0]; out = {}
            try:
                if not url.lower().startswith(("http://", "https://")): raise ValueError("not a web address")
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tiny-oracle arena)"}); raw = urllib.request.urlopen(req, timeout=8).read(600000).decode("utf-8", "replace")
                title = R.search(r"<title[^>]*>(.*?)</title>", raw, R.S | R.I); raw = R.sub(r"(?is)<(script|style|noscript|svg|nav|footer|header)[^>]*>.*?</\1>", " ", raw)
                paras = R.findall(r"(?is)<p[^>]*>(.*?)</p>", raw); body_html = " ".join(paras) if sum(len(x) for x in paras) > 400 else raw
                text = H.unescape(R.sub(r"<[^>]+>", " ", body_html)); text = R.sub(r"\[\d+\]", " ", text); text = R.sub(r"\s+", " ", text).strip()[:20000]
                out = {"title": H.unescape(title.group(1).strip())[:120] if title else url, "text": text}
            except Exception as e: out = {"error": str(e)[:200]}
            body = json.dumps(out).encode(); self.send_response(200); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body); return
        return super().do_GET()
    def do_POST(self):
        if self.path.startswith("/lab/sync"):   # the page pushes its browser-local data here so nothing depends on pressing export
            n = int(self.headers.get("Content-Length", 0)); data = json.loads(self.rfile.read(n) or b"{}"); os.makedirs(LAB, exist_ok=True)
            for k in ("engagement", "chat_scores", "corrections", "visits", "arena_snapshots"):
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
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory="demo")).serve_forever()
