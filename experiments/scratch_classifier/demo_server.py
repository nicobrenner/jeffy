"""Demo server with cross-origin isolation headers (COOP/COEP) so onnxruntime-web can use multi-threaded WASM (SharedArrayBuffer).
python3 -m http.server sends no such headers -> ort runs single-threaded (the page sets numThreads = crossOriginIsolated ? hw : 1).
  python3 demo_server.py 8766
"""
import sys, http.server, functools
class H(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin"); self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        self.send_header("Cross-Origin-Resource-Policy", "cross-origin"); self.send_header("Cache-Control", "no-cache"); super().end_headers()
    def log_message(self, *a): pass
port = int(sys.argv[1]) if len(sys.argv) > 1 else 8766
http.server.ThreadingHTTPServer(("127.0.0.1", port), functools.partial(H, directory="demo")).serve_forever()
