"""Serve a folder only under a random secret path prefix (for viewing private results through a tunnel).
Every request outside /<secret>/ gets 404. args: <folder> <port> <secret>"""
import sys, http.server, functools
folder, port, secret = sys.argv[1], int(sys.argv[2]), sys.argv[3]
class H(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        p = '/' + secret
        if self.path == p: self.send_response(301); self.send_header('Location', p + '/'); self.end_headers(); return
        if not self.path.startswith(p + '/'): self.send_error(404); return
        self.path = self.path[len(p):]; super().do_GET()
    def do_HEAD(self): self.do_GET()
    def end_headers(self): self.send_header('Cache-Control', 'private, no-store'); self.send_header('X-Robots-Tag', 'noindex'); super().end_headers()
    def log_message(self, *a): pass
http.server.ThreadingHTTPServer(('127.0.0.1', port), functools.partial(H, directory=folder)).serve_forever()
