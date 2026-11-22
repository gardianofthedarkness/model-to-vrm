"""Model-to-VRM pipeline in a browser (phone friendly), no agent needed:
  E:\\splat\\.venv\\Scripts\\python.exe server.py [port]          (default 8765, listens on 127.0.0.1 only)
Upload a .glb (or register a model already on disk), then run the pipeline step by step or to the end; every step shows
its pictures and log so a person can check it before going on, and any step can be redone.
Access: every request needs the secret key (kept in <runs>/.webkey, printed at start); the page keeps it in a cookie.
Runs live in E:\\figures\\runs\\<name>\\ (model.glb + the pipeline's output folders); runs.json also lists runs elsewhere.
One pipeline job at a time; it is the same model_to_vrm/pipeline.py the notebooks call, logs in <run>/logs/<step>.log."""
import os, sys, json, glob, time, secrets, hmac, threading, subprocess, urllib.parse, mimetypes, re, html
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.dirname(HERE)
PY = r'E:\splat\.venv\Scripts\python.exe'; PIPE = os.path.join(REPO, 'model_to_vrm', 'pipeline.py')
RUNS = os.environ.get('MTV_RUNS', r'E:\figures\runs'); os.makedirs(RUNS, exist_ok=True)
KEYF = os.path.join(RUNS, '.webkey')
if not os.path.exists(KEYF): open(KEYF, 'w').write(secrets.token_urlsafe(24))
KEY = open(KEYF).read().strip(); REG = os.path.join(RUNS, 'runs.json')
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8765; MAX_UPLOAD = 1 << 30
# pictures per step (glob patterns relative to the run folder), what a person checks before going on
PICS = {'views': ['body/tex_a*.png'], 'vis': ['body/clay_a*.png'], 'joints': ['body/joints_check.jpg'],
        'sticks': ['body/sticks_on_tex_a*.png', 'body/sticks_*.png'], 'seglabel': ['sam_body/seg_cards/seg_*.jpg'],
        'cards': ['contacts/sub_cards/sub_*.jpg'], 'show_contacts': ['contacts/sub_final/all_*.png'],
        'flesh': ['flesh/material_*.png'], 'ct_fit': ['anatomy/slices3d_body_*.png', 'anatomy/*_fit_sheet.png'], 'geo_paint': ['geo/organs_*.png'], 'geo_views': ['geo/face_*.png']}
TEXT = {'materials': 'sam_body/materials.json', 'seglabel': 'sam_body/segment_labels.json', 'sublabel': 'contacts/sub_labels.json', 'matfix': 'contacts/sub_fix.json', 'ct_fit': 'anatomy/ct_fit.json'}
JOB = {'proc': None, 'run': None, 'what': '', 'started': 0, 'rc': None}; LOCK = threading.Lock()
def registry():
    R = json.load(open(REG)) if os.path.exists(REG) else {}
    for d in sorted(os.listdir(RUNS)):
        p = os.path.join(RUNS, d)
        if os.path.isdir(p) and os.path.exists(os.path.join(p, 'model.glb')): R.setdefault(d, {'model': os.path.join(p, 'model.glb'), 'out': p})
    return R
def steps(run):
    r = registry()[run]; o = subprocess.run([PY, PIPE, r['model'], r['out'], '--face', '--list'], capture_output=True, text=True, cwd=REPO).stdout
    out = []
    for line in o.splitlines():
        m = re.match(r'(\S+)\s+->\s+(.*)', line)
        if not m: continue
        name, outs = m.group(1), [x.strip() for x in m.group(2).split(',')]
        done = all(os.path.exists(os.path.join(r['out'], x)) for x in outs); log = os.path.join(r['out'], 'logs', name + '.log')
        tail = ''
        if os.path.exists(log):
            L = [l for l in open(log, encoding='utf-8', errors='replace').read().splitlines() if 'it/s]' not in l and l.strip()]; tail = '\n'.join(L[-12:])
        failed = (not done) and os.path.exists(log) and ('Traceback' in tail or 'Error' in tail)
        pics = sorted({os.path.relpath(f, r['out']).replace('\\', '/') for pat in PICS.get(name, []) for f in glob.glob(os.path.join(r['out'], pat))})[:24]
        txt = open(os.path.join(r['out'], TEXT[name]), encoding='utf-8').read()[:3000] if name in TEXT and os.path.exists(os.path.join(r['out'], TEXT[name])) else ''
        out.append({'name': name, 'done': done, 'failed': failed, 'log': tail, 'pics': pics, 'text': txt})
    return out
def start(run, args, what):
    with LOCK:
        if JOB['proc'] is not None and JOB['proc'].poll() is None: return 'busy: %s on %s' % (JOB['what'], JOB['run'])
        r = registry()[run]; cmd = [PY, '-u', PIPE, r['model'], r['out'], '--face'] + args
        os.makedirs(os.path.join(r['out'], 'logs'), exist_ok=True)
        f = open(os.path.join(r['out'], 'logs', '_web.log'), 'a', encoding='utf-8'); f.write('\n$ %s\n' % ' '.join(args)); f.flush()
        JOB.update(proc=subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=REPO), run=run, what=what, started=time.time(), rc=None)
        return 'started: ' + what
def job():
    p = JOB['proc']; rc = None if p is None else p.poll(); running = p is not None and rc is None; step = ''
    if running:                                                   # the step at work = the step log written last since the job began
        logs = [(os.path.getmtime(f), f) for f in glob.glob(os.path.join(registry()[JOB['run']]['out'], 'logs', '*.log')) if not f.endswith('_web.log')]
        logs = [x for x in logs if x[0] >= JOB['started'] - 1]
        if logs: step = os.path.basename(max(logs)[1])[:-4]
    return {'run': JOB['run'], 'what': JOB['what'], 'running': running, 'step': step, 'rc': rc, 'seconds': int(time.time() - JOB['started']) if p else 0}
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def ok(self, body, ctype='application/json', code=200, extra=None):
        b = body if isinstance(body, bytes) else (json.dumps(body) if ctype == 'application/json' else body).encode('utf-8')
        self.send_response(code); self.send_header('Content-Type', ctype); self.send_header('Content-Length', str(len(b)))
        for k, v in (extra or {}).items(): self.send_header(k, v)
        self.end_headers(); self.wfile.write(b)
    def authed(self, q):
        k = q.get('key', [''])[0] or dict(c.strip().split('=', 1) for c in (self.headers.get('Cookie') or '').split(';') if '=' in c).get('mtvkey', '')
        return hmac.compare_digest(k, KEY), bool(q.get('key'))
    def route(self, method):
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query); good, via_q = self.authed(q)
        if not good: return self.ok('<h3>Needs the access key: open the link with ?key=...</h3>', 'text/html', 403)
        cookie = {'Set-Cookie': 'mtvkey=%s; Path=/; HttpOnly; SameSite=Strict; Max-Age=2592000' % KEY} if via_q else None
        run = q.get('run', [''])[0]; R = registry()
        if run and run not in R: return self.ok({'error': 'no such run'}, code=404)
        if method == 'GET' and u.path == '/': return self.ok(open(os.path.join(HERE, 'index.html'), encoding='utf-8').read(), 'text/html; charset=utf-8', extra=cookie)
        if method == 'GET' and u.path == '/api/runs': return self.ok({'runs': {k: {'out': v['out']} for k, v in R.items()}, 'job': job()})
        if method == 'GET' and u.path == '/api/status': return self.ok({'steps': steps(run), 'job': job()})
        if method == 'GET' and u.path == '/api/graph':                # steps with the files they read and write (node view)
            r = R[run]; g = json.loads(subprocess.run([PY, PIPE, r['model'], r['out'], '--face', '--graph'], capture_output=True, text=True, cwd=REPO).stdout)
            for n in g: n['have'] = {f: os.path.exists(os.path.join(r['out'], f)) or f == 'model.glb' for f in n['reads'] + n['writes']}
            return self.ok({'graph': g, 'job': job()})
        if method == 'GET' and u.path == '/file':
            base = os.path.realpath(R[run]['out']); p = os.path.realpath(os.path.join(base, q.get('path', [''])[0]))
            if not p.startswith(base + os.sep) or not os.path.isfile(p): return self.ok({'error': 'no file'}, code=404)
            return self.ok(open(p, 'rb').read(), mimetypes.guess_type(p)[0] or 'application/octet-stream', extra={'Cache-Control': 'max-age=60'})
        if method == 'GET' and u.path == '/thumb':                    # small JPEG of a picture, cached in <run>/_thumbs
            base = os.path.realpath(R[run]['out']); p = os.path.realpath(os.path.join(base, q.get('path', [''])[0]))
            if not p.startswith(base + os.sep) or not os.path.isfile(p): return self.ok({'error': 'no file'}, code=404)
            t = os.path.join(base, '_thumbs', re.sub(r'[^A-Za-z0-9_.-]', '_', os.path.relpath(p, base)) + '.jpg')
            if not os.path.exists(t) or os.path.getmtime(t) < os.path.getmtime(p):
                import cv2; im = cv2.imread(p); h_, w_ = im.shape[:2]; sc = 360 / max(h_, w_)
                os.makedirs(os.path.dirname(t), exist_ok=True); cv2.imwrite(t, cv2.resize(im, (int(w_ * sc), int(h_ * sc)), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 80])
            return self.ok(open(t, 'rb').read(), 'image/jpeg', extra={'Cache-Control': 'max-age=300'})
        if method == 'POST' and u.path == '/api/upload':
            name = re.sub(r'[^A-Za-z0-9_-]', '_', q.get('name', [''])[0])[:40]; n = int(self.headers.get('Content-Length', '0'))
            if not name or n <= 0 or n > MAX_UPLOAD: return self.ok({'error': 'name and a .glb under 1 GB needed'}, code=400)
            d = os.path.join(RUNS, name)
            if os.path.exists(d): return self.ok({'error': 'that name is taken'}, code=409)
            os.makedirs(d); left = n
            with open(os.path.join(d, 'model.glb'), 'wb') as f:
                while left > 0:
                    b = self.rfile.read(min(left, 1 << 20))
                    if not b: break
                    f.write(b); left -= len(b)
            head = open(os.path.join(d, 'model.glb'), 'rb').read(4)
            if head != b'glTF': os.remove(os.path.join(d, 'model.glb')); os.rmdir(d); return self.ok({'error': 'not a binary glTF (.glb)'}, code=400)
            return self.ok({'ok': name})
        if method == 'POST' and u.path == '/api/run':
            mode, st = q.get('mode', [''])[0], q.get('step', [''])[0]
            if mode == 'all': return self.ok({'msg': start(run, [], 'all remaining steps')})
            if mode == 'step' and st: return self.ok({'msg': start(run, ['--step', st], st)})
            if mode == 'redo' and st: return self.ok({'msg': start(run, ['--only', st], 'redo ' + st)})
            if mode == 'from' and st: return self.ok({'msg': start(run, ['--from', st], 'from ' + st + ' to the end')})
            return self.ok({'error': 'mode all, or step/redo/from with step=NAME'}, code=400)
        if method == 'POST' and u.path == '/api/stop':
            p = JOB['proc']
            if p is not None and p.poll() is None: subprocess.run(['taskkill', '/T', '/F', '/PID', str(p.pid)], capture_output=True); return self.ok({'msg': 'stopped'})
            return self.ok({'msg': 'nothing running'})
        return self.ok({'error': 'not found'}, code=404)
    def do_GET(self): self.route('GET')
    def do_POST(self): self.route('POST')
if __name__ == '__main__':
    print('MTV web: http://127.0.0.1:%d/?key=%s' % (PORT, KEY), flush=True)
    ThreadingHTTPServer(('127.0.0.1', PORT), H).serve_forever()
