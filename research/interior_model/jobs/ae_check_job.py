# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "scikit-image"]
# ///
"""Autoencoder variants side by side, then the stage gate (ae_check.py) on validation heads (user, 2026-11-26: check the
encoding before generator training). Trains VARIANTS in parallel on one card (name:KEY=VAL;... sets ARCH conv|tvae, WARMUP, SEED, KLW, KIND
per variant), also gates REFS (earlier autoencoders in the repo).
Uploads <repo>/field_exp/<TAG>/. env: COMMIT, TAG, FIELDS, AE_STEPS, VARIANTS, REFS (comma list of repo paths). args: <repo>"""
import os, sys, time, json, subprocess, threading, urllib.request
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; C = os.environ['COMMIT']; TAG = os.environ.get('TAG', 'ae_check'); FIELDS = os.environ.get('FIELDS', 'head_fields_v1')
ROOT = '/data/fx'; os.makedirs('code', exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py', 'flow_f1.py', 'field_exp.py', 'ae_check.py', 'tvae.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join('code', f))
snapshot_download(repo, repo_type='dataset', local_dir=ROOT, max_workers=16, allow_patterns=['%s/*' % FIELDS])
fields = os.path.join(ROOT, FIELDS); out = os.path.join(ROOT, 'field_exp', TAG); os.makedirs(out, exist_ok=True)
V = os.environ.get('VARIANTS', 'soft,soft_emb,hard').split(','); procs = []
def run(name, cmd, env):
    p = subprocess.Popen(cmd, cwd='code', env=dict(os.environ, **env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); lines = []
    for l in p.stdout: lines.append(l); print('[%s] %s' % (name, l.rstrip()[:1500]), flush=True)
    p.wait(); log('exit', name, p.returncode); os.makedirs(os.path.join(out, name), exist_ok=True); open(os.path.join(out, name, 'log.txt'), 'w').write(''.join(lines))
th = []
VK = {}
for spec in V:                                                                        # name or name:KEY=VAL;KEY=VAL (e.g. tvae_s1:ARCH=tvae;WARMUP=1;SEED=1)
    v, _, kv = spec.partition(':'); env = dict(x.split('=', 1) for x in kv.split(';') if x)
    if v.endswith('emb'): env.setdefault('SOFT_IN', 'emb')
    kind = env.pop('KIND', 'hard' if v.startswith('hard') else 'soft'); VK[v] = kind
    th.append(threading.Thread(target=run, args=(v, [sys.executable, '-u', 'field_exp.py', fields, os.environ.get('AE_STEPS', '12000'), kind, os.path.join(out, v)], env)))
for t in th: t.start()
for t in th: t.join()
aes = [os.path.join(out, v, 'ae_%s.pt' % k) for v, k in VK.items()]
for r in [x for x in os.environ.get('REFS', '').split(',') if x]:
    d = os.path.join(out, 'ref_' + r.replace('/', '_')); os.makedirs(d, exist_ok=True); p = hf_hub_download(repo, r, repo_type='dataset', local_dir='/data/refs'); dst = os.path.join(d, os.path.basename(p)); os.replace(p, dst); aes.append(dst)
gt = [threading.Thread(target=run, args=('gate_%d' % i, [sys.executable, '-u', 'ae_check.py', fields, os.path.join(out, 'gate_%d.json' % i), a], {})) for i, a in enumerate(aes)]   # one process per autoencoder
for t in gt: t.start()
for t in gt: t.join()
G = {}
for i in range(len(aes)):
    f = os.path.join(out, 'gate_%d.json' % i)
    if os.path.exists(f): G.update(json.load(open(f)))
json.dump(G, open(os.path.join(out, 'gate.json'), 'w'), indent=1); log('SUMMARY gate', json.dumps(G))
for a in range(5):
    try: api.upload_folder(folder_path=out, path_in_repo='field_exp/' + TAG, repo_id=repo, repo_type='dataset', commit_message='autoencoder check ' + TAG); break
    except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
log('SUMMARY done')
