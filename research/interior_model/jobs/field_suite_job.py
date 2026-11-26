# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "scikit-image"]
# ///
"""Field experiment, all arms packed on one card (user, 2026-11-26: same conditions, compare clean input, tissue + eye
tagging, shell loss, masked partial faces). Stages, each as parallel processes on the same GPU:
  1. autoencoders: hard (old labels) and soft (continuous field), trained on full-face heads (HaN-Seg + PDDCA train)
  2. preparation: codes and shell tokens for every head with each autoencoder (SHARDS processes per kind)
  3. flows (ARMS): A0 hard; N soft; NS soft + shell loss; NSP soft + shell loss + SynthRAD's masked-face heads
Test heads are full faces only (HaN-Seg and PDDCA test splits). nvidia-smi and free memory are logged every GPU_LOG s.
Uploads <repo>/field_exp/<SUITE>/... after each stage.
env: COMMIT, SUITE, FIELDS, AE_STEPS, STEPS, SHAPE, SEEDS (e.g. 0,1), ARMS (comma: A0,N,NS,NSP), SHARDS, TEST_LIMIT, LIMIT_HEADS, REUSE (earlier suite: flows only), MAXPAR (processes at once).
args: <repo>"""
import os, sys, time, json, subprocess, threading, urllib.request
from huggingface_hub import HfApi, snapshot_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; C = os.environ['COMMIT']; SUITE = os.environ.get('SUITE', 'suite1'); FIELDS = os.environ.get('FIELDS', 'head_fields_v1')
AE_STEPS = os.environ.get('AE_STEPS', '12000'); STEPS = os.environ.get('STEPS', '10000'); SHAPE = os.environ.get('SHAPE', '0.2')
SEEDS = os.environ.get('SEEDS', '0,1').split(','); ARMS = os.environ.get('ARMS', 'A0,N,NS,NSP').split(','); SHARDS = int(os.environ.get('SHARDS', '2'))
ROOT = '/data/fx'; os.makedirs('code', exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py', 'flow_f1.py', 'field_exp.py', 'field_prep.py', 'field_flow.py', 'tvae.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join('code', f))
snapshot_download(repo, repo_type='dataset', local_dir=ROOT, max_workers=16, allow_patterns=['%s/*' % FIELDS])
fields = os.path.join(ROOT, FIELDS); out = os.path.join(ROOT, 'field_exp', SUITE); os.makedirs(out, exist_ok=True)
if os.environ.get('LIMIT_HEADS'):                                                       # smoke: a few heads per source and split
    idx = json.load(open(os.path.join(fields, 'index.json'))); n = int(os.environ['LIMIT_HEADS']); keep = {}
    for c, v in sorted(idx['heads'].items()):
        k = (v['source'], v['split'])
        if sum(1 for x in keep.values() if (x['source'], x['split']) == k) < n: keep[c] = v
    idx['heads'] = keep; json.dump(idx, open(os.path.join(fields, 'index.json'), 'w'))
log('data ready', len(json.load(open(os.path.join(fields, 'index.json')))['heads']), 'heads')
stop = False
def gpu_logger():
    while not stop:
        try:
            q = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout.strip()
            mem = [l for l in open('/proc/meminfo') if l.startswith(('MemTotal', 'MemAvailable'))]
            log('GPU', q, '| RAM', ' '.join(l.split()[1] for l in mem), 'kB total/available')
        except Exception as e: log('gpu log', type(e).__name__)
        time.sleep(int(os.environ.get('GPU_LOG', '300')))
threading.Thread(target=gpu_logger, daemon=True).start()
def parallel(jobs):
    """jobs: list of (name, cmd, env); run all at once, stream lines prefixed by name, return {name: (rc, text)}"""
    res = {}
    def one(name, cmd, env):
        p = subprocess.Popen(cmd, cwd='code', env=dict(os.environ, **env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); lines = []
        for l in p.stdout: lines.append(l); print('[%s] %s' % (name, l.rstrip()[:1500]), flush=True)
        p.wait(); res[name] = (p.returncode, ''.join(lines)); log('exit', name, p.returncode)
    sem = threading.Semaphore(int(os.environ.get('MAXPAR', '99')))                     # at most MAXPAR processes on the card at once
    def gated(*j):
        with sem: one(*j)
    th = [threading.Thread(target=gated, args=j) for j in jobs]
    for t in th: t.start()
    for t in th: t.join()
    return res
def up(folder, path):
    for a in range(5):
        try: api.upload_folder(folder_path=folder, path_in_repo=path, repo_id=repo, repo_type='dataset', commit_message='field suite ' + path); return
        except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
kinds = sorted({'hard' if a == 'A0' else 'soft' for a in ARMS})
py = [sys.executable, '-u']
REUSE = os.environ.get('REUSE')                                                        # an earlier suite: take its autoencoders and preparation, run flows only
if REUSE:
    snapshot_download(repo, repo_type='dataset', local_dir=ROOT, max_workers=16, allow_patterns=['field_exp/%s/ae_*/*' % REUSE, 'field_exp/%s/prep_*/*' % REUSE])
    src = os.path.join(ROOT, 'field_exp', REUSE)
    for d in os.listdir(src):
        if d.startswith(('ae_', 'prep_')): os.symlink(os.path.join(src, d), os.path.join(out, d))
    log('reusing autoencoders and preparation of', REUSE, sorted(os.listdir(out)))
# 1. autoencoders
if not REUSE: r = parallel([('ae_' + k, py + ['field_exp.py', fields, AE_STEPS, k, os.path.join(out, 'ae_' + k)], {}) for k in kinds])
if not REUSE:
    for k in kinds: open(os.path.join(out, 'ae_' + k, 'log.txt'), 'w').write(r['ae_' + k][1])
    up(out, 'field_exp/' + SUITE); log('stage autoencoders done')
# 2. preparation (all heads, both kinds, SHARDS processes each)
if not REUSE: r = parallel([('prep_%s_%d' % (k, i), py + ['field_prep.py', fields, os.path.join(out, 'ae_' + k, 'ae_%s.pt' % k), os.path.join(out, 'prep_' + k)], {'SHARD': '%d/%d' % (i, SHARDS)}) for k in kinds for i in range(SHARDS)])
if not REUSE: log('stage preparation done', {k: v[0] for k, v in r.items()})
# 3. flows
arm_env = dict(A0=('hard', '0', 'hanseg,pddca'), N=('soft', '0', 'hanseg,pddca'), NS=('soft', SHAPE, 'hanseg,pddca'), NSP=('soft', SHAPE, 'hanseg,pddca,synthrad'))
jobs = []
for a in ARMS:
    k, shp, ts = arm_env[a]
    for s in (SEEDS if a != 'A0' else SEEDS[:1]):
        o = os.path.join(out, '%s_s%s' % (a, s)); jobs.append(('%s_s%s' % (a, s), py + ['field_flow.py', fields, os.path.join(out, 'prep_' + k), os.path.join(out, 'ae_' + k, 'ae_%s.pt' % k)],
                                                         dict(OUT=o, SHAPE=shp, SEED=s, STEPS=STEPS, TRAIN_SOURCES=ts, HEADCACHE='/data/hc')))
r = parallel(jobs)
for n, (rc, txt) in r.items(): os.makedirs(os.path.join(out, n), exist_ok=True); open(os.path.join(out, n, 'log.txt'), 'w').write(txt)
summ = {}
for n in r:
    f = os.path.join(out, n, 'field_flow.json')
    if os.path.exists(f): summ[n] = json.load(open(f))['summary']
json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=1); log('SUMMARY', json.dumps(summ))
stop = True
if REUSE:
    for d in os.listdir(out):
        if os.path.islink(os.path.join(out, d)): os.unlink(os.path.join(out, d))   # reused parts stay where they are
up(out, 'field_exp/' + SUITE); log('SUMMARY done')
