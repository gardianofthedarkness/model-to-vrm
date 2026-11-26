# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "scikit-image"]
# ///
"""Field experiment on one card (field_exp.py, field_prep.py, field_flow.py). Downloads the new head labels
(<repo>/head_fields_v1), the code at COMMIT, and earlier outputs of the same experiment; runs the stages asked for and
uploads <repo>/field_exp/<TAG>/.
env: COMMIT, TAG, STAGES (comma list of ae, prep, flow), KIND (hard|soft), AE_STEPS, STEPS, SHAPE, SEED, FIELDS (folder name).
args: <repo>"""
import os, sys, time, subprocess, urllib.request, shutil
from huggingface_hub import HfApi, snapshot_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; C = os.environ['COMMIT']; TAG = os.environ['TAG']; KIND = os.environ.get('KIND', 'soft'); STAGES = os.environ.get('STAGES', 'ae,prep,flow').split(',')
FIELDS = os.environ.get('FIELDS', 'head_fields_v1'); ROOT = '/data/fx'; os.makedirs('code', exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py', 'flow_f1.py', 'field_exp.py', 'field_prep.py', 'field_flow.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join('code', f))
snapshot_download(repo, repo_type='dataset', local_dir=ROOT, max_workers=16, allow_patterns=['%s/*' % FIELDS, 'field_exp/ae_%s/*' % KIND, 'field_exp/prep_%s/*' % KIND])
fields = os.path.join(ROOT, FIELDS); aedir = os.path.join(ROOT, 'field_exp', 'ae_' + KIND); prepdir = os.path.join(ROOT, 'field_exp', 'prep_' + KIND); out = os.path.join(ROOT, 'field_exp', TAG)
for d in (aedir, prepdir, out): os.makedirs(d, exist_ok=True)
log('data ready', len(os.listdir(fields)), 'files')
def run(cmd, env=None):
    p = subprocess.Popen(cmd, cwd='code', env=dict(os.environ, **(env or {})), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); lines = []
    for l in p.stdout: lines.append(l); print(l.rstrip()[:2000], flush=True)
    p.wait(); log('exit', p.returncode, cmd[2]); return p.returncode, ''.join(lines)
def up(folder, path):
    for a in range(5):
        try: api.upload_folder(folder_path=folder, path_in_repo=path, repo_id=repo, repo_type='dataset', commit_message='field experiment ' + path); return
        except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
if 'ae' in STAGES:
    rc, txt = run([sys.executable, '-u', 'field_exp.py', fields, os.environ.get('AE_STEPS', '12000'), KIND, aedir]); open(os.path.join(aedir, 'log_ae.txt'), 'w').write(txt); up(aedir, 'field_exp/ae_' + KIND)
if 'prep' in STAGES:
    rc, txt = run([sys.executable, '-u', 'field_prep.py', fields, os.path.join(aedir, 'ae_%s.pt' % KIND), prepdir]); up(prepdir, 'field_exp/prep_' + KIND)
if 'flow' in STAGES:
    rc, txt = run([sys.executable, '-u', 'field_flow.py', fields, prepdir, os.path.join(aedir, 'ae_%s.pt' % KIND)], dict(OUT=out)); open(os.path.join(out, 'log_flow.txt'), 'w').write(txt); up(out, 'field_exp/' + TAG)
log('SUMMARY done')
