# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "scikit-image"]
# ///
"""Truth, autoencoder rebuild and generated interior of one real test head as surfaces (viewer/case_gen.py), on one card.
PRIVATE output: <repo>/private_views/<TAG>/<case>/ in the private repo, for the private gallery only.
env: COMMIT, TAG, CASE, FLOW (repo path), AE (repo path), PREP (repo folder of the preparation), FIELDS. args: <repo>"""
import os, sys, time, subprocess, urllib.request
from huggingface_hub import HfApi, hf_hub_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; C = os.environ['COMMIT']; TAG = os.environ.get('TAG', 'case_gen'); case = os.environ['CASE']; FIELDS = os.environ.get('FIELDS', 'head_fields_v1')
os.makedirs('code/viewer', exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py', 'flow_f1.py', 'field_exp.py', 'tvae.py', 'viewer/case_gen.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join('code', f))
fld = os.path.dirname(hf_hub_download(repo, '%s/%s.npz' % (FIELDS, case), repo_type='dataset', local_dir='/data/in'))
prp = os.path.dirname(hf_hub_download(repo, '%s/%s.npz' % (os.environ['PREP'], case), repo_type='dataset', local_dir='/data/in'))
flow = hf_hub_download(repo, os.environ['FLOW'], repo_type='dataset', local_dir='/data/in'); ae = hf_hub_download(repo, os.environ['AE'], repo_type='dataset', local_dir='/data/in')
out = '/data/out/' + case
p = subprocess.run([sys.executable, '-u', 'viewer/case_gen.py', fld, prp, flow, ae, case, out], cwd='code', env=dict(os.environ, PYTHONWARNINGS='ignore'), capture_output=True, text=True)
for l in (p.stdout + p.stderr).splitlines()[-40:]: print('TS sub', l, flush=True)
log('exit', p.returncode)
for a in range(5):
    try: api.upload_folder(folder_path='/data/out', path_in_repo='private_views/' + TAG, repo_id=repo, repo_type='dataset', commit_message='private view ' + TAG); break
    except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
log('SUMMARY done')
