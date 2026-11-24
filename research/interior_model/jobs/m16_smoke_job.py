# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch"]
# ///
"""Milestones 15 (Tokens) and 16 (Transformer), smoke and profile on a small card.
Fetches the part data (face heads in head frame, pelvises in pelvis frame) and the code at a fixed commit, then:
  1. face label autoencoders t4 / t8 / t12 and the pelvis one (t12), AE_STEPS each (research/interior_model/face_compress.py)
  2. transformer smoke over pelvis + face tokens, TF = 8 for TF_STEPS, then TF = 4 for a short memory and speed probe
     (research/interior_model/transformer_smoke.py)
Everything (weights, predictions, JSON, log) is uploaded. env: COMMIT, AE_STEPS, TF_STEPS, LIMIT, TEST_LIMIT. args: <repo>"""
import os, sys, json, time, subprocess, urllib.request
from huggingface_hub import HfApi, snapshot_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; HS = 'gardianofthedarkness/hanseg-ct-mr'; ROOT = '/data/interior'; os.makedirs(ROOT, exist_ok=True)
C = os.environ['COMMIT']; code = 'code'; os.makedirs(code, exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join(code, f))
for r in (HS, repo): snapshot_download(r, repo_type='dataset', allow_patterns=['face_v1_headframe/*'], local_dir=ROOT, max_workers=16)
snapshot_download(repo, repo_type='dataset', allow_patterns=['pelvis_v1/*'], local_dir=ROOT, max_workers=16)
log('data in', sum(len(fs) for _, _, fs in os.walk(ROOT)), 'files')
def run(args, env):
    e = dict(os.environ, **env); t = time.time(); p = subprocess.run([sys.executable, '-u'] + args, cwd=code, env=e, capture_output=True, text=True)
    out = p.stdout + p.stderr; open(os.path.join(ROOT, 'log_%s.txt' % '_'.join(a for a in args[1:] if '/' not in a)[:60]), 'w').write(out)
    for l in out.splitlines():
        if 'SUMMARY' in l or 'Error' in l or 'Traceback' in l or ' t4 {' in l or ' t8 {' in l or ' t12 {' in l or 'prepared' in l or 'step 0 ' in l: log(l[:1500])
    log(args[0], 'exit', p.returncode, '%.1f min' % ((time.time() - t) / 60)); return p.returncode
AE, TFS, TL = os.environ.get('AE_STEPS', '1500'), os.environ.get('TF_STEPS', '300'), os.environ.get('TEST_LIMIT', '5')
run(['face_compress.py', ROOT, AE, 't4,t8,t12'], dict(PART='face', TEST_LIMIT=TL))
run(['face_compress.py', ROOT, AE], dict(PART='pelvis', TEST_LIMIT=TL))
run(['transformer_smoke.py', ROOT, TFS, '8', os.environ.get('LIMIT', '40')], dict(TEST_LIMIT=TL))
run(['transformer_smoke.py', ROOT, '30', '4', '12'], dict(TEST_LIMIT='2'))
for d in ('face_compress', 'pelvis_compress', 'transformer_smoke'):
    if os.path.isdir(os.path.join(ROOT, d)): pass
up = os.path.join(ROOT, 'upload'); os.makedirs(up, exist_ok=True)
for d in ('face_compress', 'pelvis_compress', 'transformer_smoke'):
    if os.path.isdir(os.path.join(ROOT, d)): os.rename(os.path.join(ROOT, d), os.path.join(up, d))
for f in os.listdir(ROOT):
    if f.startswith('log_'): os.rename(os.path.join(ROOT, f), os.path.join(up, f))
for attempt in range(6):
    try: api.upload_folder(folder_path=up, path_in_repo='m16_smoke', repo_id=repo, repo_type='dataset', commit_message='milestones 15/16 smoke'); break
    except Exception as e: log('upload retry', attempt, type(e).__name__); time.sleep(300)
log('SUMMARY done')
