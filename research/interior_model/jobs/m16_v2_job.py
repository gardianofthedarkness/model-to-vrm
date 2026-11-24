# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch"]
# ///
"""Milestone 16 transformer version 2 (transformer_v2.py) on one small card. Downloads the label sets, the frozen
autoencoders, the cached preparation and any earlier checkpoint of the same run (so a restarted job continues), runs
the trainer with live log, and uploads the output folder (weights, predictions, JSON, log) at the end; the trainer
itself uploads checkpoints at every validation. env: COMMIT, STAGE (run name, default v2), TF, TF_STEPS, LIMIT,
TEST_LIMIT, PART=prep (fill and upload the preparation cache only, in parallel). args: <repo>"""
import os, sys, time, subprocess, urllib.request, shutil
from huggingface_hub import HfApi, snapshot_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; HS = 'gardianofthedarkness/hanseg-ct-mr'; ROOT = '/data/interior'; os.makedirs(ROOT, exist_ok=True)
C = os.environ['COMMIT']; TF = os.environ.get('TF', '8'); OUT = 'transformer_' + os.environ.get('STAGE', 'v2'); LIMIT = os.environ.get('LIMIT', '')
code = 'code'; os.makedirs(code, exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py', 'transformer_v2.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join(code, f))
for r in (HS, repo): snapshot_download(r, repo_type='dataset', allow_patterns=['face_v1_headframe/*'], local_dir=ROOT, max_workers=16)
snapshot_download(repo, repo_type='dataset', local_dir=ROOT, max_workers=16, allow_patterns=['pelvis_v1/*', 'm16_full/face_compress/autoencoder_*.pt', 'm16_full/pelvis_compress/autoencoder_*.pt',
                  'm16_full/prep_v2_TF%s/*' % TF, 'm16_full/%s/last.pt' % OUT, 'm16_full/%s/net_v2_*.pt' % OUT])
for d in ('face_compress', 'pelvis_compress', 'prep_v2_TF%s' % TF, OUT):
    if os.path.isdir(os.path.join(ROOT, 'm16_full', d)): shutil.copytree(os.path.join(ROOT, 'm16_full', d), os.path.join(ROOT, d), dirs_exist_ok=True)
log('data ready; cached preparations', len(os.listdir(os.path.join(ROOT, 'prep_v2_TF%s' % TF))) if os.path.isdir(os.path.join(ROOT, 'prep_v2_TF%s' % TF)) else 0)
if os.environ.get('PART') == 'prep':                                                                     # preparation only: one process per vCPU, each takes every n-th person; then upload the cache
    n = os.cpu_count() or 4; cache = os.path.join(ROOT, 'prep_v2_TF%s' % TF)
    ps = [subprocess.Popen([sys.executable, '-u', 'transformer_v2.py', ROOT, '1', TF], cwd=code, env=dict(os.environ, PREP_ONLY='1', SHARD='%d/%d' % (i, n))) for i in range(n)]
    rc = [q.wait() for q in ps]; log('prep exit codes', rc, 'files', len(os.listdir(cache)))
    for attempt in range(6):
        try: api.upload_folder(folder_path=cache, path_in_repo='m16_full/prep_v2_TF%s' % TF, repo_id=repo, repo_type='dataset', commit_message='m16 v2 preparation cache'); break
        except Exception as ex: log('upload retry', attempt, type(ex).__name__); time.sleep(120)
    log('SUMMARY prep done'); sys.exit(max(rc))
e = dict(os.environ, OUT=OUT, REPO=repo); lines = []
p = subprocess.Popen([sys.executable, '-u', 'transformer_v2.py', ROOT, os.environ.get('TF_STEPS', '40000'), TF] + ([LIMIT] if LIMIT else []), cwd=code, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for l in p.stdout: lines.append(l); print(l.rstrip()[:1500], flush=True)
p.wait(); log('trainer exit', p.returncode)
src = os.path.join(ROOT, OUT); os.makedirs(src, exist_ok=True); open(os.path.join(src, 'log_%s_%d.txt' % (OUT, int(t0))), 'w').write(''.join(lines))
for attempt in range(6):
    try: api.upload_folder(folder_path=src, path_in_repo='m16_full/%s' % OUT, repo_id=repo, repo_type='dataset', commit_message='m16 %s' % OUT); break
    except Exception as ex: log('upload retry', attempt, type(ex).__name__); time.sleep(300)
log('SUMMARY done')
