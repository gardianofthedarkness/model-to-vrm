# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch"]
# ///
"""Plan v8 flow-matching tests (flow_f1.py) on one small card. Downloads the label sets of the parts asked for, the
frozen autoencoders, the cached preparation and any earlier checkpoint of the same run (a restarted job continues),
runs the trainer with live log, and uploads the output folder (weights, predictions, JSON, log) at the end.
env: COMMIT, STAGE (run name), TF, TF_STEPS, LIMIT, TEST_LIMIT, CFG (passed to the trainer). args: <repo>"""
import os, sys, time, subprocess, urllib.request, shutil
from huggingface_hub import HfApi, snapshot_download, hf_hub_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; HS = 'gardianofthedarkness/hanseg-ct-mr'; ROOT = '/data/interior'; os.makedirs(ROOT, exist_ok=True)
C = os.environ['COMMIT']; TF = os.environ.get('TF', '8'); OUT = 'flow_' + os.environ.get('STAGE', 'f1'); LIMIT = os.environ.get('LIMIT', '')
parts = dict(kv.split('=') for kv in os.environ.get('CFG', '').split(',') if kv).get('parts', 'pelvis').split('+')
code = 'code'; os.makedirs(code, exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py', 'flow_f1.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join(code, f))
for r in (HS, repo): snapshot_download(r, repo_type='dataset', allow_patterns=['face_v1_headframe/*' if 'face' in parts and os.environ.get('PREP', 'prep_v2').startswith('prep_v2') else 'face_v1_headframe/index.json'], local_dir=ROOT, max_workers=16)
PREP = os.environ.get('PREP', 'prep_v2_TF%s' % TF); FAE = os.environ.get('FACE_AE_DIR', 'face_compress')                    # PREP=prep_v3_TF8 FACE_AE_DIR=face_compress_v3 for the v3 data
snapshot_download(repo, repo_type='dataset', local_dir=ROOT, max_workers=16, allow_patterns=['pelvis_v1/*' if 'pelvis' in parts and PREP.startswith('prep_v2') else 'pelvis_v1/index.json', 'm16_full/%s/autoencoder_*.pt' % FAE,
                  'm16_full/pelvis_compress/autoencoder_*.pt', 'm16_full/%s/last.pt' % OUT, 'm16_full/%s/flow_f1.pt' % OUT, 'm16_full/align/prep_v3_exclude.json'] + ['m16_full/%s/*_%s.npz' % (PREP, p) for p in parts])
for d in (FAE, 'pelvis_compress', PREP, OUT):
    if os.path.isdir(os.path.join(ROOT, 'm16_full', d)): shutil.copytree(os.path.join(ROOT, 'm16_full', d), os.path.join(ROOT, d), dirs_exist_ok=True)
if os.environ.get('PREP_PELVIS'):                                                               # Test 2.5: pelvis people from another preparation
    PP = os.environ['PREP_PELVIS']; snapshot_download(repo, repo_type='dataset', local_dir=ROOT, max_workers=16, allow_patterns=['m16_full/%s/*_pelvis.npz' % PP])
    for f in os.listdir(os.path.join(ROOT, 'm16_full', PP)): shutil.copy(os.path.join(ROOT, 'm16_full', PP, f), os.path.join(ROOT, PREP, f))
if os.environ.get('INIT'):                                                                      # Test 2.5: start from another run's weights (path in the repo)
    os.environ['INIT'] = hf_hub_download(repo, os.environ['INIT'], repo_type='dataset', local_dir=ROOT)
log('data ready; cached preparations', len(os.listdir(os.path.join(ROOT, PREP))))
e = dict(os.environ, OUT=OUT, REPO=repo, EXCLUDE=os.path.join(ROOT, 'm16_full', 'align', 'prep_v3_exclude.json')); lines = []
p = subprocess.Popen([sys.executable, '-u', 'flow_f1.py', ROOT, os.environ.get('TF_STEPS', '4000'), TF] + ([LIMIT] if LIMIT else []), cwd=code, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for l in p.stdout: lines.append(l); print(l.rstrip()[:2500], flush=True)
p.wait(); log('trainer exit', p.returncode)
src = os.path.join(ROOT, OUT); os.makedirs(src, exist_ok=True); open(os.path.join(src, 'log_%s_%d.txt' % (OUT, int(t0))), 'w').write(''.join(lines))
for attempt in range(6):
    try: api.upload_folder(folder_path=src, path_in_repo='m16_full/%s' % OUT, repo_id=repo, repo_type='dataset', commit_message='flow %s' % OUT); break
    except Exception as ex: log('upload retry', attempt, type(ex).__name__); time.sleep(300)
log('SUMMARY done')
