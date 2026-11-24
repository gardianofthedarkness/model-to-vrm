# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch"]
# ///
"""Milestones 15 (Tokens) and 16 (Transformer), full runs on the smallest card (T4-small).
STAGE=compress: face label autoencoders t4, t8, t12 and the pelvis one, AE_STEPS each, batch BS, all test people scored.
STAGE=transformer: transformer over pelvis 12 mm + face TF mm tokens, TF_STEPS, all people, autoencoders from the
compress stage (downloaded from the repo). Output is streamed to the log as it comes and each finished script's
results are uploaded at once (weights, predictions, JSON, log). env: COMMIT, STAGE, AE_STEPS, BS, TF, TF_STEPS. args: <repo>"""
import os, sys, time, subprocess, urllib.request, shutil
from huggingface_hub import HfApi, snapshot_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; HS = 'gardianofthedarkness/hanseg-ct-mr'; ROOT = '/data/interior'; os.makedirs(ROOT, exist_ok=True)
C = os.environ['COMMIT']; STAGE = os.environ.get('STAGE', 'compress'); code = 'code'; os.makedirs(code, exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join(code, f))
for r in (HS, repo): snapshot_download(r, repo_type='dataset', allow_patterns=['face_v1_headframe/*'], local_dir=ROOT, max_workers=16)
snapshot_download(repo, repo_type='dataset', allow_patterns=['pelvis_v1/*'] + (['m16_full/face_compress/autoencoder_*.pt', 'm16_full/pelvis_compress/autoencoder_*.pt'] if STAGE == 'transformer' else []), local_dir=ROOT, max_workers=16)
if STAGE == 'transformer':
    for d in ('face_compress', 'pelvis_compress'): shutil.copytree(os.path.join(ROOT, 'm16_full', d), os.path.join(ROOT, d), dirs_exist_ok=True)
log('data ready')
def run(args, env, outdir, tag):
    e = dict(os.environ, **env); t = time.time(); lines = []
    p = subprocess.Popen([sys.executable, '-u'] + args, cwd=code, env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for l in p.stdout: lines.append(l); print(l.rstrip()[:1500], flush=True)                         # live
    p.wait(); log(tag, 'exit', p.returncode, '%.1f min' % ((time.time() - t) / 60))
    src = os.path.join(ROOT, outdir); os.makedirs(src, exist_ok=True); open(os.path.join(src, 'log_%s.txt' % tag), 'w').write(''.join(lines))
    for attempt in range(6):
        try: api.upload_folder(folder_path=src, path_in_repo='m16_full/%s' % outdir, repo_id=repo, repo_type='dataset', commit_message='m16 full %s' % tag); break
        except Exception as ex: log('upload retry', attempt, type(ex).__name__); time.sleep(300)
    return p.returncode
if STAGE == 'compress':
    AE, BS = os.environ.get('AE_STEPS', '12000'), os.environ.get('BS', '4')
    run(['face_compress.py', ROOT, AE, 't4,t8,t12'], dict(PART='face', BS=BS), 'face_compress', 'face')
    run(['face_compress.py', ROOT, AE], dict(PART='pelvis', BS=BS), 'pelvis_compress', 'pelvis')
else:
    TF = os.environ.get('TF', '8')
    run(['transformer_smoke.py', ROOT, os.environ.get('TF_STEPS', '20000'), TF], {}, 'transformer_smoke', 'transformer_TF%s' % TF)
log('SUMMARY done')
