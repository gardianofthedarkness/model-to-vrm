# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch"]
# ///
"""Retrain the face label autoencoder (face_compress.py) on one card: downloads the head-frame face sets of both sources,
runs face_compress.py with the env given (MARGINAL, OUTDIR, BS, TEST_LIMIT), uploads the output to m16_full/<OUTDIR>.
env: COMMIT, AE_STEPS, VARIANTS (default t8). args: <repo>"""
import os, sys, time, subprocess, urllib.request
from huggingface_hub import HfApi, snapshot_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; ROOT = '/data/interior'; os.makedirs(ROOT, exist_ok=True); C = os.environ['COMMIT']; OUT = os.environ.get('OUTDIR', 'face_compress_v2')
os.makedirs('code', exist_ok=True)
for f in ('face_compress.py',): urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join('code', f))
for r in ('gardianofthedarkness/hanseg-ct-mr', repo): snapshot_download(r, repo_type='dataset', allow_patterns=['face_v1_headframe/*'], local_dir=ROOT, max_workers=16)
log('data ready', len(os.listdir(os.path.join(ROOT, 'face_v1_headframe'))), 'files')
p = subprocess.Popen([sys.executable, '-u', 'face_compress.py', ROOT, os.environ.get('AE_STEPS', '12000'), os.environ.get('VARIANTS', 't8')], cwd='code', stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True); lines = []
for l in p.stdout: lines.append(l); print(l.rstrip()[:1500], flush=True)
p.wait(); log('trainer exit', p.returncode); d = os.path.join(ROOT, OUT); os.makedirs(d, exist_ok=True); open(os.path.join(d, 'log_%d.txt' % int(t0)), 'w').write(''.join(lines))
for a in range(6):
    try: api.upload_folder(folder_path=d, path_in_repo='m16_full/' + OUT, repo_id=repo, repo_type='dataset', commit_message='face autoencoder ' + OUT); break
    except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(120)
log('SUMMARY done')
