# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch"]
# ///
"""Preparation v3 (prep_v3.py) for all people on one card: downloads the label sets, the autoencoders, the mean shells and
fit-recovery results (m16_full/align), the cover templates (m16_full/covers); runs prep_v3.py in parallel shards (one per
core the memory allows); uploads the result to m16_full/<OUT>. env: COMMIT, OUT (default prep_v3_TF8), FACE_AE, PARTS. args: <repo>"""
import os, sys, time, subprocess, urllib.request, shutil
from huggingface_hub import HfApi, snapshot_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; ROOT = '/data/interior'; os.makedirs(ROOT, exist_ok=True); C = os.environ['COMMIT']; OUT = os.environ.get('OUT', 'prep_v3_TF8')
os.makedirs('code', exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py', 'transformer_v2.py', 'shell_align.py', 'prep_v3.py', 'tissue_map.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join('code', f))
for r in ('gardianofthedarkness/hanseg-ct-mr', repo): snapshot_download(r, repo_type='dataset', allow_patterns=['face_v1_headframe/*'], local_dir=ROOT, max_workers=16)
snapshot_download(repo, repo_type='dataset', local_dir=ROOT, max_workers=16, allow_patterns=['pelvis_v1/*', 'm16_full/face_compress_v3/autoencoder_*.pt', 'm16_full/%s/autoencoder_*.pt' % os.environ.get('FACE_AE', 'face_compress_v3/x').split('/')[0], 'm16_full/pelvis_compress/autoencoder_*.pt', 'm16_full/align/*', 'm16_full/covers/*'])
for d in ('face_compress_v3', os.environ.get('FACE_AE', 'face_compress_v3/x').split('/')[0], 'pelvis_compress', 'align', 'covers'): shutil.copytree(os.path.join(ROOT, 'm16_full', d), os.path.join(ROOT, d), dirs_exist_ok=True)
try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
except Exception: ram = 30
try: q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) // int(per)
except Exception: cpus = len(os.sched_getaffinity(0))
n = max(1, min(cpus, int(ram // 3.5))); log('data ready; processes', n, '(cores %s, RAM %.0f GB)' % (cpus, ram))
env = dict(os.environ, OUT=OUT, FACE_AE=os.environ.get('FACE_AE', 'face_compress_v3/autoencoder_t8.pt'))
ps = [subprocess.Popen([sys.executable, '-u', 'prep_v3.py', ROOT, os.path.join(ROOT, 'align'), os.path.join(ROOT, 'covers')], cwd='code', env=dict(env, SHARD='%d/%d' % (i, n))) for i in range(n)]
rc = [p.wait() for p in ps]; log('exit codes', rc, 'files', len(os.listdir(os.path.join(ROOT, OUT))))
for a in range(6):
    try: api.upload_folder(folder_path=os.path.join(ROOT, OUT), path_in_repo='m16_full/' + OUT, repo_id=repo, repo_type='dataset', commit_message='preparation ' + OUT); break
    except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(120)
log('SUMMARY done')
