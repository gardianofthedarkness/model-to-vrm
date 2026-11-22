"""Runs ON a Hugging Face Job: dense atlas slicing or dense slice fits for some segments, inputs and results in a private
dataset repo (launched by hf_fit_jobs.py):
  slice mode  args: <repo id> slice <atlas set> <segment> [...]   -> za_slices.py --dense on atlas/atlas.npz,
              results to atlas_sets/<atlas set>/<segment>.{npz,json,_sheet.png}
  fit mode    args: <repo id> <model name> <segment> [...]        -> fit_slices.py with atlas/<segment>.npz (or, with
              env ATLAS_SET, atlas_sets/<set>/<segment>.npz), results to results/<model>/<segment>_fit.*
  repo layout  code/ (segments.py, planes.py, fit_slices.py, za_slices.py), atlas/atlas.npz, atlas_sets/..., models/<model>/
               {parts,contacts,body}/..., results/<model>/..."""
# /// script
# dependencies = ["numpy", "scipy", "cvxpy", "clarabel", "opencv-python-headless", "huggingface_hub"]
# ///
import os, sys, subprocess, glob
from huggingface_hub import snapshot_download, HfApi
repo = sys.argv[1]; api = HfApi()
if sys.argv[2] == 'slice':
    aset, segs = sys.argv[3], sys.argv[4:]
    local = snapshot_download(repo, repo_type='dataset', allow_patterns=['code/*', 'atlas/atlas.npz'])
    out = os.path.join(local, 'out'); os.makedirs(out, exist_ok=True); env = dict(os.environ, PYTHONPATH=os.path.join(local, 'code'))
    for s in segs:
        print('=== slice %s (%d CPUs)' % (s, os.cpu_count()), flush=True)
        r = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(local, 'code', 'za_slices.py'), os.path.join(local, 'atlas', 'atlas.npz'), s, out, '--dense'],
                           capture_output=True, text=True, env=env)
        print((r.stdout + r.stderr)[-3000:], flush=True)
        files = [os.path.join(out, s + x) for x in ('.npz', '.json', '_sheet.png') if os.path.exists(os.path.join(out, s + x))]
        for f in files: api.upload_file(path_or_fileobj=f, path_in_repo='atlas_sets/%s/%s' % (aset, os.path.basename(f)), repo_id=repo, repo_type='dataset')
        print('uploaded %d files for %s' % (len(files), s), flush=True)
        if not any(f.endswith('.npz') for f in files): sys.exit('no atlas slices for %s' % s)
    sys.exit(0)
model, segs = sys.argv[2], sys.argv[3:]; aset = os.environ.get('ATLAS_SET')
apat = ['atlas_sets/%s/%s.npz' % (aset, s) for s in segs] if aset else ['atlas/%s.npz' % s for s in segs]
local = snapshot_download(repo, repo_type='dataset', allow_patterns=['code/*'] + apat + ['models/%s/**' % model])
out = os.path.join(local, 'out'); os.makedirs(out, exist_ok=True)
for s in segs:
    print('=== %s (%d CPUs)' % (s, os.cpu_count()), flush=True)
    env = dict(os.environ, PYTHONPATH=os.path.join(local, 'code'))   # the hub cache links files, so the script's own folder is not the code folder
    ap = os.path.join(local, 'atlas_sets', aset, s + '.npz') if aset else os.path.join(local, 'atlas', s + '.npz')
    r = subprocess.run([sys.executable, '-W', 'ignore', os.path.join(local, 'code', 'fit_slices.py'), os.path.join(local, 'models', model), s, ap, out],
                       capture_output=True, text=True, env=env)
    print('\n'.join(l for l in (r.stdout + r.stderr).splitlines() if 'first-surface' not in l)[-3000:], flush=True)
    files = [f for f in glob.glob(os.path.join(out, s + '_fit*'))]
    for f in files:
        api.upload_file(path_or_fileobj=f, path_in_repo='results/%s/%s' % (model, os.path.basename(f)), repo_id=repo, repo_type='dataset')
    print('uploaded %d files for %s' % (len(files), s), flush=True)
    if not files: sys.exit('no results for %s' % s)
