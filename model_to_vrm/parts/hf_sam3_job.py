"""Runs ON a Hugging Face GPU Job: SAM 3 masks (sam3_masks.py) for some views folders kept in a private dataset repo.
  args: <repo id> <name> [name ...]     inputs  views/<name>/{cameras.json, tex_*.png}, code/sam3_masks.py
                                       results views/<name>/masks/<render>.npz (uploaded back)"""
# /// script
# dependencies = ["torch", "torchvision", "transformers>=4.57", "accelerate", "pillow", "numpy", "huggingface_hub"]
# ///
import os, sys, subprocess, glob
from huggingface_hub import snapshot_download, HfApi
repo, names = sys.argv[1], sys.argv[2:]; api = HfApi()
local = snapshot_download(repo, repo_type='dataset', allow_patterns=['code/sam3_masks.py'] + ['views/%s/*' % n for n in names])
for n in names:
    d = os.path.join(local, 'views', n); print('=== %s' % n, flush=True)
    r = subprocess.run([sys.executable, os.path.join(local, 'code', 'sam3_masks.py'), d], capture_output=True, text=True)
    print((r.stdout + r.stderr)[-2500:], flush=True)
    files = sorted(glob.glob(os.path.join(d, 'masks', '*.npz')))
    for f in files: api.upload_file(path_or_fileobj=f, path_in_repo='views/%s/masks/%s' % (n, os.path.basename(f)), repo_id=repo, repo_type='dataset')
    print('uploaded %d mask files for %s' % (len(files), n), flush=True)
    if not files: sys.exit('no masks for %s' % n)
