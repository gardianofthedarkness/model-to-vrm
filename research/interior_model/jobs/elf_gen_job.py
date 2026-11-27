# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "scikit-image", "matplotlib"]
# ///
"""The elf's head interior from several generators on one card (viewer/elf_field_gen.py per generator), for a human look
(user, 2026-11-27: scores do not show whether it looks right). Downloads the elf shell (<repo>/elf/elf_face_sym.npz), the
code at COMMIT, each generator and the autoencoder; uploads <repo>/elf/<TAG>/<name>/ (probabilities, picture, web meshes).
env: COMMIT, TAG, GENS (comma list of name=repo path of flow.pt), AE (repo path). args: <repo>"""
import os, sys, time, subprocess, urllib.request
from huggingface_hub import HfApi, hf_hub_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; C = os.environ['COMMIT']; TAG = os.environ.get('TAG', 'elf_gen'); os.makedirs('code/viewer', exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py', 'flow_f1.py', 'field_exp.py', 'tvae.py', 'viewer/elf_field_gen.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join('code', f))
shell = hf_hub_download(repo, 'elf/elf_face_sym.npz', repo_type='dataset', local_dir='/data/in'); ae = hf_hub_download(repo, os.environ['AE'], repo_type='dataset', local_dir='/data/in')
for spec in os.environ['GENS'].split(','):
    name, path = spec.split('='); fl = hf_hub_download(repo, path, repo_type='dataset', local_dir='/data/in/' + name)
    out = '/data/out/' + name; log('generator', name, path)
    p = subprocess.run([sys.executable, '-u', 'viewer/elf_field_gen.py', shell, fl, ae, out], cwd='code', env=dict(os.environ, WEB=out + '/web', PYTHONWARNINGS='ignore'), capture_output=True, text=True)
    print(p.stdout[-4000:], p.stderr[-2000:], flush=True); log('exit', name, p.returncode)
for a in range(5):
    try: api.upload_folder(folder_path='/data/out', path_in_repo='elf/' + TAG, repo_id=repo, repo_type='dataset', commit_message='elf interiors ' + TAG); break
    except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
log('SUMMARY done')
