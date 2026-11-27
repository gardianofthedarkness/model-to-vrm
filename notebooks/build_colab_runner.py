"""Writes colab_runner.ipynb: runs one of the interior-model job scripts (research/interior_model/jobs/*_job.py) on a Colab
GPU paid from the Google AI plan's compute units, with the same settings as on Hugging Face Jobs. Results go to the private
Hugging Face repos exactly as there. The Hugging Face token is read only from Colab's Secrets (key icon, name HF_TOKEN,
notebook access on); it is never typed into the notebook.  Run: python build_colab_runner.py"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
cells = [
('markdown', '''
# Interior model: run a job on a Colab GPU
1. Runtime -> Change runtime type -> pick a GPU (A100 or L4 for training, T4 for small checks).
2. Secrets (key icon, left): `HF_TOKEN` with write access to your private dataset repos, notebook access on.
3. Set the job in the next cell, then Runtime -> Run all. Keep this tab open (on the Pro plan a closed tab stops the run).
Results are uploaded to the private Hugging Face repo by the job itself, as on Hugging Face Jobs.
'''),
('code', '''
# ---- the job. PRESET reads it from notebooks/colab_jobs/<PRESET>.json on GitHub main (Claude sets it there);
# or set PRESET = None and fill JOB / ENV here by hand (same names and settings as research/interior_model/jobs/launch_script.py)
PRESET = 'next'
JOB, REPO_SUFFIX, COMMIT, ENV = 'elf_gen_job.py', 'hanseg-ct-mr', 'main', {}
if PRESET:
    import json, urllib.request
    _p = json.load(urllib.request.urlopen('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/main/notebooks/colab_jobs/%s.json' % PRESET))
    JOB, REPO_SUFFIX, COMMIT, ENV = _p['job'], _p.get('repo_suffix', REPO_SUFFIX), _p.get('commit', 'main'), _p.get('env', {})
    print('job', JOB, '| note:', _p.get('note', '')); print('settings', ENV)
'''),
('code', '''
# ---- setup: GPU, packages, token (from Colab Secrets only), code version
import os, subprocess, sys, json, time, urllib.request
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], capture_output=True, text=True).stdout or 'NO GPU: set Runtime -> Change runtime type')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'huggingface_hub>=1.0', 'hf_xet', 'scikit-image', 'scipy', 'SimpleITK', 'nibabel'], check=True)
from google.colab import userdata
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
from huggingface_hub import HfApi
user = HfApi().whoami()['name']; repo = '%s/%s' % (user, REPO_SUFFIX)
print('private repo', repo, 'private:', HfApi().repo_info(repo, repo_type='dataset').private)
if COMMIT == 'main':
    COMMIT = json.load(urllib.request.urlopen('https://api.github.com/repos/gardianofthedarkness/model-to-vrm/commits/main'))['sha']
print('code version', COMMIT)
'''),
('code', '''
# ---- run the job (output streams below; the job uploads its results itself)
work = '/content/run'; os.makedirs(work, exist_ok=True)
src = 'https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/jobs/%s' % (COMMIT, JOB)
urllib.request.urlretrieve(src, os.path.join(work, JOB))
env = dict(os.environ, COMMIT=COMMIT, PYTHONUNBUFFERED='1', **ENV)
t0 = time.time()
p = subprocess.Popen([sys.executable, '-u', JOB, repo], cwd=work, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout:
    if 'it/s]' not in line and 'B/s]' not in line: print(line.rstrip()[:1500], flush=True)
p.wait(); print('job exit', p.returncode, 'after %.1f min' % ((time.time() - t0) / 60))
'''),
('code', '''
# ---- what the job saved in the private repo
tag = ENV.get('TAG') or ENV.get('SUITE') or ''
files = [f for f in HfApi().list_repo_files(repo, repo_type='dataset') if tag and ('/' + tag + '/') in ('/' + f)]
print(len(files), 'files'); print('\\n'.join(files[:40]))
'''),
]
nb = {'cells': [{'cell_type': t, 'metadata': {}, 'source': s.strip('\n').splitlines(True), **({'outputs': [], 'execution_count': None} if t == 'code' else {})} for t, s in cells],
      'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}, 'language_info': {'name': 'python'},
                   'accelerator': 'GPU', 'colab': {'provenance': []}}, 'nbformat': 4, 'nbformat_minor': 5}
json.dump(nb, open(os.path.join(HERE, 'colab_runner.ipynb'), 'w', encoding='utf-8'), indent=1)
print('wrote colab_runner.ipynb')
