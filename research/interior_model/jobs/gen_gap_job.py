# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "scikit-image"]
# ///
"""Check 0 before S1.1a (gen_gap.py) on a released MARROW suite: assemble the game heads, prepare tokens and codes for the
validation heads and N_REF training heads with the suite's autoencoder (no hair variants), then score the gap.
Uploads <repo>/field_exp/<SUITE>/gen_gap/ (gen_gap.json, log.txt).
env: COMMIT, SUITE (default marrow_s1_0_t16), ARM (default NH_s0), N_REF (default 30), N_TRAIN (default 8), SHARDS (default 2), LIMIT_VAL (smoke). args: <repo>"""
import os, sys, json, time, subprocess, urllib.request, threading
from huggingface_hub import HfApi, hf_hub_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
api = HfApi(); repo = sys.argv[1]; C = os.environ['COMMIT']; SUITE = os.environ.get('SUITE', 'marrow_s1_0_t16'); ARM = os.environ.get('ARM', 'NH_s0')
os.makedirs('code', exist_ok=True)
for f in ('face_compress.py', 'transformer_smoke.py', 'flow_f1.py', 'field_exp.py', 'field_prep.py', 'tvae.py', 'game.py', 'gen_gap.py'):
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.join('code', f))
os.environ.update(CLASSES='game', TRAIN_SOURCES='ts,hanseg', TEST_SOURCES='ts,hanseg', HEADCACHE='/data/hc')
sys.path.insert(0, 'code'); import game
fields = '/data/fx/game_heads'; game.assemble(fields)
idx = json.load(open(os.path.join(fields, 'index.json'))); import numpy as np
tr = sorted(c for c, v in idx['heads'].items() if v['split'] == 'train'); va = sorted(c for c, v in idx['heads'].items() if v['split'] == 'val')
va = va[:int(os.environ['LIMIT_VAL'])] if os.environ.get('LIMIT_VAL') else va   # smoke
rng = np.random.default_rng(0); ref = sorted(tr[i] for i in rng.choice(len(tr), min(int(os.environ.get('N_REF', '30')), len(tr)), replace=False))
idx['heads'] = {c: idx['heads'][c] for c in ref + va}; json.dump(idx, open(os.path.join(fields, 'index.json'), 'w'))
log('heads: reference train', len(ref), 'val', len(va))
aef = hf_hub_download(repo, 'field_exp/%s/ae_soft/ae_soft.pt' % SUITE, repo_type='dataset', local_dir='/data/in')
flf = hf_hub_download(repo, 'field_exp/%s/%s/flow.pt' % (SUITE, ARM), repo_type='dataset', local_dir='/data/in')
prep = '/data/prep'; S = int(os.environ.get('SHARDS', '2')); env = dict(os.environ, PYTHONWARNINGS='ignore')
ps = [subprocess.Popen([sys.executable, '-u', 'field_prep.py', fields, aef, prep], cwd='code', env=dict(env, SHARD='%d/%d' % (i, S), HAIR='0')) for i in range(S)]
for p in ps: p.wait()
log('preparation done', [p.returncode for p in ps])
out = '/data/out'; os.makedirs(out, exist_ok=True)
p = subprocess.run([sys.executable, '-u', 'gen_gap.py', fields, prep, aef, flf, os.path.join(out, 'gen_gap.json')], cwd='code', env=env, capture_output=True, text=True)
print(p.stdout[-20000:], p.stderr[-4000:], flush=True); open(os.path.join(out, 'log.txt'), 'w').write(p.stdout + p.stderr); log('exit gen_gap', p.returncode)
for a in range(5):
    try: api.upload_folder(folder_path=out, path_in_repo='field_exp/%s/gen_gap_%s' % (SUITE, ARM), repo_id=repo, repo_type='dataset', commit_message='gen gap check ' + SUITE); break
    except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
log('SUMMARY done')
