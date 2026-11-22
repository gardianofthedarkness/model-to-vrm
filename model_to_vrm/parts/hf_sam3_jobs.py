"""SAM 3 masks for several views folders at once, as parallel Hugging Face GPU Jobs (one per folder), from this PC:
  python hf_sam3_jobs.py <views dir> [<views dir> ...] [--flavor=a10g-small]
Each <views dir> holds cameras.json and tex_*.png (body_views.py); its name on the hub is the path below the figures
folder with separators replaced. Uploads the views and sam3_masks.py to the private dataset repo
<user>/model-to-vrm-anatomy-jobs, runs hf_sam3_job.py per folder, waits, and downloads <views dir>/masks/*.npz.
The token comes from the vault (vault.py 'huggingface'), never a file."""
import sys, os, time, glob, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
from vault import get
from huggingface_hub import HfApi, run_uv_job, snapshot_download, CommitOperationAdd
dirs = [os.path.abspath(a) for a in sys.argv[1:] if not a.startswith('--')]; flags = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
flavor = flags.get('flavor', 'a10g-small'); tok = get('huggingface'); api = HfApi(token=tok); repo = '%s/model-to-vrm-anatomy-jobs' % api.whoami()['name']
api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
key = lambda d: d.replace('\\', '/').split('/figures/')[-1].replace('/', '__').replace(':', '')
ops = [CommitOperationAdd(path_in_repo='code/sam3_masks.py', path_or_fileobj=os.path.join(HERE, 'sam3_masks.py'))]
for d in dirs:
    for f in [os.path.join(d, 'cameras.json')] + sorted(glob.glob(os.path.join(d, 'tex_*.png'))):
        ops.append(CommitOperationAdd(path_in_repo='views/%s/%s' % (key(d), os.path.basename(f)), path_or_fileobj=f))
api.create_commit(repo, repo_type='dataset', operations=ops, commit_message='SAM 3 views'); print('S3J uploaded %d files' % len(ops), flush=True)
jobs = {d: run_uv_job(os.path.join(HERE, 'hf_sam3_job.py'), script_args=[repo, key(d)], flavor=flavor, secrets={'HF_TOKEN': tok}, timeout='1h', token=tok, name='sam3-' + key(d)[-40:]).id for d in dirs}
for d, j in jobs.items(): print('S3J started %s: job %s' % (key(d), j), flush=True)
t0 = time.time(); done = {}
while len(done) < len(jobs):
    for d, jid in jobs.items():
        if d in done: continue
        st = api.inspect_job(job_id=jid).status.stage
        if st in ('COMPLETED', 'ERROR', 'CANCELED', 'DELETED'):
            done[d] = st; tail = list(api.fetch_job_logs(job_id=jid))[-5:]
            print('S3J %s %s after %.0f s\n   %s' % (key(d), st, time.time() - t0, '\n   '.join(l.rstrip() for l in tail)), flush=True)
    time.sleep(15)
for d in dirs:
    tmp = os.path.join(d, '_hf'); snapshot_download(repo, repo_type='dataset', allow_patterns=['views/%s/masks/*' % key(d)], local_dir=tmp, token=tok)
    os.makedirs(os.path.join(d, 'masks'), exist_ok=True)
    for f in glob.glob(os.path.join(tmp, 'views', key(d), 'masks', '*.npz')): shutil.copy(f, os.path.join(d, 'masks'))
    shutil.rmtree(tmp, ignore_errors=True)
print('S3J done: %s' % {key(d): v for d, v in done.items()}, flush=True)
