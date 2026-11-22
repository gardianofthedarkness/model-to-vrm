"""Dense atlas slicing and dense slice fits on Hugging Face Jobs (one CPU job per segment, in parallel), from this PC:
  python hf_fit_jobs.py slice <atlas.npz> <atlas set name> <out dir> [--flavor=cpu-upgrade] [--segments=a,b]
      za_slices.py --dense per segment (minimal-area planes); the atlas is uploaded once; results land in the repo's
      atlas_sets/<name>/ and in <out dir>. Atlas-only: done once, reused for every model.
  python hf_fit_jobs.py <model dir> <dense atlas slices dir | --atlas-set=NAME> <out dir> [--flavor=..] [--segments=..]
      fit_slices.py per segment on the model
  1. a private dataset repo <your user>/model-to-vrm-anatomy-jobs holds the code (segments.py, planes.py, fit_slices.py,
     za_slices.py), the atlas, the dense atlas slices and the model's inputs (parts/orig_colours.npz,
     parts/orig_normals.npz, contacts/material.npz, body/frames.npz - keypoint or bone frames - or body/skeleton.npz)
  2. one job per segment runs hf_fit_job.py (uv script) and uploads its results
  3. this script waits for the jobs, prints each one's tail, and downloads the results into <out dir>
  The Hugging Face token comes from the system's credential store (vault.py 'huggingface'), never from a file."""
import sys, os, time, shutil, glob
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'common'))
from vault import get
from huggingface_hub import HfApi, run_uv_job, snapshot_download, CommitOperationAdd
from segments import ORDER
args = [a for a in sys.argv[1:] if not a.startswith('--')]; flags = dict(a[2:].split('=', 1) for a in sys.argv[1:] if a.startswith('--') and '=' in a)
segs = flags.get('segments', ','.join(ORDER)).split(','); flavor = flags.get('flavor', 'cpu-upgrade')
tok = get('huggingface'); api = HfApi(token=tok); user = api.whoami()['name']; repo = '%s/model-to-vrm-anatomy-jobs' % user
api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
have = set(api.list_repo_files(repo, repo_type='dataset'))
code = [(os.path.join(HERE, f), 'code/' + f) for f in ('segments.py', 'planes.py', 'fit_slices.py', 'za_slices.py')]
def push(ops, msg):
    missing = [a for a, _ in ops if not os.path.exists(a)]
    if missing: raise SystemExit('missing inputs: %s' % missing)
    api.create_commit(repo, repo_type='dataset', operations=[CommitOperationAdd(path_in_repo=b, path_or_fileobj=a) for a, b in ops], commit_message=msg)
    print('HJ uploaded %d files to %s' % (len(ops), repo), flush=True)
def wait(jobs):
    t0 = time.time(); done = {}
    while len(done) < len(jobs):
        for s, jid in jobs.items():
            if s in done: continue
            st = api.inspect_job(job_id=jid).status.stage
            if st in ('COMPLETED', 'ERROR', 'CANCELED', 'DELETED'):
                done[s] = st; tail = list(api.fetch_job_logs(job_id=jid))[-6:]
                print('HJ %s %s after %.0f s\n   %s' % (s, st, time.time() - t0, '\n   '.join(l.rstrip() for l in tail)), flush=True)
        time.sleep(20)
    return done
def fetch(pattern, sub, OUT):
    os.makedirs(OUT, exist_ok=True); snapshot_download(repo, repo_type='dataset', allow_patterns=[pattern], local_dir=os.path.join(OUT, '_hf'), token=tok)
    for f in glob.glob(os.path.join(OUT, '_hf', *sub.split('/'), '*')): shutil.copy(f, OUT)
if args[0] == 'slice':                                             # dense atlas slicing
    ap, aset, OUT = os.path.abspath(args[1]), args[2], os.path.abspath(args[3])
    push(code + ([] if 'atlas/atlas.npz' in have else [(ap, 'atlas/atlas.npz')]), 'atlas slicing code' + ('' if 'atlas/atlas.npz' in have else ' and atlas'))
    jobs = {s: run_uv_job(os.path.join(HERE, 'hf_fit_job.py'), script_args=[repo, 'slice', aset, s], flavor=flavor, secrets={'HF_TOKEN': tok}, timeout='3h', token=tok, name='slice-%s-%s' % (aset, s)).id for s in segs}
    for s, j in jobs.items(): print('HJ started slice %s: job %s' % (s, j), flush=True)
    done = wait(jobs); fetch('atlas_sets/%s/*' % aset, 'atlas_sets/%s' % aset, OUT); print('HJ done: %s' % done, flush=True); sys.exit(0)
R, OUT = os.path.abspath(args[0]), os.path.abspath(args[-1]); model = os.path.basename(R.rstrip('\\/')); aset = flags.get('atlas-set')
ops = list(code)
if not aset: SD = os.path.abspath(args[1]); ops += [(os.path.join(SD, s + '.npz'), 'atlas/%s.npz' % s) for s in segs]
frames = 'body/frames.npz' if os.path.exists(os.path.join(R, 'body', 'frames.npz')) else 'body/skeleton.npz'
ops += [(os.path.join(R, *p.split('/')), 'models/%s/%s' % (model, p)) for p in ('parts/orig_colours.npz', 'parts/orig_normals.npz', 'contacts/material.npz', frames)]
push(ops, 'inputs for %s' % model)
env = {'ATLAS_SET': aset} if aset else None
jobs = {}
for s in segs:
    j = run_uv_job(os.path.join(HERE, 'hf_fit_job.py'), script_args=[repo, model, s], flavor=flavor, secrets={'HF_TOKEN': tok}, env=env, timeout='2h', token=tok, name='fit-%s-%s' % (model, s))
    jobs[s] = j.id; print('HJ started %s: job %s' % (s, j.id), flush=True)
done = wait(jobs); fetch('results/%s/*' % model, 'results/%s' % model, OUT)
print('HJ done: %s' % {s: v for s, v in done.items()}, flush=True)
