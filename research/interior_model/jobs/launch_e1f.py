"""Start one fetch/census CPU Job and follow it.  args: <job name> <repo suffix> <script args...>  (token only from the vault)."""
import sys, os, time
sys.path.insert(0, r'E:\code\model-to-vrm\model_to_vrm\common')
from vault import get
from huggingface_hub import HfApi, run_uv_job
name, suffix, rest = sys.argv[1], sys.argv[2], sys.argv[3:]
tok = get('huggingface'); api = HfApi(token=tok); repo = '%s/%s' % (api.whoami()['name'], suffix)
api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
print('repo', suffix, 'private:', api.repo_info(repo, repo_type='dataset').private, flush=True)
args = [a.replace('{repo}', repo) for a in rest]
job = run_uv_job(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'e1_figs_job.py'), script_args=args,
                 flavor='a10g-small', secrets={'HF_TOKEN': tok}, timeout='1h', token=tok, name=name)
print('job', job.id, 'started', time.strftime('%H:%M:%S'), flush=True); t0 = time.time(); seen = 0; tail = []
while True:
    try:
        for i, l in enumerate(api.fetch_job_logs(job_id=job.id, follow=True)):
            if i < seen: continue
            seen = i + 1; tail = (tail + [l.rstrip()])[-25:]
            if 'TS ' in l or 'rror' in l or 'Traceback' in l: print(l.rstrip(), flush=True)
    except Exception as e: print('log stream:', type(e).__name__, flush=True)
    st = api.inspect_job(job_id=job.id).status.stage
    if st in ('COMPLETED', 'ERROR', 'CANCELED', 'DELETED'):
        print('job %s after %.1f min' % (st, (time.time() - t0) / 60)); print('\n'.join(tail)); break
    time.sleep(30)
