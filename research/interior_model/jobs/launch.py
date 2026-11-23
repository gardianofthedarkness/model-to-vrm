"""Start the fetch-and-count CPU Job, then wait for it and print the log tail. The token comes from the vault only."""
import sys, os, time
sys.path.insert(0, r'E:\code\model-to-vrm\model_to_vrm\common')
from vault import get
from huggingface_hub import HfApi, run_uv_job
tok = get('huggingface'); api = HfApi(token=tok); repo = '%s/totalsegmentator-ct-v201' % api.whoami()['name']
api.create_repo(repo, repo_type='dataset', private=True, exist_ok=True)
print('repo private:', api.repo_info(repo, repo_type='dataset').private, flush=True)
job = run_uv_job(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ts_fetch_count_job.py'), script_args=[repo],
                 flavor='cpu-upgrade', secrets={'HF_TOKEN': tok}, timeout='4h', token=tok, name='ts-fetch-count')
print('job', job.id, 'started', time.strftime('%H:%M:%S'), flush=True); t0 = time.time(); tail = []; seen = 0
while True:
    try:
        for i, l in enumerate(api.fetch_job_logs(job_id=job.id, follow=True)):    # streams while the job runs
            if i < seen: continue
            seen = i + 1; tail = (tail + [l.rstrip()])[-15:]
            if 'TS ' in l or 'rror' in l: print(l.rstrip(), flush=True)
    except Exception as e: print('log stream:', type(e).__name__, flush=True)
    st = api.inspect_job(job_id=job.id).status.stage
    if st in ('COMPLETED', 'ERROR', 'CANCELED', 'DELETED'):
        print('job %s after %.1f min' % (st, (time.time() - t0) / 60)); print('\n'.join(tail)); break
    time.sleep(30)
