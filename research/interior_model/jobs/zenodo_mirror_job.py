# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet"]
# ///
"""Copy files of open Zenodo records into a private Hugging Face dataset, one file at a time (stream to disk, upload,
delete), CPU only. env: RECORDS = comma list of <record id>[:<file name>] (no file name: every file of the record),
DEST_PREFIX (default raw), README, PAR (files copied at once, default 1). args: <private repo>"""
import os, sys, json, time, shutil, urllib.request
from huggingface_hub import HfApi
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
repo = sys.argv[1]; pre = os.environ.get('DEST_PREFIX', 'raw'); api = HfApi(); have = set(api.list_repo_files(repo, repo_type='dataset'))
if os.environ.get('README'): api.upload_file(path_or_fileobj=os.environ['README'].encode(), path_in_repo='README.md', repo_id=repo, repo_type='dataset', commit_message='README')
for item in os.environ['RECORDS'].split(','):
    rid, _, want = item.partition(':'); rec = json.load(urllib.request.urlopen('https://zenodo.org/api/records/' + rid, timeout=60))
    lic = rec.get('metadata', {}).get('license', {}).get('id'); log('record', rid, rec['metadata'].get('title'), 'licence', lic, 'access', rec['metadata'].get('access_right'))
    def copy(f):
        dest = '%s/%s/%s' % (pre, rid, f['key'])
        if dest in have: log('already copied', dest); return
        for a in range(5):
            try:
                p = '/tmp/z_' + f['key']
                with urllib.request.urlopen(f['links']['self'], timeout=120) as r, open(p, 'wb') as o: shutil.copyfileobj(r, o, 16 * 2**20)
                api.upload_file(path_or_fileobj=p, path_in_repo=dest, repo_id=repo, repo_type='dataset', commit_message='copy ' + f['key']); os.remove(p); break
            except Exception as e: log('retry', a, f['key'], type(e).__name__, str(e)[:120]); time.sleep(60)
        log(f['key'], round(f['size'] / 1e9, 2), 'GB copied')
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(int(os.environ.get('PAR', '1'))) as ex: list(ex.map(copy, [f for f in rec.get('files', []) if not want or f['key'] == want]))
log('SUMMARY done')
