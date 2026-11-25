# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet"]
# ///
"""Copy a public Hugging Face dataset into a private one, file by file (download, upload, delete), CPU only, so the job's
disk never holds more than one file. env: SRC (source dataset), DEST_PREFIX (path in the private repo, default raw),
README (text for the private repo's README). args: <private repo>"""
import os, sys, time
from huggingface_hub import HfApi, hf_hub_download
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
repo = sys.argv[1]; src = os.environ['SRC']; pre = os.environ.get('DEST_PREFIX', 'raw'); api = HfApi()
have = set(api.list_repo_files(repo, repo_type='dataset'))
files = [s for s in api.dataset_info(src, files_metadata=True).siblings if not s.rfilename.startswith('.')]
log('source', src, len(files), 'files', round(sum(s.size or 0 for s in files) / 1e9, 1), 'GB; already copied', sum(pre + '/' + s.rfilename in have for s in files))
if os.environ.get('README'): api.upload_file(path_or_fileobj=os.environ['README'].encode(), path_in_repo='README.md', repo_id=repo, repo_type='dataset', commit_message='README')
for i, s in enumerate(files):
    dest = pre + '/' + s.rfilename
    if dest in have: continue
    for a in range(5):
        try:
            p = hf_hub_download(src, s.rfilename, repo_type='dataset', local_dir='/tmp/m')
            api.upload_file(path_or_fileobj=p, path_in_repo=dest, repo_id=repo, repo_type='dataset', commit_message='copy ' + s.rfilename); os.remove(p); break
        except Exception as e: log('retry', a, s.rfilename, type(e).__name__, str(e)[:120]); time.sleep(60)
    log(i + 1, '/', len(files), s.rfilename, round((s.size or 0) / 1e9, 2), 'GB copied')
log('SUMMARY done', len(files), 'files')
