"""Wake-up watcher for a suite running elsewhere (Colab or Hugging Face Jobs): polls the private repo and exits with one
line when there is something to act on, so a background shell wakes the assistant.
  WAIT_FOR=gate     the autoencoder gate has landed (field_exp/<suite>/ae_*/gate.json)
  WAIT_FOR=summary  the suite has finished (field_exp/<suite>/summary.json)
Also exits on a stale heartbeat (field_exp/<suite>/heartbeat.json older than STALE_MIN minutes, default 30: the run was
probably dropped) or after MAX_H hours (default 8). args: <suite> [repo suffix, default hanseg-ct-mr]"""
import os, sys, json, time, datetime
sys.path.insert(0, r'E:\code\model-to-vrm\model_to_vrm\common')
from vault import get
from huggingface_hub import HfApi, hf_hub_download
suite = sys.argv[1]; suffix = sys.argv[2] if len(sys.argv) > 2 else 'hanseg-ct-mr'
WAIT = os.environ.get('WAIT_FOR', 'summary'); STALE = float(os.environ.get('STALE_MIN', '30')); MAXH = float(os.environ.get('MAX_H', '8'))
tok = get('huggingface'); api = HfApi(token=tok); repo = '%s/%s' % (api.whoami()['name'], suffix); t0 = time.time(); seen_hb = None
while True:
    try:
        fs = [f for f in api.list_repo_files(repo, repo_type='dataset') if f.startswith('field_exp/%s/' % suite)]
        if WAIT == 'gate':
            g = [f for f in fs if f.endswith('/gate.json') and '/ae_' in f]
            if g:
                for f in g: print('GATE', f, open(hf_hub_download(repo, f, repo_type='dataset', token=tok, force_download=True)).read().replace('\n', ' ')[:3000], flush=True)
                sys.exit(0)
        if WAIT == 'summary' and 'field_exp/%s/summary.json' % suite in fs:
            print('DONE', open(hf_hub_download(repo, 'field_exp/%s/summary.json' % suite, repo_type='dataset', token=tok, force_download=True)).read().replace('\n', ' ')[:4000], flush=True); sys.exit(0)
        if 'field_exp/%s/heartbeat.json' % suite in fs:
            hb = json.load(open(hf_hub_download(repo, 'field_exp/%s/heartbeat.json' % suite, repo_type='dataset', token=tok, force_download=True)))
            age = (datetime.datetime.utcnow() - datetime.datetime.strptime(hb['utc'], '%Y-%m-%dT%H:%M:%SZ')).total_seconds() / 60
            seen_hb = hb
            if age > STALE: print('STALE heartbeat %.0f min old; last stage %s after %s s' % (age, hb['stage'], hb['elapsed_s']), flush=True); sys.exit(2)
        elif time.time() - t0 > STALE * 60 * 2: print('NO HEARTBEAT after %.0f min: the run may not have started' % ((time.time() - t0) / 60), flush=True); sys.exit(3)
    except Exception as e: print('poll error', type(e).__name__, str(e)[:120], file=sys.stderr, flush=True)
    if time.time() - t0 > MAXH * 3600: print('TIMEOUT after %.1f h; last heartbeat %s' % (MAXH, seen_hb), flush=True); sys.exit(4)
    time.sleep(120)
