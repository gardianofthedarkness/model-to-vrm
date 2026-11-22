"""What hangs loose on the figure, named and boxed by Jev (the decision model) on each textured body view:
  python loose_names.py <model dir>        -> <model dir>/loose/names.json and the views copied to <model dir>/loose/
One request per view (8 in all; the budget guard LLM_MAX_CALLS stops a run that would exceed it). Per view Jev answers
  garment   which loose outer garment hangs away from the body, from prepared names (or none)   -> the word SAM 3 is asked
  box       where it is in the picture                                                         -> picks SAM 3's instances
The key comes from the vault, is passed only in the request header, and is never printed or written."""
import sys, os, json, base64, shutil, time, urllib.request, urllib.error
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common')); from vault import get
from models import JEV_MODEL, JEV_ENDPOINT
MD = sys.argv[1]; OD = os.path.join(MD, 'loose'); os.makedirs(OD, exist_ok=True); cams = json.load(open(os.path.join(MD, 'body', 'cameras.json'))); views = sorted(n for n in cams if n.startswith('tex_'))
if len(views) > int(os.environ.get('LLM_MAX_CALLS', '200')): raise SystemExit('over the LLM call budget')
MODEL, ENDPOINT = JEV_MODEL, JEV_ENDPOINT; KEY = get('jev')
if not KEY: raise SystemExit('no Jev key in the vault')
NAMES = ['cape', 'cloak', 'robe', 'coat', 'long skirt', 'scarf', 'veil']
Q = {'garment': {'type': 'choice', 'instructions': 'Which loose outer garment, if any, hangs from this figure away from the body (not tight clothing, not hair)?', 'criteria': dict({n: 'a ' + n for n in NAMES}, none='nothing hangs loose')},
     'box': {'type': 'box', 'instructions': 'In the "view" picture, the whole loose outer garment that hangs away from the body (cape, cloak, robe, coat, long skirt, scarf or veil), not the body and not the hair.'}}
def ask(az, png):
    body = {'model': MODEL, 'state': {'state': 'One render of a 3D figure standing with arms out, seen from %d degrees around it (0 = front, 180 = back). We are sorting its surface into body and loose clothing.' % az, 'view': 'view_image'},
            'assets': {'view_image': {'type': 'image', 'format': 'png', 'data_base64': base64.b64encode(open(png, 'rb').read()).decode()}}, 'questions': Q}
    req = urllib.request.Request(ENDPOINT, json.dumps(body).encode(), {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY, 'TypeSafe-Danger-Zone-Images': 'true'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as res: return json.loads(res.read())['answers']
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429: raise SystemExit('Jev refused: %d %s' % (e.code, e.read()[:300].decode('utf-8', 'replace')))
        except (urllib.error.URLError, TimeoutError): pass
        time.sleep(1 + 2 * attempt)
    raise SystemExit('Jev failed 3 times')
out = {}
for n in views:
    src = os.path.join(MD, 'body', n + '.png'); shutil.copyfile(src, os.path.join(OD, n + '.png')); a = ask(int(cams[n].get('az', 0)), src)
    out[n] = dict(garment=a.get('garment', {}).get('choice'), probabilities=a.get('garment', {}).get('probabilities'), box=a.get('box', {}).get('box'), confidence=a.get('box', {}).get('confidence'))
    print('LN', n, out[n]['garment'], out[n]['box'], out[n]['confidence'], flush=True)
json.dump(dict(model=MODEL, views=out), open(os.path.join(OD, 'names.json'), 'w'), indent=1); print('LN done: %d requests' % len(views))
