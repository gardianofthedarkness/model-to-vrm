"""Ask the vision LLM (DeepSeek through Pi, key from the vault) per sub-contact: what are the two sides, and are they two
separate things pressing together or one thing folded on itself. Materials come from SAM 3 (given as fact in the hints);
the LLM only flags it when one looks wrong.
  python subcontact_label.py <contacts dir> [workers]   ->  <dir>/sub_labels.json   (re-run keeps finished ones)
Calls run in parallel (default 8 at once); each finished answer is written straight away.
Answer: {"A": "...", "B": "...", "kind": "two_things|fold|noise", "material_wrong": "" or "A: hair" ..., "why": "..."}"""
import sys, os, json, re, subprocess, threading
from concurrent.futures import ThreadPoolExecutor
D = sys.argv[1]; G = json.load(open(os.path.join(D, 'sub.json'))); out_p = os.path.join(D, 'sub_labels.json')
done = json.load(open(out_p)) if os.path.exists(out_p) else {}
RUN = r'E:\code\novel-ontology-pi\engine\characters\pipeline\pi_run.py'; PY = r'E:\splat\.venv\Scripts\python.exe'
Q = ('The picture shows one contact on a 3D model of a dressed person, in three panels: the textured model, a close-up of '
     'the bare model, and the same close-up cut through by a plane so surfaces hidden inside creases show. Side A is '
     'MAGENTA, side B is CYAN. A segmenter measured: side A is {ma} ({ca}), side B is {mb} ({cb}); nearest skeleton joints '
     'A {ja}, B {jb}; gap {gap} mm. Say what each side is (e.g. "left inner thigh", "skirt hem", "bag strap"). Then say the '
     'kind: "two_things" if A and B are two separate things pressing together (two garments, a garment on skin, a carried '
     'thing on the body, or two different body parts such as calf on thigh or chin on neck), "fold" if A and B are one thing '
     'folded or creased on itself (one garment\'s pleats, one bag\'s folds, one lock of hair), or "noise". If the '
     'segmenter\'s material for a side is clearly wrong, say so in material_wrong (e.g. "B: hair"), else "". JSON only: '
     '{{"A": "...", "B": "...", "kind": "...", "material_wrong": "", "why": "..."}}')
lock = threading.Lock(); W = int(sys.argv[2]) if len(sys.argv) > 2 else 8
def one(g):
    k = str(g['sub'])
    q = Q.format(ma=g['A']['material'], ca=g['A']['concepts'], mb=g['B']['material'], cb=g['B']['concepts'], ja=g['A']['near_joint'], jb=g['B']['near_joint'], gap=round(g['gap_median_m'] * 1000, 1))
    card = os.path.join(D, 'sub_cards', 'sub_%s.jpg' % k)
    r = subprocess.run([PY, RUN, '--no-session', '-p', '@' + card, q], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    m = re.search(r'\{.*\}', r.stdout, re.S)
    try: ans = json.loads(m.group(0)) if m else {'error': r.stdout[-300:] + r.stderr[-300:]}
    except Exception: ans = {'error': 'bad json: ' + m.group(0)[:300]}
    with lock:
        done[k] = ans; json.dump(done, open(out_p, 'w'), indent=1)
        print('SL sub %s %s: %s | %s | %s' % (k, g['pair'], ans.get('kind', ans.get('error', '?'))[:80], ans.get('A', ''), ans.get('B', '')), flush=True)
todo = [g for g in G if str(g['sub']) not in done or 'error' in done[str(g['sub'])]]; print('SL %d to do, %d at once' % (len(todo), W), flush=True)
with ThreadPoolExecutor(W) as ex: list(ex.map(one, todo))
print('SL done: %d of %d labelled' % (sum('error' not in v for v in done.values()), len(G)))
