"""Material per segment, judged by the vision LLM (DeepSeek through Pi, key from the vault) instead of per-vertex votes:
  python segment_label.py <model dir> [workers]     -> sam_body/segment_material.npz, sam_body/segment_labels.json,
                                                         sam_body/seg_cards/
  segments  the model's own parts (parts/orig_colours.npz 'part', e.g. Tripo's); a model in one piece falls back to
            colour clusters (Gaussian mixture in Lab, count by BIC) joined along the mesh; crumbs below the vertex-weighted Otsu
            split of the piece sizes merged into the neighbour they share the most border with
  card      the 3 textured body views where the segment is most visible: the segment in its own colours with a magenta
            outline, the rest faded (a magenta tint made bare legs look like tights)
  ask       "what is the magenta area": flesh | cloth | hair | prop | mixed, and its name. SAM 3's mean shares for the
            segment go along as a hint. Calls run in parallel; the budget guard (LLM_MAX_CALLS, default 200) stops a run
            that would exceed it.
  mixed     a segment answered "mixed" is split by colour (mixture in Lab, 1..6 by BIC) into connected pieces above the
            Otsu split of its piece sizes, and each piece is asked again (one more round only)
  writes    material per vertex (-1: invisible or a crumb; the material step fills those along the mesh)"""
import sys, os, json, re, subprocess, threading, numpy as np, cv2
from concurrent.futures import ThreadPoolExecutor
from sklearn.mixture import GaussianMixture
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
R = sys.argv[1]; W = int(sys.argv[2]) if len(sys.argv) > 2 else 8; j = lambda *p: os.path.join(R, *p)
BUDGET = int(os.environ.get('LLM_MAX_CALLS', '200'))
RUN = r'E:\code\novel-ontology-pi\engine\characters\pipeline\pi_run.py'; PY = r'E:\splat\.venv\Scripts\python.exe'
O = np.load(j('parts', 'orig_colours.npz')); V, C, part = O['V'], O['C'].astype(np.float32), O['part']; F = np.load(j('parts', 'orig_normals.npz'))['F']
VZ = np.load(j('body', 'vis.npz')); vn = [str(x) for x in VZ['names']]; VIS = VZ['vis']; cams = json.load(open(j('body', 'cameras.json')))   # arrays read once: threads share them
Sh = np.load(j('sam_body', 'shares.npz')); share, con = np.asarray(Sh['share']), [str(x) for x in Sh['concepts']]
MN = ['flesh', 'cloth', 'hair', 'prop']; od = j('sam_body', 'seg_cards'); os.makedirs(od, exist_ok=True)
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[0] - 1 if m else 0
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
_, rep, inv = np.unique(V, axis=0, return_index=True, return_inverse=True); inv = inv.ravel(); dup = np.flatnonzero(rep[inv] != np.arange(len(V)))
EDGES = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]], np.c_[dup, rep[inv[dup]]]]
LAB = cv2.cvtColor(C.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
def colour_pieces(idx, kmax):                                     # colour clusters inside idx, joined along the mesh, crumbs dropped
    AB = LAB[:, 1:]; sub = AB[idx][np.random.default_rng(0).choice(len(idx), min(30000, len(idx)), replace=False)]   # true colour: chroma only, shading left out
    fits = [GaussianMixture(k, random_state=0).fit(sub) for k in range(1, kmax + 1)]; g = fits[int(np.argmin([f.bic(sub) for f in fits]))]
    lab = -np.ones(len(V), int); lab[idx] = g.predict(AB[idx]); e = EDGES[(lab[EDGES[:, 0]] >= 0) & (lab[EDGES[:, 0]] == lab[EDGES[:, 1]])]
    n, comp = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))), directed=False)
    if MERGE:                                                     # crumbs merged into the neighbour sharing the most border (a realistic texture
        comp = merge_crumbs(comp, idx); return [idx[comp[idx] == k] for k in np.unique(comp[idx])]   # breaks into thousands of shading specks)
    sz = np.bincount(comp[idx], minlength=n); keep = np.flatnonzero(sz > np.exp(otsu(np.log(sz[sz > 0]))))
    return [idx[comp[idx] == k] for k in keep]
MERGE = True                                                      # crumbs always merged (first split and the "mixed" re-split)
def otsu_w(v, w):                                                 # Otsu split weighted (here: by vertices, so big pieces count by their size)
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w); cv = np.cumsum(w * v); Wt = c[-1]
    w0 = c[:-1]; w1 = Wt - w0; ok = (w0 > 0) & (w1 > 0); m0 = cv[:-1] / np.maximum(w0, 1e-30); m1 = (cv[-1] - cv[:-1]) / np.maximum(w1, 1e-30)
    s = np.where(ok, w0 * w1 * (m0 - m1) ** 2, -1); return v[int(np.argmax(s))]
def merge_crumbs(comp, idx):
    if len(idx) == len(V): return merge_crumbs_whole(comp, idx)   # the whole model: the first split (its pieces key the saved answers)
    return merge_crumbs_part(comp, idx)
def merge_crumbs_whole(comp, idx):
    comp = comp.copy(); u = np.unique(comp[idx]); sz = np.bincount(comp, minlength=comp.max() + 1); thr = np.exp(otsu_w(np.log(sz[u]), sz[u].astype(float)))
    for _ in range(100):
        sz = np.bincount(comp, minlength=comp.max() + 1); small = sz[comp] < thr
        b = EDGES[comp[EDGES[:, 0]] != comp[EDGES[:, 1]]]; b = np.r_[b, b[:, ::-1]]; b = b[small[b[:, 0]]]
        if not len(b): break
        N = comp.max() + 1; pair = comp[b[:, 0]].astype(np.int64) * N + comp[b[:, 1]]; up, cnt = np.unique(pair, return_counts=True); a, t = up // N, up % N
        o = np.lexsort((-cnt, a)); first = o[np.r_[True, a[o][1:] != a[o][:-1]]]; moved = 0
        for k, tg in sorted(zip(a[first], t[first]), key=lambda kv: sz[kv[0]]):
            if sz[tg] >= sz[k]: comp[comp == k] = tg; moved += 1
        if not moved: break
    print('SG crumbs merged: %d pieces left (size split %.0f vertices)' % (len(np.unique(comp[idx])), thr), flush=True); return comp
def merge_crumbs_part(comp, idx):
    """pieces smaller than the vertex-weighted Otsu split of the log piece sizes go into the neighbouring piece they share
    the most border with (smallest first, only into a larger piece), until none is left below the split"""
    comp = comp.copy(); inn = np.zeros(len(V), bool); inn[idx] = True   # only inside idx: outside vertices are not pieces
    E_ = EDGES[inn[EDGES[:, 0]] & inn[EDGES[:, 1]]]; N = comp.max() + 1
    u = np.unique(comp[idx]); sz = np.bincount(comp[idx], minlength=N); thr = np.exp(otsu_w(np.log(sz[u]), sz[u].astype(float)))
    for _ in range(100):
        sz = np.bincount(comp[idx], minlength=N); small = sz[comp] < thr
        b = E_[comp[E_[:, 0]] != comp[E_[:, 1]]]; b = np.r_[b, b[:, ::-1]]; b = b[small[b[:, 0]]]
        if not len(b): break
        pair = comp[b[:, 0]].astype(np.int64) * N + comp[b[:, 1]]; up, cnt = np.unique(pair, return_counts=True); a, t = up // N, up % N
        o = np.lexsort((-cnt, a)); first = o[np.r_[True, a[o][1:] != a[o][:-1]]]; a, t = a[first], t[first]
        ok = (sz[t] > sz[a]) | ((sz[t] == sz[a]) & (t > a))       # only into a larger piece (ties by id: no swapping pairs)
        if not ok.any(): break
        rm = np.arange(N); rm[a[ok]] = t[ok]
        for _ in range(64):                                       # follow chains (a -> b -> c) in one relabel
            nxt = rm[rm]
            if (nxt == rm).all(): break
            rm = nxt
        comp[idx] = rm[comp[idx]]
    print('SG crumbs merged: %d pieces left (size split %.0f vertices)' % (len(np.unique(comp[idx])), thr), flush=True); return comp
def proj(name, idx):
    cm = cams[name]; Rr = cm['res']; q = np.c_[V[idx], np.ones(len(idx))] @ np.array(cm['matrix_inv']).T
    u = ((q[:, 0] / cm['scale'] + 0.5) * Rr).astype(int); v = ((0.5 - q[:, 1] / cm['scale']) * Rr).astype(int)
    ok = VIS[vn.index(name)][idx] & (u >= 0) & (v >= 0) & (u < Rr) & (v < Rr); return u[ok], v[ok]
def card(key, idx):
    views = sorted(vn, key=lambda n: -len(proj(n, idx)[0]))[:3]
    if len(proj(views[0], idx)[0]) == 0: return None
    tiles = []
    for name in views:
        im = cv2.imread(j('body', name + '.png')); dim = (im * 0.45 + 255 * 0.55).astype(np.uint8); u, v = proj(name, idx)
        mk = np.zeros(im.shape[:2], np.uint8); mk[v, u] = 255; mk = cv2.morphologyEx(mk, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        dim[mk > 0] = im[mk > 0]                                      # the segment in its own colours (a tint made skin look like tights)
        cs, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(dim, cs, -1, (255, 0, 255), 2)
        H_, W_ = dim.shape[:2]                                    # zoom: three times the segment's own extent (its surroundings for context)
        if len(u):
            cu, cv_ = (u.min() + u.max()) / 2, (v.min() + v.max()) / 2; half = min(max(np.ptp(u), np.ptp(v), 1) * 1.5, max(H_, W_) / 2)
            x0, x1, y0, y1 = int(max(0, cu - half)), int(min(W_, cu + half)), int(max(0, cv_ - half)), int(min(H_, cv_ + half)); dim = dim[y0:y1, x0:x1]
        tiles.append(cv2.resize(dim, (480, 480), interpolation=cv2.INTER_CUBIC))
    p = os.path.join(od, 'seg_%s.jpg' % key); cv2.imwrite(p, np.hstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85]); return p
Q = ('The picture shows close-ups of a 3D model of a dressed person from three directions. One area is OUTLINED IN MAGENTA and shown in its '
     'true colours; everything else is faded. What is the outlined area made of? flesh (bare skin), cloth (anything worn: garment, sock, shoe, hair band), hair, prop (a carried thing), '
     'or mixed (it clearly covers two or more of these, e.g. a sleeve and the hand). A segmenter guessed: {hint}. '
     'Answer with JSON only: {{"material": "one of flesh, cloth, hair, prop, mixed", "what": "...", "why": "..."}}')   # no | : pi runs via a .cmd and cmd would pipe
JUDGE = os.environ.get('LABEL_JUDGE', 'pi')                       # pi (DeepSeek through Pi) or jev (the project's decision model, much faster)
CACHE_F = j('sam_body', 'segment_answers_%s_parts.json' % JUDGE); CACHE = json.load(open(CACHE_F, encoding='utf-8')) if os.path.exists(CACHE_F) else {}
CLOCK = threading.Lock()                                          # answers kept as they arrive: a stopped run never pays for a card twice
JEV_STATE = ('Three close-up renders of a 3D model of a dressed fictional figure, seen from three directions. One area is OUTLINED '
             'IN MAGENTA and shown in its true colours; everything else is faded.')
JEV_Q = {'material': {'type': 'choice', 'instructions': 'What is the area outlined in magenta made of?',
                      'criteria': {'flesh': 'bare skin', 'cloth': 'anything worn: garment, glove, sleeve, sock, shoe, hair band',
                                   'hair': 'hair', 'prop': 'a carried or separate object', 'mixed': 'it clearly covers two or more of these'}}}
def ask_jev(p):
    import base64, urllib.request, urllib.error, time
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_to_vrm', 'common')); from vault import get
    body = {'model': os.environ.get('JEV_MODEL', 'jev-1.14.0'), 'state': {'state': JEV_STATE, 'card': 'card_image'},
            'assets': {'card_image': {'type': 'image', 'format': 'jpeg', 'data_base64': base64.b64encode(open(p, 'rb').read()).decode()}}, 'questions': JEV_Q}
    req = urllib.request.Request('https://api.typesafe.ai/v1/systemone', json.dumps(body).encode(),
                                 {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + get('jev'), 'TypeSafe-Danger-Zone-Images': 'true'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as res: a = json.loads(res.read())['answers']['material']
            return {'material': a.get('choice'), 'p': a.get('probabilities'), 'what': ''}
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429: return {'material': None, 'why': 'Jev refused %d %s' % (e.code, e.read()[:200].decode('utf-8', 'replace'))}
        except (urllib.error.URLError, TimeoutError): pass
        time.sleep(1 + 2 * attempt)
    return {'material': None, 'why': 'Jev failed 3 times'}
def ask(key, idx):
    p = card(key, idx)
    if p is None: return {'material': None, 'why': 'not visible in any view'}
    if key in CACHE: return CACHE[key]
    a = ask_jev(p) if JUDGE == 'jev' else ask_pi(key, idx, p)
    if a.get('material') is not None:
        with CLOCK: CACHE[key] = a; json.dump(CACHE, open(CACHE_F, 'w', encoding='utf-8'), indent=1)
    return a
def ask_pi(key, idx, p):
    ms = share[idx][share[idx].sum(1) > 0].mean(0) if (share[idx].sum(1) > 0).any() else np.zeros(len(con)); o = np.argsort(-ms)[:3]
    hint = ', '.join('%s %.2f' % (con[i], ms[i]) for i in o)
    try: r = subprocess.run([PY, RUN, '--no-session', '-p', '@' + p, Q.format(hint=hint)], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    except subprocess.TimeoutExpired: return {'material': None, 'why': 'no answer within 600 s'}   # left unlabelled: filled along the mesh
    m = re.search(r'\{.*\}', r.stdout, re.S)
    try: return json.loads(m.group(0)) if m else {'material': None, 'why': 'no answer: ' + (r.stdout[-150:] + ' / ' + r.stderr[-250:]).replace(chr(10), ' ')}
    except Exception: return {'material': None, 'why': 'bad json'}
def round_(segs):
    new = [k for k in segs if k not in CACHE]
    if len(new) > BUDGET: sys.exit('SG %d segments > LLM_MAX_CALLS %d' % (len(new), BUDGET))
    print('SG asking %s about %d segments (%d answered before), %d at once' % (JUDGE, len(new), len(segs) - len(new), W), flush=True); out = {}; lock = threading.Lock()
    def one(kv):
        k, idx = kv; a = ask(k, idx)
        with lock: out[k] = a; print('SG %-8s %7d vertices: %-6s %s' % (k, len(idx), a.get('material'), str(a.get('what', a.get('why', '')))[:60]), flush=True)
    with ThreadPoolExecutor(W) as ex: list(ex.map(one, segs.items()))
    return out
pu = np.unique(part)
if len(pu) > 1: segs = {'p%d' % k: np.flatnonzero(part == k) for k in pu}; print('SG segments = the model\'s own %d parts' % len(pu))
else:                                                             # one piece: shape first, then true colour (geo_colour_parts.py)
    gp = j('sam_body', 'geo_parts.npz')
    if not os.path.exists(gp): subprocess.run([PY, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'geo_colour_parts.py'), R], check=True)
    s1v = np.load(gp)['stage1']; sz1 = np.bincount(s1v[s1v >= 0]); u1 = np.flatnonzero(sz1)
    big1 = u1[sz1[u1] >= np.exp(otsu(np.log(sz1[u1])))]          # parts above the Otsu split of the log part sizes; the rest are filled along the mesh
    segs = {'b%d' % i: np.flatnonzero(s1v == i) for i in big1}
    print('SG one-piece model: %d big parts of %d (relief + SAM 3 + colour ranges), covering %.1f%% of the vertices' % (len(segs), len(u1), 100 * sz1[big1].sum() / sz1.sum()))
lab1 = round_(segs); mat = -np.ones(len(V), int); labels = {}
for k, a in lab1.items():
    labels[k] = dict(a, vertices=int(len(segs[k])))
    if a.get('material') in MN: mat[segs[k]] = MN.index(a['material'])
mixed = {k: segs[k] for k, a in lab1.items() if a.get('material') == 'mixed'}
if mixed:
    sub = {}
    for k, idx in mixed.items():
        for i, s in enumerate(colour_pieces(idx, 6)): sub['%s.%d' % (k, i)] = s
    lab2 = round_(sub)
    for k, a in lab2.items():
        labels[k] = dict(a, vertices=int(len(sub[k])))
        if a.get('material') in MN: mat[sub[k]] = MN.index(a['material'])
json.dump(labels, open(j('sam_body', 'segment_labels%s.json' % ('' if JUDGE == 'pi' else '_' + JUDGE)), 'w'), indent=1)
np.savez(j('sam_body', 'segment_material%s.npz' % ('' if JUDGE == 'pi' else '_' + JUDGE)), material=mat)
print('SG done: %s; unlabelled %d vertices' % ({MN[i]: int((mat == i).sum()) for i in range(4)}, (mat < 0).sum()))
