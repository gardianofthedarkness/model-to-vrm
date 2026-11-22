"""Split a part that holds two things (skin and hair, a garment and its trim):
  python tools/split.py <part id>
The part's points are cut by colour: projected on their main colour direction (Lab), cut at the Otsu split, then each
side cut into connected pieces (8 nearest points). Pieces get new part ids (printed); the old id is gone. If the colours
are not two groups (Otsu's between-group share of the variance is below the share the part's own colour spread would
give by chance) nothing is changed. Pieces inherit the old part's node if it had one; reassign them as needed."""
import sys, os, json, numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
sys.path.insert(0, os.path.dirname(__file__)); from common import *
i = int(sys.argv[1]); f = os.path.join(WS, 'parts.npz'); Z = dict(np.load(f)); reg = Z['region']; m = np.flatnonzero(reg == i)
if len(m) < 10: sys.exit('part %d has %d points' % (i, len(m)))
O = np.load(CFG['colours']); V, C = O['V'][m], O['C'][m]
lab = cv2.cvtColor(C.astype(np.float32)[None], cv2.COLOR_RGB2LAB)[0]; lab -= lab.mean(0)
_, s, vt = np.linalg.svd(lab, full_matrices=False); x = lab @ vt[0]
xs = np.sort(x); n = len(xs); cs = np.cumsum(xs); k = np.arange(1, n); m0 = cs[:-1] / k; m1 = (cs[-1] - cs[:-1]) / (n - k)
bv = k * (n - k) * (m0 - m1) ** 2 / n ** 2; j = np.argmax(bv); cut = xs[j]; eta = bv[j] / xs.var()
# a single Gaussian blob cut at its middle gives eta = 2/pi; a real two-group split is above that
if eta <= 2 / np.pi: sys.exit('part %d is one colour group (separation %.2f <= %.2f); not split' % (i, eta, 2 / np.pi))
P = parts(); A = all_parts(); src = A[i]; t = tree(); node = t['assign'].pop(str(i), None); nxt = reg.max() + 1; made = []
for side in (x <= cut, x > cut):
    q = np.flatnonzero(side)
    if len(q) == 0: continue
    _, nb = cKDTree(V[q]).query(V[q], k=min(9, len(q))); r = np.repeat(np.arange(len(q)), nb.shape[1] - 1)
    nc, comp = connected_components(coo_matrix((np.ones(len(r)), (r, nb[:, 1:].ravel())), shape=(len(q), len(q))), directed=False)
    for u in range(nc):
        pts = m[q[comp == u]]; reg[pts] = nxt
        e = {'id': int(nxt), 'points': int(len(pts)), 'centre': O['V'][pts].mean(0).round(3).tolist(), 'colour': np.median(O['C'][pts], 0).round(3).tolist(),
             'split_from': i}
        if src['kind'] == 'colour region': e.update({k: src[k] for k in ('tripo_part', 'tripo_part_name', 'sam3_hint') if k in src}); P['colour_regions'].append(e)
        else: e.update({k: src[k] for k in ('concept', 'inside_concept', 'on_colour_region', 'tripo_part', 'mean_share')}); P['sam3_details'].append(e)
        if node: t['assign'][str(nxt)] = node
        made.append((nxt, len(pts), hexc(e['colour']))); nxt += 1
P['colour_regions'] = [p for p in P['colour_regions'] if p['id'] != i]; P['sam3_details'] = [p for p in P['sam3_details'] if p['id'] != i]
Z['region'] = reg; np.savez(f, **Z); json.dump(P, open(os.path.join(WS, 'part_list.json'), 'w'), indent=1); save_tree(t)
print('part %d split (separation %.2f) into: %s' % (i, eta, ', '.join('%d (%d pts, %s)' % mm for mm in made)))
