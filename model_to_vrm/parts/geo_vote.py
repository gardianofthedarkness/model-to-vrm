"""SAM 3 masks from bare-geometry views voted onto the head's vertices, then claimed per concept:
  python geo_vote.py <views dir>   ->  <dir>/organs.npz (label per head point, names), <dir>/kept_<view>.jpg panels
  found   a concept counts only if its best score over all views is above the Otsu split of all concepts' best scores
  keep    per concept, a mask counts when its score is above the Otsu split of that concept's pooled scores (all views),
          and a view's top mask counts when it is above the pool's median
  vote    a counted mask adds its score to the head points the view sees (ray-cast visibility) inside it;
          share = votes / number of views that asked the concept and see the point
  claim   a point is claimed by a concept when its share clears the Otsu split of that concept's nonzero shares;
          concepts are tried from the fewest claimed points (most specific) to the most; 'face' and 'hair' last
  pieces  connected pieces per concept (8 nearest points); pieces below the Otsu split of the concept's log piece sizes
          are dropped"""
import sys, os, json, numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
D = sys.argv[1]; md = os.path.join(D, 'masks'); cams = json.load(open(os.path.join(D, 'cameras.json'))); HP = np.load(os.path.join(D, 'head_points.npz'))
V, VIS = HP['V'], HP['vis']; Ph = np.c_[V, np.ones(len(V))]
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[0] - 1e-9 if m else 1
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
F = {f[:-4]: np.load(os.path.join(md, f)) for f in sorted(os.listdir(md)) if f.endswith('.npz')}
ALL = sorted({k[2:] for z in F.values() for k in z.files if k.startswith('s_')})
pool = {c: np.concatenate([z['s_' + c] for z in F.values() if 's_' + c in z.files]) for c in ALL}
split = {c: otsu(pool[c]) for c in ALL}; med = {c: np.median(pool[c]) for c in ALL}
best = {c: pool[c].max() for c in ALL}; found = otsu(list(best.values()))                 # concepts SAM 3 never really saw
ALL = [c for c in ALL if best[c] > found]; print('GVOTE dropped (best score below %.2f): %s' % (found, [c for c in best if c not in ALL]))
votes = np.zeros((len(V), len(ALL)), np.float32); seen = np.zeros((len(V), len(ALL)), np.float32)
for name, z in F.items():
    cm = cams[name]; R = cm['res']; q = Ph @ np.array(cm['matrix_inv']).T; u = (q[:, 0] / cm['scale'] + 0.5) * R; v = (0.5 - q[:, 1] / cm['scale']) * R
    idx = np.flatnonzero(VIS[cm['view']] & (u >= 0) & (v >= 0) & (u < R) & (v < R)); pix = v[idx].astype(int) * R + u[idx].astype(int)
    base = cv2.imread(os.path.join(D, name + '.png')); panels = []
    for c in ALL:
        if 's_' + c not in z.files: continue
        ci = ALL.index(c); seen[idx, ci] += 1; S = z['s_' + c]
        if len(S) == 0: continue
        keep = S > split[c]; keep[np.argmax(S)] |= S.max() > med[c]
        if not keep.any(): continue
        M = np.unpackbits(z['m_' + c][keep], axis=1, count=R * R).astype(bool); best = (M * S[keep][:, None]).max(0); votes[idx, ci] += best[pix]
        mk = best.reshape(R, R) > 0; pn = base.copy(); pn[mk] = (pn[mk] * 0.35 + np.array([255, 0, 255]) * 0.65).astype(np.uint8)
        pn = cv2.resize(pn, (320, 320)); cv2.putText(pn, '%s %.2f' % (c, S[keep].max()), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2); panels.append(pn)
    while len(panels) % 4: panels.append(np.full((320, 320, 3), 255, np.uint8))
    cv2.imwrite(os.path.join(D, 'kept_' + name + '.jpg'), np.vstack([np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
share = np.where(seen > 0, votes / np.maximum(seen, 1), 0)
claimed = {}
for ci in range(len(ALL)):
    t = share[:, ci][share[:, ci] > 0]
    if len(t) >= 3: claimed[ci] = share[:, ci] > otsu(t)
REGION = {'face', 'hair'}                                   # whole regions: everything else sits on them, so they go last
size = {ci: (claimed[ci].sum() + (1e12 if ALL[ci] in REGION else 0)) for ci in claimed}
order = sorted(claimed, key=lambda ci: size[ci]); lab = -np.ones(len(V), int)
_, nb = cKDTree(V).query(V, k=9)
for ci in order:
    m = np.flatnonzero((lab < 0) & claimed[ci])
    if len(m) < 2: continue
    sub = -np.ones(len(V), int); sub[m] = np.arange(len(m)); e = sub[nb[m, 1:]]; r = np.repeat(np.arange(len(m)), 8); ok = e.ravel() >= 0
    n, comp = connected_components(coo_matrix((np.ones(ok.sum()), (r[ok], e.ravel()[ok])), shape=(len(m), len(m))), directed=False)
    sz = np.bincount(comp); keep = sz > np.exp(otsu(np.log(sz))) if n >= 3 else np.ones(n, bool)
    lab[m[keep[comp]]] = ci; print('GVOTE %-8s split %.2f  %6d points in %d pieces (of %d)' % (ALL[ci], split[ALL[ci]], keep[comp].sum(), keep.sum(), n))
np.savez(os.path.join(D, 'organs.npz'), label=lab, names=np.array(ALL), share=share, idx=HP['idx'])
print('GVOTE done')
