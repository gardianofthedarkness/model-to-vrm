"""Carve a unit into colour regions:  python tools/split.py <unit id>
  colours   the unit's texture colours clustered in Lab, cluster count by BIC (1..6)
  pieces    each cluster cut into connected pieces (each point joined to its 8 nearest points of the same cluster)
  tiny      pieces smaller than the split Otsu finds in the pieces' log sizes are merged into the region their points'
            nearest neighbours mostly belong to
The unit is replaced by regions <unit>.1, <unit>.2, ...; a card is made for each. Undo: python tools/merge.py <unit>."""
import sys, numpy as np, ws
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture
uid = sys.argv[1]; o = ws.owner(); idx = np.flatnonzero(o == uid)
if len(idx) < 20: sys.exit('%s: too few points to split (%d)' % (uid, len(idx)))
X = ws.lab(ws.COL[idx]); P = ws.V[idx]; _, nb16 = cKDTree(P).query(P, k=min(16, len(idx)))
for it in range(3): X = X[nb16].mean(1)                                                   # NUMERIC: smoothing - each colour the mean of its 16 nearest points, 3 passes (texture noise out, region edges kept)
samp = X[np.random.default_rng(0).choice(len(X), min(len(X), 8000), replace=False)]
g = min((GaussianMixture(k, random_state=0).fit(samp) for k in range(1, 7)), key=lambda g: g.bic(samp)); cl = g.predict(X)
_, nb = cKDTree(P).query(P, k=9); r = np.repeat(np.arange(len(idx)), 8); c = nb[:, 1:].ravel(); same = cl[r] == cl[c]
A = coo_matrix((np.ones(same.sum()), (r[same], c[same])), shape=(len(idx), len(idx))); n, comp = connected_components(A, directed=False)
lab = comp.copy()
for rnd in range(20):                                                                     # merge small pieces into the neighbours they touch most, until only the big class (Otsu) is left
    ids, sizes = np.unique(lab, return_counts=True)
    if len(ids) < 2: break
    cut = np.exp(ws.otsu(np.log(sizes))); small = set(ids[sizes <= cut].tolist())
    if not small or len(small) == len(ids): break
    for i in np.flatnonzero(np.isin(lab, list(small))):
        nl = lab[nb[i, 1:]]; nl = nl[~np.isin(nl, list(small))]
        if len(nl): lab[i] = np.bincount(nl).argmax()
keep = np.unique(lab[lab >= 0]); new = {k: '%s.%d' % (uid, j + 1) for j, k in enumerate(keep)}
o = o.astype('<U12');
for k, nid in new.items(): o[idx[lab == k]] = nid
o[idx[lab < 0]] = uid + '.%d' % (len(keep) + 1) if (lab < 0).any() else o[idx[lab < 0]]
ws.save_owner(o); s = ws.sheet(); s.setdefault(uid, {})['split_into'] = sorted(set(o[idx].tolist()))
for nid in sorted(set(o[idx].tolist())):
    j = np.flatnonzero(o == nid); p = ws.card(nid, j); c = ws.candidates(j); s[nid] = {'parent': uid, 'points': int(len(j)), 'candidates': c}
    print('%s: %d points; card %s; candidates %s' % (nid, len(j), p, ', '.join('%s (%.0f%%)' % (x['hex'], 100 * x['share']) for x in c)))
ws.save_sheet(s); print('split %s into %d regions (%d colour clusters by BIC)' % (uid, len(set(o[idx].tolist())), g.n_components))
