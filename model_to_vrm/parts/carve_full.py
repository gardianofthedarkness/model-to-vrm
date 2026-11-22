"""Candidate subparts on the full-detail original, two kinds: python carve_full.py <orig_colours.npz> <orig_normals.npz> <out.npz>
  graph   per Tripo part: points joined by triangle edges; points at the same place (texture seams) joined too
  colour  each point's Lab colour smoothed over its graph neighbours (3 passes); clustered (count by BIC, 1..12); clusters cut
          into connected pieces; pieces below the Otsu split of the pieces' log sizes merged into the piece they touch most
          (one round); edges cleaned by majority vote over graph neighbours (3 passes)
  shape   depth = n . (mean of the point's 2-ring neighbours - point): > 0 the point sits below its surroundings (a dent),
          < 0 above (a bump); per part, dents and bumps each split from the rest by Otsu on |depth|; then pieces, merging
          and edge cleaning as above
Writes colour_region, dent_region, bump_region (global ids, -1 = none), and for colour regions their median colour."""
import sys, numpy as np, cv2, scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture
C_, N_, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
V, COL, P = C_['V'], C_['C'], C_['part']; Nn, F = N_['N'], N_['F']; n = len(V)
def otsu(v):
    v = np.sort(np.asarray(v, float))
    if len(v) < 3: return v[0] if len(v) else 0
    m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i)
    return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
def graph(idx):
    loc = -np.ones(n, int); loc[idx] = np.arange(len(idx)); f = F[(loc[F] >= 0).all(1)]; f = loc[f]
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    _, u = np.unique(V[idx].round(7), axis=0, return_inverse=True); u = u.ravel(); first = np.full(u.max() + 1, -1); first[u[::-1]] = np.arange(len(idx))[::-1]
    e = np.vstack([e, np.c_[np.arange(len(idx)), first[u]]]); m = len(idx)
    A = sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(m, m)); A = ((A + A.T) > 0).astype(np.float32).tocsr(); A.setdiag(0); A.eliminate_zeros(); return A
def smooth(A, X, k):
    deg = np.asarray(A.sum(1)).ravel() + 1
    for _ in range(k): X = (A @ X + X) / deg[:, None]
    return X
def pieces(A, lab):
    """connected pieces of equal label, small pieces merged into the neighbour they touch most, edges cleaned"""
    r, c = A.nonzero(); same = lab[r] == lab[c]; m = len(lab)
    B = sp.coo_matrix((np.ones(same.sum()), (r[same], c[same])), shape=(m, m)); _, comp = connected_components(B, directed=False)
    for rnd in range(1):                                                                 # one merge round: repeating it re-splits the shrinking set and collapses everything (test model R, 2026-11-19)
        ids, sz = np.unique(comp, return_counts=True)
        if len(ids) < 2: break
        cut = np.exp(otsu(np.log(sz))); small = ids[sz <= cut]
        if len(small) == 0 or len(small) == len(ids): break
        smask = np.isin(comp, small); rr, cc = r[smask[r] & ~smask[c]], c[smask[r] & ~smask[c]]
        if len(rr) == 0: break
        best = {}
        for a_, b_ in zip(rr, cc): best.setdefault(comp[a_], []).append(comp[b_])
        for s_, lst in best.items(): comp[comp == s_] = np.bincount(lst).argmax()
    for _ in range(3):                                                                    # edge cleaning: majority of the graph neighbours
        Y = sp.coo_matrix((np.ones(len(r)), (r, comp[c])), shape=(m, comp.max() + 1)).tocsr(); maj = np.asarray(Y.argmax(1)).ravel(); cnt = np.asarray(Y.max(1).todense()).ravel()
        deg = np.asarray(A.sum(1)).ravel(); flip = cnt > deg / 2; comp = np.where(flip, maj, comp)
    return comp
colour_region = -np.ones(n, int); dent_region = -np.ones(n, int); bump_region = -np.ones(n, int); next_c = next_d = next_b = 0; med = {}
for p in np.unique(P):
    idx = np.flatnonzero(P == p); A = graph(idx)
    lab = cv2.cvtColor((np.clip(COL[idx], 0, 1)[None] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(np.float32); Xs = smooth(A, lab, 3)
    smp = Xs[np.random.default_rng(0).choice(len(idx), min(len(idx), 10000), replace=False)]
    g = min((GaussianMixture(k, random_state=0).fit(smp) for k in range(1, 13)), key=lambda g: g.bic(smp)); comp = pieces(A, g.predict(Xs))
    for u in np.unique(comp):
        m = idx[comp == u]; colour_region[m] = next_c; med[next_c] = np.median(COL[m], 0); next_c += 1
    A2 = A @ A; A2.setdiag(0); A2.eliminate_zeros(); A2.data[:] = 1; deg = np.asarray(A2.sum(1)).ravel()
    depth = ((A2 @ V[idx]) / np.maximum(deg, 1)[:, None] - V[idx]) * Nn[idx]; depth = depth.sum(1)
    for sign, store, name in ((1, dent_region, 'dent'), (-1, bump_region, 'bump')):
        d = sign * depth; pos = d > 0
        if pos.sum() < 20: continue
        thr = otsu(d[pos]); sel = (d > thr).astype(int)
        comp = pieces(A, sel); keep = [u for u in np.unique(comp) if sel[comp == u].mean() > 0.5]
        for u in keep:
            m = idx[comp == u]
            if name == 'dent': dent_region[m] = next_d; next_d += 1
            else: bump_region[m] = next_b; next_b += 1
    print('CF part %2d: %6d points; colour regions so far %d (this part %d clusters); dents %d bumps %d' % (p, len(idx), next_c, g.n_components, next_d, next_b), flush=True)
np.savez(out, colour_region=colour_region, dent_region=dent_region, bump_region=bump_region, colour_median=np.array([med[i] for i in range(next_c)]))
print('CF done: %d colour regions, %d dents, %d bumps' % (next_c, next_d, next_b))
