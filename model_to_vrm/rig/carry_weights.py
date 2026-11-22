"""The bent twin's skin weights onto the original's points: python carry_weights.py <fit.npz> <orig.npz> <out.npz>
  carry    each original point takes the weights of the three nearest bent-twin points of its paired part (a surface point is
           spanned by the three corners of the triangle it lies on), each by inverse distance
  trust    a carried weight is trusted where its reach is short: reaches split in two by Otsu on their logarithm (the data's
           own split between "lies on the twin" and "lies off it", as the pipeline splits a picture's tones)
  fill     the untrusted points get their weights by diffusion over the original's own surface from the trusted points of the
           same piece (harmonic: each such point the mean of its neighbours), as in robust skin-weight transfer by weight
           inpainting (Abdrashitov et al., SIGGRAPH Asia 2023); a piece with no trusted point keeps what it carried"""
import sys, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sl
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
fit, org, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
Vb, W, pt = fit['Vbent'], fit['W'], fit['part_t']; pair = dict(fit['pair'].tolist()); Vo, po, F = org['V'], org['part'], org['F']
Wo = np.zeros((len(Vo), W.shape[1])); reach = np.zeros(len(Vo))
for p in np.unique(po):
    mo = np.flatnonzero(po == p); mt = np.flatnonzero(pt == pair[int(p)]); d, i = cKDTree(Vb[mt]).query(Vo[mo], k=3)
    w = 1 / np.maximum(d, 1e-9); w /= w.sum(1, keepdims=True); Wo[mo] = (w[..., None] * W[mt][i]).sum(1); reach[mo] = d[:, 0]
def otsu(v):
    v = np.sort(np.asarray(v, float).ravel()); n = len(v); c = np.cumsum(v); i = np.arange(1, n); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (n - i)
    return v[np.argmax(i * (n - i) * (m0 - m1) ** 2)]
lr = np.log(np.maximum(reach, 1e-6)); cut = np.exp(otsu(lr)); trust = reach <= cut
# the surface graph (uniform weights): neighbours share an edge; points at the same place (split along texture seams by the
# file format) are joined, so the surface is one piece wherever the model's surface is
_, uid = np.unique(Vo, axis=0, return_inverse=True); uid = uid.ravel(); n = len(Vo)
E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); E = np.vstack([E, np.c_[np.arange(n), np.zeros(n, int)]])
first = np.full(uid.max() + 1, -1); first[uid[::-1]] = np.arange(n)[::-1]; E[-n:, 1] = first[uid]          # each point tied to the first point at its place
A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); A = ((A + A.T) > 0).astype(float).tocsr()
L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
nc, comp = connected_components(A, directed=False); filled = 0
U = np.flatnonzero(~trust)
for c in np.unique(comp[U]):
    m = comp == c; u = np.flatnonzero(m & ~trust); k = np.flatnonzero(m & trust)
    if len(k) == 0: continue
    Luu = L[u][:, u].tocsc(); Luk = L[u][:, k]; Wo[u] = sl.splu(Luu).solve(-(Luk @ Wo[k])); filled += len(u)
Wo = np.clip(Wo, 0, None); Wo /= Wo.sum(1, keepdims=True)
print('CW %d points; reach median %.4f, 95%% %.4f; trusted where reach <= %.4f (Otsu): %d untrusted, %d filled over the surface (%d pieces)' % (n, np.median(reach), np.percentile(reach, 95), cut, len(U), filled, nc))
np.savez(out, W=Wo, names=fit['names'], parent=fit['parent'], head=fit['head'], tail=fit['tail'], reach=reach, trust=trust)
