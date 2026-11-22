"""The T-pose twin corrected to the original's shape and coloured from its texture:
  python style_correct.py <ab_free.npz> <twin_rig.npz> <orig_colours.npz> <twin_full.npz> <out.npz>
  unpose   each original point (full detail) is matched to the nearest point of the bent twin within its paired part and
           carried back to T-pose through that twin point's own turn (its skin weights' blend of the fitted bone turns,
           made a rotation): x_T = twin_rest + R^T (x - twin_bent)
  trust    a match is trusted where its reach is short: reaches split by Otsu on their logarithm (as in carry_weights.py)
  shape    each twin point's correction is the mean gap to the unposed original points that match it (trusted only);
           twin points with none get theirs by harmonic fill over the twin's welded surface (hidden places keep the
           twin's own shape, smoothly joined)
  colour   each full-detail twin point takes the colour of the nearest unposed original point of its paired part(s)
  check    the corrected twin bent again by the same fit, against the original: median and 95% miss, before and after"""
import sys, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sl
from scipy.spatial import cKDTree
from scipy.sparse.csgraph import connected_components
fit, rig, oc, tf, out = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), np.load(sys.argv[4]), sys.argv[5]
V0, Vb, W, pt = fit['V0'], fit['Vbent'], fit['W'], fit['part_t']; Rg = fit['Rg']; pair = dict(fit['pair'].tolist())
Xo, Co, po = oc['V'], oc['C'], oc['part']
def otsu(v):
    v = np.sort(np.asarray(v, float)); n = len(v); c = np.cumsum(v); i = np.arange(1, n); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (n - i)
    return v[np.argmax(i * (n - i) * (m0 - m1) ** 2)]
# each twin point's turn at the fitted pose
M = np.einsum('nj,jab->nab', W, Rg); U, _, Vt = np.linalg.svd(M); R = U @ Vt; bad = np.linalg.det(R) < 0
U[bad, :, -1] *= -1; R = U @ Vt
# match and unpose
mi = np.zeros(len(Xo), int); reach = np.zeros(len(Xo))
for p in np.unique(po):
    mo = np.flatnonzero(po == p); mt = np.flatnonzero(pt == pair[int(p)]); d, i = cKDTree(Vb[mt]).query(Xo[mo]); mi[mo] = mt[i]; reach[mo] = d
XT = V0[mi] + np.einsum('nba,nb->na', R[mi], Xo - Vb[mi])                         # R^T (x - bent)
cut = np.exp(otsu(np.log(np.maximum(reach[::5], 1e-6)))); trust = reach <= cut
print('SC original points %d unposed; trusted where reach <= %.4f (Otsu): %d (%.0f%%)' % (len(Xo), cut, trust.sum(), 100 * trust.mean()))
# shape correction on the twin rig points
n = len(V0); acc = np.zeros((n, 3)); cnt = np.zeros(n)
np.add.at(acc, mi[trust], XT[trust] - V0[mi[trust]]); np.add.at(cnt, mi[trust], 1)
known = cnt > 0; Dsp = np.zeros((n, 3)); Dsp[known] = acc[known] / cnt[known][:, None]
F = rig['F']; _, uid = np.unique(V0.round(7), axis=0, return_inverse=True); uid = uid.ravel()
E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]); first = np.full(uid.max() + 1, -1); first[uid[::-1]] = np.arange(n)[::-1]
E = np.vstack([E, np.c_[np.arange(n), first[uid]]])
# also join each point to its nearest neighbours (the rig copy is torn by decimation; the field must still be one piece per part)
d_, nb = cKDTree(V0).query(V0, k=4); same = pt[nb] == pt[:, None]; E = np.vstack([E] + [np.c_[np.arange(n), nb[:, k]][same[:, k]] for k in range(1, 4)])
A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); A = ((A + A.T) > 0).astype(float).tocsr(); Lp = sp.diags(np.asarray(A.sum(1)).ravel()) - A
nc, comp = connected_components(A, directed=False); filled = 0
for c in np.unique(comp[~known]):
    m = comp == c; u = np.flatnonzero(m & ~known); k = np.flatnonzero(m & known)
    if len(k) == 0: continue
    Dsp[u] = sl.splu(Lp[u][:, u].tocsc()).solve(-(Lp[u][:, k] @ Dsp[k])); filled += len(u)
print('SC twin points %d: %d with a trusted match, %d filled smoothly (%d pieces); correction median %.4f, 95%% %.4f' % (n, known.sum(), filled, nc, np.median(np.linalg.norm(Dsp, axis=1)), np.percentile(np.linalg.norm(Dsp, axis=1), 95)))
# check: bend the corrected twin and compare with the original (rig-copy points vs the original's full points)
H0, P = fit['head0'], fit['head']; Vc = V0 + Dsp
def bend(X): return np.einsum('nj,jab,njb->na', W, Rg, X[:, None, :] - H0[None]) + (W @ P)
Vb2 = bend(Vc); kd_o = cKDTree(Xo)
for name, B_ in (('before', Vb), ('after', Vb2)):
    d = kd_o.query(B_)[0]; print('SC %s: twin bent onto the original, miss median %.4f, 95%% %.4f' % (name, np.median(d), np.percentile(d, 95)))
# full-detail twin: correction by its nearest rig points (inverse distance, same part), colour by the nearest unposed original
Vf, Pf = tf['V'], tf['part']; Df = np.zeros_like(Vf); Cf = np.zeros_like(Vf)
inv_pair = {}
for o, t in pair.items(): inv_pair.setdefault(t, []).append(o)
for p in np.unique(Pf):
    mf = np.flatnonzero(Pf == p); mr = np.flatnonzero(pt == p)
    if len(mr): d, i = cKDTree(V0[mr]).query(Vf[mf], k=4); w = 1 / np.maximum(d, 1e-9); w /= w.sum(1, keepdims=True); Df[mf] = (w[..., None] * Dsp[mr][i]).sum(1)
    src = np.flatnonzero(np.isin(po, inv_pair.get(int(p), [])) & trust)
    if len(src) == 0: src = np.flatnonzero(trust)
    Cf[mf] = Co[src][cKDTree(XT[src]).query(Vf[mf] + Df[mf])[1]]
np.savez(out, V=Vf + Df, F=tf['F'], part=Pf, C=Cf.astype(np.float32), D=Df.astype(np.float32), rig_D=Dsp, XT=XT.astype(np.float32), trust=trust)
print('SC full twin %d points corrected and coloured' % len(Vf))
