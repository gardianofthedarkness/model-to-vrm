"""The body filled with fTetWild tetrahedra, and the body's own skin embedded in them:
  E:\\splat\\.venv-sim\\Scripts\\python.exe body_tetwild.py <body_fit.npz> <out.npz>
  size     ideal tet edge = half the body's median flesh thickness (skin point to nearest bone), so flesh is >= 2 tets deep
  embed    each skin point of the body lies in (or nearest to) one tet and moves with it by barycentric weights; the
           baked skin is the stage-2 body itself, not fTetWild's own surface
Writes X (nodes), T (tets), and for the skin: emb_tet, emb_bary."""
import sys, numpy as np, pytetwild
from scipy.spatial import cKDTree
B = np.load(sys.argv[1]); out = sys.argv[2]; V, F, J = B['V'].astype(np.float64), B['F'].astype(np.int32), B['J']
S = np.load('E:/code/ardy/ardy/assets/skeletons/somaskel77/skin_standard.npz'); con = S['rig_joint_connections']
def segdist(P):
    best = np.full(len(P), np.inf)
    for p, q in zip(J[con[:, 0]], J[con[:, 1]]):
        ab = q - p; t = np.clip(((P - p) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(P - (p + t[:, None] * ab), axis=1))
    return best
thick = np.median(segdist(V)); h = thick / 2
X, T = pytetwild.tetrahedralize(V, F, edge_length_abs=h)
X = np.asarray(X, np.float64); T = np.asarray(T, np.int64)
P = X[T]; vol = np.einsum('ij,ij->i', np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0]) / 6
T[vol < 0] = T[vol < 0][:, [0, 2, 1, 3]]; P = X[T]; vol = np.abs(vol)
L = np.stack([np.linalg.norm(P[:, a] - P[:, b], axis=1) for a in range(4) for b in range(a + 1, 4)], 1)
q = 6 * np.sqrt(2) * vol / (L ** 2).mean(1) ** 1.5
print('TW flesh thickness median %.4f -> tet edge %.4f; nodes %d, tets %d; quality min %.3f, 1%% %.3f, median %.3f; volume %.5f' % (
    thick, h, len(X), len(T), q.min(), np.percentile(q, 1), np.median(q), vol.sum()))
# embed the skin: candidate tets by centre distance, pick the one whose barycentric weights are least negative
C = P.mean(1); kd = cKDTree(C); _, cand = kd.query(V, k=16)
Dm = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], 2); Di = np.linalg.inv(Dm)
best_t = np.zeros(len(V), int); best_b = np.zeros((len(V), 4))
for k in range(cand.shape[1]):
    t = cand[:, k]; l = np.einsum('nij,nj->ni', Di[t], V - P[t, 0]); b = np.c_[1 - l.sum(1), l]
    score = b.min(1); better = (k == 0) | (score > best_b.min(1))
    best_t[better], best_b[better] = t[better], b[better]
print('TW skin embedded: %d points; inside a tet %d, worst outside weight %.3f' % (len(V), (best_b.min(1) >= -1e-6).sum(), best_b.min()))
np.savez(out, X=X, T=T, h=h, emb_tet=best_t, emb_bary=best_b)
