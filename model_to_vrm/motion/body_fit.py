"""Fit the stretched SOMA body to the twin: python body_fit.py <soma_stretched.npz> <twin_full.npz> <twin_layers.npz> <out.npz>
Rounds of: fix correspondences (KD-tree), then gradient steps on the body points (GPU).
  skin     a body point whose nearest twin surface is skin goes onto that skin point; a twin skin point pulls the body
           point nearest it when that body point is itself on skin (mutual pairs only: an open skin edge, like the top of
           a leg piece, cannot drag the hip into it)
  inside   a body point whose nearest twin surface is garment, hair or trim may not lie farther from its own bone than
           that surface point does (outward = away from the bone; Tripo's normals are not consistent, so not used)
  shape    every body edge keeps its stretched-body vector (a local change costs the distance it moves the edge)
All terms are mean distances in model units, so none needs a weight."""
import sys, numpy as np, torch
from scipy.spatial import cKDTree
B, T, L, out = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
TAB = {}
for i, n in enumerate(L['names']):
    for p in np.unique(L['part'][L['label'] == i]): TAB[int(p)] = str(n)
lay = np.array([TAB[int(p)] for p in T['part']]); keep = lay != 'organ'; Vt, lay = T['V'][keep], lay[keep]
skin = lay == 'skin'; kd_all = cKDTree(Vt); Vs_skin = Vt[skin]
V0 = B['V'].astype(float); F = B['F']; n = len(V0); J = B['J']
# each body point's bone: its strongest SOMA joint and that joint's farthest child (the segment the radial test uses)
li, lw = B['lbs_indices'], B['lbs_weights']; jb = li[np.arange(n), lw.argmax(1)]
import numpy as _np
S = _np.load(r'E:\code\ardy\ardy\assets\skeletons\somaskel77\skin_standard.npz'); con = S['rig_joint_connections']; kids = {}
for p_, c_ in con: kids.setdefault(int(p_), []).append(int(c_))
seg_end = np.array([J[max(kids[j], key=lambda c: np.linalg.norm(J[c] - J[j]))] if j in kids else J[j] for j in range(len(J))])
A0, A1 = J[jb], seg_end[jb]
def radial(P, a, b):   # unit vector from the segment a-b out to each point P
    ab = b - a; t = np.clip(((P - a) * ab).sum(1) / np.maximum((ab * ab).sum(1), 1e-12), 0, 1); q = a + t[:, None] * ab; u = P - q
    return u / np.maximum(np.linalg.norm(u, axis=1, keepdims=True), 1e-9)
E = np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0); Ei = torch.tensor(E, device=dev)
X = tt(V0).clone().requires_grad_(True); e0 = tt(V0[E[:, 1]] - V0[E[:, 0]])
for rnd in range(12):                                                             # NUMERIC: correspondence rounds
    Xn = X.detach().cpu().numpy(); d, i = kd_all.query(Xn); on_skin = skin[i]
    U = radial(Vt[i], A0, A1)                                                       # outward at the matched surface point
    kd_x = cKDTree(Xn); j = kd_x.query(Vs_skin)[1]; mutual = on_skin[j]
    ii, ms, Ut = torch.tensor(i, device=dev), torch.tensor(on_skin, device=dev), tt(U)
    jj, Ps = torch.tensor(j[mutual], device=dev), tt(Vs_skin[mutual]); G = tt(Vt[i])
    opt = torch.optim.Adam([X], lr=5e-4)                                          # NUMERIC: step (~0.5 mm)
    for it in range(150):                                                         # NUMERIC: steps per round
        e_skin = (X[ms] - G[ms]).norm(dim=1).mean() + (X[jj] - Ps).norm(dim=1).mean()
        e_in = torch.relu(((X - G) * Ut).sum(1))[~ms].mean()
        e_shape = ((X[Ei[:, 1]] - X[Ei[:, 0]]) - e0).norm(dim=1).mean()
        loss = e_skin + e_in + e_shape; opt.zero_grad(); loss.backward(); opt.step()
    print('BF round %2d: skin %.4f  outside %.4f  shape %.4f  (body points on skin %d, mutual skin pulls %d)' % (rnd, e_skin.item(), e_in.item(), e_shape.item(), on_skin.sum(), mutual.sum()), flush=True)
Xn = X.detach().cpu().numpy(); d, i = kd_all.query(Xn); U = radial(Vt[i], A0, A1); r = ((Xn - Vt[i]) * U).sum(1)
outside = (~skin[i]) & (r > 0); cov = cKDTree(Xn).query(Vs_skin)[0]
print('BF final: body points outside clothes/hair %d (max %.4f); twin skin covered: median %.4f, 95%% %.4f' % (outside.sum(), r[outside].max() if outside.any() else 0, np.median(cov), np.percentile(cov, 95)))
np.savez(out, V=Xn, F=F, J=J, names=B['names'], lbs_indices=li, lbs_weights=lw, on_skin=skin[i])
