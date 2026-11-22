"""Flesh on the bones with NVIDIA Newton (VBD, neo-Hookean tets), baked:
  E:\\splat\\.venv-sim\\Scripts\\python.exe newton_flesh.py <tetwild.npz> <body_fit.npz> <drive.npz> <out.npz> [substeps] [iterations]
  weights  boundary nodes take the nearest skin point's SOMA weights; the interior is harmonic over the tet mesh
  pins     every interior node nearer a bone than the skin: zero mass, placed by skinning each substep
  flesh    Newton's VBD solver on the tets: neo-Hookean, soft tissue E 30 kPa, Poisson 0.45, density 1000 kg/m^3
           (textbook values); no gravity (the rest shape already stands under it)
  start    at rest (T-pose), eased into the clip's first frame over one second, then the clip
Writes the surface per clip frame (V), the skinning-only surface (lbs), inverted tets and volume per frame."""
import sys, time, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sl, warp as wp, newton
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.sparse.csgraph import connected_components
Tm, B, Dv, out = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]
nsub = int(sys.argv[5]) if len(sys.argv) > 5 else 10; nit = int(sys.argv[6]) if len(sys.argv) > 6 else 10   # NUMERIC: Newton example defaults
X0, T = Tm['X'].astype(np.float64), Tm['T']; J = B['J']; nJ = len(J); N = len(X0); Vb = B['V'].astype(np.float64); ns = len(Vb)
emb_t, emb_b = Tm['emb_tet'], Tm['emb_bary']                                      # the body's skin, embedded in the tets
li, lw = B['lbs_indices'], B['lbs_weights']
Ws = np.zeros((ns, nJ)); np.add.at(Ws, (np.repeat(np.arange(ns), li.shape[1]), li.ravel()), lw.ravel()); Ws /= Ws.sum(1, keepdims=True)
E_ = np.unique(np.sort(np.vstack([T[:, [a, b]] for a in range(4) for b in range(a + 1, 4)]), 1), axis=0)
A = sp.coo_matrix((np.ones(len(E_)), (E_[:, 0], E_[:, 1])), shape=(N, N)); A = (A + A.T).tocsr()
# boundary nodes (on faces used by one tet) take the nearest skin point's weights; the interior is harmonic between them
Fc = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), 1); uf, cf = np.unique(Fc, axis=0, return_counts=True)
bnd = np.zeros(N, bool); bnd[uf[cf == 1].ravel()] = True
W = np.zeros((N, nJ)); kdb = cKDTree(Vb); W[bnd] = Ws[kdb.query(X0[bnd])[1]]; known = bnd.copy()
nc, comp = connected_components(A, directed=False); lone = np.zeros(N, bool)
for c_ in np.unique(comp[~known]):
    m_ = comp == c_
    if not (m_ & known).any(): lone |= m_ & ~known
W[lone] = Ws[kdb.query(X0[lone])[1]]; known |= lone
fixed, free = np.flatnonzero(known), np.flatnonzero(~known); Lp = sp.diags(np.asarray(A.sum(1)).ravel()) - A
W[free] = np.clip(sl.splu(Lp[free][:, free].tocsc()).solve(-(Lp[free][:, fixed] @ W[fixed])), 0, None); W /= W.sum(1, keepdims=True)
S = np.load('E:/code/ardy/ardy/assets/skeletons/somaskel77/skin_standard.npz'); con = S['rig_joint_connections']
def segdist(P):
    best = np.full(len(P), np.inf)
    for p, q in zip(J[con[:, 0]], J[con[:, 1]]):
        ab = q - p; t = np.clip(((P - p) @ ab) / max(ab @ ab, 1e-12), 0, 1)
        best = np.minimum(best, np.linalg.norm(P - (p + t[:, None] * ab), axis=1))
    return best
pin = (segdist(X0) < kdb.query(X0)[0]) & ~bnd; pin |= lone
def skin(q): return (emb_b[..., None] * q[T[emb_t]]).sum(1)                       # the body's skin from the tet nodes
pidx = np.flatnonzero(pin); print('NF nodes %d (boundary %d), tets %d; pinned %d' % (N, bnd.sum(), len(T), len(pidx)), flush=True)
fps = float(Dv['fps']); K = int(round(fps))                                         # NUMERIC: one second of easing in
D_, P_ = Dv['D'], Dv['P']; r1 = Rot.from_matrix(D_[0]); I3 = Rot.from_matrix(np.eye(3))
Dw = np.stack([np.stack([Slerp([0, 1], Rot.concatenate([I3, r1[j]]))([s / K]).as_matrix()[0] for j in range(nJ)]) for s in range(K)])
Pw = np.stack([J + (P_[0] - J) * s / K for s in range(K)])
Dall, Pall = np.concatenate([Dw, D_]), np.concatenate([Pw, P_]); nF = len(Dall)
def lbs(t, idx):
    Y = np.einsum('jab,njb->nja', Dall[t], X0[idx][:, None, :] - J[None]) + Pall[t][None]; return (W[idx][..., None] * Y).sum(1)
def lbs_body(t):   # the skin by skinning alone (the reference the flesh is compared with)
    Y = np.einsum('jab,njb->nja', Dall[t], Vb[:, None, :] - J[None]) + Pall[t][None]; return (Ws[..., None] * Y).sum(1)
E, nu, rho = 30e3, 0.45, 1000.0; mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
builder = newton.ModelBuilder(); builder.gravity = (0.0, 0.0, 0.0)
builder.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=X0.tolist(), indices=T.ravel().tolist(),
                      density=rho, k_mu=mu, k_lambda=lam, k_damp=0.0)
for i in pidx: builder.particle_mass[i] = 0.0
builder.color(); model = builder.finalize()
solver = newton.solvers.SolverVBD(model=model, iterations=nit, particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0, s1, ctl = model.state(), model.state(), model.control()
pidx_w = wp.array(pidx, dtype=wp.int32); ptgt = wp.zeros(len(pidx), dtype=wp.vec3)
@wp.kernel
def place(q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3), idx: wp.array(dtype=wp.int32), tgt: wp.array(dtype=wp.vec3), dt: float):
    i = wp.tid(); k = idx[i]; qd[k] = (tgt[i] - q[k]) / dt; q[k] = tgt[i]
dt = 1.0 / fps / nsub; Tv = T; outV, inv, vr = [], [], []
Dm0 = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2); vol0 = np.abs(np.linalg.det(Dm0)) / 6; Dinv0 = np.linalg.inv(Dm0)
prev = X0[pidx].copy(); t0 = time.time()
for f in range(nF):
    nxt = lbs(f, pidx)
    for s in range(nsub):
        ptgt.assign((prev + (nxt - prev) * (s + 1) / nsub).astype(np.float32))
        wp.launch(place, dim=len(pidx), inputs=[s0.particle_q, s0.particle_qd, pidx_w, ptgt, dt])
        s0.clear_forces(); solver.step(s0, s1, ctl, None, dt); s0, s1 = s1, s0
    prev = nxt
    q = s0.particle_q.numpy().astype(np.float64)
    if not np.isfinite(q).all(): print('NF frame %d: not finite, stopping' % (f - K)); break
    Ds = np.stack([q[T[:, 1]] - q[T[:, 0]], q[T[:, 2]] - q[T[:, 0]], q[T[:, 3]] - q[T[:, 0]]], 2); dets = np.linalg.det(Ds @ Dinv0)
    inv.append(int((dets <= 0).sum())); vr.append(float((dets * vol0).sum() / vol0.sum()))
    if f >= K: outV.append(skin(q).astype(np.float32))
    if f >= K and (f - K) % 60 == 0: np.save(out.replace('.npz', '_partial.npy'), np.stack(outV))   # for looking at it partway
    if f % 30 == 0:
        print('NF frame %3d%s: inverted %d, volume %.4f, off skinning %.4f  (%.0f s)' % (f - K, ' (easing in)' if f < K else '', inv[-1], vr[-1],
              np.linalg.norm(skin(q) - lbs_body(f), axis=1).mean(), time.time() - t0), flush=True)
outV = np.stack(outV); lbsV = np.stack([lbs_body(f) for f in range(K, K + len(outV))]).astype(np.float32)
dv = np.linalg.norm(outV - lbsV, axis=2); inv = np.array(inv[K:]); vr = np.array(vr[K:])
print('NF done in %.0f s: substeps %d, iterations %d; inverted tets max %d; volume %.4f..%.4f; surface vs skinning: mean %.4f, 99%% %.4f, max %.4f' % (
    time.time() - t0, nsub, nit, inv.max(), vr.min(), vr.max(), dv.mean(), np.percentile(dv, 99), dv.max()))
np.savez(out, V=outV, lbs=lbsV, inverted=inv, volume=vr, pin=pin)
