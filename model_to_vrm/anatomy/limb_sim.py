"""Move a built limb segment (limb_build.py) with flesh simulation - NVIDIA Newton VBD on fTetWild tets:
  E:\\splat\\.venv-sim\\Scripts\\python.exe limb_sim.py <limb.npz> <skeleton.npz> <out.npz> [bend_deg twist_deg]
  tets      fTetWild in the closed skin; ideal edge = half the median skin-to-bone distance (flesh >= 2 tets deep)
  skeleton  the stick skeleton is only the hidden driver: it sets each bone's rigid motion
  bones     tet nodes inside the real (smoothed) bone meshes - winding number, works on the clipped ends - move rigidly
  layers    tets labelled skin (outermost), muscle (centre inside a muscle), sticky layer (between), bone; per-layer
            textbook stiffness (see below); muscle strands are carried by the flesh (drawn, not yet a force)
  holder    the nodes at the segment's near end (one edge from the end cap) stay put: they belong to the upper arm
  moves     'bend' turns the bones about the near joint, about the axis across upper arm and forearm; 'twist' turns them
            about the segment's own axis. Each eased in over frames, the flesh simulated in between.
  flesh     neo-Hookean per tet, Poisson 0.45, 1000 kg/m^3, no gravity
  writes    per move: skin, muscles, bones moved (embedded in the tets), inverted tet count, volume ratio"""
import sys, numpy as np, warp as wp, newton, pytetwild
from scipy.spatial import cKDTree, ConvexHull, Delaunay
from scipy.spatial.transform import Rotation as Rot
lp, skp, out = sys.argv[1:4]; bend = float(sys.argv[4]) if len(sys.argv) > 4 else 45.0; twist = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
Z = np.load(lp, allow_pickle=True); SV, SF = Z['skin_V'], Z['skin_F']; J0, ax, L = Z['J0'], Z['ax'], float(Z['L'])
St = [(str(n), str(k), np.asarray(v, float), np.asarray(f)) for n, k, v, f in zip(Z['names'], Z['kinds'], Z['Vs'], Z['Fs'])]
BV = np.vstack([s[2] for s in St if s[1] == 'bone']); MV = [s[2] for s in St if s[1] == 'muscle']
K = np.load(skp); H, lab = K['heads'], [str(x) for x in K['labels']]
up = J0 - H[lab.index('r_uparm')]; up /= np.linalg.norm(up); nb = np.cross(up, ax); nb /= np.linalg.norm(nb)
h = float(np.median(cKDTree(BV).query(SV)[0])) / 2
X, T = pytetwild.tetrahedralize(SV, SF.astype(np.int32), edge_length_abs=h); X = np.asarray(X, float); T = np.asarray(T, np.int64)
P = X[T]; vol = np.einsum('ij,ij->i', np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]), P[:, 3] - P[:, 0]) / 6; T[vol < 0] = T[vol < 0][:, [0, 2, 1, 3]]
def winding(Q, V_, F_, chunk=2000):                              # generalised winding number (works on open, clipped meshes)
    A_, B_, C_ = V_[F_[:, 0]], V_[F_[:, 1]], V_[F_[:, 2]]; w = np.zeros(len(Q))
    for i in range(0, len(Q), chunk):
        q = Q[i:i + chunk, None]; a, b, c = A_ - q, B_ - q, C_ - q; la, lb, lc = [np.linalg.norm(x, axis=2) for x in (a, b, c)]
        det = np.einsum('nij,nij->ni', a, np.cross(b, c)); den = la * lb * lc + np.einsum('nij,nij->ni', a, b) * lc + np.einsum('nij,nij->ni', b, c) * la + np.einsum('nij,nij->ni', c, a) * lb
        w[i:i + chunk] = 2 * np.arctan2(det, den).sum(1) / (4 * np.pi)
    return w
inb = np.zeros(len(X), bool)                                      # the real (smoothed) bones, not the stick line, hold the flesh
for s in St:
    if s[1] == 'bone': inb |= (np.abs(winding(X, s[2], s[3])) > 0.5) | (cKDTree(s[2]).query(X)[0] < h / 4)
Cn = X[T].mean(1); inm = np.zeros(len(T), bool)
for s in St:
    if s[1] == 'muscle': inm |= np.abs(winding(Cn, s[2], s[3])) > 0.5
Fc = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), 1); uf, ic, cf = np.unique(Fc, axis=0, return_inverse=True, return_counts=True)
outer = (cf[ic.ravel()] == 1).reshape(4, -1).any(0)
tb = inb[T].all(1); layer = np.where(tb, 3, np.where(outer, 0, np.where(inm, 2, 1)))     # 0 skin, 1 sticky layer, 2 muscle, 3 bone
print('LS tets: skin %d, sticky layer %d, muscle %d, bone %d' % tuple((layer == i).sum() for i in range(4)), flush=True)
along = (X - J0) @ ax; hold = along < along.min() + h
print('LS tet edge %.4f m: nodes %d, tets %d; bone nodes %d, held at the near end %d' % (h, len(X), len(T), inb.sum(), hold.sum()), flush=True)
P = X[T]; Dm = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0], P[:, 3] - P[:, 0]], 2); Di = np.linalg.inv(Dm); kd = cKDTree(P.mean(1))
def embed(Q):                                                     # tet + barycentric weights for points (least-outside tet among 16 near)
    _, cand = kd.query(Q, k=16); bt = np.zeros(len(Q), int); bb = np.full((len(Q), 4), -np.inf)
    for c in range(cand.shape[1]):
        t = cand[:, c]; l = np.einsum('nij,nj->ni', Di[t], Q - P[t, 0]); b = np.c_[1 - l.sum(1), l]; better = b.min(1) > bb.min(1)
        bt[better], bb[better] = t[better], b[better]
    return bt, bb
STR = [np.asarray(q, float) for q in Z['strands']]
emb = [embed(Q) for Q in [SV] + MV + STR]
pos = lambda q, e: (e[1][..., None] * q[T[e[0]]]).sum(1)
# textbook stiffness per layer (Young's modulus, Pa): skin ~150 kPa (in-vivo dermis 0.1-0.2 MPa), subcutaneous fat / fascia
# ~2 kPa (the soft sticky layer the skin slides on), relaxed skeletal muscle ~20 kPa (shear-wave elastography 10-30 kPa);
# bone tets are pinned. Poisson 0.45, density 1000 kg/m^3.
E_ = np.array([150e3, 2e3, 20e3, 20e3])[layer]; nu, rho = 0.45, 1000.0; mu = E_ / (2 * (1 + nu)); lam = E_ * nu / ((1 + nu) * (1 - 2 * nu))
pidx = np.flatnonzero(inb | hold); vol0 = np.abs(np.linalg.det(Dm)) / 6
@wp.kernel
def place(q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3), idx: wp.array(dtype=wp.int32), tgt: wp.array(dtype=wp.vec3), dt: float):
    i = wp.tid(); k = idx[i]; qd[k] = (tgt[i] - q[k]) / dt; q[k] = tgt[i]
def run(axis, deg, frames=60, nsub=10, nit=10):
    b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, 0.0)
    b.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=X.tolist(), indices=T.ravel().tolist(), density=rho, k_mu=mu, k_lambda=lam, k_damp=np.zeros(len(T)))
    for i in pidx: b.particle_mass[i] = 0.0
    b.color(); m = b.finalize(); sol = newton.solvers.SolverVBD(model=m, iterations=nit, particle_enable_self_contact=False, particle_enable_tile_solve=False)
    s0, s1, ctl = m.state(), m.state(), m.control(); iw = wp.array(pidx, dtype=wp.int32); tg = wp.zeros(len(pidx), dtype=wp.vec3); dt = 1.0 / 60 / nsub
    for f in range(frames + 30):                                  # ease in, then hold 30 frames to settle
        a = deg * min(1.0, (f + 1) / frames); R = Rot.from_rotvec(np.radians(a) * axis).as_matrix()
        tgt = X[pidx].copy(); mb = inb[pidx] & ~hold[pidx]; tgt[mb] = (X[pidx][mb] - J0) @ R.T + J0
        for s in range(nsub):
            tg.assign(tgt.astype(np.float32)); wp.launch(place, dim=len(pidx), inputs=[s0.particle_q, s0.particle_qd, iw, tg, dt])
            s0.clear_forces(); sol.step(s0, s1, ctl, None, dt); s0, s1 = s1, s0
    q = s0.particle_q.numpy().astype(float); Pq = q[T]; det = np.linalg.det(np.stack([Pq[:, 1] - Pq[:, 0], Pq[:, 2] - Pq[:, 0], Pq[:, 3] - Pq[:, 0]], 2)) / 6
    R = Rot.from_rotvec(np.radians(deg) * axis).as_matrix(); bones = [(s[2] - J0) @ R.T + J0 for s in St if s[1] == 'bone']
    nM = len(MV); return dict(q=q, skin=pos(q, emb[0]), muscles=[pos(q, e) for e in emb[1:1 + nM]], strands=[pos(q, e) for e in emb[1 + nM:]], bones=bones, inverted=int((det <= 0).sum()), volume=float(det.sum() / vol0.sum()))
res = {'rest': dict(q=X, skin=SV, muscles=MV, strands=STR, bones=[s[2] for s in St if s[1] == 'bone'], inverted=0, volume=1.0)}
for nm, axis, deg in (('bend', nb, bend), ('twist', ax, twist)):
    res[nm] = run(axis, deg); r = res[nm]
    print('LS %s %+.0f deg: inverted tets %d of %d, volume ratio %.4f, skin moved max %.3f m' % (nm, deg, r['inverted'], len(T), r['volume'], np.linalg.norm(r['skin'] - SV, axis=1).max()), flush=True)
np.savez(out, X=X, T=T, hold=hold, inb=inb, layer=layer, skin_F=SF, mus_F=np.array([s[3] for s in St if s[1] == 'muscle'], dtype=object),
         bone_F=np.array([s[3] for s in St if s[1] == 'bone'], dtype=object),
         **{'%s_%s' % (k, f): (np.array(v[f], dtype=object) if f in ('muscles', 'bones', 'strands') else v[f]) for k, v in res.items() for f in ('skin', 'muscles', 'strands', 'bones', 'q', 'inverted', 'volume')}, allow_pickle=True)
print('LS done')
