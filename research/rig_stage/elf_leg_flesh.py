"""Flesh test (rig-stage strategy c): the elf's left-leg flesh as Newton VBD tetrahedra around its fitted bones, driven by a
calibrated ARDY clip (ardy_to_elf_leg_calibrated.py's <clip>_leg_transforms.npz).  Compares the skin with the nearest-vertex
binding of the one-leg audit (model_to_vrm/rig/leg_bend_test.py).
  E:\\splat\\.venv-sim\\Scripts\\python.exe elf_leg_flesh.py <fits dir> <model dir> <leg_transforms.npz> <out dir> [substeps] [iterations]

  skin     the leg's own flesh piece of the model surface (the island holding the femur, as leg_bend_test.py)
  tets     fTetWild (pytetwild) inside it; ideal edge = half the leg's median flesh thickness (skin point to nearest bone),
           so flesh is at least two tets deep (rule of flesh/body_tetwild.py)
  pins     nodes nearer a bone surface than the skin (rule of flesh/newton_flesh.py) ride rigidly with that bone: hip bone
           fixed, femur with the thigh, tibia/fibula/patella/foot bones with the shank; nodes on the cap that closes the open
           hip cut (boundary nodes farther from the skin than the Otsu split of boundary-node distances) are held fixed
  flesh    Newton VBD, neo-Hookean, E 30 kPa, Poisson 0.45, density 1000 kg/m^3 (the textbook values newton_flesh.py uses);
           no gravity; one second easing in from rest, then the clip
  skin out every skin vertex embedded in its tet (barycentric), so the measured skin is the model's own surface
Checks per frame (same as the audit): skin edge stretch p1/median/p99, flipped skin triangles, inverted tets, volume.
Writes elf_leg_flesh.json and elf_leg_flesh.npz (skin per frame)."""
import sys, os, json, time, numpy as np, warp as wp, newton, pytetwild
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp
FD, MD, TRF, OUT = sys.argv[1:5]; nsub = int(sys.argv[5]) if len(sys.argv) > 5 else 10; nit = int(sys.argv[6]) if len(sys.argv) > 6 else 10   # Newton example defaults
os.makedirs(OUT, exist_ok=True)
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
S = '.l'; hipb, fem, tib, fib, pat = (nid[n + S] for n in ('Hip bone', 'Femur', 'Tibia', 'Fibula', 'Patella'))
import re
foot = [i for i, n in enumerate(names) if n.endswith(S) and re.search(r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform|phalan', n)]
GRP = {hipb: 0, fem: 1, tib: 2, fib: 2, pat: 2, **{b: 2 for b in foot}}            # 0 pelvis (fixed), 1 thigh, 2 shank
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']
ISL = np.load(os.path.join(MD, 'flesh', 'islands.npz'))['island']
near = ISL[cKDTree(MV).query(V[OB == fem])[1]]; leg = int(np.bincount(near[near >= 0]).argmax())
sk = np.flatnonzero(ISL == leg); SF = MF[(ISL[MF] == leg).all(1)]; loc = np.full(len(MV), -1); loc[sk] = np.arange(len(sk)); SFl = loc[SF]; Vs = MV[sk]
bidx = np.flatnonzero(np.isin(OB, list(GRP))); BP = V[bidx]; bgrp = np.array([GRP[b] for b in OB[bidx]]); kb = cKDTree(BP)
thick = kb.query(Vs)[0]; h = 0.5 * float(np.median(thick))
print('LF leg skin: island %d, %d vertices, %d faces; median flesh %.1f mm -> tet edge %.1f mm' % (leg, len(Vs), len(SF), 1000 * np.median(thick), 1000 * h), flush=True)
cache = os.path.join(OUT, 'elf_leg_tets.npz'); t0 = time.time()
if os.path.exists(cache): C_ = np.load(cache); X0, T = C_['X'], C_['T']
else: X0, T = pytetwild.tetrahedralize(Vs, SFl.astype(np.int32), edge_length_abs=h); X0 = X0.astype(np.float64); T = T.astype(np.int64); np.savez(cache, X=X0, T=T)
print('LF tets: %d nodes, %d tets (%.0f s)' % (len(X0), len(T), time.time() - t0), flush=True)
# Newton's add_tetrahedron silently DROPS any tet whose signed rest volume det[x1-x0, x2-x0, x3-x0]/6 is <= 0, and
# fTetWild writes the opposite winding, so every tet must be reordered (swap two vertices) before building the model
sv = np.linalg.det(np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2)) / 6
print('LF winding: %d tets negative, %d positive, %d zero; smallest |volume| / median %.2e' % ((sv < 0).sum(), (sv > 0).sum(), (sv == 0).sum(), np.abs(sv).min() / np.median(np.abs(sv))), flush=True)
if os.environ.get('KEEP_WINDING') != '1': T = np.where((sv < 0)[:, None], T[:, [0, 1, 3, 2]], T)
Fc = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), 1); uf, cf = np.unique(Fc, axis=0, return_counts=True)
bnd = np.zeros(len(X0), bool); bnd[uf[cf == 1].ravel()] = True
ks = cKDTree(Vs); dskin = ks.query(X0)[0]; dbone, bi = kb.query(X0)
cap = bnd & (dskin > otsu(dskin[bnd]))
pin = (dbone < dskin) & ~bnd | cap; grp = np.where(cap, 0, bgrp[bi])
print('LF pinned %d nodes (%d on the hip cut cap); by group pelvis/thigh/shank: %s' % (pin.sum(), cap.sum(), np.bincount(grp[pin], minlength=3).tolist()), flush=True)
# skin embedding
cen = X0[T].mean(1); kt = cKDTree(cen); cand = kt.query(Vs, k=8)[1]
def bary(p, tet):
    A = np.stack([X0[tet[:, 1]] - X0[tet[:, 0]], X0[tet[:, 2]] - X0[tet[:, 0]], X0[tet[:, 3]] - X0[tet[:, 0]]], 2)
    l = np.linalg.solve(A, (p - X0[tet[:, 0]])[..., None])[..., 0]; return np.c_[1 - l.sum(1), l]
emb_t = np.zeros(len(Vs), int); emb_b = np.zeros((len(Vs), 4)); best = np.full(len(Vs), np.inf)
for k in range(cand.shape[1]):
    b = bary(Vs, T[cand[:, k]]); out_ = -np.minimum(b, 0).sum(1); better = out_ < best
    emb_t[better] = cand[better, k]; emb_b[better] = b[better]; best[better] = out_[better]
def skin(q): return (emb_b[..., None] * q[T[emb_t]]).sum(1)
# drive
D = np.load(TRF); fps = float(D['fps']); K = int(round(fps))
RT = [(np.repeat(np.eye(3)[None], len(D['thigh_R']), 0), np.zeros((len(D['thigh_R']), 3))), (D['thigh_R'], D['thigh_t']), (D['shank_R'], D['shank_t'])]
def target(f):
    """pinned node places: rigid per bone group; during the first second, straight from rest towards the clip's first frame"""
    g_ = grp[pin]; P = X0[pin]
    def at(k):
        out = np.empty_like(P)
        for g in range(3): out[g_ == g] = P[g_ == g] @ RT[g][0][k].T + RT[g][1][k]
        return out
    return at(f - K) if f >= K else P + (at(0) - P) * (f / K)
nF = K + min(len(D['thigh_R']), int(os.environ.get('MAXF', 10 ** 9)))
E, nu, rho = 30e3, 0.45, 1000.0; mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))
builder = newton.ModelBuilder(); builder.gravity = (0.0, 0.0, 0.0)
builder.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=X0.tolist(), indices=T.ravel().tolist(), density=rho, k_mu=mu, k_lambda=lam, k_damp=0.0)
pidx = np.flatnonzero(pin)
for i in pidx: builder.particle_mass[i] = 0.0
builder.color(); model = builder.finalize()
print('LF model holds %d of %d tets' % (model.tet_count, len(T)), flush=True)
solver = newton.solvers.SolverVBD(model=model, iterations=nit, particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0, s1, ctl = model.state(), model.state(), model.control(); pidx_w = wp.array(pidx, dtype=wp.int32); ptgt = wp.zeros(len(pidx), dtype=wp.vec3)
@wp.kernel
def place(q: wp.array(dtype=wp.vec3), qd: wp.array(dtype=wp.vec3), idx: wp.array(dtype=wp.int32), tgt: wp.array(dtype=wp.vec3), dt: float):
    i = wp.tid(); k = idx[i]; qd[k] = (tgt[i] - q[k]) / dt; q[k] = tgt[i]
dt = 1.0 / fps / nsub
Dm0 = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2); vol0 = np.abs(np.linalg.det(Dm0)) / 6; Dinv0 = np.linalg.inv(Dm0)
E_ = np.r_[SFl[:, [0, 1]], SFl[:, [1, 2]], SFl[:, [2, 0]]]; e0 = np.linalg.norm(Vs[E_[:, 0]] - Vs[E_[:, 1]], axis=1); ok = e0 > 0
n0 = np.cross(Vs[SFl[:, 1]] - Vs[SFl[:, 0]], Vs[SFl[:, 2]] - Vs[SFl[:, 0]])
def stage(label, q):
    """inverted tets, split by how many of their 4 nodes are pinned and whether the pinned ones ride different bone groups"""
    Ds = np.stack([q[T[:, 1]] - q[T[:, 0]], q[T[:, 2]] - q[T[:, 0]], q[T[:, 3]] - q[T[:, 0]]], 2); J = np.linalg.det(Ds @ Dinv0)
    npin = pin[T].sum(1); gs = np.where(pin[T], grp[T], -1); mixed = np.array([len(set(r[r >= 0])) > 1 for r in gs])
    bad = J <= 0; near = (J > 0) & (J < 0.1)
    print('LF STAGE %-40s inverted %5d (%.2f%%), near-singular J<0.1 %d; inverted by pinned nodes 0..4: %s; inverted with pins on two bone groups %d; J p1/50/99 %.3f/%.3f/%.3f'
          % (label, bad.sum(), 100 * bad.mean(), near.sum(), np.bincount(npin[bad], minlength=5).tolist(), (bad & mixed).sum(), *np.percentile(J, [1, 50, 99])), flush=True)
    return dict(label=label, inverted=int(bad.sum()), near_singular=int(near.sum()), by_pinned=np.bincount(npin[bad], minlength=5).tolist(), mixed_groups=int((bad & mixed).sum()))
stages = [stage('rest', X0)]
qi = X0.copy(); qi[pidx] = target(K); stages.append(stage('pins at clip frame 0, free nodes at rest', qi))
prev = X0[pidx].copy(); rows = []; keep = []; t0 = time.time(); SAVE_NODES = os.environ.get('SAVE_NODES') == '1'; nodes = []
for f in range(nF):
    nxt = target(f)
    for s in range(nsub):
        ptgt.assign((prev + (nxt - prev) * (s + 1) / nsub).astype(np.float32))
        wp.launch(place, dim=len(pidx), inputs=[s0.particle_q, s0.particle_qd, pidx_w, ptgt, dt])
        s0.clear_forces(); solver.step(s0, s1, ctl, None, dt); s0, s1 = s1, s0
    prev = nxt; q = s0.particle_q.numpy().astype(np.float64)
    if f in (1, K // 2, K): stages.append(stage('after ease-in frame %d (%d substeps solved)' % (f, (f + 1) * nsub), q))
    if f in (1, K // 2, K):
        fr = np.setdiff1d(np.arange(len(X0)), pidx)
        print('LF debug frame %d: pinned nodes off their target %.2f mm (target moved %.1f mm from rest); free nodes moved %.2f mm' % (f, 1000 * np.linalg.norm(q[pidx] - nxt, axis=1).mean(), 1000 * np.linalg.norm(nxt - X0[pidx], axis=1).mean(), 1000 * np.linalg.norm(q[fr] - X0[fr], axis=1).mean()), flush=True)
    if not np.isfinite(q).all(): print('LF frame %d: not finite, stopping' % (f - K)); break
    if f < K: continue
    Ds = np.stack([q[T[:, 1]] - q[T[:, 0]], q[T[:, 2]] - q[T[:, 0]], q[T[:, 3]] - q[T[:, 0]]], 2); dets = np.linalg.det(Ds @ Dinv0)
    X = skin(q); e1 = np.linalg.norm(X[E_[:, 0]] - X[E_[:, 1]], axis=1); st = e1[ok] / e0[ok]
    n1 = np.cross(X[SFl[:, 1]] - X[SFl[:, 0]], X[SFl[:, 2]] - X[SFl[:, 0]]); flip = float(((n0 * n1).sum(1) < 0).mean())
    rows.append(dict(frame=f - K, inverted_tets=int((dets <= 0).sum()), volume=float((dets * vol0).sum() / vol0.sum()), skin_edge=dict(p1=float(np.percentile(st, 1)), median=float(np.median(st)), p99=float(np.percentile(st, 99))), skin_flipped=flip))
    if (f - K) % 15 == 0: keep.append(X.astype(np.float32))
    if SAVE_NODES: nodes.append(q.astype(np.float32))
    if (f - K) % 30 == 0: print('LF frame %3d: inverted %d, volume %.4f, skin edge p99 x%.2f, flipped %.2f%%  (%.0f s)' % (f - K, rows[-1]['inverted_tets'], rows[-1]['volume'], rows[-1]['skin_edge']['p99'], 100 * flip, time.time() - t0), flush=True)
summ = dict(tets=int(len(T)), nodes=int(len(X0)), pinned=int(pin.sum()), tet_edge_mm=1000 * h, substeps=nsub, iterations=nit, seconds=round(time.time() - t0),
            inverted_tets_max=max(r['inverted_tets'] for r in rows), volume_range=[min(r['volume'] for r in rows), max(r['volume'] for r in rows)],
            skin_edge_p99_max=max(r['skin_edge']['p99'] for r in rows), skin_edge_p1_min=min(r['skin_edge']['p1'] for r in rows), skin_flipped_max=max(r['skin_flipped'] for r in rows))
json.dump(dict(summary=summ, stages=stages, frames=rows), open(os.path.join(OUT, 'elf_leg_flesh.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(OUT, 'elf_leg_flesh.npz'), skin=np.array(keep), faces=SFl, rest=Vs.astype(np.float32))
if SAVE_NODES: np.savez_compressed(os.path.join(OUT, 'elf_leg_nodes.npz'), q=np.array(nodes), X0=X0.astype(np.float32), T=T, pin=pin, grp=grp, fps=fps)   # every clip frame, for the viewer
print('LF SUMMARY ' + json.dumps(summ))
