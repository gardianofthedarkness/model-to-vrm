"""Muscles, step 1: every atlas muscle (and tendon, fascia, bursa - the atlas's 'muscle' system) carried into the figure
by the bones it lies on, after the bones were placed by body_assembly.py.
  E:\\splat\\.venv\\Scripts\\python.exe muscle_carry.py <fits dir> <model dir>
  bones     each bone's move atlas -> figure as one affine map (least squares on its own vertices; exact for the free
            bones, whose fit is such a map), split into a turn and a stretch (polar decomposition)
  contact   a muscle vertex within one edge of that muscle's own mesh of a bone lies on that bone: it moves with it
  between   on the muscle's own mesh (seams welded) the bone shares are the harmonic function with those vertices as
            fixed values (Joshi et al. 2007, harmonic coordinates; uniform graph Laplacian): no falloff number
  loose     a piece of mesh touching no bone takes the shares of the nearest vertex that has them
  blend     position = dual-quaternion blend of the bones' turns and shifts (Kavan et al. 2008), applied to the
            share-weighted stretch of the atlas point: a muscle over a joint that turned 90 degrees bends round it
            and keeps its thickness, where a plain weighted mean of positions would cut the corner
Checks (nothing is corrected here; this step only shows what the bones alone give):
  outside   muscle vertices outside the solid model of the body, against the earlier slice warp (atlas_warp.npz)
  in bone   muscle vertices inside a solid model of the placed bones, against the same measure in the atlas itself
  stretch   how much each muscle's mesh edges changed length
Writes <fits dir>/muscle_carry.npz (V: every atlas vertex in the figure; W: the bone shares) and muscle_carry.json."""
import sys, os, json, time, numpy as np
from scipy import ndimage
from scipy.linalg import polar
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.sparse import coo_matrix, csr_matrix, diags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu
FD, MD = sys.argv[1:3]; t00 = time.time()
def log(*a): print('MC', *a, flush=True)
W_ = np.load(os.path.join(FD, 'atlas_warp.npz')); V0 = W_['V0'].astype(np.float64); OB = W_['obj']; names = [str(x) for x in W_['names']]; sysn = [str(x) for x in W_['system']]; FB, FM = W_['F_bone'], W_['F_muscle']; Vold = W_['V'].astype(np.float64)
BA = np.load(os.path.join(FD, 'body_assembly.npz')); V = BA['V'].copy()
bones = [int(b) for b in np.unique(OB) if sysn[b] == 'bone']; bcol = {b: k for k, b in enumerate(bones)}; NB = len(bones)
muscles = [int(m) for m in np.unique(OB) if sysn[m] == 'muscle' and not names[m].endswith(('.g', '.j'))]

# --- each bone's move as turn R, stretch S, shift t:  x -> R (S x) + t
Rb = np.zeros((NB, 3, 3)); Sb = np.zeros((NB, 3, 3)); tb = np.zeros((NB, 3)); res = []
for b, k in bcol.items():
    i = np.flatnonzero(OB == b); X = np.c_[V0[i], np.ones(len(i))]; M = np.linalg.lstsq(X, V[i], rcond=None)[0]; A = M[:3].T
    R, S = polar(A)
    if np.linalg.det(R) < 0: R, S = -R, -S
    Rb[k], Sb[k], tb[k] = R, S, M[3]; res.append(np.abs(X @ M - V[i]).max())
log('%d bones as affine moves: worst vertex off by %.2f mm (median bone %.3f mm); turns up to %.0f deg' % (NB, 1000 * max(res), 1000 * np.median(res), np.degrees(np.max(Rot.from_matrix(Rb).magnitude()))))
qb = Rot.from_matrix(Rb).as_quat()[:, [3, 0, 1, 2]]                                  # w, x, y, z
ref = qb[np.argmin(Rot.from_matrix(Rb).magnitude())]; qb[qb @ ref < 0] *= -1         # one hemisphere, so blending never goes the long way round
def qmul(a, b):
    return np.stack([a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1), *(a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])).T], -1)
qd = 0.5 * qmul(np.c_[np.zeros(NB), tb], qb)                                         # dual part
SbF = Sb.reshape(NB, 9)

# --- bone shares for every muscle vertex
bm = np.isin(OB, bones); bidx = np.flatnonzero(bm); bt = cKDTree(V0[bidx]); bo = np.array([bcol[b] for b in OB[bidx]])
rows, cols, vals = [], [], []; done = np.zeros(len(V0), bool); stat = dict(contact=0, harmonic=0, loose=0); touched = {}
loc = -np.ones(len(V0), np.int64)
contact = np.zeros(len(V0), bool)
for mi, m in enumerate(muscles):
    vi = np.flatnonzero(OB == m); loc[vi] = np.arange(len(vi)); F = FM[(OB[FM[:, 0]] == m)]; F = loc[F]
    _, wid = np.unique(np.round(V0[vi], 6), axis=0, return_inverse=True); wid = wid.ravel(); n = wid.max() + 1; rep = np.zeros(n, np.int64); rep[wid] = np.arange(len(vi))
    Fw = wid[F]; E = np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]; E = E[E[:, 0] != E[:, 1]]
    edge = float(np.median(np.linalg.norm(V0[vi][F[:, 0]] - V0[vi][F[:, 1]], axis=1))) if len(F) else 0.0
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(np.float64)
    d, j = bt.query(V0[vi][rep], distance_upper_bound=edge if edge > 0 else 1e-9); fix = np.isfinite(d); own = np.where(fix, bo[np.minimum(j, len(bo) - 1)], -1)
    ub = np.unique(own[fix]); Wm = np.zeros((n, len(ub))); ok = np.zeros(n, bool); touched[m] = [names[bones[u]] for u in ub]
    if fix.any():
        Wm[fix, np.searchsorted(ub, own[fix])] = 1; ok[fix] = True; contact[vi] = fix[wid]
        nc, lab = connected_components(A, directed=False); has = np.zeros(nc, bool); has[lab[fix]] = True; free = ~fix & has[lab]
        if free.any():
            L = (diags(np.asarray(A.sum(1)).ravel()) - A).tocsr(); fi = np.flatnonzero(free); lu = splu(L[fi][:, fi].tocsc()); Wm[fi] = lu.solve(-(L[fi][:, np.flatnonzero(fix)] @ Wm[fix]))
            ok[fi] = True
        if (~ok).any():                                                              # pieces of this muscle touching no bone: the nearest vertex of the muscle that has shares
            Wm[~ok] = Wm[np.flatnonzero(ok)[cKDTree(V0[vi][rep][ok]).query(V0[vi][rep][~ok])[1]]]
        stat['contact'] += int(fix[wid].sum()); stat['harmonic'] += int((free)[wid].sum()); stat['loose'] += int((~ok)[wid].sum())
        Wm = np.clip(Wm, 0, None); Wm /= Wm.sum(1, keepdims=True); r_, c_ = np.nonzero(Wm[wid]); rows.append(vi[r_]); cols.append(ub[c_]); vals.append(Wm[wid][r_, c_]); done[vi] = True
    if mi % 100 == 0: log('  muscle %d of %d (%.0f s)' % (mi, len(muscles), time.time() - t00))
Wt = csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(len(V0), NB))
mall = np.isin(OB, muscles); rest = np.flatnonzero(mall & ~done)
if len(rest):                                                                        # whole items touching no bone (a bursa, a fascia): the nearest muscle vertex that has shares
    dn = np.flatnonzero(done); Wt = Wt.tolil(); src = dn[cKDTree(V0[dn]).query(V0[rest])[1]]; Wt = Wt.tocsr(); Wt = Wt + csr_matrix((np.ones(len(rest)), (rest, src)), shape=(len(V0), len(V0))) @ Wt
log('shares: %d muscle vertices; on a bone %d, between (harmonic) %d, loose pieces %d, whole loose items %d (%s)' % (mall.sum(), stat['contact'], stat['harmonic'], stat['loose'], len(rest), ', '.join(sorted({names[m] for m in np.unique(OB[rest])})[:8])))

# --- blend
mi_ = np.flatnonzero(mall); Wm_ = Wt[mi_]; x = V0[mi_]
Sx = (Wm_ @ SbF).reshape(-1, 3, 3); p = np.einsum('nij,nj->ni', Sx, x)
Q = Wm_ @ qb; Qd = Wm_ @ qd; nrm = np.linalg.norm(Q, axis=1, keepdims=True); Q, Qd = Q / nrm, Qd / nrm
conj = Q * np.array([1, -1, -1, -1]); tr = 2 * qmul(Qd, conj)[:, 1:]
V[mi_] = Rot.from_quat(Q[:, [1, 2, 3, 0]]).apply(p) + tr
one = np.flatnonzero(np.asarray((Wm_ > 0.999999).sum(1)).ravel() > 0)[:2000]; kb = np.asarray(Wm_[one].argmax(1)).ravel()
log('blend check: a vertex wholly on one bone lands where that bone\'s own move puts it to within %.4f mm' % (1000 * np.abs(V[mi_][one] - (np.einsum('nij,nj->ni', Rb[kb] @ Sb[kb], x[one]) + tb[kb])).max()))

# --- checks
def solid_of(P, F, step):
    lo = P.min(0) - 4 * step; gn = np.ceil((P.max(0) + 4 * step - lo) / step).astype(int) + 1; T = P[F]; occ = np.zeros(gn, bool)
    ns = np.clip(np.ceil(np.maximum(np.linalg.norm(T[:, 1] - T[:, 0], axis=1), np.linalg.norm(T[:, 2] - T[:, 0], axis=1)) / (step / 2)).astype(int), 1, 64)
    for n_ in np.unique(ns):
        Tn = T[ns == n_]; u, v = np.meshgrid(np.arange(n_ + 1) / n_, np.arange(n_ + 1) / n_); k = (u + v) <= 1 + 1e-9; u, v = u[k], v[k]
        for c0 in range(0, len(Tn), 20000):
            Tc = Tn[c0:c0 + 20000]; Pn = (Tc[:, None, 0] * (1 - u - v)[None, :, None] + Tc[:, None, 1] * u[None, :, None] + Tc[:, None, 2] * v[None, :, None]).reshape(-1, 3); ii = np.rint((Pn - lo) / step).astype(int); occ[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    return occ, lo
def sdf_of(sol, step): return ((ndimage.distance_transform_edt(sol) - ndimage.distance_transform_edt(~sol)) * step).astype(np.float32)
def sample(sd, lo, step, P): return ndimage.map_coordinates(sd, ((P - lo) / step).T, order=1, mode='nearest')
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']; hstep = 2 * float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1)))
occ, glo = solid_of(MV, MF, hstep)
for seal in range(1, 17):                                                             # the same solid as body_assembly.py
    thick = ndimage.binary_dilation(occ, iterations=seal); fill = ndimage.binary_fill_holes(thick)
    if fill.sum() - thick.sum() > thick.sum(): break
else: print('WARNING muscle_carry.py: the solid did not close within 17 sealing steps; the widest sealing is used and the result may be wrong', flush=True)
body = ndimage.binary_erosion(fill, iterations=seal) | occ; sdf = sdf_of(body, hstep); vol_body = body.sum() * hstep ** 3
dep = sample(sdf, glo, hstep, V[mi_]); dep_old = sample(sdf, glo, hstep, Vold[mi_])
def bone_solid(P, step): o, lo = solid_of(P, FB, step); return sdf_of(ndimage.binary_fill_holes(o), step), lo, ndimage.binary_fill_holes(o).sum() * step ** 3
sb, lob, vol_bone = bone_solid(V, hstep); inb = sample(sb, lob, hstep, V[mi_])
s_at = float(BA['s1']); sa, loa, _ = bone_solid(V0, hstep / s_at); inb_at = sample(sa, loa, hstep / s_at, V0[mi_]) * s_at
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solids import closed_volumes
cv = closed_volumes(V0, FM, OB[FM[:, 0]], muscles, V, V0); vol_mus = sum(v[0] for v in cv.values()); vol_mus_at = sum(v[1] for v in cv.values())   # closed meshes only: a fascia sheet has no volume
e0 = np.linalg.norm(V0[FM[:, 0]] - V0[FM[:, 1]], axis=1); e1 = np.linalg.norm(V[FM[:, 0]] - V[FM[:, 1]], axis=1); ok = e0 > 0; st = e1[ok] / (e0[ok] * s_at)
om = OB[mi_]; per = []
for m in muscles:
    k = om == m; per.append(dict(name=names[m], bones=touched.get(m, []), vertices=int(k.sum()), outside=float((dep[k] < 0).mean()), outside_worst_mm=1000 * float(max(0, -dep[k].min())), in_bone=float((inb[k] > hstep).mean())))
rep = dict(muscles=len(muscles), vertices=int(mall.sum()), shares=stat, loose_items=sorted({names[m] for m in np.unique(OB[rest])}),
           outside=dict(carried=float((dep < 0).mean()), slice_warp=float((dep_old < 0).mean()), carried_over_one_cell=float((dep < -hstep).mean()), slice_warp_over_one_cell=float((dep_old < -hstep).mean()), carried_worst_mm=1000 * float(-dep.min()), carried_p99_mm=1000 * float(max(0, -np.percentile(dep, 1)))),
           in_bone=dict(carried=float((inb > hstep).mean()), atlas=float((inb_at > hstep).mean())), grid_mm=1000 * hstep,
           closed_meshes=len(cv), volume_cm3=dict(body=1e6 * vol_body, bones=1e6 * vol_bone, muscles=1e6 * vol_mus, muscles_if_atlas_only_scaled=1e6 * vol_mus_at * s_at ** 3),
           edge_stretch=dict(median=float(np.median(st)), p1=float(np.percentile(st, 1)), p99=float(np.percentile(st, 99))), per_muscle=per, seconds=time.time() - t00)
json.dump(rep, open(os.path.join(FD, 'muscle_carry.json'), 'w'), indent=1)
np.savez(os.path.join(FD, 'muscle_carry.npz'), V=V.astype(np.float32), F_muscle=FM, F_bone=FB, obj=OB, W_data=Wt.data.astype(np.float32), W_indices=Wt.indices, W_indptr=Wt.indptr, bones=np.array(bones), muscles=np.array(muscles), contact=contact, depth=dep.astype(np.float32), in_bone=inb.astype(np.float32), mverts=mi_)
log('outside the body: carried %.1f%% of muscle vertices (%.1f%% by more than one %.1f mm cell; 99th percentile %.1f mm, worst %.1f mm); the earlier slice warp %.1f%% (%.1f%%)' % (100 * rep['outside']['carried'], 100 * rep['outside']['carried_over_one_cell'], 1000 * hstep, rep['outside']['carried_p99_mm'], rep['outside']['carried_worst_mm'], 100 * rep['outside']['slice_warp'], 100 * rep['outside']['slice_warp_over_one_cell']))
log('inside bone by more than one cell: carried %.1f%%; in the atlas itself %.1f%%' % (100 * rep['in_bone']['carried'], 100 * rep['in_bone']['atlas']))
log('volumes: body %.0f cm3, bones %.0f, muscles (closed meshes) %.0f (%.0f if the atlas were only scaled to her height)' % tuple(rep['volume_cm3'][k] for k in ('body', 'bones', 'muscles', 'muscles_if_atlas_only_scaled')))
log('mesh edges changed length by x%.2f median (1%%..99%%: x%.2f..x%.2f) beyond the whole-body size' % (rep['edge_stretch']['median'], rep['edge_stretch']['p1'], rep['edge_stretch']['p99']))
for p_ in sorted(per, key=lambda p_: -p_['outside'] * p_['vertices'])[:12]: log('  most outside: %-46s %3.0f%% of %6d vertices, worst %.0f mm; on %s' % (p_['name'], 100 * p_['outside'], p_['vertices'], p_['outside_worst_mm'], ', '.join(p_['bones'][:4])))
log('done in %.0f s' % (time.time() - t00))
