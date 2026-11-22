"""The figure's own skeleton: every atlas bone placed inside the model, kept bone-like (never sheared), fitted to the
slice-fitted atlas warp (atlas_warp.py), joints held together, bones kept inside the skin:
  E:\\splat\\.venv\\Scripts\\python.exe skeleton_fit.py <atlas_warp.npz> <model dir>
Per bone b, 9 numbers (see the page "Own Skeleton Fit"):
  rotation r_b (on top of the closed-form start), position t_b, size s_b, and two shape numbers e_b along the bone's own
  principal axes (length vs. the two widths, volume kept: diag(e1, e2, -e1-e2)), so a bone may be longer or slimmer than
  the atlas's but never sheared.
  T_b(x) = Rot(r_b) R0_b ( s_b P_b diag(exp e_b) P_b^T (x - c_b) ) + t_b
Energy (robust least squares, every scale from the data):
  data     T_b(x) - warp(x) for every atlas bone vertex                       / sigma_d
  joints   T_a(x) - T_b(x) for the contact points of every touching pair     / sigma_c
           (touching: atlas gap under the median nearest-other-bone gap; contact points: Otsu split of the log distances)
  skin     max(0, -(inside depth)) for every bone vertex (depth from the nearest model vertex along its normal) / sigma_d
  priors   (log s_b - median) / sigma_s,  e_b / sigma_e  - spreads estimated across all bones (empirical Bayes)
Solver: start from Umeyama's closed form per bone (1991); then trust-region least squares (scipy least_squares 'trf',
sparse Jacobian, soft-L1 loss): a step is taken only if the energy drops.
Scales: variance-component estimation - after each solve every term's sigma is re-estimated from its own residuals
(1.4826 x median absolute residual); solve again until no bone moves more than sigma_d.
Writes <fits dir>/skeleton_fit.npz (per-bone parameters, V_rigid for every atlas vertex: bones placed, others as warped)
and skeleton_fit.json (scales, rounds, and the checks: residual to the warp, joint gaps, bone outside the skin, bone in
bone, left/right size difference, which bones the data pins down vs. the prior alone)."""
import sys, os, json, time, numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix, csr_matrix
fp, MD = sys.argv[1], sys.argv[2]; od = os.path.dirname(os.path.abspath(fp))
W = np.load(fp); V0, Vw, OB = W['V0'].astype(np.float64), W['V'].astype(np.float64), W['obj']
names = [str(x) for x in W['names']]; sysn = np.array([str(x) for x in W['system']])
FB = W['F_bone']; isb = sysn[OB] == 'bone'; bones = list(np.unique(OB[isb])); B = len(bones); bi = {b: i for i, b in enumerate(bones)}
idx = {b: np.flatnonzero(OB == b) for b in bones}
from scipy.spatial import ConvexHull
def hull_ids(ii):                                                   # a bone's extent: its convex-hull vertices (all of them for tiny bones)
    if len(ii) < 8: return ii
    try: return ii[np.unique(ConvexHull(V0[ii]).vertices)]
    except Exception: return ii
idx_all = idx; idx = {b: hull_ids(idx_all[b]) for b in bones}       # data, skin and fitting use the hull points; results are written for all vertices
bv = np.concatenate([idx[b] for b in bones]); bown = np.concatenate([np.full(len(idx[b]), bi[b]) for b in bones])
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MN = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['N'].astype(np.float64)
mtree = cKDTree(MV); t00 = time.time()
LOGF = open(os.path.join(od, 'skeleton_fit.log'), 'w', encoding='utf-8')
log = lambda *a: (print('SK', *a, flush=True), LOGF.write('SK ' + ' '.join(map(str, a)) + chr(10)), LOGF.flush())
# --- step 1: closed form per bone (Umeyama), principal frame
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, S_, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt; s = np.trace(np.diag(S_) @ D) / (Xc ** 2).sum(1).mean()
    return R, s, my - s * R @ mx
C = np.zeros((B, 3)); P = np.zeros((B, 3, 3)); R0 = np.zeros((B, 3, 3)); x0 = np.zeros((B, 9)); res1 = np.zeros(B)
for b in bones:
    i = bi[b]; X, Y = V0[idx[b]], Vw[idx[b]]; R, s, t = umeyama(X, Y); C[i] = X.mean(0)
    ev, evec = np.linalg.eigh(np.cov((X - C[i]).T) if len(X) > 3 else np.eye(3)); P[i] = evec[:, ::-1]   # columns: longest axis first
    if np.linalg.det(P[i]) < 0: P[i][:, 2] *= -1
    R0[i] = R; x0[i, 3:6] = s * R @ C[i] + t; x0[i, 6] = np.log(s)
    res1[i] = np.median(np.linalg.norm(X @ (s * R).T + t - Y, axis=1))
log('step 1: %d bones, closed-form similarity to the warp: median residual %.2f mm (90%% of bones under %.2f mm)' % (B, 1000 * np.median(res1), 1000 * np.percentile(res1, 90)))
# --- contacts in the atlas
trees = {b: cKDTree(V0[idx_all[b]]) for b in bones}; cent = {b: V0[idx_all[b]].mean(0) for b in bones}; rad = {b: np.linalg.norm(V0[idx_all[b]] - cent[b], axis=1).max() for b in bones}
gap = {}
for a in bones:
    for b in bones:
        if b <= a or np.linalg.norm(cent[a] - cent[b]) > rad[a] + rad[b]: continue
        gap[(a, b)] = float(trees[b].query(V0[idx_all[a]])[0].min())
nearest = {}
for (a, b), g in gap.items(): nearest[a] = min(nearest.get(a, np.inf), g); nearest[b] = min(nearest.get(b, np.inf), g)
thr = float(np.median(list(nearest.values())))
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i)
    return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
pairs = []                                                         # (i_a, i_b, atlas contact points of both sides)
for (a, b), g in gap.items():
    if g > thr: continue
    pts = []
    for u, v in ((a, b), (b, a)):
        d = trees[v].query(V0[idx_all[u]])[0]; l = np.log(d + 1e-9); pts.append(idx_all[u][l < otsu(l)] if len(l) > 2 else idx_all[u])
    pairs.append((bi[a], bi[b], np.concatenate(pts)))
cpa = np.concatenate([np.full(len(p[2]), p[0]) for p in pairs]); cpb = np.concatenate([np.full(len(p[2]), p[1]) for p in pairs]); cpx = np.concatenate([p[2] for p in pairs])
log('contacts: %d touching bone pairs (gap under %.2f mm), %d contact points' % (len(pairs), 1000 * thr, len(cpx)))
# --- the model
def mats(x):
    x = x.reshape(B, 9); Rr = Rot.from_rotvec(x[:, :3]).as_matrix() @ R0; e = np.c_[x[:, 7], x[:, 8], -x[:, 7] - x[:, 8]]
    A = np.exp(x[:, 6])[:, None, None] * Rr @ P @ (np.exp(e)[:, :, None] * np.transpose(P, (0, 2, 1)))   # s R P diag(e^e) P^T
    return A, x[:, 3:6]
def T(A, t, ids, own): return np.einsum('nij,nj->ni', A[own], V0[ids] - C[own]) + t[own]
def depth(p):                                                       # inside depth below the model surface (+ inside)
    _, k = mtree.query(p); return -np.einsum('ij,ij->i', p - MV[k], MN[k])
nd, nc, nk, npr = 3 * len(bv), 3 * len(cpx), len(bv), 3 * B
def resid(x, S):
    A, t = mats(x); x = x.reshape(B, 9); pb = T(A, t, bv, bown)
    rd = ((pb - Vw[bv]) / S['d']).ravel()
    rc = ((T(A, t, cpx, cpa) - T(A, t, cpx, cpb)) / S['c']).ravel()
    rk = np.maximum(0, -depth(pb)) / S['d']
    rp = np.r_[(x[:, 6] - S['mu']) / S['s'], x[:, 7] / S['e'], x[:, 8] / S['e']]
    return np.r_[rd, rc, rk, rp]
def dblock(x, ids, own):
    """exact derivatives of the placed points T_b(x) by their bone's 9 numbers, (n, 3, 9); the rotation part is the
    small-angle one (exact at r = 0; the rotation is folded into R0 after every solve)"""
    xb = x.reshape(B, 9); Rr = Rot.from_rotvec(xb[:, :3]).as_matrix() @ R0; s = np.exp(xb[:, 6])
    ee = np.exp(np.c_[xb[:, 7], xb[:, 8], -xb[:, 7] - xb[:, 8]]); d = V0[ids] - C[own]; Pt = np.einsum('nji,nj->ni', P[own], d)
    q = s[own, None] * np.einsum('nij,nj->ni', P[own], ee[own] * Pt); Rq = np.einsum('nij,nj->ni', Rr[own], q); n = len(ids); J = np.zeros((n, 3, 9))
    J[:, 0, 1], J[:, 0, 2], J[:, 1, 0], J[:, 1, 2], J[:, 2, 0], J[:, 2, 1] = Rq[:, 2], -Rq[:, 1], -Rq[:, 2], Rq[:, 0], Rq[:, 1], -Rq[:, 0]
    J[:, 0, 3] = J[:, 1, 4] = J[:, 2, 5] = 1; J[:, :, 6] = Rq
    for g, c in ((7, np.array([1.0, 0, -1])), (8, np.array([0, 1.0, -1]))):
        dq = s[own, None] * np.einsum('nij,nj->ni', P[own], ee[own] * c * Pt); J[:, :, g] = np.einsum('nij,nj->ni', Rr[own], dq)
    return J
def jac(x, S):
    rows, cols, vals = [], [], []
    def put(r0, own, Jb, k):                                         # Jb (n, k, 9) -> rows r0 + k*i + a, cols 9*own + j
        n = len(own); rr = r0 + np.arange(n)[:, None, None] * k + np.arange(k)[None, :, None]; cc = own[:, None, None] * 9 + np.arange(9)[None, None, :]
        rows.append(np.broadcast_to(rr, Jb.shape).ravel()); cols.append(np.broadcast_to(cc, Jb.shape).ravel()); vals.append(Jb.ravel())
    Jd = dblock(x, bv, bown); put(0, bown, Jd / S['d'], 3)
    put(nd, cpa, dblock(x, cpx, cpa) / S['c'], 3); put(nd, cpb, -dblock(x, cpx, cpb) / S['c'], 3)
    A, t = mats(x); pb = T(A, t, bv, bown); _, k = mtree.query(pb); act = ((pb - MV[k]) * MN[k]).sum(1) > 0
    put(nd + nc, bown, (np.einsum('ni,nij->nj', MN[k], Jd) * act[:, None] / S['d'])[:, None, :], 1)
    for g, sc in ((0, S['s']), (1, S['e']), (2, S['e'])):
        rows.append(nd + nc + nk + g * B + np.arange(B)); cols.append(np.arange(B) * 9 + 6 + g); vals.append(np.full(B, 1 / sc))
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nd + nc + nk + npr, 9 * B)).tocsr()
# sparsity: which rows touch which bone's 9 columns
R_, C_ = [], []
def mark(r0, own, k):
    rows = r0 + np.arange(len(own) * k); cols = np.repeat(own, k) * 9
    for j in range(9): R_.append(rows); C_.append(cols + j)
mark(0, bown, 3); mark(nd, cpa, 3); mark(nd, cpb, 3); mark(nd + nc, bown, 1)
for g in range(3): R_.append(nd + nc + nk + g * B + np.arange(B)); C_.append(np.arange(B) * 9 + 6 + g)
R_, C_ = np.concatenate(R_), np.concatenate(C_)
Jp = csr_matrix(coo_matrix((np.ones(len(R_), np.int8), (R_, C_)), shape=(nd + nc + nk + npr, 9 * B))); Jp.data[:] = 1
mad = lambda r: 1.4826 * float(np.median(np.abs(r))) + 1e-12
S = {'d': float(np.median(res1)) / np.sqrt(3) + 1e-9, 'c': 0.0, 'mu': float(np.median(x0[:, 6])), 's': mad(x0[:, 6] - np.median(x0[:, 6]))}
A_, t_ = mats(x0.ravel()); S['c'] = mad(T(A_, t_, cpx, cpa) - T(A_, t_, cpx, cpb))
ea = []                                                             # shape spread start: per-bone free affine fit's anisotropy
for b in bones:
    i = bi[b]; X = V0[idx[b]] - C[i]; Y = Vw[idx[b]] - Vw[idx[b]].mean(0)
    if len(X) < 4: continue
    M = np.linalg.lstsq(X, Y, rcond=None)[0].T; sv = np.linalg.norm(M @ P[i], axis=0); l = np.log(sv) - np.log(sv).mean(); ea += [l[0], l[1]]
S['e'] = mad(np.array(ea)); log('starting scales: data %.2f mm, joints %.2f mm, size spread %.3f, shape spread %.3f (log units)' % (1000 * S['d'], 1000 * S['c'], S['s'], S['e']))
EDGE = float(np.median(np.linalg.norm(Vw[FB[:, 0]] - Vw[FB[:, 1]], axis=1)))   # the warped bone meshes' median edge: the finest scale they can resolve
log('scale floor: median bone edge %.2f mm' % (1000 * EDGE))
x = x0.ravel().copy(); hist = []
for rnd in range(20):
    r = least_squares(resid, x, jac=jac, args=(S,), method='trf', tr_solver='lsmr', loss='soft_l1', x_scale='jac', max_nfev=60, verbose=0)   # exact derivatives
    A0, t0_ = mats(x); A1, t1 = mats(r.x); move = float(np.linalg.norm(T(A1, t1, bv, bown) - T(A0, t0_, bv, bown), axis=1).max()); x = r.x
    rr = resid(x, S); xb = x.reshape(B, 9)
    newS = {'d': max(mad(rr[:nd] * S['d']), EDGE), 'c': max(mad(rr[nd:nd + nc] * S['c']), EDGE), 'mu': float(np.median(xb[:, 6])), 's': mad(xb[:, 6] - np.median(xb[:, 6])), 'e': mad(xb[:, 7:9])}   # scales never finer than the bone meshes' own edge length
    hist.append({'round': rnd, 'cost': float(r.cost), 'nfev': int(r.nfev), 'largest_move_mm': 1000 * move, **{k: float(v) for k, v in S.items()}})
    log('round %d: cost %.0f after %d evaluations, largest bone move %.2f mm; new scales data %.2f mm, joints %.2f mm, size %.3f, shape %.3f (%.0f s)' % (
        rnd, r.cost, r.nfev, 1000 * move, 1000 * newS['d'], 1000 * newS['c'], newS['s'], newS['e'], time.time() - t00))
    xb_ = x.reshape(B, 9); R0[:] = Rot.from_rotvec(xb_[:, :3]).as_matrix() @ R0; xb_[:, :3] = 0; x = xb_.ravel()   # fold the rotation in: derivatives exact again
    if rnd > 0 and move < S['d']: break
    S = newS
# --- results and checks
A, t = mats(x); xb = x.reshape(B, 9); Vr = Vw.copy()
bva = np.concatenate([idx_all[b] for b in bones]); bowna = np.concatenate([np.full(len(idx_all[b]), bi[b]) for b in bones]); Vr[bva] = T(A, t, bva, bowna)
res = np.array([np.median(np.linalg.norm(Vr[idx_all[b]] - Vw[idx_all[b]], axis=1)) for b in bones])
gapj = np.linalg.norm(T(A, t, cpx, cpa) - T(A, t, cpx, cpb), axis=1)
dep = depth(Vr[bv]); out_b = np.bincount(bown, dep < 0, B) / np.bincount(bown, minlength=B)
def winding(Q, Vm, Fm, ch=2000):
    a, b_, c = Vm[Fm[:, 0]], Vm[Fm[:, 1]], Vm[Fm[:, 2]]; w = np.zeros(len(Q))
    for k in range(0, len(Q), ch):
        q = Q[k:k + ch, None]; A_, B_, C_ = a - q, b_ - q, c - q; la, lb, lc = [np.linalg.norm(z, axis=2) for z in (A_, B_, C_)]
        num = np.einsum('ijk,ijk->ij', A_, np.cross(B_, C_)); den = la * lb * lc + (A_ * B_).sum(2) * lc + (B_ * C_).sum(2) * la + (C_ * A_).sum(2) * lb
        w[k:k + ch] = np.arctan2(num, den).sum(1) / (2 * np.pi)
    return w
fob = OB[FB[:, 0]]; pen = []
for ia, ib, _ in pairs:                                            # bone b's vertices inside bone a, both ways
    for u, v in ((ia, ib), (ib, ia)):
        Fv = FB[fob == bones[u]]; q = Vr[idx_all[bones[v]]]
        if not len(Fv): continue
        near = cKDTree(Vr[idx_all[bones[u]]]).query(q)[0] < 3 * S['d']
        n_in = int((np.abs(winding(q[near], Vr, Fv)) > 0.5).sum()) if near.any() else 0
        pen.append((names[bones[v]], names[bones[u]], n_in, len(q)))
pen_bad = [p for p in pen if p[2] > 0]
lr = []
for b in bones:
    n = names[b]
    if n.endswith('.l') and n[:-2] + '.r' in names:
        o = names.index(n[:-2] + '.r')
        if o in bi: lr.append((n[:-2], float(np.exp(xb[bi[b], 6] - xb[bi[o], 6]) - 1)))
# which bones' size and shape the data pins down: information of the data rows vs. the prior's (1 per number in these units)
J = r.jac.tocsc() if hasattr(r.jac, 'tocsc') else csr_matrix(r.jac).tocsc(); Jd = J[:nd + nc + nk]; pinned = []
for i in range(B):
    blk = Jd[:, 9 * i + 6: 9 * i + 9]; ev = np.linalg.eigvalsh((blk.T @ blk).toarray()); pinned.append(int((ev > 1.0).sum()))
js = {'bones': B, 'contact_pairs': len(pairs), 'scales_final': {k: float(v) for k, v in S.items()}, 'rounds': hist,
      'residual_to_warp_mm': {'median': 1000 * float(np.median(res)), 'p90': 1000 * float(np.percentile(res, 90)), 'step1_median': 1000 * float(np.median(res1))},
      'joint_gap_mm': {'median': 1000 * float(np.median(gapj)), 'p90': 1000 * float(np.percentile(gapj, 90)), 'start_scale': 1000 * float(hist[0]['c'])},
      'bone_outside_skin': {'vertices_share': float((dep < 0).mean()), 'bones_with_any': [names[bones[i]] for i in np.flatnonzero(out_b > 0)][:60], 'max_out_mm': 1000 * float(max(0, -dep.min()))},
      'bone_in_bone': {'pairs_checked': len(pen), 'pairs_with_overlap': len(pen_bad), 'worst': sorted(pen_bad, key=lambda p: -p[2] / p[3])[:15]},
      'left_right_size_diff': {'median_abs': float(np.median([abs(v) for _, v in lr])) if lr else None, 'largest': sorted(lr, key=lambda z: -abs(z[1]))[:10]},
      'size_shape_pinned_by_data': {'all_3': int(sum(p == 3 for p in pinned)), 'some': int(sum(0 < p < 3 for p in pinned)), 'prior_only': [names[bones[i]] for i in range(B) if pinned[i] == 0]},
      'seconds': round(time.time() - t00)}
json.dump(js, open(os.path.join(od, 'skeleton_fit.json'), 'w'), indent=1)
np.savez(os.path.join(od, 'skeleton_fit.npz'), bones=np.array(bones), bone_names=np.array([names[b] for b in bones]), params=xb, R0=R0, P=P, C=C, A=A, t=t,
         V_rigid=Vr.astype(np.float32), F=W['F'], F_bone=FB, F_muscle=W['F_muscle'], obj=OB, residual=res, residual_step1=res1, pinned=np.array(pinned),
         contact_pairs=np.array([(bones[a], bones[b]) for a, b, _ in pairs]))
log('done: residual to warp median %.2f mm (step 1: %.2f); joint gap median %.2f mm; %.2f%% bone vertices outside the skin; %d of %d touching pairs overlap; size+shape pinned by data for %d bones, prior only for %d (%.0f s)' % (
    js['residual_to_warp_mm']['median'], js['residual_to_warp_mm']['step1_median'], js['joint_gap_mm']['median'], 100 * js['bone_outside_skin']['vertices_share'],
    len(pen_bad), len(pen), js['size_shape_pinned_by_data']['all_3'], len(js['size_shape_pinned_by_data']['prior_only']), time.time() - t00))
