"""The skull placed and shaped under the face by paired face points (forensic reconstruction run backwards: from the face
to the skull), bone by bone, the rest of the skeleton held where skeleton_fit.py put it:
  E:\\splat\\.venv\\Scripts\\python.exe skull_landmark_fit.py <fits dir> <model dir> <atlas head dir>
  pairs     the same detector's face points on the atlas's own head skin and on the model (zoom_joints3d.py on both):
            brows, nose, eyes, mouth (51 points). The 17 jaw-outline points follow the visible outline and shift with
            the view, so they are not anatomical points and are left out.
  carrier   each atlas face point rides on the head bone nearest to it in the atlas (its soft tissue: the atlas's own
            thickness there, scaled with the bone), so  T_b(atlas point) = model point  is the constraint
  start     one similarity (Umeyama) of all pairs for every head bone: the whole skull as one piece
  energy    pairs / sigma_l  +  joints (contact points of touching pairs, skeleton_fit.py; pairs with a fixed bone hold
            the skull on the neck)  +  bone inside the head surface  +  size and shape priors (skeleton_fit.json's
            spreads; size centred on the head bones' own median, re-estimated each round)
            The warp's data term is left out for the head: its slices put the skull too deep (tissue 3x the atlas's).
  scales    sigma_l = 1.4826 x median absolute pair residual, never under the model's median edge; joints from
            skeleton_fit.json; re-estimated each round (variance components)
  solver    scipy least_squares (trf, soft-L1, exact sparse Jacobian), rotation folded in after each round, until no bone
            moves more than sigma_l
Writes <fits dir>/skull_fit.npz (skeleton_fit.npz's keys, head bones replaced) and skull_fit.json (pair residuals,
tissue thickness against the atlas's per face group, joint gaps incl. skull-neck, sizes, outside-surface share)."""
import sys, os, json, re, time, numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segments import SEG, ORDER
od, MD, AH = sys.argv[1:4]; t00 = time.time(); log = lambda *a: print('SL', *a, flush=True)
W = np.load(os.path.join(od, 'atlas_warp.npz')); V0 = W['V0'].astype(np.float64); OB = W['obj']; names = [str(x) for x in W['names']]; sysn = [str(x) for x in W['system']]
K = np.load(os.path.join(od, 'skeleton_fit.npz'), allow_pickle=True); SJ = json.load(open(os.path.join(od, 'skeleton_fit.json')))['scales_final']
bones = [int(b) for b in K['bones']]; B = len(bones); bi = {b: i for i, b in enumerate(bones)}; R0, P, C = K['R0'].copy(), K['P'], K['C']; x = K['params'].astype(np.float64).ravel().copy()
idx_all = {b: np.flatnonzero(OB == b) for b in bones}
def bone_names(spec): return [n for n in spec if n in names] if isinstance(spec, list) else [n for n, s in zip(names, sysn) if s == 'bone' and re.search(spec, n)]
listed = {}
for s in ORDER:
    for n in bone_names(SEG[s]['bones']): listed.setdefault(n, s)
lb = [b for b in bones if names[b] in listed]; lv = np.concatenate([idx_all[b] for b in lb]); ltree = cKDTree(V0[lv])
seg_of = {b: listed[names[b]] if names[b] in listed else listed[names[OB[lv[ltree.query(V0[idx_all[b]].mean(0))[1]]]]] for b in bones}
free = np.array([seg_of[b] == 'head' for b in bones]); fi_ = np.flatnonzero(free)
def mats(x):
    x = x.reshape(B, 9); Rr = Rot.from_rotvec(x[:, :3]).as_matrix() @ R0; e = np.c_[x[:, 7], x[:, 8], -x[:, 7] - x[:, 8]]
    return np.exp(x[:, 6])[:, None, None] * Rr @ P @ (np.exp(e)[:, :, None] * np.transpose(P, (0, 2, 1))), x[:, 3:6]
def Tp(A, t, X, own): return np.einsum('nij,nj->ni', A[own], X - C[own]) + t[own]
def dpts(x, X, own):
    xb = x.reshape(B, 9); Rr = Rot.from_rotvec(xb[:, :3]).as_matrix() @ R0; s = np.exp(xb[:, 6])
    ee = np.exp(np.c_[xb[:, 7], xb[:, 8], -xb[:, 7] - xb[:, 8]]); Pt = np.einsum('nji,nj->ni', P[own], X - C[own])
    q = s[own, None] * np.einsum('nij,nj->ni', P[own], ee[own] * Pt); Rq = np.einsum('nij,nj->ni', Rr[own], q); J = np.zeros((len(X), 3, 9))
    J[:, 0, 1], J[:, 0, 2], J[:, 1, 0], J[:, 1, 2], J[:, 2, 0], J[:, 2, 1] = Rq[:, 2], -Rq[:, 1], -Rq[:, 2], Rq[:, 0], Rq[:, 1], -Rq[:, 0]
    J[:, 0, 3] = J[:, 1, 4] = J[:, 2, 5] = 1; J[:, :, 6] = Rq
    for g, c in ((7, np.array([1.0, 0, -1])), (8, np.array([0, 1.0, -1]))):
        J[:, :, g] = np.einsum('nij,nj->ni', Rr[own], s[own, None] * np.einsum('nij,nj->ni', P[own], ee[own] * c * Pt))
    return J
def hull_ids(ii):
    if len(ii) < 8: return ii
    try: return ii[np.unique(ConvexHull(V0[ii]).vertices)]
    except Exception: return ii
# --- the pairs
LA = np.load(os.path.join(AH, 'parts_zoom', 'head', 'points3d.npz'))['X'][:68]; LE = np.load(os.path.join(MD, 'parts_zoom', 'head', 'points3d.npz'))['X'][:68]
GR = {'brows': range(17, 27), 'nose bridge': range(27, 31), 'nose base': range(31, 36), 'eyes': range(36, 48), 'mouth': range(48, 68)}
sel = np.array([i for i in range(17, 68) if np.isfinite(LA[i]).all() and np.isfinite(LE[i]).all()]); la, le = LA[sel], LE[sel]
hv_all = np.concatenate([idx_all[bones[i]] for i in fi_]); ho_all = np.concatenate([np.full(len(idx_all[bones[i]]), i) for i in fi_]); htree = cKDTree(V0[hv_all])
tk_a, kk = htree.query(la); lown = ho_all[kk]                       # atlas tissue thickness at each point and the bone that carries it
log('%d pairs on %d carrier bones: %s' % (len(sel), len(np.unique(lown)), ', '.join('%s %d' % (names[bones[i]], (lown == i).sum()) for i in np.unique(lown))))
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, S_, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt; s = np.trace(np.diag(S_) @ D) / (Xc ** 2).sum(1).mean(); return R, s, my - s * R @ mx
Ru, su, tu = umeyama(la, le); xb = x.reshape(B, 9); x_fit = x.copy()
for i in fi_: R0[i] = Ru; xb[i, :3] = 0; xb[i, 3:6] = su * Ru @ C[i] + tu; xb[i, 6] = np.log(su); xb[i, 7:9] = 0
x = xb.ravel(); R0_fit = K['R0']
# --- the model surface (head), contacts
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MNo = np.load(os.path.join(MD, 'parts', 'orig_normals.npz')); MF, MN = MNo['F'], MNo['N'].astype(np.float64)
vid = np.load(os.path.join(MD, 'parts_zoom', 'head_vids.npy')); HV, HN = MV[vid], MN[vid]; mtree = cKDTree(HV)
MEDGE = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1)))
def depth(p): _, k = mtree.query(p); return -np.einsum('ij,ij->i', p - HV[k], HN[k])
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
trees = {}; cpa, cpb, cpx, cpn = [], [], [], []
for a, b in K['contact_pairs']:
    a, b = int(a), int(b)
    if not (free[bi[a]] or free[bi[b]]): continue
    for u_, v_ in ((a, b), (b, a)):
        if v_ not in trees: trees[v_] = cKDTree(V0[idx_all[v_]])
        dd = trees[v_].query(V0[idx_all[u_]])[0]; l = np.log(dd + 1e-9); pts = idx_all[u_][l < otsu(l)] if len(l) > 2 else idx_all[u_]
        pts = hull_ids(pts) if len(pts) > 200 else pts              # a suture's thousands of contact points: their hull spans it
        cpa.append(np.full(len(pts), bi[a])); cpb.append(np.full(len(pts), bi[b])); cpx.append(pts)
cpa, cpb, cpx = np.concatenate(cpa), np.concatenate(cpb), np.concatenate(cpx); neck = free[cpa] != free[cpb]
hv = np.concatenate([hull_ids(idx_all[bones[i]]) for i in fi_]); ho = np.concatenate([np.full(len(hull_ids(idx_all[bones[i]])), i) for i in fi_])
log('%d free head bones, %d contact points (%d to fixed bones), %d hull points' % (len(fi_), len(cpx), neck.sum(), len(hv)))
cols = (fi_[:, None] * 9 + np.arange(9)).ravel(); colmap = -np.ones(9 * B, int); colmap[cols] = np.arange(len(cols)); nf = len(fi_)
def full(z): xx = xbase.copy(); xx[cols] = z; return xx
def resid(z, S):
    xx = full(z); A, t = mats(xx); xb = xx.reshape(B, 9)
    rl = ((Tp(A, t, la, lown) - le) / S['l']).ravel(); rc = ((Tp(A, t, V0[cpx], cpa) - Tp(A, t, V0[cpx], cpb)) / S['c']).ravel()
    ri = np.maximum(0, -depth(Tp(A, t, V0[hv], ho))) / S['l']
    return np.r_[rl, rc, ri, (xb[fi_, 6] - S['mu']) / S['s'], xb[fi_, 7] / S['e'], xb[fi_, 8] / S['e']]
def jac(z, S):
    xx = full(z); A, t = mats(xx); rows, cl, vals = [], [], []
    def put(r_idx, own, Jb):
        m = free[own]
        if m.any(): rows.append(np.repeat(r_idx[m], 9)); cl.append(colmap[(own[m][:, None] * 9 + np.arange(9)).ravel()]); vals.append(Jb[m].ravel())
    Jl = dpts(xx, la, lown) / S['l']; n0 = 0
    for a_ in range(3): put(3 * np.arange(len(la)) + a_, lown, Jl[:, a_])
    n0 = 3 * len(la); Ja, Jb_ = dpts(xx, V0[cpx], cpa) / S['c'], -dpts(xx, V0[cpx], cpb) / S['c']
    for a_ in range(3): put(n0 + 3 * np.arange(len(cpx)) + a_, cpa, Ja[:, a_]); put(n0 + 3 * np.arange(len(cpx)) + a_, cpb, Jb_[:, a_])
    n0 += 3 * len(cpx); ph = Tp(A, t, V0[hv], ho); _, k = mtree.query(ph); ac = ((ph - HV[k]) * HN[k]).sum(1) > 0
    put(n0 + np.arange(len(hv)), ho, np.einsum('ni,nij->nj', HN[k], dpts(xx, V0[hv], ho)) * (ac / S['l'])[:, None]); n0 += len(hv)
    for g, sc in ((0, S['s']), (1, S['e']), (2, S['e'])): rows.append(n0 + g * nf + np.arange(nf)); cl.append(colmap[fi_ * 9 + 6 + g]); vals.append(np.full(nf, 1 / sc))
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cl))), shape=(n0 + 3 * nf, len(cols))).tocsr()
mad = lambda r: 1.4826 * float(np.median(np.abs(r))) + 1e-12
def report(x_, R0_):
    global R0
    keep = R0; R0 = R0_; A, t = mats(x_); R0 = keep; xb = x_.reshape(B, 9)
    rl = np.linalg.norm(Tp(A, t, la, lown) - le, axis=1); gap = np.linalg.norm(Tp(A, t, V0[cpx], cpa) - Tp(A, t, V0[cpx], cpb), axis=1)
    Vh = Tp(A, t, V0[hv_all], ho_all); tk = cKDTree(Vh).query(le)[0]; sz = np.exp(xb[fi_, 6]); ratio = tk / (np.median(sz) * tk_a); dep = depth(Tp(A, t, V0[hv], ho))
    out = {'pair_residual_mm': {'median': 1000 * float(np.median(rl)), 'p90': 1000 * float(np.percentile(rl, 90))},
           'tissue_vs_atlas': {'median': float(np.median(ratio)), **{g: {'atlas_scaled_mm': 1000 * float(np.median(sz) * np.median(tk_a[np.isin(sel, list(r))])), 'model_mm': 1000 * float(np.median(tk[np.isin(sel, list(r))])),
                               'ratio': float(np.median(ratio[np.isin(sel, list(r))]))} for g, r in GR.items()}},
           'joint_gap_mm': {'skull_median': 1000 * float(np.median(gap[~neck])), 'skull_p90': 1000 * float(np.percentile(gap[~neck], 90)), 'skull_to_neck_median': 1000 * float(np.median(gap[neck])) if neck.any() else None},
           'size': {'median': float(np.median(sz)), 'min': float(sz.min()), 'max': float(sz.max())}, 'outside_surface_share': float((dep < 0).mean()), 'outside_worst_mm': 1000 * float(max(0, -dep.min()))}
    return out, A, t
rep = {}
rep['slice_route'], _, _ = report(x_fit, R0_fit); rep['one_piece'], A, t = report(x, R0)
for k in rep: log(k, json.dumps(rep[k]))
S = {'l': max(mad(np.linalg.norm(Tp(A, t, la, lown) - le, axis=1)) / np.sqrt(3), MEDGE), 'c': float(SJ['c']), 'mu': float(np.log(su)), 's': float(SJ['s']), 'e': float(SJ['e'])}
hist = []
for rnd in range(12):
    xbase = x.copy(); r = least_squares(resid, x[cols].copy(), jac=jac, args=(S,), method='trf', tr_solver='lsmr', loss='soft_l1', x_scale='jac', max_nfev=60)
    xn = full(r.x); A0, t0 = mats(x); A1, t1 = mats(xn); move = float(np.linalg.norm(Tp(A1, t1, V0[hv], ho) - Tp(A0, t0, V0[hv], ho), axis=1).max()); x = xn
    xb_ = x.reshape(B, 9); R0[fi_] = Rot.from_rotvec(xb_[fi_, :3]).as_matrix() @ R0[fi_]; xb_[fi_, :3] = 0; x = xb_.ravel(); A, t = mats(x)
    newl = max(mad((Tp(A, t, la, lown) - le).ravel()), MEDGE); newc = max(mad((Tp(A, t, V0[cpx], cpa) - Tp(A, t, V0[cpx], cpb)).ravel()), MEDGE)
    hist.append({'round': rnd, 'cost': float(r.cost), 'nfev': int(r.nfev), 'largest_move_mm': 1000 * move, 'sigma_l_mm': 1000 * S['l'], 'sigma_c_mm': 1000 * S['c']})
    log('round %d: cost %.0f (%d evaluations), largest move %.2f mm; scales pairs %.2f -> %.2f mm, joints %.2f -> %.2f mm (%.0f s)' % (rnd, r.cost, r.nfev, 1000 * move, 1000 * S['l'], 1000 * newl, 1000 * S['c'], 1000 * newc, time.time() - t00))
    if rnd > 0 and move < S['l']: break
    S['l'], S['c'], S['mu'] = newl, newc, float(np.median(xb_[fi_, 6]))
rep['per_bone'], A, t = report(x, R0); log('per_bone', json.dumps(rep['per_bone'])); xb = x.reshape(B, 9)
rep['rounds'] = hist; rep['pairs'] = int(len(sel)); rep['carriers'] = {names[bones[i]]: int((lown == i).sum()) for i in np.unique(lown)}
rep['sizes'] = {names[bones[i]]: float(np.exp(xb[i, 6])) for i in fi_}; rep['seconds'] = round(time.time() - t00)
json.dump(rep, open(os.path.join(od, 'skull_fit.json'), 'w'), indent=1)
Vr = K['V_rigid'].astype(np.float64).copy(); Vr[hv_all] = Tp(A, t, V0[hv_all], ho_all); FB = K['F_bone']; hm = np.zeros(len(V0), bool); hm[hv_all] = True
np.savez(os.path.join(od, 'skull_fit.npz'), bones=K['bones'], bone_names=K['bone_names'], params=xb, R0=R0, P=P, C=C, A=A, t=t, V_rigid=Vr.astype(np.float32), F=K['F'], F_bone=FB,
         F_head=FB[hm[FB].all(1)], F_muscle=K['F_muscle'], obj=OB, free=free, contact_pairs=K['contact_pairs'])
log('done (%.0f s)' % (time.time() - t00))
