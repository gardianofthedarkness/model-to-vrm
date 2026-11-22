"""Fine-tuning of the fitted skeleton (skeleton_fit.py) at the body's ends - head, hands, feet - by CT slices per bone:
  E:\\splat\\.venv\\Scripts\\python.exe bone_slice_refine.py <atlas_warp.npz> <model dir> [regions]   regions: head,hand_r,hand_l,foot_r,foot_l
Why: the segment slices (fit_slices.py) fit one atlas outline per level of a whole segment. At the ends that is too coarse:
a slice through the toes or fingers holds several digits in one outline, high heels bend the foot away from the atlas's
flat one, and the head has no joint at its top (its length came from the hair). Here every bone of those regions is
refined against the model's own skin, seen in slices across that bone.
  regions   a bone belongs to the segment that lists it (segments.py); an unlisted bone (finger phalanges) to the segment
            of its nearest listed bone, as in atlas_warp.py. Bones outside the regions stay where skeleton_fit put them.
  skin      every atlas skin vertex rides on its nearest atlas bone (rest pose); the model's vertices belong to the bone
            nearest to them in the current placement (fit_slices.py's ownership). A region's skin = the skin its bones carry.
  slices    per bone that carries skin: planes across its longest principal axis, through its whole length, spaced as the
            72 rays around the section are (2 pi r / 72, r = the carried skin's median distance from the axis), so a
            slice samples along and around the bone alike. In the atlas once, in the model per round: the same plane
            carried by the bone's current transform. In each plane, per ray from the bone's axis point, the first surface
            out (planes.py) - in a slice across one finger that is that finger's own skin, not the next finger's.
  residual  each model point to the nearest place on the plane's atlas skin outline, along its outward normal (ICP,
            correspondences fixed during a solve, as in fit_slices.py): seen skin two-sided; cloth, hair, prop one-sided
            (the skin may lie inside them, never outside), so hair no longer sets the head's length; tight cloth also
            weakly near (scale: the section's own spread, as fit_slices.py)
  energy    slices + joints (contact points of touching pairs, as in skeleton_fit.py; a pair with a fixed bone holds the
            region on to the rest of the skeleton) + bones inside the model surface + the size / shape priors of
            skeleton_fit. Scales: skin = 1.4826 x median absolute skin residual, re-estimated each round
            (variance components), never finer than the model's median edge; joints and priors from skeleton_fit.json.
  solver    scipy least_squares (trf, soft-L1, exact sparse Jacobian of the placed outline points); new slices and
            correspondences each round, until no bone moves more than the skin scale.
Writes <fits dir>/skeleton_refined[_<regions>].npz (same keys as skeleton_fit.npz), skeleton_refined.json (per region: slices,
points, skin residual and cloth/hair overlap before / after, bone outside the skin), refine_<region>_sheet.png (slices:
model points by material, atlas outline before grey / after red, the bone's own section before / after)."""
import sys, os, json, re, time, numpy as np, cv2
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segments import SEG, ORDER
import planes as PL
fp, MD = sys.argv[1], sys.argv[2]; REG = sys.argv[3].split(',') if len(sys.argv) > 3 else ['head', 'hand_r', 'hand_l', 'foot_r', 'foot_l']
od = os.path.dirname(os.path.abspath(fp)); t00 = time.time(); TAG = 'skeleton_refined' + ('' if len(sys.argv) <= 3 else '_' + '_'.join(REG))   # one run per region set can go in parallel
LOGF = open(os.path.join(od, TAG + '.log'), 'w', encoding='utf-8')
log = lambda *a: (print('BR', *a, flush=True), LOGF.write('BR ' + ' '.join(map(str, a)) + chr(10)), LOGF.flush())
W = np.load(fp); V0 = W['V0'].astype(np.float64); OB = W['obj']; names = [str(x) for x in W['names']]; sysn = np.array([str(x) for x in W['system']])
K = np.load(os.path.join(od, 'skeleton_fit.npz'), allow_pickle=True); SJ = json.load(open(os.path.join(od, 'skeleton_fit.json')))['scales_final']
bones = [int(b) for b in K['bones']]; B = len(bones); bi = {b: i for i, b in enumerate(bones)}
R0, P, C = K['R0'].copy(), K['P'], K['C']; x = K['params'].astype(np.float64).ravel().copy()
idx_all = {b: np.flatnonzero(OB == b) for b in bones}
# --- regions
def bone_names(spec):
    return [n for n in spec if n in names] if isinstance(spec, list) else [n for n, s in zip(names, sysn) if s == 'bone' and re.search(spec, n)]
listed = {}
for s in ORDER:
    for n in bone_names(SEG[s]['bones']): listed.setdefault(n, s)
lb = [b for b in bones if names[b] in listed]; lv = np.concatenate([idx_all[b] for b in lb]); lo = np.concatenate([np.full(len(idx_all[b]), names[b]) for b in lb])
ltree = cKDTree(V0[lv]); seg_of = {}
for b in bones: seg_of[b] = listed[names[b]] if names[b] in listed else listed[lo[ltree.query(V0[idx_all[b]].mean(0))[1]]]
free = np.array([seg_of[b] in REG for b in bones]); fi_ = np.flatnonzero(free)
log('regions %s: %d free bones (%s), %d fixed' % (','.join(REG), free.sum(), ', '.join('%s %d' % (r, sum(seg_of[b] == r for b in bones)) for r in REG), B - free.sum()))
# --- the bone model (skeleton_fit.py)
def mats(x):
    x = x.reshape(B, 9); Rr = Rot.from_rotvec(x[:, :3]).as_matrix() @ R0; e = np.c_[x[:, 7], x[:, 8], -x[:, 7] - x[:, 8]]
    return np.exp(x[:, 6])[:, None, None] * Rr @ P @ (np.exp(e)[:, :, None] * np.transpose(P, (0, 2, 1))), x[:, 3:6]
def Tp(A, t, X, own): return np.einsum('nij,nj->ni', A[own], X - C[own]) + t[own]
def dpts(x, X, own):
    """exact derivatives (n, 3, 9) of the placed points T_b(X) by their bone's 9 numbers (rotation: small-angle, exact at r = 0)"""
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
hid = {b: hull_ids(idx_all[b]) for b in bones}
# --- the model
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MNo = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))
MF, MN = MNo['F'], MNo['N'].astype(np.float64); M_ = np.load(os.path.join(MD, 'contacts', 'material.npz')); MAT = [str(m) for m in M_['materials']]
fmat = M_['material'][MF[:, 0]]; FL = MAT.index('flesh'); CLO = MAT.index('cloth'); mtree = cKDTree(MV)
MEDGE = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1)))
FB = W['F_bone']; EDGE = float(np.median(np.linalg.norm(V0[FB[:, 0]] - V0[FB[:, 1]], axis=1)))
def depth(p): _, k = mtree.query(p); return -np.einsum('ij,ij->i', p - MV[k], MN[k])
# --- atlas skin carried by bones
FS = W['F_skin']; sv = np.unique(FS); ab = np.concatenate([hid[b] for b in bones]); abo = np.concatenate([np.full(len(hid[b]), bi[b]) for b in bones])
carry = -np.ones(len(V0), int); carry[sv] = abo[cKDTree(V0[ab]).query(V0[sv])[1]]   # bone index (into bones) per skin vertex
log('atlas skin: %d vertices carried by %d bones; model median edge %.2f mm, bone edge %.2f mm' % (len(sv), len(np.unique(carry[sv])), 1000 * MEDGE, 1000 * EDGE))
RB = {r: np.array([seg_of[b] == r for b in bones]) for r in REG}   # per region: which bones (index) are in it
def region_faces_atlas(r): return FS[RB[r][carry[FS]].all(1)]
def model_owner(x):
    """per model vertex: the bone (index) nearest to it in the current placement (hull points)"""
    A, t = mats(x); pts = Tp(A, t, V0[ab], abo); return abo[cKDTree(pts).query(MV)[1]]
def first_surface(Vx, Fx, p, n, e1, e2):
    """per ray (TH angle bins) from p in the plane: the nearest cut point; returns 2D points, face ids, bins"""
    sg, fi = PL.cut(Vx, Fx, p, n, e1, e2, ret_face=True)
    if not len(sg): return np.zeros((0, 2)), np.zeros(0, int), np.zeros(0, int)
    u = np.linspace(0, 1, 5)[None, :, None]; Q = (sg[:, :1] + u * (sg[:, 1:] - sg[:, :1])).reshape(-1, 2); ff = np.repeat(fi, 5)
    b = ((np.arctan2(Q[:, 1], Q[:, 0]) + np.pi) / (2 * np.pi) * PL.TH).astype(int) % PL.TH; r = np.hypot(Q[:, 0], Q[:, 1])
    o = np.lexsort((r, b)); f = o[np.r_[True, b[o][1:] != b[o][:-1]]]; return Q[f], ff[f], b[f]
# --- atlas slices per carrying bone (once)
SL = []                                                            # per slice: bone index, atlas plane (p, n, e1), outline atlas points + carriers
for r in REG:
    Fr = region_faces_atlas(r)
    for i in fi_:
        b = bones[i]
        if seg_of[b] != r: continue
        sk = sv[carry[sv] == i]
        if len(sk) < PL.TH // 4: continue                           # carries (almost) no skin: nothing to see in its slices
        a, e1r = P[i][:, 0], P[i][:, 1]; d = V0[sk] - C[i]; rr = float(np.median(np.linalg.norm(d - np.outer(d @ a, a), axis=1)))
        pr = (V0[idx_all[b]] - C[i]) @ a; sp = 2 * np.pi * rr / PL.TH; lev = np.arange(pr.min() + sp / 2, pr.max(), sp)
        for l in lev:
            p = C[i] + l * a; e1, e2 = PL.frame(a, e1r); Q, ff, bb = first_surface(V0, Fr, p, a, e1, e2)
            if len(Q) < PL.TH // 4: continue                        # the plane misses the carried skin on most rays
            Q3 = p + Q @ np.c_[e1, e2].T; vv = Fr[ff]; nv = vv[np.arange(len(vv)), np.argmin(np.linalg.norm(V0[vv] - Q3[:, None], axis=2), 1)]
            o = np.argsort(bb); SL.append(dict(i=i, reg=r, p=p, n=a, e1=e1, A=Q3[o], own=np.full(len(o), i)))   # the outline rides on the plane's own bone: carried by
                                                                  # each skin vertex's nearest bone it tore into spikes as soon as neighbouring bones parted
log('atlas slices: %d planes across %d bones (%s) (%.0f s)' % (len(SL), len({s['i'] for s in SL}), ', '.join('%s %d' % (r, sum(s['reg'] == r for s in SL)) for r in REG), time.time() - t00))
# --- per round: the planes carried into the model, model points, correspondences
def world_plane(A, t, s):
    i = s['i']; p = A[i] @ (s['p'] - C[i]) + t[i]; n = np.linalg.solve(A[i].T, s['n']); n /= np.linalg.norm(n)
    e1 = A[i] @ s['e1']; e1 -= n * (e1 @ n); e1 /= np.linalg.norm(e1); return p, n, e1, np.cross(n, e1)
def correspond(x):
    A, t = mats(x); own = model_owner(x); rows = []; per = []; Fm = {rg: MF[RB[rg][own[MF]].all(1)] for rg in REG}
    for k, s in enumerate(SL):
        p, n, e1, e2 = world_plane(A, t, s); Q3 = Tp(A, t, s['A'], s['own']); q = np.c_[(Q3 - p) @ e1, (Q3 - p) @ e2]
        Pm, ffm, _ = first_surface(MV, Fm[s['reg']], p, n, e1, e2)
        if len(Pm) < 3: per.append(None); continue
        qa, qb = q, np.roll(q, -1, 0); d = qb - qa; u = np.clip(((Pm[:, None] - qa) * d).sum(2) / ((d * d).sum(1) + 1e-30), 0, 1)
        D = np.linalg.norm(Pm[:, None] - qa - u[..., None] * d, axis=2); j = D.argmin(1); uu = u[np.arange(len(Pm)), j]
        sgn = np.sign(np.sum(qa[:, 0] * qb[:, 1] - qb[:, 0] * qa[:, 1])) or 1.0; dj = d[j]; n2 = sgn * np.c_[dj[:, 1], -dj[:, 0]]
        n2 /= np.linalg.norm(n2, axis=1, keepdims=True) + 1e-30; n3 = n2[:, :1] * e1 + n2[:, 1:] * e2; j2 = (j + 1) % len(q)
        rows.append(dict(a=s['A'][j], b=s['A'][j2], oa=s['own'][j], ob=s['own'][j2], u=uu, n=n3, P=p + Pm @ np.c_[e1, e2].T,
                         flesh=fmat[ffm] == FL, cloth=fmat[ffm] == CLO, slice=np.full(len(Pm), k),
                         spread=np.full(len(Pm), 1.4826 * float(np.median(np.abs(np.hypot(*Pm.T) - np.median(np.hypot(*Pm.T))))) + MEDGE)))
        per.append((p, n, e1, e2, Pm, fmat[ffm]))
    cat = {key: np.concatenate([r_[key] for r_ in rows]) for key in rows[0]}; return cat, per
# --- contacts touching a free bone (skeleton_fit.py's contact points)
trees = {}
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i)
    return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
cpa, cpb, cpx = [], [], []
for a, b in K['contact_pairs']:
    a, b = int(a), int(b)
    if not (free[bi[a]] or free[bi[b]]): continue
    for u_, v_ in ((a, b), (b, a)):
        if v_ not in trees: trees[v_] = cKDTree(V0[idx_all[v_]])
        dd = trees[v_].query(V0[idx_all[u_]])[0]; l = np.log(dd + 1e-9); pts = idx_all[u_][l < otsu(l)] if len(l) > 2 else idx_all[u_]
        cpa.append(np.full(len(pts), bi[a])); cpb.append(np.full(len(pts), bi[b])); cpx.append(pts)
cpa, cpb, cpx = np.concatenate(cpa), np.concatenate(cpb), np.concatenate(cpx)
hv = np.concatenate([hid[bones[i]] for i in fi_]); ho = np.concatenate([np.full(len(hid[bones[i]]), i) for i in fi_])
log('contacts with a free bone: %d points; bone hull points kept inside: %d' % (len(cpx), len(hv)))
cols = (fi_[:, None] * 9 + np.arange(9)).ravel(); colmap = -np.ones(9 * B, int); colmap[cols] = np.arange(len(cols))
def full(z): xx = xbase.copy(); xx[cols] = z; return xx
def resid(z, cr, S):
    xx = full(z); A, t = mats(xx); xb = xx.reshape(B, 9)
    q = (1 - cr['u'])[:, None] * Tp(A, t, cr['a'], cr['oa']) + cr['u'][:, None] * Tp(A, t, cr['b'], cr['ob']); r = ((q - cr['P']) * cr['n']).sum(1)
    rk = np.where(cr['flesh'], r, np.maximum(0, r)) / S['k']; rw = r[cr['cloth']] / cr['spread'][cr['cloth']]   # tight cloth: also weakly near (the section's own spread)
    rc = ((Tp(A, t, V0[cpx], cpa) - Tp(A, t, V0[cpx], cpb)) / S['c']).ravel()
    ri = np.maximum(0, -depth(Tp(A, t, V0[hv], ho))) / S['k']
    rp = np.r_[(xb[fi_, 6] - S['mu']) / S['s'], xb[fi_, 7] / S['e'], xb[fi_, 8] / S['e']]
    return np.r_[rk, rw, rc, ri, rp]
def jac(z, cr, S):
    xx = full(z); A, t = mats(xx); rows, cl, vals = [], [], []; n0 = 0
    def put(r_idx, own, Jb):                                       # Jb (n, 9) rows r_idx, columns of bone own (free only)
        m = free[own]
        if not m.any(): return
        rows.append(np.repeat(r_idx[m], 9)); cl.append(colmap[(own[m][:, None] * 9 + np.arange(9)).ravel()]); vals.append(Jb[m].ravel())
    q = (1 - cr['u'])[:, None] * Tp(A, t, cr['a'], cr['oa']) + cr['u'][:, None] * Tp(A, t, cr['b'], cr['ob']); r = ((q - cr['P']) * cr['n']).sum(1)
    act = (cr['flesh'] | (r > 0)) / S['k']; ri_ = np.arange(len(r))
    put(ri_, cr['oa'], np.einsum('ni,nij->nj', cr['n'], dpts(xx, cr['a'], cr['oa'])) * ((1 - cr['u']) * act)[:, None])
    put(ri_, cr['ob'], np.einsum('ni,nij->nj', cr['n'], dpts(xx, cr['b'], cr['ob'])) * (cr['u'] * act)[:, None])
    n0 = len(r); cm = np.flatnonzero(cr['cloth']); w_ = 1 / cr['spread'][cm]; rr_ = n0 + np.arange(len(cm))
    put(rr_, cr['oa'][cm], np.einsum('ni,nij->nj', cr['n'][cm], dpts(xx, cr['a'][cm], cr['oa'][cm])) * ((1 - cr['u'][cm]) * w_)[:, None])
    put(rr_, cr['ob'][cm], np.einsum('ni,nij->nj', cr['n'][cm], dpts(xx, cr['b'][cm], cr['ob'][cm])) * (cr['u'][cm] * w_)[:, None])
    n0 += len(cm); Ja, Jb_ = dpts(xx, V0[cpx], cpa) / S['c'], -dpts(xx, V0[cpx], cpb) / S['c']
    for a_ in range(3): put(n0 + 3 * np.arange(len(cpx)) + a_, cpa, Ja[:, a_]); put(n0 + 3 * np.arange(len(cpx)) + a_, cpb, Jb_[:, a_])
    n0 += 3 * len(cpx); ph = Tp(A, t, V0[hv], ho); _, k = mtree.query(ph); ac = ((ph - MV[k]) * MN[k]).sum(1) > 0
    put(n0 + np.arange(len(hv)), ho, np.einsum('ni,nij->nj', MN[k], dpts(xx, V0[hv], ho)) * (ac / S['k'])[:, None]); n0 += len(hv)
    nf = len(fi_)
    for g, sc in ((0, S['s']), (1, S['e']), (2, S['e'])):
        rows.append(n0 + g * nf + np.arange(nf)); cl.append(colmap[fi_ * 9 + 6 + g]); vals.append(np.full(nf, 1 / sc))
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cl))), shape=(n0 + 3 * nf, len(cols))).tocsr()
mad = lambda r: 1.4826 * float(np.median(np.abs(r))) + 1e-12
def stats(cr, x_):
    A, t = mats(x_); q = (1 - cr['u'])[:, None] * Tp(A, t, cr['a'], cr['oa']) + cr['u'][:, None] * Tp(A, t, cr['b'], cr['ob']); r = ((q - cr['P']) * cr['n']).sum(1)
    out = {}
    for rg in REG:
        m = SLREG[cr['slice']] == rg
        f, o = m & cr['flesh'], m & ~cr['flesh']
        out[rg] = {'skin_points': int(f.sum()), 'skin_residual_median_mm': 1000 * float(np.median(np.abs(r[f]))) if f.any() else None,
                   'other_points': int(o.sum()), 'skin_outside_cloth_hair_median_mm': 1000 * float(np.median(np.maximum(0, r[o]))) if o.any() else None,
                   'skin_outside_cloth_hair_p90_mm': 1000 * float(np.percentile(np.maximum(0, r[o]), 90)) if o.any() else None}
    return out, r
SLREG = np.array([s_['reg'] for s_ in SL]); cr, per0 = correspond(x); xbase = x.copy(); st0, r0 = stats(cr, x); per_start = per0; x_start = x.copy()
S = {'k': max(mad(r0[cr['flesh']]), MEDGE), 'c': float(SJ['c']), 'mu': float(SJ['mu']), 's': float(SJ['s']), 'e': float(SJ['e'])}
log('start: %d model points in %d slices; skin scale %.2f mm; %s (%.0f s)' % (len(r0), len(SL), 1000 * S['k'], json.dumps(st0), time.time() - t00))
hist = []
for rnd in range(12):
    xbase = x.copy(); z0 = x[cols].copy()
    res = least_squares(resid, z0, jac=jac, args=(cr, S), method='trf', tr_solver='lsmr', loss='soft_l1', x_scale='jac', max_nfev=40)
    xn = full(res.x); A0, t0 = mats(x); A1, t1 = mats(xn); move = float(np.linalg.norm(Tp(A1, t1, V0[hv], ho) - Tp(A0, t0, V0[hv], ho), axis=1).max())
    x = xn; xb_ = x.reshape(B, 9); R0[:] = Rot.from_rotvec(xb_[:, :3]).as_matrix() @ R0; xb_[:, :3] = 0; x = xb_.ravel()   # fold the rotation in
    cr, per = correspond(x); st, r = stats(cr, x); newk = max(mad(r[cr['flesh']]), MEDGE)
    hist.append({'round': rnd, 'cost': float(res.cost), 'nfev': int(res.nfev), 'largest_move_mm': 1000 * move, 'skin_scale_mm': 1000 * S['k'], **{rg: v['skin_residual_median_mm'] for rg, v in st.items()}})
    log('round %d: cost %.0f (%d evaluations), largest bone move %.2f mm, skin scale %.2f -> %.2f mm; skin residual median %s (%.0f s)' % (
        rnd, res.cost, res.nfev, 1000 * move, 1000 * S['k'], 1000 * newk, {rg: (None if v['skin_residual_median_mm'] is None else round(v['skin_residual_median_mm'], 2)) for rg, v in st.items()}, time.time() - t00))
    if rnd > 0 and move < S['k']: break
    S['k'] = newk
# --- results
A, t = mats(x); xb = x.reshape(B, 9); Vr = K['V_rigid'].astype(np.float64).copy()
for i in fi_: Vr[idx_all[bones[i]]] = Tp(A, t, V0[idx_all[bones[i]]], np.full(len(idx_all[bones[i]]), i))
As, ts = mats(x_start); dep0 = depth(Tp(As, ts, V0[hv], ho)); dep1 = depth(Tp(A, t, V0[hv], ho))
gap = np.linalg.norm(Tp(A, t, V0[cpx], cpa) - Tp(A, t, V0[cpx], cpb), axis=1)
js = {'regions': REG, 'free_bones': int(free.sum()), 'slices': len(SL), 'rounds': hist, 'scales': {k: float(v) for k, v in S.items()},
      'before': st0, 'after': st, 'joint_gap_mm': {'median': 1000 * float(np.median(gap)), 'p90': 1000 * float(np.percentile(gap, 90))},
      'bone_outside_skin_share': {'before': float((dep0 < 0).mean()), 'after': float((dep1 < 0).mean())},
      'size_change': {names[bones[i]]: float(np.exp(xb[i, 6] - x_start.reshape(B, 9)[i, 6]) - 1) for i in fi_}, 'seconds': round(time.time() - t00)}
json.dump(js, open(os.path.join(od, TAG + '.json'), 'w'), indent=1)
np.savez(os.path.join(od, TAG + '.npz'), bones=K['bones'], bone_names=K['bone_names'], params=xb, R0=R0, P=P, C=C, A=A, t=t, V_rigid=Vr.astype(np.float32),
         F=K['F'], F_bone=K['F_bone'], F_muscle=K['F_muscle'], obj=OB, free=free, contact_pairs=K['contact_pairs'])
# --- sheets: per region up to 16 slices spread over its bones
CM = {MAT.index('flesh'): (150, 140, 245), MAT.index('cloth'): (230, 140, 80), MAT.index('hair'): (40, 80, 140), MAT.index('prop'): (90, 190, 90)}
Sz = 300
for rg in REG:
    ks = [k for k in range(len(SL)) if SL[k]['reg'] == rg and per[k] is not None]
    if not ks: continue
    ks = [ks[int(round(v))] for v in np.linspace(0, len(ks) - 1, min(16, len(ks)))]; tiles = []
    for k in ks:
        s = SL[k]; p, n, e1, e2, Pm, mm = per[k]; im = np.full((Sz, Sz, 3), 255, np.uint8)
        q0 = Tp(As, ts, s['A'], s['own']); q1 = Tp(A, t, s['A'], s['own']); to2 = lambda Q: np.c_[(Q - p) @ e1, (Q - p) @ e2]
        bsec = PL.cut(V0, FB[OB[FB[:, 0]] == bones[s['i']]], s['p'], s['n'], s['e1'], np.cross(s['n'], s['e1']))
        def bsec_w(Am, tm):
            if not len(bsec): return np.zeros((0, 2, 2))
            w = s['p'] + bsec @ np.c_[s['e1'], np.cross(s['n'], s['e1'])].T; return to2(Tp(Am, tm, w.reshape(-1, 3), np.full(w.size // 3, s['i']))).reshape(-1, 2, 2)
        allp = np.r_[Pm, to2(q0), to2(q1)]; sc = Sz * 0.42 / max(np.abs(allp).max(), 1e-6); toi = lambda Q: (np.c_[Q[:, 0], -Q[:, 1]] * sc + Sz / 2).astype(np.int32)
        for (u_, v_), m_ in zip(toi(Pm), mm): cv2.circle(im, (int(u_), int(v_)), 2, CM.get(int(m_), (150, 150, 150)), -1)
        for Q, col, w in ((to2(q0), (170, 170, 170), 1), (to2(q1), (40, 40, 220), 2)): cv2.polylines(im, [toi(np.r_[Q, Q[:1]])], True, col, w)
        for sg_, col in ((bsec_w(As, ts), (170, 200, 215)), (bsec_w(A, t), (40, 120, 170))):
            for ab_ in sg_: cv2.line(im, tuple(toi(ab_[:1])[0]), tuple(toi(ab_[1:])[0]), col, 2)
        cv2.putText(im, names[bones[s['i']]][:38], (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1); tiles.append(im)
    nc = 4; nr = -(-len(tiles) // nc); tiles += [np.full((Sz, Sz, 3), 255, np.uint8)] * (nr * nc - len(tiles))
    cv2.imwrite(os.path.join(od, 'refine_%s_sheet.png' % rg), np.vstack([np.hstack(tiles[a_ * nc:(a_ + 1) * nc]) for a_ in range(nr)]))
log('done: %s; joint gap median %.2f mm; bone hull points outside the skin %.1f%% -> %.1f%% (%.0f s)' % (
    '; '.join('%s skin %s -> %s mm' % (rg, None if st0[rg]['skin_residual_median_mm'] is None else round(st0[rg]['skin_residual_median_mm'], 2),
                                        None if st[rg]['skin_residual_median_mm'] is None else round(st[rg]['skin_residual_median_mm'], 2)) for rg in REG),
    js['joint_gap_mm']['median'], 100 * (dep0 < 0).mean(), 100 * (dep1 < 0).mean(), time.time() - t00))
