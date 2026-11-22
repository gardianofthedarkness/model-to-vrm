"""Fit the atlas's critical cross-sections (za_slices.py) to a model's limb, slice by slice, each on the model's own minimal-area
plane (planes.py; path through the model's skeleton joints between the segment's ends):
  python fit_slices.py <real dir> <segment> <atlas slices.npz> <out dir>       segment: any of segments.ORDER
  frame     the model's own skeleton (body/skeleton.npz): axis joint -> joint, roll toward the same landmark as the atlas
            (segments.py: thumb, front of the knee, toes, shin, the body's front, the eyes); the head has no top joint,
            so its length = the farthest model surface point along the neck-head axis owned by the head's bones
  section   the model mesh cut at each critical level; a point counts when this segment is its nearest skeleton bone
            (so torso, other limbs stay out), and per angle only the first surface out from the bone (the limb's own
            outside; what lies beyond it is something else); material per point from contacts/material.npz
  model     per slice: atlas skin outline -> shift (cx, cy) + roll + widths (sx, sy >= 0) in the slice plane
  residual  true signed distance from each point to the outline (not along rays from the skeleton line, which can sit near
            the skin and make rays run tangent); start = centre and size of the slice's own seen skin, roll 0 (both frames aim at the same landmark)
  fit       ICP with a convex step (cvxpy, Clarabel): fix each seen point's nearest place on the current outline and
            linearise the roll, then the outline is linear in the 5 numbers, so the step is a convex problem; repeat until the correspondences stop
            changing or a round gains less than one point at the residual scale. Robust (Huber, knee at the residual scale) on seen skin - points seen through a hole count less;
            seen flesh on the outline; cloth, hair, prop outside it (one-sided); neighbouring slices' parameters alike.
            Weights from the data: residual scale = median flesh residual of a first per-slice fit; parameter scale =
            median neighbour difference of slices that see flesh. Slices that see no flesh are filled by neighbours.
            A slice sees skin when it has more skin points than its 5 unknowns. A slice that sees no skin keeps its roll
            (0 = the landmark frame; in the joint fit, interpolated from the slices that see skin): cloth alone let the
            roll wander (test model R's left upper arm: -95 to +203 degrees), twisting the atlas limb. Clothed segments (no slice sees skin):
            each slice starts at the centre and size of the seen cloth surface and the one-sided term keeps the skin
            inside the cloth; the parameter scale then comes from all slices' first fits.
  writes    <segment>_fit.npz (params, world polylines of fitted skin/muscle/bone per slice), <segment>_fit_sheet.png"""
import sys, os, json, numpy as np, cv2
R, seg, sp, od = sys.argv[1:5]; os.makedirs(od, exist_ok=True); j = lambda *p: os.path.join(R, *p)
A = np.load(sp); TH = int(A['TH']); crit = A['crit']; tl = A['levels'][crit]; Xa = A['X'][crit, :TH]; La = float(A['L'])
V = np.load(j('parts', 'orig_colours.npz'))['V'].astype(np.float64); F = np.load(j('parts', 'orig_normals.npz'))['F']
M = np.load(j('contacts', 'material.npz')); mat = M['material']; MN = [str(x) for x in M['materials']]
S = np.load(j('body', 'frames.npz') if os.path.exists(j('body', 'frames.npz')) else j('body', 'skeleton.npz'))   # frames: keypoints or fitted bones (never a stick fit)
H, par, lab = S['heads'], S['parents'], [str(x) for x in S['labels']]; h = lambda n: H[lab.index(n)]
from segments import model_frame
hl = None
if seg == 'head':                                                 # no joint at the top: the farthest surface point along the axis that the head's bones own
    from segments import SEG as SG_
    u_ = h('c_head') - h('c_neck'); u_ /= np.linalg.norm(u_); oi = [lab.index(n) for n in SG_['head']['model']['own']]
    Bh = [(par[i], i) for i in range(len(H)) if par[i] >= 0]; Dh = []
    for a_, b_ in Bh:
        d_ = H[b_] - H[a_]; w_ = np.clip(((V - H[a_]) @ d_) / (d_ @ d_ + 1e-12), 0, 1); Dh.append(np.linalg.norm(V - H[a_] - w_[:, None] * d_, axis=1))
    mine = np.isin(np.argmin(np.array(Dh), 0), [k for k, (p_, c_) in enumerate(Bh) if c_ in oi]); hl = float(((V[mine] - h('c_head')) @ u_).max())
J0, J1, tgt, own = model_frame(seg, H, lab, hl); ax = (J1 - J0) / np.linalg.norm(J1 - J0)
ex = tgt - J0; ex -= ax * (ex @ ax); ex /= np.linalg.norm(ex); ey = np.cross(ax, ex); L = np.linalg.norm(J1 - J0); s0 = L / La
B = [(par[i], i) for i in range(len(H)) if par[i] >= 0]; chain = [k for k, (p_, c_) in enumerate(B) if c_ in own]   # the segment's own skeleton bones
def segdist(P):                                                   # distance of world points to every skeleton bone segment
    D = []
    for a, b in B:
        d = H[b] - H[a]; u = np.clip(((P - H[a]) @ d) / (d @ d + 1e-12), 0, 1); D.append(np.linalg.norm(P - H[a] - u[:, None] * d, axis=1))
    return np.array(D).T
# minimal-area planes (planes.py): the path runs through the model's own skeleton joints between the segment's ends (it
# bends with the spine); at each level the plane through the path point with the smallest cut of the segment's own surface
import planes as PL
from segments import SEG as SG2
m_ = SG2[seg]['model']; path = [J0]
if len(m_['j0']) == 1 and isinstance(m_['j1'], list) and len(m_['j1']) == 1:
    a_, b_ = lab.index(m_['j0'][0]), lab.index(m_['j1'][0]); up = []
    while b_ >= 0 and b_ != a_: up.append(b_); b_ = par[b_]
    if b_ == a_: path += [H[i] for i in up[::-1][:-1]]
PATH = PL.Path(path + [J1]); L = PATH.L; s0 = L / La
print('FS %s: model length %.3f m along %d joints (atlas %.3f, scale %.2f)' % (seg, L, len(PATH.pts), La, s0))
vown = np.isin(np.concatenate([segdist(V[i:i + 50000]).argmin(1) for i in range(0, len(V), 50000)]), chain)   # vertices this segment's bones own
Fown = F[vown[F].all(1)]
tg = np.linspace(tl.min(), tl.max(), max(101, len(tl)))          # planes on an even grid, then read at the atlas levels
PG = PL.min_area_planes(V, Fown, PATH, tg, tgt - J0, log=lambda s: print('FS', s, flush=True))
nI = np.array([np.interp(tl, tg, PG['n'][:, c]) for c in range(3)]).T; nI /= np.linalg.norm(nI, axis=1, keepdims=True)
PLN = {'p': np.array([PATH.at(t)[0] for t in tl]), 'n': nI}; PLN['e1'], PLN['e2'] = map(np.array, zip(*[PL.frame(n_, tgt - J0) for n_ in nI]))
Fm_all = mat[F[:, 0]]
obs = []
for kk, t in enumerate(tl):
    p_, n_, e1_, e2_ = PLN['p'][kk], PLN['n'][kk], PLN['e1'][kk], PLN['e2'][kk]
    sg_, fi_ = PL.cut(V, F, p_, n_, e1_, e2_, ret_face=True); P = sg_.mean(1); mm = Fm_all[fi_]
    wpt = p_ + P @ np.c_[e1_, e2_].T; own = np.isin(segdist(wpt).argmin(1), chain); P, mm = P[own], mm[own]
    th_, r_ = np.arctan2(P[:, 1], P[:, 0]), np.hypot(P[:, 0], P[:, 1]); b_ = ((th_ + np.pi) / (2 * np.pi) * 2 * TH).astype(int) % (2 * TH)
    o_ = np.lexsort((r_, b_)); first = o_[np.r_[True, b_[o_][1:] != b_[o_][:-1]]] if len(o_) else o_   # per angle: the first surface out from the bone
    obs.append((th_[first], r_[first], mm[first]))
    print('FS t=%.2f  first-surface points %3d: %s' % (t, len(obs[-1][0]), {MN[i]: int((obs[-1][2] == i).sum()) for i in range(4) if (obs[-1][2] == i).any()}))
th_a = (np.arange(TH) + 0.5) / TH * 2 * np.pi - np.pi; pa = lambda k: np.c_[Xa[k] * np.cos(th_a), Xa[k] * np.sin(th_a)]
def contour(k, p):                                                # atlas outline k under params p -> points in the slice plane
    cx, cy, ph, sx, sy = p; c, s_ = np.cos(ph), np.sin(ph); return (pa(k) * [sx, sy]) @ np.array([[c, s_], [-s_, c]]) + [cx, cy]
def rmodel(k, p, th):                                             # model radius along the observation angles
    q = contour(k, p); a = np.arctan2(q[:, 1], q[:, 0]); r = np.hypot(q[:, 0], q[:, 1]); o = np.argsort(a)
    return np.interp(th, a[o], r[o], period=2 * np.pi)
FL = MN.index('flesh')
OBXY = [np.c_[o[1] * np.cos(o[0]), o[1] * np.sin(o[0])] for o in obs]
def inside(q, P):                                                 # crossing-number point-in-polygon
    a, b = q, np.roll(q, -1, 0); ay, by = a[:, 1][None], b[:, 1][None]; py, px = P[:, 1][:, None], P[:, 0][:, None]
    cr = (ay > py) != (by > py); xi = a[:, 0][None] + (py - ay) / (by - ay + 1e-30) * (b[:, 0] - a[:, 0])[None]
    return ((cr & (px < xi)).sum(1) % 2) == 1
def sdist(q, P):                                                  # signed distance of points P to the closed outline q (+ outside)
    a, b = q, np.roll(q, -1, 0); d = b - a; u = np.clip(((P[:, None] - a) * d).sum(2) / (d * d).sum(1), 0, 1)
    dist = np.linalg.norm(P[:, None] - a - u[..., None] * d, axis=2).min(1)
    return np.where(inside(q, P), -dist, dist)
def res_slice(k, p):
    P = OBXY[k]; mm = obs[k][2]; f = mm == FL; sd = sdist(contour(k, p), P)
    return np.r_[sd[f], np.maximum(0, -sd[~f])]                    # seen skin on the outline; anything else not inside it
import cvxpy as cp
nk = len(tl); FLm = [o[2] == FL for o in obs]; CL = MN.index('cloth'); CLm = [o[2] == CL for o in obs]
def mad(k):                                                       # robust scale of a slice's observed radii
    r = obs[k][1]; return float(np.median(np.abs(r - np.median(r)))) + 1e-12 if len(r) else np.nan
def rr_(k):                                                       # the skeleton line stays inside the limb: shift within the slice's observed flesh radius
    f = FLm[k]; return float(np.median(obs[k][1][f])) if f.any() else s0 * float(Xa[k].mean())
seen = np.array([f.sum() > 5 for f in FLm])                       # sees skin: more skin points than the slice's 5 unknowns (else exactly fitted, no information)
def start(k):                                                     # centre and size of the slice's own seen skin (clothed: of the seen surface); roll 0
    f = FLm[k] if seen[k] else np.ones(len(FLm[k]), bool)          # (both frames point at the same landmark)
    if f.sum() <= 5: return np.array([s0, s0, 0.0, 0, 0])
    c = np.median(OBXY[k][f], 0); r = rr_(k); c = np.clip(c, -r, r); sc_ = np.median(np.linalg.norm(OBXY[k][f] - c, axis=1)) / float(Xa[k].mean())
    return np.array([sc_, sc_, 0.0, c[0], c[1]])
def MC(th):                                                       # theta = (sx, sy, roll, cx, cy): outline point = R(roll) diag(sx, sy) a + c
    c_, s_ = np.cos(th[2]), np.sin(th[2]); return np.array([[c_, -s_], [s_, c_]]) @ np.diag(th[:2]), th[3:]
def match(k, th):
    """ICP correspondences: per seen point, the nearest place on the current outline (atlas point a on that segment)
    and the outline's outward normal n there. With the roll linearised (R (D + W), W a small rotation w), the residual
    n.(M a + c - P) is linear in (sx, sy, w, cx, cy)"""
    M, C = MC(th); a = pa(k); b = np.roll(a, -1, 0); q, qb = a @ M.T + C, b @ M.T + C; d = qb - q; P = OBXY[k]
    u = np.clip(((P[:, None] - q) * d).sum(2) / ((d * d).sum(1) + 1e-30), 0, 1); D = np.linalg.norm(P[:, None] - q - u[..., None] * d, axis=2)
    j = D.argmin(1); uu = u[np.arange(len(P)), j]; Am = a[j] + uu[:, None] * (b[j] - a[j]); t_ = d[j]
    sgn = np.sign(np.sum(q[:, 0] * qb[:, 1] - qb[:, 0] * q[:, 1])) or 1.0     # outline orientation (counter-clockwise: outward = (dy, -dx))
    n = sgn * np.c_[t_[:, 1], -t_[:, 0]]; n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-30
    c_, s_ = np.cos(th[2]), np.sin(th[2]); m = n @ np.array([[c_, -s_], [s_, c_]])        # R^T n
    G = np.c_[m[:, 0] * Am[:, 0], m[:, 1] * Am[:, 1], m[:, 1] * Am[:, 0] - m[:, 0] * Am[:, 1], n]; h = (n * P).sum(1)
    return G, h, j
WMED = WMAD = None                                               # the segment's own widths (set after the per-slice fits)
# bone continuity: each bone's section centre (atlas plane coordinates per fitted level) must move between neighbouring
# slices as it does in the atlas (scaled to the model). A nearly round skin section cannot fix its own roll; the bones
# inside can. Linear in the slice numbers: point = R(roll0) (diag(sx, sy) a + w (-a_y, a_x)) + c
_SG, _SK, _SL, _SS = A['seg_xy'], A['seg_kind'], A['seg_level'], A['seg_struct']
BC = []                                                           # per level: {bone id: centre in the plane}
for k in range(len(tl)):
    m = (_SL == k) & (_SK == 2); BC.append({int(b): _SG[m & (_SS == b)].mean((0, 1)) for b in np.unique(_SS[m])})
STEP = [np.linalg.norm(BC[k + 1][b] - BC[k][b]) for k in range(len(tl) - 1) for b in BC[k] if b in BC[k + 1]]
def lin_pt(th, a):                                                # 2x3 matrix: point = L @ (sx, sy, w) + (cx, cy)
    c_, s_ = np.cos(th[2]), np.sin(th[2]); R0_ = np.array([[c_, -s_], [s_, c_]]); return R0_ @ np.array([[a[0], 0, -a[1]], [0, a[1], a[0]]])
def solve(ks, TH0, sr, sp, smooth):
    """one convex step per ICP round (cvxpy, Clarabel) until the correspondences stop changing or a round gains less
    than one point at the residual scale: seen skin on the outline (Huber, knee at sr); cloth / hair / prop not inside
    it (one-sided, squared, at sr); cloth also weakly near it (Huber at the section's own spread: tight clothing, so
    clothed slices do not collapse); neighbouring slices alike (scale sp); widths >= 0 and
    near the segment's own median width (scale: their MAD), so a slice seen from one side cannot collapse"""
    TH_ = TH0.copy(); last, best = None, np.inf; nround = 0
    while True:
        TH_[ks, 2] = np.unwrap(TH_[ks, 2])
        mt = {k: match(k, TH_[k]) for k in ks if len(OBXY[k])}
        key = tuple(np.concatenate([mt[k][2] for k in sorted(mt)]).tolist()) if mt else ()
        if key == last: return TH_
        last = key; X = cp.Variable((len(ks), 5)); terms, cons = [], [X[:, :2] >= 0]; nround += 1
        sbar = np.maximum(TH_[ks, :2].mean(1), 1e-12); ang = TH_[ks, 2] + cp.multiply(X[:, 2], 1 / sbar)   # roll after the step (small-angle)
        for i, k in enumerate(ks):
            if k not in mt: continue
            G, h, _ = mt[k]; f, cl = FLm[k], CLm[k]
            if f.any(): terms.append(cp.sum(cp.huber((G[f] @ X[i] - h[f]) / sr, 1)))
            if (~f).any(): terms.append(cp.sum_squares(cp.pos(G[~f] @ X[i] - h[~f]) / sr))
            if cl.any(): terms.append(cp.sum(cp.huber((G[cl] @ X[i] - h[cl]) / mad(k), 1)))
            cons.append(cp.abs(X[i, 3:]) <= rr_(k))
            if not seen[k]: cons.append(X[i, 2] == 0)                # no skin seen: no information on the roll; it stays where it starts
                                                                  # (0, the landmark frame; in the joint fit, interpolated from slices that see skin)
        if smooth and len(ks) > 1:
            Y = cp.hstack([X[:, :2], cp.reshape(ang, (len(ks), 1), order='C'), X[:, 3:]])
            terms.append(cp.sum_squares(cp.multiply(Y[1:] - Y[:-1], np.tile(1 / sp, (len(ks) - 1, 1)))))
            terms.append(cp.sum_squares((X[:, :2] - WMED) / WMAD))  # widths stay within the segment's own spread (median, MAD of the per-slice fits)
            if STEP:                                              # bones continuous between neighbouring slices (scale: the atlas's own step, scaled, plus the residual scale)
                sb = s0 * float(np.median(STEP)) + sr
                for i in range(len(ks) - 1):
                    k, k2 = ks[i], ks[i + 1]
                    for b in BC[k]:
                        if b not in BC[k2]: continue
                        pa_ = lin_pt(TH_[k], BC[k][b]) @ X[i, :3] + X[i, 3:]; pb_ = lin_pt(TH_[k2], BC[k2][b]) @ X[i + 1, :3] + X[i + 1, 3:]
                        terms.append(cp.sum_squares((pb_ - pa_ - s0 * (BC[k2][b] - BC[k][b])) / sb))
        if not terms: return TH_
        pr = cp.Problem(cp.Minimize(cp.sum(terms)), cons)
        try: pr.solve(solver=cp.CLARABEL)
        except cp.error.SolverError:                              # numerical stall: try SCS; failing that, keep the last good round
            try: pr.solve(solver=cp.SCS)
            except cp.error.SolverError: return TH_
        if X.value is None: return TH_
        if os.environ.get('FS_DEBUG'): print('FS   ICP round %d: cost %.3f' % (nround, pr.value), flush=True)
        new = X.value.copy(); new[:, 2] = TH_[ks, 2] + np.arctan2(X.value[:, 2], sbar); TH_[ks] = new
        if best - pr.value < 1: return TH_                        # gain below one point at the residual scale: within the data's noise
        best = pr.value
TH0 = np.array([start(k) for k in range(nk)])
sr0 = np.array([mad(k) for k in range(nk)])
TH1 = TH0.copy()
for k in range(nk): TH1[k] = solve([k], TH0, sr0[k], None, False)[k]          # each slice on its own first
def sres(k, th, m):                                               # true signed distance of points to the outline (+ outside)
    M, C = MC(th); return sdist(pa(k) @ M.T + C, OBXY[k][m])
fr = np.concatenate([np.zeros(0)] + [np.abs(sres(k, TH1[k], FLm[k])) for k in range(nk) if seen[k]])
sr = float(np.median(fr)) if len(fr) else float(np.nanmedian(sr0)); sk = np.flatnonzero(seen)   # residual scale: median skin residual (clothed: the seen surface's own spread)
TH1[:, 2] = np.unwrap(TH1[:, 2]); sk_ = sk if len(sk) > 2 else np.arange(nk); sp_ = np.median(np.abs(np.diff(TH1[sk_], axis=0)), 0)
sp_ = np.where(sp_ > 0, sp_, np.abs(TH1[sk_]).mean(0)); sp_ = np.where(sp_ > 0, sp_, np.inf)   # parameter scale: neighbour differences of the
                                                                  # per-slice fits; a parameter with no spread at all (fixed) gets no smoothness term
print('FS per-slice fit: median flesh residual %.4f m; slices seeing flesh %d of %d' % (sr, seen.sum(), nk))
W_ = TH1[sk_, :2]; WMED = np.median(W_); WMAD = float(np.median(np.abs(W_ - WMED))) + 1e-12
X0 = TH1.copy(); X0[:, 2] = 0.0                                # joint solve starts in the landmark frame (roll 0): a near-round skin section cannot fix its own roll,
                                                                  # the per-slice rolls are arbitrary; skin + bone continuity move it only where the data asks
if len(sk): X0[~seen] = np.array([np.interp(np.flatnonzero(~seen), sk, TH1[sk, c]) for c in range(5)]).T
TH2 = solve(list(range(nk)), X0, sr, sp_, True)
P1 = TH1[:, [3, 4, 2, 0, 1]]; P2 = TH2[:, [3, 4, 2, 0, 1]]; shear = np.zeros(nk)   # params as before: cx, cy, roll, sx, sy
Mf = np.array([MC(t)[0] for t in TH2]); Cf = TH2[:, 3:]
def contour(k, p=None):                                           # fitted outline k in the slice plane
    M, C = MC(TH2[k]); return pa(k) @ M.T + C
fr2 = [np.median(np.abs(sres(k, TH2[k], FLm[k]))) if seen[k] else np.nan for k in range(len(tl))]
pen = [float(np.maximum(0, -sres(k, TH2[k], ~FLm[k])).max(initial=0)) if (obs[k][2] != FL).any() else 0.0 for k in range(len(tl))]
for k in range(len(tl)): print('FS t=%.2f  sx %.2f sy %.2f roll %+5.1f deg shift %+.3f %+.3f m  flesh residual %s  deepest cloth/prop overlap %.4f m' % (
    tl[k], P2[k, 3] / s0, P2[k, 4] / s0, np.degrees(P2[k, 2]), P2[k, 0], P2[k, 1], ('%.4f m' % fr2[k]) if seen[k] else 'no skin seen', pen[k]))
# world polylines for Blender: fitted skin outline + atlas muscles/bones moved by the same transform
SG, SKd, SL = A['seg_xy'], A['seg_kind'], A['seg_level']; lines, kinds, levs = [], [], []
W3 = lambda k, q2: PLN['p'][k] + q2 @ np.c_[PLN['e1'][k], PLN['e2'][k]].T   # slice-plane coordinates -> world
def xf(k, q):
    M, C = MC(TH2[k]); return W3(k, q @ M.T + C)
for k in range(len(tl)):
    q = contour(k, P2[k]); lines.append(W3(k, np.r_[q, q[:1]])); kinds.append(0); levs.append(k)
    for kind in (1, 2):
        m = (SL == k) & (SKd == kind)
        for sgm in SG[m]: lines.append(xf(k, sgm)); kinds.append(kind); levs.append(k)
np.savez(os.path.join(od, seg + '_fit.npz'), params=P2, params_first=P1, M=Mf, C=Cf, t=tl, seen=seen, J0=J0, J1=J1, ax=ax, ex=ex, ey=ey, L=L,
         plane_p=PLN['p'], plane_n=PLN['n'], plane_e1=PLN['e1'], plane_e2=PLN['e2'], path=PATH.pts,
         lines=np.array(lines, dtype=object), kinds=np.array(kinds), levels=np.array(levs), allow_pickle=True)
# sheet: per slice, model section points by material + fitted outline + atlas muscles/bones moved into it
Sz = 300; Rm = max(np.hypot(*contour(k, P2[k]).T).max() for k in range(len(tl))); sc = Sz / (3 * Rm); tiles = []
CM = {MN.index('flesh'): (150, 140, 245), MN.index('cloth'): (230, 140, 80), MN.index('hair'): (40, 80, 140), MN.index('prop'): (90, 190, 90)}
toimg = lambda q: (np.stack([q[..., 0], -q[..., 1]], -1) * sc + Sz / 2).astype(np.int32)
for k in range(len(tl)):
    im = np.full((Sz, Sz, 3), 255, np.uint8); th, r, mm = obs[k]
    for x_, y_, m_ in zip(r * np.cos(th), r * np.sin(th), mm): cv2.circle(im, tuple(toimg(np.array([x_, y_]))), 1, CM.get(int(m_), (150, 150, 150)), -1)
    for kind, col, wd in ((1, (120, 60, 170), 1), (2, (120, 170, 200), 2)):
        m = (SL == k) & (SKd == kind)
        if m.any():
            M, C = MC(TH2[k]); q2 = SG[m] @ M.T + C
            cv2.polylines(im, list(toimg(q2)), False, col, wd)
    q = contour(k, P2[k]); cv2.polylines(im, [toimg(np.r_[q, q[:1]])], True, (40, 40, 220), 2)
    cv2.putText(im, 't=%.2f %s' % (tl[k], '' if seen[k] else '(no skin seen: from neighbours)'), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1); tiles.append(im)
cols = 4; rows = -(-len(tiles) // cols); tiles += [np.full((Sz, Sz, 3), 255, np.uint8)] * (rows * cols - len(tiles))
cv2.imwrite(os.path.join(od, seg + '_fit_sheet.png'), np.vstack([np.hstack(tiles[r_ * cols:(r_ + 1) * cols]) for r_ in range(rows)]))
json.dump({'segment': seg, 'length_m': float(L), 'median_flesh_residual_m': [None if not np.isfinite(x) else float(x) for x in fr2]}, open(os.path.join(od, seg + '_fit.json'), 'w'), indent=1)
print('FS done')
