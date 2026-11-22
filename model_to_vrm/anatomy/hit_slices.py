"""Option 3: HIT's tissues forced onto the model, slice by slice, at minimal-area cross-sections:
  E:\\splat\\.venv\\Scripts\\python.exe hit_slices.py <model dir> <segment> [smpl model .npz]
  inputs    body/skeleton.npz (support line), anatomy/smpl_fit.npz (SMPL body fitted to MHR), anatomy/hit_tissues.npz
            (HIT lean / fat / bone surfaces posed in the world), the model mesh and contacts/material.npz
  support   the segment's skeleton path (segments.model_frame ends, the skeleton joints between them), 101 points
            evenly along its length
  ownership the SMPL body's own parts: each SMPL vertex belongs to its strongest skinning joint (SMPL_PARTS below);
            a model / tissue point belongs to the part of its nearest SMPL vertex. Unlike the nearest stick bone this
            follows the body's surface, so a thigh pressed against the belly keeps its own side
  plane     at each support point: the plane through it whose cut of the body (SMPL, the body under the clothes; only
            this segment's part; per angle the first surface out from the point) has the least area. Searched over
            the plane's two tilt angles (Powell), starting from the previous slice's plane (the first from the path
            direction); a slice may turn from its neighbour by at most atan(spacing / neighbour's mean radius), so
            neighbouring planes cannot cross inside the body
  stretch   per slice: first HIT's ring is shifted onto the seen skin (mean per-angle offset, repeated until it moves
            less than a HIT voxel); then per angle (72): factor = seen skin radius / HIT radius where the model shows
            flesh, interpolated around the circle between those angles, and capped so the skin stays inside cloth / hair /
            prop. Every HIT tissue point -> shift + radius * factor(its angle): the skin matches the seen flesh at every
            angle, the inside follows proportionally. No skin seen in a slice: HIT kept, only pushed inside the cloth
  writes    anatomy/<segment>_hit.npz (planes, radii, factors, world polylines: kind 0 skin, 1 lean, 2 bone, 3 fat-lean
            boundary drawn with lean) for show_slices3d.py / show_slice_planes.py; <segment>_hit_sheet.png"""
import sys, os, numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.optimize import minimize
from segments import SEG, model_frame
R, seg = sys.argv[1:3]; smpl_npz = sys.argv[3] if len(sys.argv) > 3 else None
j = lambda *p: os.path.join(R, *p)
SMPL_PARTS = {  # SMPL joint ids (0 pelvis 1 L_hip 2 R_hip 3 spine1 4 L_knee 5 R_knee 6 spine2 7 L_ankle 8 R_ankle 9 spine3 10 L_foot
    # 11 R_foot 12 neck 13 L_collar 14 R_collar 15 head 16 L_shoulder 17 R_shoulder 18 L_elbow 19 R_elbow 20 L_wrist 21 R_wrist
    # 22 L_hand 23 R_hand): a vertex follows the joint that rotates it most, so joint 2 (R_hip) moves the right thigh
    'pelvis': [0], 'trunk': [3, 6, 9, 13, 14], 'neck': [12], 'head': [15],
    'thigh_r': [2], 'calf_r': [5], 'foot_r': [8, 11], 'uparm_r': [17], 'forearm_r': [19], 'hand_r': [21, 23],
    'thigh_l': [1], 'calf_l': [4], 'foot_l': [7, 10], 'uparm_l': [16], 'forearm_l': [18], 'hand_l': [20, 22]}
SP = np.load(j('anatomy', 'smpl_fit.npz')); gender = str(SP['gender'])
smpl_npz = smpl_npz or r'E:\figures\models\hit\smpl_models\smpl\SMPL_%s.npz' % gender.upper()
part = np.load(smpl_npz)['weights'].argmax(1); mine = np.isin(part, SMPL_PARTS[seg])
HT = np.load(j('anatomy', 'hit_tissues.npz')); SV, SF = HT['smpl_V'], HT['smpl_F']   # the SMPL body HIT ran on (its shape kept in HIT's range)
sk = cKDTree(SV)
K = np.load(j('body', 'skeleton.npz')); H, par, lab = K['heads'], K['parents'], [str(x) for x in K['labels']]
hl = None
if seg == 'head':                                                 # top: the farthest SMPL head-part vertex along neck -> head
    u_ = H[lab.index('c_head')] - H[lab.index('c_neck')]; u_ /= np.linalg.norm(u_); hl = float(((SV[mine] - H[lab.index('c_head')]) @ u_).max())
J0, J1, tgt, _ = model_frame(seg, H, lab, hl)
# support path: the skeleton joints strictly between the two ends (when both ends are single joints on one chain)
m_ = SEG[seg]['model']; path = [J0]
if len(m_['j0']) == 1 and isinstance(m_['j1'], list) and len(m_['j1']) == 1:
    a, b = lab.index(m_['j0'][0]), lab.index(m_['j1'][0]); up = []
    while b >= 0 and b != a: up.append(b); b = par[b]
    if b == a: path += [H[i] for i in up[::-1][:-1]]
path = np.array(path + [J1]); path = path[np.r_[True, np.linalg.norm(np.diff(path, axis=0), axis=1) > 0]]   # joints at the same place once
seglen = np.linalg.norm(np.diff(path, axis=0), axis=1); s_ = np.r_[0, np.cumsum(seglen)]; L = s_[-1]
NL = 101; ts = np.linspace(0, 1, NL)
def along(t):                                                     # point and direction on the path at fraction t
    d = t * L; k = min(np.searchsorted(s_, d, side='right') - 1, len(seglen) - 1); w = (d - s_[k]) / seglen[k]
    return path[k] + w * (path[k + 1] - path[k]), (path[k + 1] - path[k]) / seglen[k]
TH = 72; th_c = (np.arange(TH) + 0.5) / TH * 2 * np.pi - np.pi
def frame(nrm, ref):                                              # plane basis: ex toward the roll landmark, ey = n x ex
    e1 = ref - nrm * (ref @ nrm); e1 /= np.linalg.norm(e1); return e1, np.cross(nrm, e1)
def cut(V, F, fsel, p, nrm, e1, e2):
    """crossing segments (k, 2, 2) in plane coordinates of the triangles F[fsel] with the plane (p, nrm)"""
    T = V[F[fsel]] - p; d = T @ nrm; s = np.sign(d); cr = s[:, [0, 1, 2]] != s[:, [1, 2, 0]]; k = np.flatnonzero(cr.sum(1) == 2)
    if not len(k): return np.zeros((0, 2, 2))
    P = []
    for e in range(3):
        a, b = e, (e + 1) % 3; w = d[k, a] / (d[k, a] - d[k, b] + 1e-30); q = T[k, a] + w[:, None] * (T[k, b] - T[k, a])
        P.append(np.where(cr[k, e][:, None], q, np.nan))
    P = np.stack(P, 1); out = np.array([q[~np.isnan(q[:, 0])][:2] for q in P])
    return np.stack([out @ e1, out @ e2], -1)
def polar_first(segs, ret_idx=False):
    """per angle bin: the first surface out from the point (smallest radius); empty bins nan"""
    if not len(segs): return np.full(TH, np.nan), (np.full(TH, -1) if ret_idx else None)
    u = np.linspace(0, 1, 9)[None, :, None]; Q = (segs[:, :1] + u * (segs[:, 1:] - segs[:, :1])).reshape(-1, 2)
    b = ((np.arctan2(Q[:, 1], Q[:, 0]) + np.pi) / (2 * np.pi) * TH).astype(int) % TH; r = np.hypot(Q[:, 0], Q[:, 1])
    out = np.full(TH, np.inf); np.minimum.at(out, b, r); out[np.isinf(out)] = np.nan
    if ret_idx:
        first = np.full(TH, -1); o = np.lexsort((r, b)); f = o[np.r_[True, b[o][1:] != b[o][:-1]]]; first[b[f]] = f // 9; return out, first
    return out, None
def fill(r):                                                      # gaps around the circle: periodic interpolation
    ok = np.isfinite(r)
    if ok.sum() < 2: return r
    k = np.arange(TH); return np.interp(k, np.r_[k[ok] - TH, k[ok], k[ok] + TH], np.tile(r[ok], 3))
fS = np.flatnonzero(mine[SF].all(1))                              # this segment's SMPL faces
def area(p, nrm, ref):
    e1, e2 = frame(nrm, ref); r = fill(polar_first(cut(SV, SF, fS, p, nrm, e1, e2))[0])
    return 0.5 * np.nansum(r ** 2) * (2 * np.pi / TH) if np.isfinite(r).any() else np.inf, r
def tilt(n0, ab):                                                 # unit normal tilted from n0 by (a, b) along its own basis
    e1, e2 = frame(n0, tgt - J0 if abs((tgt - J0) @ n0) < 0.99 * np.linalg.norm(tgt - J0) else np.cross(n0, [0, 0, 1.0]))
    n = n0 + ab[0] * e1 + ab[1] * e2; return n / np.linalg.norm(n)
planes, rads, areas, turns = [], [], [], []; ds = L / (NL - 1); prev = None
for t in ts:
    p, d = along(t); n0 = d if prev is None else prev[0]; ref = tgt - J0
    A0, r0 = area(p, n0, ref); Rm = float(np.nanmean(r0)) if np.isfinite(r0).any() else ds
    lim = np.tan(np.arctan(ds / (prev[1] if prev is not None else Rm)))   # the most a plane may turn from its neighbour
    f = lambda ab: area(p, tilt(n0, np.clip(ab, -lim, lim)), ref)[0]
    res = minimize(f, np.zeros(2), method='Powell', bounds=[(-lim, lim)] * 2) if np.isfinite(A0) else None
    n = tilt(n0, res.x) if res is not None and res.fun < A0 else n0
    A, r = area(p, n, ref); planes.append((p, n)); rads.append(r); areas.append(A); turns.append(np.degrees(np.arccos(np.clip(n @ d, -1, 1))))
    prev = (n, float(np.nanmean(r)) if np.isfinite(r).any() else Rm)
print('HS %s: path %d points, length %.3f m; minimal-area planes: tilt from the path median %.1f deg, max %.1f deg' % (seg, len(path), L, np.median(turns), np.max(turns)), flush=True)
# the model's surface in each plane: first surface out from the point, among model points owned by this segment
V = np.load(j('parts', 'orig_colours.npz'))['V'].astype(np.float64); F = np.load(j('parts', 'orig_normals.npz'))['F']
M = np.load(j('contacts', 'material.npz')); mat = M['material']; MN = [str(x) for x in M['materials']]; FL = MN.index('flesh'); CL = MN.index('cloth')
fown = np.flatnonzero(mine[sk.query(V[F].mean(1))[1]])           # model faces whose centre is nearest this segment's SMPL part
fmat = mat[F[:, 0]]
POSE = os.environ.get('HIT_POSE', 'Vhit')   # Vhit: HIT's own learned posing; V: anchored to the nearest SMPL vertex
TIS = [(k, HT['%s_%s' % (k, POSE)], HT['%s_F' % k]) for k in ('fat', 'lean', 'bone')]
town = {k: np.flatnonzero(mine[sk.query(Vt[Ft].mean(1))[1]]) for k, Vt, Ft in TIS}
lines, kinds, levels, out = [], [], [], dict(target=[], model_r=[], model_mat=[], factor=[], seen_flesh=[])
for li, ((p, n), r_s) in enumerate(zip(planes, rads)):
    e1, e2 = frame(n, tgt - J0); sg = cut(V, F, fown, p, n, e1, e2); r_m, first = polar_first(sg, True)
    mm = np.full(TH, -1)                                          # material per angle: of the face that gave the first surface
    T_ = V[F[fown]] - p; dd = T_ @ n; s_g = np.sign(dd); crs = s_g[:, [0, 1, 2]] != s_g[:, [1, 2, 0]]; kf = fown[np.flatnonzero(crs.sum(1) == 2)]
    ok = first >= 0; mm[ok] = fmat[kf[first[ok]]]
    rs = fill(r_s); U = np.c_[np.cos(th_c), np.sin(th_c)]; Pm = r_m[:, None] * U       # model first-surface points per angle
    fl0 = (mm == FL) & np.isfinite(r_m); cl0 = (mm >= 0) & ~fl0 & np.isfinite(r_m); c = np.zeros(2)
    def rebin(Q, m):                                              # radius per angle bin of points Q (only where m)
        r = np.full(TH, np.nan)
        if m.any():
            q = Q[m]; bb = ((np.arctan2(q[:, 1], q[:, 0]) + np.pi) / (2 * np.pi) * TH).astype(int) % TH; rr = np.hypot(q[:, 0], q[:, 1])
            for b_, r_ in zip(bb, rr): r[b_] = r_ if not np.isfinite(r[b_]) else min(r[b_], r_)
        return r
    for _ in range(TH):                                           # shift HIT's ring onto the seen skin (same angle): until it moves < a HIT voxel
        tf = rebin(Pm - c, fl0); both = np.isfinite(tf)
        if not both.any(): break
        dc = ((tf[both] - rs[both])[:, None] * U[both]).mean(0); c = c + dc
        if np.linalg.norm(dc) < float(HT['voxel']): break
    tf = rebin(Pm - c, fl0); tc = rebin(Pm - c, cl0)
    fac = np.where(np.isfinite(tf), tf / rs, np.nan); fac = fill(fac) if np.isfinite(fac).any() else np.ones(TH)   # seen skin exactly; between: interpolated
    capm = np.isfinite(tc); fac[capm] = np.minimum(fac[capm], tc[capm] / rs[capm])   # under cloth / hair / prop: not outside it
    tgt_r = rs * fac; fl = np.isfinite(tf)
    out['target'].append(tgt_r); out['model_r'].append(r_m); out['model_mat'].append(mm); out['factor'].append(fac); out['seen_flesh'].append(fl.mean())
    out.setdefault('shift', []).append(c.copy()); out.setdefault('e1', []).append(e1); out.setdefault('e2', []).append(e2)
    toW = lambda q: p + q[..., :1] * e1 + q[..., 1:] * e2
    def stretch(q):                                               # shift with the ring, radius * factor at the point's own angle
        a_ = np.arctan2(q[..., 1], q[..., 0]); kf_ = (a_ + np.pi) / (2 * np.pi) * TH - 0.5
        f_ = np.interp(kf_, np.r_[-1, np.arange(TH), TH], np.r_[fac[-1], fac, fac[0]]); return c + q * f_[..., None]
    sk_q = c + tgt_r[:, None] * U; lines.append(toW(np.r_[sk_q, sk_q[:1]])); kinds.append(0); levels.append(li)
    for kind, (k, Vt, Ft) in zip((3, 1, 2), TIS):
        if k == 'fat': continue                                  # the fat's outer boundary is the skin; its inner one is the lean surface
        for q in cut(Vt, Ft, town[k], p, n, e1, e2): lines.append(toW(stretch(q))); kinds.append(kind); levels.append(li)
sf = np.array(out['seen_flesh']); fa = np.array(out['factor'])
print('HS skin seen as flesh: %.0f%% of all angles (slices with any: %d of %d); stretch factor median %.2f (5-95%%: %.2f-%.2f)' % (
    100 * sf.mean(), (sf > 0).sum(), NL, np.nanmedian(fa), np.nanpercentile(fa, 5), np.nanpercentile(fa, 95)), flush=True)
od = j('anatomy'); P_ = np.array([p for p, _ in planes]); N_ = np.array([n for _, n in planes])
np.savez(os.path.join(od, seg + '_hit.npz'), t=ts, points=P_, normals=N_, smpl_r=np.array(rads), area=np.array(areas), tilt_deg=np.array(turns),
         **{k: np.array(v) for k, v in out.items()}, path=path, J0=J0, J1=J1, L=L, ax=(J1 - J0) / np.linalg.norm(J1 - J0),
         lines=np.array(lines, dtype=object), kinds=np.array(kinds), levels=np.array(levels), allow_pickle=True)
# sheet: 12 levels; model points by material (dots), HIT skin before (grey) and after (red), lean (purple), bone (tan)
Sz = 260; show = np.linspace(0, NL - 1, 12).astype(int); Rm_ = np.nanmax([np.nanmax(np.r_[x, y]) for x, y in zip(np.array(rads)[show], np.array(out['target'])[show])])
sc = Sz / (2.6 * Rm_); toI = lambda q: (np.stack([q[..., 0], -q[..., 1]], -1) * sc + Sz / 2).astype(np.int32); tiles = []
CM = {FL: (150, 140, 245), CL: (230, 140, 80)}
for li in show:
    im = np.full((Sz, Sz, 3), 255, np.uint8); p, n = planes[li]; e1, e2 = frame(n, tgt - J0)
    rm, mm = out['model_r'][li], out['model_mat'][li]
    for a, r_, m_ in zip(th_c, rm, mm):
        if np.isfinite(r_): cv2.circle(im, tuple(toI(np.array([r_ * np.cos(a), r_ * np.sin(a)]))), 2, CM.get(int(m_), (90, 190, 90)), -1)
    rs = fill(rads[li]); q = np.c_[rs * np.cos(th_c), rs * np.sin(th_c)]; cv2.polylines(im, [toI(q)], True, (170, 170, 170), 1)
    for l_, k_, v_ in zip(lines, kinds, levels):
        if v_ != li: continue
        q2 = np.stack([(l_ - p) @ e1, (l_ - p) @ e2], -1); cv2.polylines(im, [toI(q2)], k_ == 0, {0: (40, 40, 220), 1: (150, 60, 160), 2: (120, 170, 200)}[k_], 2 if k_ != 1 else 1)
    cv2.putText(im, 't=%.2f tilt %.0f flesh %.0f%%' % (ts[li], turns[li], 100 * out['seen_flesh'][li]), (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1); tiles.append(im)
cv2.imwrite(os.path.join(od, seg + '_hit_sheet.png'), np.vstack([np.hstack(tiles[r * 4:(r + 1) * 4]) for r in range(3)]))
print('HS done')
