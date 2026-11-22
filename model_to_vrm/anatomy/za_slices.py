"""CT-like cross-sections of one body segment from the Z-Anatomy atlas, sliced at its own critical levels, each level on
its minimal-area plane (planes.py: the plane through the axis point with the smallest cut of the segment's atlas skin):
  python za_slices.py <atlas.npz> <segment> <out dir> [--dense]
  --dense   slice everywhere, one level per median edge length of the segment's atlas muscles (not only critical levels)        segment: any of segments.ORDER (pelvis, trunk, neck, head,
                                                             uparm/forearm/hand/thigh/calf/foot _r/_l)
  joints    a joint centre = mean of the two bone groups' facing vertices (each group's vertices closer to the other group
            than the Otsu split of the log distances); segments.py says which bones meet at each end
  frame     axis = proximal joint -> distal joint (t = 0..1); roll = toward the segment's landmark (segments.py)
  own       skin: the atlas's own named skin regions of the segment (segments.py). Muscles and bones: a section point
            counts only when this segment's axis is its nearest among all segments' axes (the atlas's own stick skeleton:
            the other leg, the trunk, the next limb stay out). Axis ownership is not used for skin: the limbs' axes
            start deep in the joints, so by distance the flanks at hip level would belong to the thighs
  joint     where a slice passes into the trunk or the next limb, the segment owns only part of the skin ring: levels
  zones     from each end inward whose ring coverage is below the Otsu split of all levels' coverage are dropped, so the
            slices stop before the joint (hip, shoulder, crotch); the zone between two segments' end slices is a blend.
            The foot first searches past its joints to the end of its own skin (heel, toes).
  dense     101 levels across the kept range: each structure's section (bone, muscle, skin) per level, polar r(theta)
            around the axis for the skin (outermost hit per angle bin)
  critical  range ends, plus the extrema (peaks and valleys) along the bone of the skin section's mean radius, off-centre
            shift and flattening (Fourier terms 0, 1, 2 of r(theta)); ripples dropped by persistence (a peak/valley pair
            is kept when its height, on the descriptor's own 0..1 scale, is above the Otsu split of all pairs' heights)
  writes    <segment>.npz (frame, levels, per level: skin r(theta), section segments per structure) and
            <segment>_sheet.png (each critical level drawn: bone ivory, muscles red shades, skin outline)"""
import sys, os, re, json, numpy as np, cv2
from segments import SEG as ALL, ORDER, Atlas, otsu
DENSE = '--dense' in sys.argv; sys.argv = [x for x in sys.argv if x != '--dense']
ap, seg, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True); SEG = ALL[seg]; me = ORDER.index(seg)
AT = Atlas(ap); V, F, OB, names, sysn, idx = AT.V, AT.F, AT.OB, AT.names, AT.sys, AT.idx
J0, J1 = AT.axes[seg]['J0'], AT.axes[seg]['J1']; L = np.linalg.norm(J1 - J0); ax = (J1 - J0) / L
r0 = AT.roll_target(seg) - J0; roll_t = float(r0 @ ax) / L; r0 -= ax * (r0 @ ax); ex = r0 / np.linalg.norm(r0); ey = np.cross(ax, ex)
print('ZS %s: length %.3f m, joints %s -> %s' % (seg, L, J0.round(3), J1.round(3)))
# structures near the segment: bones of the segment, muscles with vertices inside the segment's slab and near the axis, skin regions by name
loc = (V - J0) @ np.c_[ax, ex, ey]; t_all, rad = loc[:, 0] / L, np.hypot(loc[:, 1], loc[:, 2])
skin_ids = AT.skin_ids(SEG['skin']); sk_v = np.isin(OB, skin_ids)             # own skin: the atlas's named regions
t_lo = min(0.0, float(t_all[sk_v].min())) if SEG['ext'][0] else 0.0; t_hi = max(1.0, float(t_all[sk_v].max())) if SEG['ext'][1] else 1.0
inr = sk_v & (t_all > t_lo) & (t_all < t_hi); R = float(rad[inr].max()) if inr.any() else L / 4
mus_ids = [i for i in np.unique(OB[(t_all > t_lo) & (t_all < t_hi) & (rad < R)]) if sysn[i] == 'muscle']
bone_ids = [idx[n] for n in AT.bone_names(SEG['bones'])]
print('ZS skin regions %d, muscles %d, bones %d, outer skin radius %.3f m, search range t %.2f..%.2f' % (len(skin_ids), len(mus_ids), len(bone_ids), R, t_lo, t_hi))
TRI = {}
def tris(i):
    if i not in TRI: TRI[i] = (V[F[OB[F[:, 0]] == i]] - J0) @ np.c_[ax, ex, ey]          # (n,3 corners,[along, x, y])
    return TRI[i]
def section(ids, t, own=True):
    """segments (k,2,2) where the structures' triangles cross the plane at level t, in the plane's (ex, ey) coordinates"""
    out = {}
    for i in ids:
        T = tris(i); d = T[:, :, 0] - t * L; s = np.sign(d); cross = s[:, [0, 1, 2]] != s[:, [1, 2, 0]]
        k = np.flatnonzero(cross.sum(1) == 2)
        if not len(k): continue
        pts = []
        for e in range(3):
            a, b = e, (e + 1) % 3; m = cross[k, e]; w = d[k, a] / (d[k, a] - d[k, b] + 1e-30)
            pts.append(np.where(m[:, None], T[k, a, 1:] + w[:, None] * (T[k, b, 1:] - T[k, a, 1:]), np.nan))
        P = np.stack(pts, 1); seg_ = np.array([p[~np.isnan(p[:, 0])][:2] for p in P])
        w_ = J0 + ax * t * L + seg_.mean(1) @ np.c_[ex, ey].T                       # segment midpoints in the world
        if own: seg_ = seg_[AT.nearest_axis(w_) == me]                                     # this segment's axis is the nearest of the atlas skeleton
        if len(seg_): out[i] = seg_
    return out
TH = 72
def cover(segs):                                                  # fraction of angle bins the outline reaches (points along each cut segment)
    if not segs: return 0.0
    u = np.linspace(0, 1, 9)[None, :, None]; P = np.vstack([(q[:, :1] + u * (q[:, 1:] - q[:, :1])).reshape(-1, 2) for q in segs.values()])
    return len(np.unique(((np.arctan2(P[:, 1], P[:, 0]) + np.pi) / (2 * np.pi) * TH).astype(int) % TH)) / TH
# joint zones: where the section passes into the trunk or the next limb, this segment owns only part of the skin ring.
# Levels from each end inward whose ring coverage is below the Otsu split of all levels' coverage are dropped: the
# slices stop before the joint, and the zone between two segments' end slices is left to a blend.
lv0 = np.linspace(t_lo, t_hi, 101); cov = np.array([cover(section(skin_ids, t, False)) for t in lv0])
if np.ptp(cov) > 0:
    ok = cov > otsu(cov); i0 = int(np.argmax(ok)); i1 = len(ok) - 1 - int(np.argmax(ok[::-1])); t_lo, t_hi = lv0[i0], lv0[i1]
print('ZS ring coverage %.2f..%.2f; slices kept from t %.2f to %.2f' % (cov.min(), cov.max(), t_lo, t_hi))
if DENSE:                                                          # dense: one slice per muscle edge length (the finest detail the atlas has here)
    sf_ = F[np.isin(OB[F[:, 0]], mus_ids)]; el_ = float(np.median(np.linalg.norm(V[sf_[:, 1]] - V[sf_[:, 0]], axis=1)))
    nlev = int(np.ceil((t_hi - t_lo) * L / el_)) + 1; print('ZS dense: spacing %.4f m (median edge of the segment atlas muscles) -> %d slices' % (el_, nlev), flush=True)
else: nlev = 101
levels = np.linspace(t_lo, t_hi, nlev); dense = []
# minimal-area planes (planes.py): each level's plane is the one through the axis point with the smallest cut of the
# segment's own atlas skin, so slices follow the body (bent spine, shoulder and hip zones) instead of one fixed direction
import planes as PL
PATH = PL.Path([J0, J1]); FSK = F[np.isin(OB[F[:, 0]], skin_ids)]
PLN = PL.min_area_planes(V, FSK, PATH, levels, AT.roll_target(seg) - J0, log=lambda s: print('ZS', s, flush=True))
FOB = OB[F[:, 0]]; FBY = {}
def faces_of(i):                                                  # each structure's faces, looked up once
    if i not in FBY: FBY[i] = F[FOB == i]
    return FBY[i]
def section_pl(ids, k, own=True):
    """segments (k,2,2) where the structures' triangles cross level k's plane, in that plane's (e1, e2) coordinates"""
    out = {}; p, n, e1, e2 = PLN['p'][k], PLN['n'][k], PLN['e1'][k], PLN['e2'][k]
    for i in ids:
        seg_ = PL.cut(V, faces_of(i), p, n, e1, e2)
        if not len(seg_): continue
        if own: seg_ = seg_[AT.nearest_axis(p + seg_.mean(1) @ np.c_[e1, e2].T) == me]
        if len(seg_): out[i] = seg_
    return out
for kk, t in enumerate(levels):
    sk = section_pl(skin_ids, kk, False); mu = section_pl(mus_ids, kk); bo = section_pl(bone_ids, kk)
    P = np.vstack([s.reshape(-1, 2) for s in sk.values()]) if sk else np.zeros((0, 2))
    th = np.arctan2(P[:, 1], P[:, 0]); b = ((th + np.pi) / (2 * np.pi) * TH).astype(int) % TH
    rr = np.zeros(TH); np.maximum.at(rr, b, np.hypot(P[:, 0], P[:, 1])); rr[rr == 0] = np.nan
    area = lambda segs: sum(abs(s[:, 0, 0] * s[:, 1, 1] - s[:, 0, 1] * s[:, 1, 0]).sum() / 2 for s in segs.values())   # sum of triangle areas to the origin
    dense.append(dict(t=t, skin_r=rr, muscles=sorted(mu), skin_area=np.nanmean(rr) ** 2 * np.pi if np.isfinite(rr).any() else 0,
                      bone_area=area(bo), sk=sk, mu=mu, bo=bo))
def polar(segs):                                                  # outermost radius per angle bin, gaps filled around the circle
    u = np.linspace(0, 1, 9)[None, :, None]                       # points along each cut segment, not just its ends
    P = np.vstack([(q[:, :1] + u * (q[:, 1:] - q[:, :1])).reshape(-1, 2) for q in segs.values()]) if segs else np.zeros((0, 2)); rr = np.zeros(TH)
    if len(P): np.maximum.at(rr, ((np.arctan2(P[:, 1], P[:, 0]) + np.pi) / (2 * np.pi) * TH).astype(int) % TH, np.hypot(P[:, 0], P[:, 1]))
    ok = rr > 0
    if ok.any() and not ok.all(): k = np.arange(TH); rr = np.interp(k, np.r_[k[ok] - TH, k[ok], k[ok] + TH], np.tile(rr[ok], 3))
    return rr
X = np.array([np.r_[polar(d['sk']), polar(d['mu']), polar(d['bo'])] for d in dense])
X3 = X.reshape(len(X), 3, TH); Fc = np.fft.rfft(X3[:, 0], axis=1) / TH              # skin r(theta) per level, Fourier terms
desc = {'mean radius': Fc[:, 0].real, 'off-centre': 2 * np.abs(Fc[:, 1]), 'flattening': 2 * np.abs(Fc[:, 2])}
def persistence(y):                                               # 1-D sublevel persistence: (birth index, death value - birth value)
    o = np.argsort(y, kind='stable'); par = -np.ones(len(y), int); low = {}; out = []
    def f(i):
        while par[i] != i: par[i] = par[par[i]]; i = par[i]
        return i
    for i in o:
        par[i] = i; low[i] = i
        for j in (i - 1, i + 1):
            if 0 <= j < len(y) and par[j] >= 0:
                a_, b_ = f(i), f(j)
                if a_ == b_: continue
                old, young = (a_, b_) if y[low[a_]] <= y[low[b_]] else (b_, a_)
                if low[young] not in (0, len(y) - 1): out.append((low[young], y[i] - y[low[young]]))
                par[young] = old
    return out
cand = []
for nm, y in desc.items():
    yn = (y - y.min()) / (np.ptp(y) + 1e-30)                     # each descriptor on its own 0..1 scale
    cand += [(k, p_, nm + ' min') for k, p_ in persistence(yn) if p_ > 0] + [(k, p_, nm + ' max') for k, p_ in persistence(-yn) if p_ > 0]
    for k, lab in ((int(yn.argmin()), ' min'), (int(yn.argmax()), ' max')):   # the global extremum never dies: always kept
        if 0 < k < len(yn) - 1: cand.append((k, np.inf, nm + lab))
pers = np.array([c[1] for c in cand]); fin = np.isfinite(pers); split = np.exp(otsu(np.log(pers[fin]))) if fin.sum() > 2 else 0
keep = pers > split
crit = {0: 'proximal end' if t_lo != 0 else 'proximal joint', len(levels) - 1: 'distal end' if t_hi != 1 else 'distal joint'}
for (k, p_, nm), kp in sorted(zip(cand, keep), key=lambda c: -c[0][1]):
    if kp: crit.setdefault(int(k), nm)
if DENSE:                                                          # every level is a slice
    for k in range(len(levels)): crit.setdefault(k, 'dense')
why = crit; crit = sorted(crit)
print('ZS extrema %d, kept %d by persistence (Otsu split %.3f of each descriptor range)' % (len(cand), keep.sum(), split))
print('ZS critical levels %d: %s' % (len(crit), ', '.join('%.2f %s' % (levels[k], why[k]) for k in crit)))
sa = np.array([d['skin_area'] for d in dense]); ba = np.array([d['bone_area'] for d in dense])
for k in crit: print('ZS   t=%.2f  muscles %2d  skin r %.4f  muscle r %.4f  bone r %.4f' % (levels[k], len(dense[k]['muscles']), X[k, :TH].mean(), X[k, TH:2 * TH].mean(), X[k, 2 * TH:].mean()))
# sheet: one tile per critical level
S = 300; sc = S / (2.2 * R); tiles = []
pal = [tuple(int(255 * c) for c in cv2.cvtColor(np.uint8([[[i * 37 % 180, 150, 210]]]), cv2.COLOR_HSV2RGB)[0, 0][::-1] / 255) for i in range(64)]
for k in crit:
    d = dense[k]; im = np.full((S, S, 3), 255, np.uint8); P = lambda s: (np.stack([s[..., 0], -s[..., 1]], -1) * sc + S / 2).astype(np.int32)
    for j, (i, s) in enumerate(d['mu'].items()):
        cv2.polylines(im, list(P(s)), False, (60 + j * 23 % 120, 50, 200 - j * 17 % 80), 2)
    for s in d['bo'].values():
        cv2.polylines(im, list(P(s)), False, (120, 170, 200), 3)
    for s in d['sk'].values():
        cv2.polylines(im, list(P(s)), False, (110, 120, 235), 1)
    cv2.putText(im, 't=%.2f  %d muscles' % (d['t'], len(d['muscles'])), (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(im, why[k], (6, S - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 60), 1); tiles.append(im)
cols = 4; rows = -(-len(tiles) // cols); tiles += [np.full((S, S, 3), 255, np.uint8)] * (rows * cols - len(tiles))
sheet = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)]); cv2.imwrite(os.path.join(od, seg + '_sheet.png'), sheet)
SG, SK, SL, SS = [], [], [], []                                   # critical slices' outlines: segment, kind (0 skin 1 muscle 2 bone), level, structure
for ci, k in enumerate(crit):
    for kind, key in ((0, 'sk'), (1, 'mu'), (2, 'bo')):
        for i, q in dense[k][key].items(): SG.append(q); SK.append(np.full(len(q), kind)); SL.append(np.full(len(q), ci)); SS.append(np.full(len(q), i))
np.savez(os.path.join(od, seg + '.npz'), seg_xy=np.concatenate(SG), seg_kind=np.concatenate(SK), seg_level=np.concatenate(SL), seg_struct=np.concatenate(SS), J0=J0, J1=J1, ax=ax, ex=ex, ey=ey, L=L, levels=levels, roll_t=roll_t, coverage=cov, cover_t=lv0, bones=np.array([names[i] for i in bone_ids]), crit=np.array(crit),
         X=X, TH=TH, skin_area=sa, bone_area=ba, R=R, why=np.array([why[k] for k in crit]),
         plane_p=PLN['p'], plane_n=PLN['n'], plane_e1=PLN['e1'], plane_e2=PLN['e2'], plane_tilt=PLN['tilt'],   # per level (all levels)
         muscles=np.array([names[i] for i in mus_ids]), present=np.array([[i in d['muscles'] for i in mus_ids] for d in dense]))
json.dump({'segment': seg, 'length_m': L, 'critical': [round(levels[k], 3) for k in crit], 'muscles': [names[i] for i in mus_ids]}, open(os.path.join(od, seg + '.json'), 'w'), indent=1)
print('ZS done')
