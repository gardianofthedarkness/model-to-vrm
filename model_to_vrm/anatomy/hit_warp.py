"""Option 3 back in 3D: HIT's tissue surfaces (hit_tissues.py) moved by the per-slice fits (hit_slices.py) of every segment:
  E:\\splat\\.venv\\Scripts\\python.exe hit_warp.py <model dir>
  owner     each tissue vertex belongs to the segment of its nearest SMPL vertex's part (hit_slices.SMPL_PARTS)
  between   the segment's slice planes are ordered along its path; the vertex lies between the two neighbouring planes
            where its signed distance changes sign (outside the first / last plane: that plane alone)
  move      in each of those two slices' frames: in-plane coords q -> shift + q * factor(angle of q), the distance off the
            plane kept; the two results blended by the vertex's distance to each plane
  writes    anatomy/hit_warped.npz: <tissue>_V / _F (moved), <tissue>_V0 (before), owner segment per vertex, and a half of
            each surface (faces on one side of the body's own left-right mid plane) for cut views"""
import sys, os, glob, numpy as np
from scipy.spatial import cKDTree
sys.argv += [] ; R = sys.argv[1]; j = lambda *p: os.path.join(R, *p)
SMPL_PARTS = {'pelvis': [0], 'trunk': [3, 6, 9, 13, 14], 'neck': [12], 'head': [15],
    'thigh_r': [2], 'calf_r': [5], 'foot_r': [8, 11], 'uparm_r': [17], 'forearm_r': [19], 'hand_r': [21, 23],
    'thigh_l': [1], 'calf_l': [4], 'foot_l': [7, 10], 'uparm_l': [16], 'forearm_l': [18], 'hand_l': [20, 22]}   # as in hit_slices.py
HT = np.load(j('anatomy', 'hit_tissues.npz')); SV = HT['smpl_V']; gender = str(HT['gender'])
part = np.load(r'E:\figures\models\hit\smpl_models\smpl\SMPL_%s.npz' % gender.upper())['weights'].argmax(1)
p2s = {p: s for s, ps in SMPL_PARTS.items() for p in ps}; sk = cKDTree(SV)
SL = {}
for s in SMPL_PARTS:
    f = j('anatomy', s + '_hit.npz')
    if os.path.exists(f):
        Z = np.load(f, allow_pickle=True)
        if 'e1' in Z: SL[s] = dict(P=Z['points'], N=Z['normals'], E1=Z['e1'], E2=Z['e2'], C=Z['shift'], FA=Z['factor'])
print('HW segments with slices: %d (%s)' % (len(SL), ', '.join(SL)), flush=True)
TH = 72
def warp_in(sl, k, v):                                            # vertices v moved by slice k (in-plane shift + per-angle stretch)
    d = v - sl['P'][k]; a = d @ sl['N'][k]; q = np.c_[d @ sl['E1'][k], d @ sl['E2'][k]]; fac = sl['FA'][k]; c = sl['C'][k]
    if not (np.isfinite(fac).all() and np.isfinite(c).all()): return v.copy(), np.zeros(len(v), bool)
    ang = np.arctan2(q[:, 1], q[:, 0]); kf = (ang + np.pi) / (2 * np.pi) * TH - 0.5
    f = np.interp(kf, np.r_[-1, np.arange(TH), TH], np.r_[fac[-1], fac, fac[0]]); q2 = c + q * f[:, None]
    return sl['P'][k] + a[:, None] * sl['N'][k] + q2[:, :1] * sl['E1'][k] + q2[:, 1:] * sl['E2'][k], np.ones(len(v), bool)
out = {}
for tis in ('fat', 'lean', 'bone'):
    V0 = HT['%s_Vhit' % tis].astype(float); F = HT['%s_F' % tis]; own = np.array([p2s.get(p, '') for p in part[sk.query(V0)[1]]])
    V = V0.copy(); moved = np.zeros(len(V), bool)
    for s, sl in SL.items():
        m = np.flatnonzero(own == s)
        if not len(m): continue
        v = V0[m]; D = np.einsum('vkd,kd->vk', v[:, None] - sl['P'][None], sl['N'])   # signed distance to every slice plane
        cross = D[:, :-1] * D[:, 1:] <= 0; score = np.where(cross, np.abs(D[:, :-1]) + np.abs(D[:, 1:]), np.inf)
        k = score.argmin(1); has = np.isfinite(score[np.arange(len(m)), k]); kn = np.abs(D).argmin(1)
        res = np.zeros_like(v); okm = np.zeros(len(m), bool)
        for kk in np.unique(np.r_[k[has], kn[~has]]):
            sel_b = has & (k == kk)                               # between kk and kk+1: blend by distance
            if sel_b.any():
                va, oa = warp_in(sl, kk, v[sel_b]); vb, ob = warp_in(sl, kk + 1, v[sel_b])
                da, db = np.abs(D[sel_b, kk]), np.abs(D[sel_b, kk + 1]); w = db / (da + db + 1e-12)
                res[sel_b] = w[:, None] * va + (1 - w[:, None]) * vb; okm[sel_b] = oa & ob
            sel_o = ~has & (kn == kk)                             # outside the slice range: the nearest plane alone
            if sel_o.any(): res[sel_o], okm[sel_o] = warp_in(sl, kk, v[sel_o])
        V[m[okm]] = res[okm]; moved[m[okm]] = True
    disp = np.linalg.norm(V - V0, axis=1)
    print('HW %s: %d vertices, moved %.0f%%, displacement median %.4f m, p95 %.4f m, max %.3f m' % (
        tis, len(V), 100 * moved.mean(), np.median(disp[moved]) if moved.any() else 0, np.percentile(disp[moved], 95) if moved.any() else 0, disp.max()), flush=True)
    out['%s_V' % tis] = V; out['%s_V0' % tis] = V0; out['%s_F' % tis] = F; out['%s_owner' % tis] = own
# halves for cut views: faces whose centre is on one side of the body's left-right mid plane (the SMPL body's widest axis
# among the two horizontal ones would be the arms; use the hips: right hip -> left hip direction)
Kp = SV[np.isin(part, [1])].mean(0) - SV[np.isin(part, [2])].mean(0); nx = Kp / np.linalg.norm(Kp); mid = SV.mean(0) @ nx
for tis in ('fat', 'lean', 'bone'):
    V, F = out['%s_V' % tis], out['%s_F' % tis]; keep = (V[F].mean(1) @ nx) < mid; out['%s_Fhalf' % tis] = F[keep]
np.savez_compressed(j('anatomy', 'hit_warped.npz'), **out, cut_normal=nx, cut_offset=mid)
print('HW done')
