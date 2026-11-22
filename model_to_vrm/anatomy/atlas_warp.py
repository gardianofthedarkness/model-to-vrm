"""Option 4, step 1: the whole Z-Anatomy atlas (bones, muscles, skin) warped into the model by every segment's slice fit,
blended across the joints:
  E:\\splat\\.venv\\Scripts\\python.exe atlas_warp.py <atlas.npz> <atlas slices dir> <model dir>
  per segment  limb_build.py's warp: an atlas point's level t along the segment's atlas axis and its (x, y) in the slice
               plane -> the fitted slice map at t (linear between the fitted slices; held at the end values outside the
               fitted range) -> the model's slice plane at t (fit_slices.py: M, C per slice)
  owner        a point belongs to the segment of its nearest atlas bone (segments.py lists each segment's bones)
  blending     a point takes its owner segment's warp; within the thinner limb's radius r (the atlas skin's mean
               radius at the joint) of a joint it hands over to the segments meeting there, linearly along the bone, 50/50 at
               the joint, so both sides meet continuously. Joints: a segment end lying inside another segment's atlas skin outline (in its own direction), or within its own radius of that axis (made symmetric)
  writes       anatomy/atlas_warp.npz: V (warped, float32) for the kept systems, F, obj, names, system, weights summary"""
import sys, os, numpy as np
from segments import ORDER, Atlas
ap, sd, R = sys.argv[1:4]; FD = sys.argv[4] if len(sys.argv) > 4 else os.path.join(R, 'anatomy')   # fits folder (also the output folder)
j = lambda *p: os.path.join(R, *p); jf = lambda *p: os.path.join(FD, *p)
AT = Atlas(ap); keep_sys = ('bone', 'muscle', 'skin')
label = np.array([n.endswith(('.g', '.j')) for n in AT.names])   # Z-Anatomy names: '.g' group titles, '.j' landmark outlines (not tissue)
vk = np.isin(np.array(AT.sys)[AT.OB], keep_sys) & ~label[AT.OB]; V = AT.V[vk]; idx_old = np.flatnonzero(vk)
remap = -np.ones(len(AT.V), np.int64); remap[idx_old] = np.arange(len(idx_old))
fk = vk[AT.F].all(1); F = remap[AT.F[fk]]; OB = AT.OB[vk]
print('AW atlas kept: %d vertices, %d faces (%s)' % (len(V), len(F), ', '.join(keep_sys)), flush=True)
SEGS = []
for s in ORDER:
    fp, ap_ = jf(s + '_fit.npz'), os.path.join(sd, s + '.npz')
    if not (os.path.exists(fp) and os.path.exists(ap_)): continue
    Fi, A = np.load(fp, allow_pickle=True), np.load(ap_); TH = int(A['TH'])
    if 'M' not in Fi: continue
    SEGS.append(dict(name=s, J0a=A['J0'], axa=A['ax'], exa=A['ex'], eya=A['ey'], La=float(A['L']), lv=A['levels'],
                     rmean=A['X'][:, :TH].mean(1), rdir=A['X'][:, :TH], TH=TH, tl=Fi['t'], MC=np.c_[Fi['M'].reshape(-1, 4), Fi['C']],
                     J0=Fi['J0'], ax=Fi['ax'], ex=Fi['ex'], ey=Fi['ey'], L=float(Fi['L'])))
    if 'plane_p' in Fi and 'plane_p' in A:                        # minimal-area planes (planes.py): atlas plane k <-> model plane k
        cr = A['crit']; SEGS[-1].update(app=A['plane_p'][cr], apn=A['plane_n'][cr], ae1=A['plane_e1'][cr], ae2=A['plane_e2'][cr],
                                        mpp=Fi['plane_p'], mpn=Fi['plane_n'], me1=Fi['plane_e1'], me2=Fi['plane_e2'], Mk=Fi['M'], Ck=Fi['C'])
print('AW segments: %d (%s)' % (len(SEGS), ', '.join(s['name'] for s in SEGS)), flush=True)
def seg_warp(s, P):
    if 'app' in s: return plane_warp(s, P)
    d = P - s['J0a']; t = d @ s['axa'] / s['La']; x, y = d @ s['exa'], d @ s['eya']
    tc = np.clip(t, s['tl'].min(), s['tl'].max()); m = np.stack([np.interp(tc, s['tl'], s['MC'][:, c]) for c in range(6)], -1)
    x2, y2 = m[:, 0] * x + m[:, 1] * y + m[:, 4], m[:, 2] * x + m[:, 3] * y + m[:, 5]
    return s['J0'] + s['ax'] * (t * s['L'])[:, None] + x2[:, None] * s['ex'] + y2[:, None] * s['ey']
def plane_warp(s, P):
    """tilted slice planes: a point between atlas planes k and k+1 (by its signed distances to them) takes the blend
    of both planes' maps; each map = the fitted 2D map in the plane plus the distance along the normal, scaled by the
    local plane spacing model / atlas. Before the first / past the last plane: that plane's map alone."""
    K = len(s['app']); D = P @ s['apn'].T - (s['app'] * s['apn']).sum(1)[None]          # (n, K) signed distances
    gap_a = np.linalg.norm(np.diff(s['app'], axis=0), axis=1); gap_m = np.linalg.norm(np.diff(s['mpp'], axis=0), axis=1)
    sc = np.r_[gap_m / np.maximum(gap_a, 1e-12), 0]; sc[-1] = sc[-2] if K > 1 else s['L'] / s['La']
    def one(k, idx):
        d = P[idx] - s['app'][k]; q = np.c_[d @ s['ae1'][k], d @ s['ae2'][k]]; q2 = q @ s['Mk'][k].T + s['Ck'][k]
        return s['mpp'][k] + np.outer(D[idx, k] * sc[min(k, K - 2) if K > 1 else 0], s['mpn'][k]) + q2[:, :1] * s['me1'][k] + q2[:, 1:] * s['me2'][k]
    k = np.clip((D >= 0).sum(1) - 1, -1, K - 1); out = np.zeros_like(P)
    for kk in range(-1, K):
        idx = np.flatnonzero(k == kk)
        if not len(idx): continue
        if kk == -1: out[idx] = one(0, idx); continue
        if kk == K - 1: out[idx] = one(K - 1, idx); continue
        w = np.clip(D[idx, kk] / (D[idx, kk] - D[idx, kk + 1] + 1e-30), 0, 1)[:, None]
        out[idx] = (1 - w) * one(kk, idx) + w * one(kk + 1, idx)
    return out
def axis_dist(s, P):                                              # distance to the segment's atlas axis, and the level t
    a, b = s['J0a'], s['J0a'] + s['axa'] * s['La']; dd = b - a; u = np.clip(((P - a) @ dd) / (dd @ dd), 0, 1)
    return np.linalg.norm(P - a - u[:, None] * dd, axis=1), u
# joints: segment ends that meet (an end of B within the limb radius of an end of A)
END = {}
def rad_at(s, u): return float(np.interp(u, s['lv'], s['rmean']))   # the atlas skin's mean radius at level u (held past the kept range)
def rad_dir(s, u, pt):                                            # the atlas skin's radius at level u in the direction of point pt
    d = pt - s['J0a']; th = np.arctan2(d @ s['eya'], d @ s['exa']); b = int((th + np.pi) / (2 * np.pi) * s['TH']) % s['TH']
    k = int(np.clip(np.searchsorted(s['lv'], u), 0, len(s['lv']) - 1)); return float(s['rdir'][k, b])
for k, s in enumerate(SEGS):
    for e, u0 in enumerate((0.0, 1.0)):
        pt = s['J0a'] + s['axa'] * s['La'] * u0; nb = []
        for k2, s2 in enumerate(SEGS):                            # B meets A's end when A's end lies within the larger of the
            if k2 == k: continue                                  # two limbs' radii from B's axis (a limb on the side of the trunk too)
            d2, u2 = axis_dist(s2, pt[None]); 
            if d2[0] < max(rad_at(s, u0), rad_dir(s2, u2[0], pt)): nb.append(k2)   # A's end inside B's skin outline
        END[(k, e)] = [rad_at(s, u0), nb]
for (k, e), (r, nb) in list(END.items()):                         # symmetric: if A's end meets B, B's nearest end meets A
    for k2 in nb:
        s2 = SEGS[k2]; pt = SEGS[k]['J0a'] + SEGS[k]['axa'] * SEGS[k]['La'] * e
        e2 = int(np.linalg.norm(pt - (s2['J0a'] + s2['axa'] * s2['La'])) < np.linalg.norm(pt - s2['J0a']))
        if k not in END[(k2, e2)][1]: END[(k2, e2)][1].append(k)
print('AW joints: ' + '; '.join('%s %s: %s' % (SEGS[k]['name'], 'start' if e == 0 else 'end', ','.join(SEGS[x]['name'] for x in v[1]))
                               for (k, e), v in END.items() if v[1]), flush=True)
# ownership: each atlas point belongs to the segment of its nearest atlas bone surface (humerus vs ribs / scapula ...), not
# to the nearest axis line: in the atlas the arms hang beside the trunk, so by axis distance the chest-side muscles and the
# armpit went with the arm and stretched into "wings" when the arm is raised
from scipy.spatial import cKDTree
from segments import SEG as SEGDEF
bone_owner = {}
for k, s in enumerate(SEGS):
    for n in AT.bone_names(SEGDEF[s['name']]['bones']): bone_owner.setdefault(n, k)
bsel = np.flatnonzero(np.array([AT.names[o] in bone_owner for o in OB]))
bown = np.array([bone_owner[AT.names[o]] for o in OB[bsel]]); btree = cKDTree(V[bsel])
print('AW ownership by nearest bone: %d bone objects in %d segments' % (len(bone_owner), len(set(bone_owner.values()))), flush=True)
out = np.zeros_like(V); wmax = np.zeros(len(V)); nseg = np.zeros(len(V))
CH = 200000
for i0 in range(0, len(V), CH):
    P = V[i0:i0 + CH]; D, U = zip(*[axis_dist(s, P) for s in SEGS]); D = np.array(D); U = np.array(U)
    near = bown[btree.query(P)[1]]; W = np.zeros_like(D); W[near, np.arange(len(P))] = 1.0
    for k, s in enumerate(SEGS):                                  # near an end with neighbours: hand over toward them, 50/50 at the joint
        m = np.flatnonzero(near == k)
        if not len(m): continue
        for e in (0, 1):
            r, nb = END[(k, e)]
            if not nb: continue
            rj = min([r] + [rad_at(SEGS[k2], 0.0 if np.linalg.norm(SEGS[k2]['J0a'] - (s['J0a'] + s['axa'] * s['La'] * e)) < np.linalg.norm(SEGS[k2]['J0a'] + SEGS[k2]['axa'] * SEGS[k2]['La'] - (s['J0a'] + s['axa'] * s['La'] * e)) else 1.0) for k2 in nb])
            dist = (U[k, m] if e == 0 else 1 - U[k, m]) * s['La']; g = 0.5 * np.clip(1 - dist / rj, 0, 1)   # hand-over zone: the thinner limb's radius
            if not (g > 0).any(): continue
            W[k, m] -= g
            Dn = D[nb][:, m]; wn = 1 / (Dn + 1e-9); wn /= wn.sum(0)   # shared among the neighbours by nearness
            for i, k2 in enumerate(nb): W[k2, m] += g * wn[i]
    acc = np.zeros_like(P)
    for k, s in enumerate(SEGS):
        m = W[k] > 1e-6
        if m.any(): acc[m] += W[k, m, None] * seg_warp(s, P[m])
    out[i0:i0 + CH] = acc; wmax[i0:i0 + CH] = W.max(0); nseg[i0:i0 + CH] = (W > 0.05).sum(0)
    print('AW %d / %d' % (min(i0 + CH, len(V)), len(V)), flush=True)
print('AW points taking one segment\'s warp (weight > 0.95): %.0f%%; blended over 2+ segments: %.0f%%' % (100 * (wmax > 0.95).mean(), 100 * (nseg >= 2).mean()), flush=True)
sysv = np.array(AT.sys)[OB[F[:, 0]]]
np.savez(jf('atlas_warp.npz'), V=out.astype(np.float32), V0=V.astype(np.float32), F=F.astype(np.int32), **{'F_' + k: F[sysv == k].astype(np.int32) for k in keep_sys}, obj=OB.astype(np.int32), names=np.array(AT.names), system=np.array(AT.sys), wmax=wmax.astype(np.float32))
print('AW done')
