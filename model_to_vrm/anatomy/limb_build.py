"""Build a limb segment's real bones, muscles and closed skin inside the model, from the atlas and the slice fit:
  python limb_build.py <atlas.npz> <atlas slices.npz> <segment_fit.npz> <out.npz>
  warp      every atlas point: its level t along the atlas segment and its (x, y) in the slice plane -> the fitted
            slice parameters at t (shift, roll, width, depth; linear between the critical slices, roll unwrapped) ->
            the model's slice plane at t. Bones and muscles of the segment keep their own triangles.
  skin      closed: the atlas's dense skin outline at every one of its 101 levels, warped the same way, joined ring to
            ring, with a cap at each end (fan to the ring's centre)
  smooth    bones and muscles Taubin-smoothed over the simulation's tet size; each muscle gets strands along its long
            axis (centreline + offsets at its local radius)
  writes    skin V/F, and per structure (bones, muscles) V/F/name, all in the model's world coordinates"""
import sys, re, numpy as np
ap, sp, fp, out = sys.argv[1:5]
Z = np.load(ap); V, F, OB = Z['V'].astype(np.float64), Z['F'], Z['obj']; names = [str(x) for x in Z['names']]; sysn = [str(x) for x in Z['system']]
A = np.load(sp); Fi = np.load(fp, allow_pickle=True)
J0a, axa, exa, eya, La = A['J0'], A['ax'], A['ex'], A['ey'], float(A['L']); TH = int(A['TH']); lv = A['levels']; Xs = A['X'][:, :TH]
P2, tl = Fi['params'].copy(), Fi['t']; J0, ax, ex, ey, L = Fi['J0'], Fi['ax'], Fi['ex'], Fi['ey'], float(Fi['L'])
P2[:, 2] = np.unwrap(P2[:, 2])
MCf = np.c_[Fi['M'].reshape(-1, 4), Fi['C']] if 'M' in Fi else None  # the fitted linear map + shift per slice (fit_slices.py)
def params(t): return np.stack([np.interp(t, tl, P2[:, c]) for c in range(5)], -1)
def warp(p):
    d = p - J0a; t = d @ axa / La; x, y = d @ exa, d @ eya
    if MCf is not None:                                           # linear between the critical slices
        m = np.stack([np.interp(t, tl, MCf[:, c]) for c in range(6)], -1); x2, y2 = m[:, 0] * x + m[:, 1] * y + m[:, 4], m[:, 2] * x + m[:, 3] * y + m[:, 5]
    else:
        cx, cy, ph, sx, sy = params(t).T; c, s = np.cos(ph), np.sin(ph); u, v = x * sx, y * sy; x2, y2 = c * u - s * v + cx, s * u + c * v + cy
    return J0 + ax * (t * L)[:, None] + x2[:, None] * ex + y2[:, None] * ey, t
bones = [str(x) for x in A['bones']] if 'bones' in A else {'forearm_r': ['Radius.r', 'Ulna.r'], 'thigh_r': ['Femur.r']}[re.sub(r'_fit\.npz$', '', fp.replace('\\', '/').split('/')[-1])]
mus = [str(x) for x in A['muscles']]; S = []
for nm in bones + mus:
    i = names.index(nm); fm = OB[F[:, 0]] == i; f = F[fm]; u, inv = np.unique(f, return_inverse=True); W, t = warp(V[u])
    keep = (t[inv.reshape(-1, 3)] > -0.02).all(1) & (t[inv.reshape(-1, 3)] < 1.02).all(1)     # the part inside the segment
    if keep.sum() < 4: continue
    f2 = inv.reshape(-1, 3)[keep]; u2, inv2 = np.unique(f2, return_inverse=True)
    S.append((nm, 'bone' if nm in bones else 'muscle', W[u2], inv2.reshape(-1, 3)))
th = (np.arange(TH) + 0.5) / TH * 2 * np.pi - np.pi; rings = []
for k, t in enumerate(lv):
    q = np.c_[Xs[k] * np.cos(th), Xs[k] * np.sin(th)]; p = J0a + axa * t * La + q[:, 0:1] * exa + q[:, 1:2] * eya; rings.append(warp(p)[0])
R = np.array(rings); nL = len(R); SV = R.reshape(-1, 3); SF = []
for k in range(nL - 1):
    for j in range(TH):
        a, b, c, d = k * TH + j, k * TH + (j + 1) % TH, (k + 1) * TH + j, (k + 1) * TH + (j + 1) % TH; SF += [(a, c, b), (b, c, d)]
c0, c1 = len(SV), len(SV) + 1; SV = np.vstack([SV, R[0].mean(0), R[-1].mean(0)])
for j in range(TH): SF += [(c0, j, (j + 1) % TH), (c1, (nL - 1) * TH + (j + 1) % TH, (nL - 1) * TH + j)]
SF = np.array(SF)
vol = np.einsum('ij,ij->i', SV[SF[:, 0]], np.cross(SV[SF[:, 1]], SV[SF[:, 2]])).sum() / 6
if vol < 0: SF = SF[:, ::-1]
# smoothing: details finer than the flesh simulation's tet size cannot move on their own, so each bone and muscle is
# Taubin-smoothed over that length (diffusion: iterations ~ (length / its own median edge)^2); h = half the median
# skin-to-bone distance, the tet edge limb_sim.py uses
import trimesh
from scipy.spatial import cKDTree
BV = np.vstack([s_[2] for s_ in S if s_[1] == 'bone']); h = float(np.median(cKDTree(BV).query(SV)[0])) / 2; S2, strands = [], []
for nm, kind, Wv, Wf in S:
    m = trimesh.Trimesh(Wv, Wf, process=False); el = float(np.median(m.edges_unique_length)) + 1e-12
    trimesh.smoothing.filter_taubin(m, iterations=int(np.ceil((h / el) ** 2))); S2.append((nm, kind, np.asarray(m.vertices), Wf))
    if kind != 'muscle' or len(Wv) < 8: continue
    P_ = np.asarray(m.vertices); c_ = P_.mean(0); w_, e_ = np.linalg.eigh(np.cov((P_ - c_).T)); a_ = e_[:, -1]; s_ax = (P_ - c_) @ a_
    nb = max(2, int(np.ptp(s_ax) / h)); edges = np.linspace(s_ax.min(), s_ax.max(), nb + 1); cen, rad = [], []
    for b in range(nb):                                           # strand = the muscle's centreline plus offsets scaled by its local radius
        q = P_[(s_ax >= edges[b]) & (s_ax <= edges[b + 1])]
        if len(q) < 3: continue
        cc = q.mean(0); d_ = (q - cc) - np.outer((q - cc) @ a_, a_); cen.append(cc); rad.append(np.median(np.linalg.norm(d_, axis=1)))
    if len(cen) < 2: continue
    cen, rad = np.array(cen), np.array(rad); u_ = np.cross(a_, [0, 0, 1.0]) if abs(a_[2]) < 0.9 else np.cross(a_, [1.0, 0, 0]); u_ /= np.linalg.norm(u_); v_ = np.cross(a_, u_)
    for ang in np.linspace(0, 2 * np.pi, 6, endpoint=False):     # 6 strands around + the centreline (a drawing choice)
        strands.append((nm, cen + 0.6 * rad[:, None] * (np.cos(ang) * u_ + np.sin(ang) * v_)))
    strands.append((nm, cen))
S = S2
print('LB smoothed over %.4f m (the tet edge); %d muscle strands' % (h, len(strands)))
print('LB skin: %d rings x %d, volume %.1f cm^3; bones %s; muscles %d' % (nL, TH, abs(vol) * 1e6, [s[0] for s in S if s[1] == 'bone'], sum(s[1] == 'muscle' for s in S)))
np.savez(out, skin_V=SV, skin_F=SF, names=np.array([s[0] for s in S]), kinds=np.array([s[1] for s in S]),
         Vs=np.array([s[2] for s in S], dtype=object), Fs=np.array([s[3] for s in S], dtype=object), J0=J0, J1=Fi['J1'], ax=ax, L=L, h=h,
         strands=np.array([q for _, q in strands], dtype=object), strand_muscle=np.array([n for n, _ in strands]), allow_pickle=True)
