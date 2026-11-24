"""Skin on the elf's left leg held by attachments to the muscle and bone beneath it, after the user's report
"Mathematical Modeling of Skin Attachments" (19 Nov 2026), used here as a provisional formulation.
  E:\\splat\\.venv\\Scripts\\python.exe skin_attach.py <fits dir> <model dir> <leg_transforms.npz> <out dir> [iterations]

  anchors   (report eq. 3-4) every skin vertex j is tied, once at rest, to the nearest point a_j of the deeper surface: the
            vertices of this leg's bones and of the muscles carried by them.  The anchor is a material point and is never
            reassigned.  Bones move rigidly (eq. 3, thigh and shank transforms of the calibrated clip; hip bone fixed);
            muscle points move by their bone shares blended as dual quaternions (the existing kinematic muscle rig).
  interface (eq. 21) w_j = 1/2 kappa_j |g_j|^2 per unit area, isotropic (kappa_t = kappa_n), with the gap measured from rest
            in the anchor's own rotating frame: g_j = (p_j - a_j) - Q_j (p_j0 - a_j0).  Connector stiffness k_j = kappa_j A_j
            (the report's mesh-scaling note), A_j = one third of the areas of the skin triangles at j.
  skin      a membrane with as-rigid-as-possible stretch energy (Sorkine & Alexa 2007), weight S on clamped cotangent edges.
  scale     only the ratio S / kappa enters a quasi-static shape (report eq. 23 and section 10), i.e. the transfer length
            l_s = sqrt(S / kappa).  It is set per vertex to the skin point's own rest depth to its anchor, l_j = |p_j0 - a_j0|,
            so kappa_j = S / l_j^2: no tuned stiffness.
  solve     (eq. 27) per frame a quasi-static minimum by local-global iterations: local = best rotation per skin vertex
            (ARAP), global = one sparse linear solve with a matrix factored once.  Warm start from the previous frame.
Writes skin_attach.json (per-frame skin stretch and flips, same measures as the audit) and skin_attach.npz (skin every 15th
frame, and the nearest-vertex binding on the same clip for comparison)."""
import sys, os, re, json, time, numpy as np, scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
FD, MD, TRF, OUT = sys.argv[1:5]; NIT = int(sys.argv[5]) if len(sys.argv) > 5 else 10
os.makedirs(OUT, exist_ok=True)
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
Wt = csr_matrix((R['W_data'], R['W_indices'], R['W_indptr'])); wb = [int(b) for b in R['bones']]; muscles = set(int(m) for m in R['muscles'])
S_ = '.l'; hipb, fem, tib, fib, pat = (nid[n + S_] for n in ('Hip bone', 'Femur', 'Tibia', 'Fibula', 'Patella'))
foot = [i for i, n in enumerate(names) if n.endswith(S_) and re.search(r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform|phalan', n)]
GRP = {hipb: 0, fem: 1, tib: 2, fib: 2, pat: 2, **{b: 2 for b in foot}}
# --- skin: the leg's own piece of the model surface
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']
ISL = np.load(os.path.join(MD, 'flesh', 'islands.npz'))['island']
near = ISL[cKDTree(MV).query(V[OB == fem])[1]]; leg = int(np.bincount(near[near >= 0]).argmax())
sk = np.flatnonzero(ISL == leg); SF = MF[(ISL[MF] == leg).all(1)]; loc = np.full(len(MV), -1); loc[sk] = np.arange(len(sk)); F = loc[SF]; P0 = MV[sk]; n = len(P0)
# --- deeper surface: this leg's bones, and muscle vertices whose largest bone share is on one of them
bone_v = np.flatnonzero(np.isin(OB, list(GRP)))
mv = np.flatnonzero(np.isin(OB, list(muscles)))
top = np.array([wb[k] for k in np.asarray(Wt[mv].argmax(1)).ravel()]); mv = mv[np.isin(top, list(GRP))]
deep = np.r_[bone_v, mv]; kd = cKDTree(V[deep]); dist, nn = kd.query(P0); anc = deep[nn]                      # material anchors, fixed
is_bone = np.isin(anc, bone_v)
print('SA skin %d vertices, %d faces; deep points %d (bone %d, muscle %d); anchors on bone %.0f%%, depth median %.1f mm' % (n, len(F), len(deep), len(bone_v), len(mv), 100 * is_bone.mean(), 1000 * np.median(dist)), flush=True)
# group shares per anchor: bones rigid to their group; muscle points by their bone shares (other bones count as the fixed pelvis)
col_grp = np.array([GRP.get(b, 0) for b in wb])
Wa = np.zeros((n, 3))
Wa[is_bone, [GRP[b] for b in OB[anc[is_bone]]]] = 1.0
Wm = Wt[anc[~is_bone]].tocoo(); np.add.at(Wa, (np.flatnonzero(~is_bone)[Wm.row], col_grp[Wm.col]), Wm.data); Wa /= Wa.sum(1, keepdims=True)
# --- membrane (ARAP, clamped cotangent weights) and areas
def cot(a, b): return (a * b).sum(1) / np.maximum(np.linalg.norm(np.cross(a, b), axis=1), 1e-12)
I, J, Wc = [], [], []
for k in range(3):
    i, j, o = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]
    w = 0.5 * np.maximum(cot(P0[i] - P0[o], P0[j] - P0[o]), 0); I += [i, j]; J += [j, i]; Wc += [w, w]
Wcot = sp.coo_matrix((np.concatenate(Wc), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr(); Wcot.sum_duplicates()
Lm = sp.diags(np.asarray(Wcot.sum(1)).ravel()) - Wcot
area = np.linalg.norm(np.cross(P0[F[:, 1]] - P0[F[:, 0]], P0[F[:, 2]] - P0[F[:, 0]]), axis=1) / 2; A = np.zeros(n); np.add.at(A, F.ravel(), np.repeat(area, 3) / 3)
ell = np.maximum(dist, np.median(np.linalg.norm(P0[F[:, 0]] - P0[F[:, 1]], axis=1)))        # depth, at least one skin edge
kap = A / ell ** 2                                                                         # S = 1: only the ratio matters
lone = np.asarray(Wcot.sum(1)).ravel() <= 0; kap[lone | (kap <= 0)] = np.median(kap[kap > 0])   # vertices in no usable triangle: held by their tie alone
print('SA vertices in no usable triangle: %d' % lone.sum(), flush=True)
Mtx = (Lm + sp.diags(kap)).tocsc(); lu = splu(Mtx)
Wcoo = Wcot.tocoo(); ei, ej, ew = Wcoo.row, Wcoo.col, Wcoo.data; E0 = P0[ei] - P0[ej]
d0 = P0 - V[anc]
def dq_blend(Pa, W, Ts):
    """dual-quaternion blend (Kavan et al. 2008) of rigid transforms Ts[g] = (R, t) with weights W (n x G); returns points and rotations"""
    q = np.array([Rot.from_matrix(Rm).as_quat() for Rm, _ in Ts]); t = np.array([tt for _, tt in Ts])
    q[q @ q[0] < 0] *= -1
    qd = 0.5 * np.c_[t[:, 0] * q[:, 3] + t[:, 1] * q[:, 2] - t[:, 2] * q[:, 1], -t[:, 0] * q[:, 2] + t[:, 1] * q[:, 3] + t[:, 2] * q[:, 0], t[:, 0] * q[:, 1] - t[:, 1] * q[:, 0] + t[:, 2] * q[:, 3], -(t * q[:, :3]).sum(1)]
    br, bd = W @ q, W @ qd; nn_ = np.linalg.norm(br, axis=1, keepdims=True); br, bd = br / nn_, bd / nn_
    Rm = Rot.from_quat(br).as_matrix(); w, x, y, z = br[:, 3], br[:, 0], br[:, 1], br[:, 2]; dw, dx, dy, dz = bd[:, 3], bd[:, 0], bd[:, 1], bd[:, 2]
    tt = 2 * np.c_[-dw * x + dx * w - dy * z + dz * y, -dw * y + dx * z + dy * w - dz * x, -dw * z - dx * y + dy * x + dz * w]
    return np.einsum('nij,nj->ni', Rm, Pa) + tt, Rm
TR = np.load(TRF); T = len(TR['thigh_R'])
def transforms(f): return [(np.eye(3), np.zeros(3)), (TR['thigh_R'][f], TR['thigh_t'][f]), (TR['shank_R'][f], TR['shank_t'][f])]
def arap_rhs(P):
    Ecur = P[ei] - P[ej]; Cov = np.zeros((n, 3, 3)); np.add.at(Cov, ei, ew[:, None, None] * E0[:, :, None] * Ecur[:, None, :])
    U, _, Vt = np.linalg.svd(Cov); Rv = np.einsum('nji,nkj->nik', Vt, U)
    bad = np.linalg.det(Rv) < 0; U[bad, :, -1] *= -1; Rv[bad] = np.einsum('nji,nkj->nik', Vt[bad], U[bad])
    b = np.zeros((n, 3)); np.add.at(b, ei, 0.5 * ew[:, None] * np.einsum('nij,nj->ni', Rv[ei] + Rv[ej], E0)); return b
Ed = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; e0 = np.linalg.norm(P0[Ed[:, 0]] - P0[Ed[:, 1]], axis=1); ok = e0 > 0
n0 = np.cross(P0[F[:, 1]] - P0[F[:, 0]], P0[F[:, 2]] - P0[F[:, 0]])
def measure(X):
    st = np.linalg.norm(X[Ed[ok, 0]] - X[Ed[ok, 1]], axis=1) / e0[ok]; fl = (n0 * np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])).sum(1) < 0
    return dict(p1=float(np.percentile(st, 1)), median=float(np.median(st)), p99=float(np.percentile(st, 99)), flipped=float(fl.mean()))
P = P0.copy(); rows = []; keep, keep_nn = [], []; t0 = time.time()
for f in range(T):
    Aa, Qa = dq_blend(V[anc], Wa, transforms(f)); tgt = Aa + np.einsum('nij,nj->ni', Qa, d0)          # where each tie wants its skin point
    X_nn = tgt                                                                                   # the nearest-vertex binding = infinitely stiff ties
    for it in range(NIT): P = lu.solve(arap_rhs(P) + kap[:, None] * tgt)
    m, mn = measure(P), measure(X_nn); gap = np.linalg.norm(P - tgt, axis=1)
    rows.append(dict(frame=f, attached=m, nearest_vertex=mn, gap_mm=dict(median=1000 * float(np.median(gap)), p99=1000 * float(np.percentile(gap, 99)))))
    if f % 15 == 0: keep.append(P.astype(np.float32)); keep_nn.append(X_nn.astype(np.float32))
    if f % 30 == 0: print('SA frame %3d: attached skin p99 x%.2f flipped %.2f%% | nearest-vertex p99 x%.2f flipped %.2f%% | gap to ties median %.1f mm  (%.0f s)' % (f, m['p99'], 100 * m['flipped'], mn['p99'], 100 * mn['flipped'], rows[-1]['gap_mm']['median'], time.time() - t0), flush=True)
summ = dict(skin_vertices=n, anchors_on_bone=float(is_bone.mean()), depth_median_mm=1000 * float(np.median(dist)), iterations=NIT, seconds=round(time.time() - t0),
            attached=dict(p99_max=max(r['attached']['p99'] for r in rows), flipped_max=max(r['attached']['flipped'] for r in rows)),
            nearest_vertex=dict(p99_max=max(r['nearest_vertex']['p99'] for r in rows), flipped_max=max(r['nearest_vertex']['flipped'] for r in rows)))
json.dump(dict(summary=summ, frames=rows), open(os.path.join(OUT, 'skin_attach.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(OUT, 'skin_attach.npz'), skin=np.array(keep), skin_nearest=np.array(keep_nn), faces=F, rest=P0.astype(np.float32))
print('SA SUMMARY ' + json.dumps(summ))
