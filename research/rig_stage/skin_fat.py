"""Skin on a fat layer that fills the room between it and the muscle and bone beneath, on the elf's left leg; after the user's
report "Mathematical Modeling of Skin Attachments" (19 Nov 2026), used as a provisional formulation: skin and dermis on
subcutaneous tissue on deep fascia / muscle / bone, deeper motion prescribed (report eq. 3), quasi-static (eq. 27).
  E:\\splat\\.venv\\Scripts\\python.exe skin_fat.py <fits dir> <model dir> <leg_transforms.npz> <out dir> [frame step] [iterations]

  room      each skin vertex j has a rest depth h_j to the nearest point of the deeper surface (this leg's bones and the muscle
            points they carry).  The fat layer is the shell between the skin and the skin moved inward along its normal by
            h_j: every skin triangle and its inner copy form a prism, split into 3 tetrahedra.  The fat fills exactly the
            measured room; no tet mesher is needed.
  inner     each inner-shell vertex is a material point of the deep tissue (report eq. 4): tied once at rest to its nearest
            deep point and moved with it: rigidly with a bone (eq. 3), or with a muscle point's bone shares blended as dual
            quaternions (the existing kinematic muscle rig).  Inner vertices are prescribed, skin vertices are free.
  fat       projective dynamics (Bouaziz et al. 2014): per tet an as-rigid-as-possible strain term (weight 2 mu V) and a
            volume term projecting the deformation onto det = 1 (weight K V); K / mu = 2(1 + nu) / (3(1 - 2 nu)) with
            nu = 0.45, the soft-tissue value flesh/newton_flesh.py already uses.  mu = 1: only ratios matter quasi-statically.
  skin      an as-rigid-as-possible membrane on the skin triangles (Sorkine & Alexa 2007).  Its stiffness S per area follows
            from the report's transfer length: a fat layer of shear modulus mu and depth h ties the skin with kappa ~ mu / h,
            and taking l_s = h gives S = mu h, so S_j = h_j (mu = 1).  No tuned stiffness.
Writes skin_fat.json (per frame: skin stretch, flipped skin triangles, inverted fat tets, fat volume) and skin_fat.npz."""
import sys, os, re, json, time, numpy as np, scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
FD, MD, TRF, OUT = sys.argv[1:5]; STEP = int(sys.argv[5]) if len(sys.argv) > 5 else 3; NIT = int(sys.argv[6]) if len(sys.argv) > 6 else 8
os.makedirs(OUT, exist_ok=True)
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
Wt = csr_matrix((R['W_data'], R['W_indices'], R['W_indptr'])); wb = [int(b) for b in R['bones']]; muscles = set(int(m) for m in R['muscles'])
S_ = '.l'; hipb, fem, tib, fib, pat = (nid[n + S_] for n in ('Hip bone', 'Femur', 'Tibia', 'Fibula', 'Patella'))
foot = [i for i, n in enumerate(names) if n.endswith(S_) and re.search(r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform|phalan', n)]
GRP = {hipb: 0, fem: 1, tib: 2, fib: 2, pat: 2, **{b: 2 for b in foot}}
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']
ISL = np.load(os.path.join(MD, 'flesh', 'islands.npz'))['island']
near = ISL[cKDTree(MV).query(V[OB == fem])[1]]; leg = int(np.bincount(near[near >= 0]).argmax())
sk = np.flatnonzero(ISL == leg); SF = MF[(ISL[MF] == leg).all(1)]; loc = np.full(len(MV), -1); loc[sk] = np.arange(len(sk)); F = loc[SF]; P0 = MV[sk]
used = np.unique(F); keep_v = np.zeros(len(P0), bool); keep_v[used] = True             # drop vertices in no triangle
re_ = np.full(len(P0), -1); re_[used] = np.arange(len(used)); F = re_[F]; P0 = P0[used]; n = len(P0)
P0, wi = np.unique(P0, axis=0, return_inverse=True); F = wi.ravel()[F]; n = len(P0)               # weld the seams: the model surface repeats vertices along them
F = F[(F[:, 0] != F[:, 1]) & (F[:, 1] != F[:, 2]) & (F[:, 0] != F[:, 2])]
bone_v = np.flatnonzero(np.isin(OB, list(GRP))); mv = np.flatnonzero(np.isin(OB, list(muscles)))
top = np.array([wb[k] for k in np.asarray(Wt[mv].argmax(1)).ravel()]); mv = mv[np.isin(top, list(GRP))]
deep = np.r_[bone_v, mv]; kd = cKDTree(V[deep])
# --- skin normals (area weighted), oriented outward from the deep tissue
fn = np.cross(P0[F[:, 1]] - P0[F[:, 0]], P0[F[:, 2]] - P0[F[:, 0]]); vn = np.zeros((n, 3)); np.add.at(vn, F.ravel(), np.repeat(fn, 3, 0))
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-12
h, nn0 = kd.query(P0); out_dir = P0 - V[deep[nn0]]
if np.mean(np.sum(vn * out_dir, 1) > 0) < 0.5: vn = -vn; F = F[:, [0, 2, 1]]               # make normals point away from the deep tissue
Q0 = P0 - vn * h[:, None]                                                                     # inner shell: the skin moved inward by its depth
dq, nq = kd.query(Q0); anc = deep[nq]; is_bone = np.isin(anc, bone_v)
print('SF skin %d vertices, %d faces; fat depth median %.1f mm (p5 %.1f, p95 %.1f); inner shell to deep surface median %.1f mm; inner on bone %.0f%%' % (
    n, len(F), 1000 * np.median(h), 1000 * np.percentile(h, 5), 1000 * np.percentile(h, 95), 1000 * np.median(dq), 100 * is_bone.mean()), flush=True)
col_grp = np.array([GRP.get(b, 0) for b in wb]); Wa = np.zeros((n, 3))
Wa[is_bone, [GRP[b] for b in OB[anc[is_bone]]]] = 1.0
Wm = Wt[anc[~is_bone]].tocoo(); np.add.at(Wa, (np.flatnonzero(~is_bone)[Wm.row], col_grp[Wm.col]), Wm.data); Wa /= Wa.sum(1, keepdims=True)
d_in = Q0 - V[anc]                                                                            # inner point relative to its deep material point
if os.environ.get('SHARES') == 'harmonic':
    # smooth motion shares for the fat floor: points on bone keep their bone, the open top loop (the hip cut) keeps the pelvis,
    # every other point takes the harmonic blend over the skin surface (uniform graph Laplacian)
    from scipy.sparse.csgraph import connected_components
    Es = np.sort(np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], 1); ue_, ce_ = np.unique(Es, axis=0, return_counts=True); be_ = ue_[ce_ == 1]
    Gb = sp.coo_matrix((np.ones(len(be_)), (be_[:, 0], be_[:, 1])), shape=(n, n)); _, lb = connected_components(Gb + Gb.T, directed=False)
    bv_ = np.unique(be_.ravel()); Lb = lb[bv_]; ids_ = np.unique(Lb); cut = bv_[Lb == ids_[np.argmax([P0[bv_[Lb == i], 2].mean() for i in ids_])]]
    Au = sp.coo_matrix((np.ones(len(ue_)), (ue_[:, 0], ue_[:, 1])), shape=(n, n)); Au = (Au + Au.T).tocsr(); Lu = sp.diags(np.asarray(Au.sum(1)).ravel()) - Au
    fixed = is_bone.copy(); fixed[cut] = True; Wb = Wa.copy(); Wb[cut] = [1, 0, 0]
    fr_, fx_ = np.flatnonzero(~fixed), np.flatnonzero(fixed)
    Wb[fr_] = np.clip(splu(Lu[fr_][:, fr_].tocsc()).solve(-(Lu[fr_][:, fx_] @ Wb[fx_])), 0, None); Wa = Wb / Wb.sum(1, keepdims=True)
    print('SF harmonic floor shares: hip cut %d vertices, on bone %d' % (len(cut), is_bone.sum()), flush=True)
# --- fat tets: prism (skin a,b,c ; inner a',b',c') -> 3 tets; nodes 0..n-1 skin (free), n..2n-1 inner (prescribed)
a, b, c = F[:, 0], F[:, 1], F[:, 2]; A_, B_, C_ = a + n, b + n, c + n
TT = np.concatenate([np.c_[a, b, c, A_], np.c_[b, c, A_, B_], np.c_[c, A_, B_, C_]])
X0 = np.r_[P0, Q0]
Dm = np.stack([X0[TT[:, 1]] - X0[TT[:, 0]], X0[TT[:, 2]] - X0[TT[:, 0]], X0[TT[:, 3]] - X0[TT[:, 0]]], 2); vol = np.linalg.det(Dm)
flip_t = vol < 0; TT[flip_t] = TT[flip_t][:, [0, 2, 1, 3]]; Dm[flip_t] = Dm[flip_t][:, :, [1, 0, 2]]; vol = np.abs(vol) / 6
good = vol > np.percentile(vol, 0.1) * 1e-3; TT, Dm, vol = TT[good], Dm[good], vol[good]; Dinv = np.linalg.inv(Dm)
print('SF fat tets %d (dropped %d degenerate)' % (len(TT), (~good).sum()), flush=True)
n_in = np.cross(Q0[F[:, 1]] - Q0[F[:, 0]], Q0[F[:, 2]] - Q0[F[:, 0]]); n_out = np.cross(P0[F[:, 1]] - P0[F[:, 0]], P0[F[:, 2]] - P0[F[:, 0]])
folded = (n_in * n_out).sum(1) <= 0
print('SF CHECK rest: inner-shell triangles folded against their skin triangle %.2f%% (%d of %d)' % (100 * folded.mean(), folded.sum(), len(F)), flush=True)
nu = 0.45; Kmu = 2 * (1 + nu) / (3 * (1 - 2 * nu)); w_str = 2.0 * vol; w_vol = Kmu * vol
# gradient operator per tet: F = X_e G with G (4x3) from Dinv
G = np.concatenate([-Dinv.sum(1, keepdims=True), Dinv], 1)                                     # 4 x 3 per tet
# --- skin membrane (ARAP on edges, clamped cotangent), weight S_j = h_j per area
def cot(p, q): return (p * q).sum(1) / np.maximum(np.linalg.norm(np.cross(p, q), axis=1), 1e-12)
I, J, Wc = [], [], []
for k in range(3):
    i, j, o = F[:, k], F[:, (k + 1) % 3], F[:, (k + 2) % 3]; w = 0.5 * np.maximum(cot(P0[i] - P0[o], P0[j] - P0[o]), 0) * 0.5 * (h[i] + h[j])
    I += [i, j]; J += [j, i]; Wc += [w, w]
Wcot = sp.coo_matrix((np.concatenate(Wc), (np.concatenate(I), np.concatenate(J))), shape=(n, n)).tocsr(); Wcot.sum_duplicates()
Lm = sp.diags(np.asarray(Wcot.sum(1)).ravel()) - Wcot; Wco = Wcot.tocoo(); ei, ej, ew = Wco.row, Wco.col, Wco.data; E0 = P0[ei] - P0[ej]
# --- global matrix over all 2n nodes: sum_t (w_str + w_vol) G G^T assembled, plus the membrane on the skin block
wt = w_str + w_vol; rows = np.repeat(TT, 4, 1).ravel(); cols = np.tile(TT, (1, 4)).ravel()
GG = np.einsum('tik,tjk->tij', G, G) * wt[:, None, None]
Kfull = sp.coo_matrix((GG.ravel(), (rows, cols)), shape=(2 * n, 2 * n)).tocsr()
Kfull = Kfull + sp.block_diag([Lm, sp.csr_matrix((n, n))]).tocsr()
Kff = Kfull[:n][:, :n].tocsc(); Kfp = Kfull[:n][:, n:].tocsr(); lu = splu(Kff)
def dq_blend(Pa, W, Ts):
    q = np.array([Rot.from_matrix(Rm).as_quat() for Rm, _ in Ts]); t = np.array([tt for _, tt in Ts]); q[q @ q[0] < 0] *= -1
    qd = 0.5 * np.c_[t[:, 0] * q[:, 3] + t[:, 1] * q[:, 2] - t[:, 2] * q[:, 1], -t[:, 0] * q[:, 2] + t[:, 1] * q[:, 3] + t[:, 2] * q[:, 0], t[:, 0] * q[:, 1] - t[:, 1] * q[:, 0] + t[:, 2] * q[:, 3], -(t * q[:, :3]).sum(1)]
    br, bd = W @ q, W @ qd; nn_ = np.linalg.norm(br, axis=1, keepdims=True); br, bd = br / nn_, bd / nn_
    Rm = Rot.from_quat(br).as_matrix(); w, x, y, z = br[:, 3], br[:, 0], br[:, 1], br[:, 2]; dw, dx, dy, dz = bd[:, 3], bd[:, 0], bd[:, 1], bd[:, 2]
    tt = 2 * np.c_[-dw * x + dx * w - dy * z + dz * y, -dw * y + dx * z + dy * w - dz * x, -dw * z - dx * y + dy * x + dz * w]
    return np.einsum('nij,nj->ni', Rm, Pa) + tt, Rm
TR = np.load(TRF); T = len(TR['thigh_R'])
REST_ONLY = os.environ.get('REST_ONLY') == '1'; EASE = int(os.environ.get('EASE', '1'))
def transforms(f):
    if REST_ONLY: return [(np.eye(3), np.zeros(3))] * 3
    return [(np.eye(3), np.zeros(3)), (TR['thigh_R'][f], TR['thigh_t'][f]), (TR['shank_R'][f], TR['shank_t'][f])]
def between(T0, T1, u):    # rigid transforms part way: rotation by slerp, translation linear
    from scipy.spatial.transform import Slerp
    return [(Slerp([0, 1], Rot.from_matrix(np.stack([a_[0], b_[0]])))([u]).as_matrix()[0], (1 - u) * a_[1] + u * b_[1]) for a_, b_ in zip(T0, T1)]
def polar(Fm):
    U, s, Vt = np.linalg.svd(Fm); d = np.linalg.det(U @ Vt) < 0; U[d, :, -1] *= -1; s[d, -1] *= -1; return U, s, Vt
def rhs(X):
    Fm = np.einsum('tki,tkj->tij', X[TT], G)                                                 # deformation gradient per tet
    U, s, Vt = polar(Fm); Rr = U @ Vt
    sv = s.copy(); sv = sv / np.cbrt(np.abs(np.prod(sv, 1)))[:, None]                          # nearest det = 1 (uniform scaling of singular values)
    Pv = np.einsum('tij,tj,tjk->tik', U, sv, Vt)
    Tgt = (w_str[:, None, None] * Rr + w_vol[:, None, None] * Pv)                              # projected targets, weighted
    f = np.einsum('tkj,tij->tki', G, Tgt)                                                      # 4 x 3 nodal contributions
    r = np.zeros((2 * n, 3)); np.add.at(r, TT.ravel(), f.reshape(-1, 3))
    Ec = X[ei] - X[ej]; Cov = np.zeros((n, 3, 3)); np.add.at(Cov, ei, ew[:, None, None] * E0[:, :, None] * Ec[:, None, :])
    Uu, _, Vv = np.linalg.svd(Cov); Rv = np.einsum('nji,nkj->nik', Vv, Uu); bad = np.linalg.det(Rv) < 0; Uu[bad, :, -1] *= -1; Rv[bad] = np.einsum('nji,nkj->nik', Vv[bad], Uu[bad])
    np.add.at(r, ei, 0.5 * ew[:, None] * np.einsum('nij,nj->ni', Rv[ei] + Rv[ej], E0)); return r, Fm
Ed = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; e0 = np.linalg.norm(P0[Ed[:, 0]] - P0[Ed[:, 1]], axis=1); ok = e0 > 0
n0 = np.cross(P0[F[:, 1]] - P0[F[:, 0]], P0[F[:, 2]] - P0[F[:, 0]])
def measure(Xs):
    st = np.linalg.norm(Xs[Ed[ok, 0]] - Xs[Ed[ok, 1]], axis=1) / e0[ok]; fl = (n0 * np.cross(Xs[F[:, 1]] - Xs[F[:, 0]], Xs[F[:, 2]] - Xs[F[:, 0]])).sum(1) < 0
    return dict(p1=float(np.percentile(st, 1)), median=float(np.median(st)), p99=float(np.percentile(st, 99)), flipped=float(fl.mean()))
X = X0.copy(); out_rows = []; keep = []; t0 = time.time(); frames = list(range(0, T, STEP))
prevT = [(np.eye(3), np.zeros(3))] * 3
if REST_ONLY: frames = frames[:3]
for f in frames:
    Tf = transforms(f)
    for e in range(1, EASE + 1):                                                                # ease from the last solved pose in EASE steps
        Qa, Ra = dq_blend(V[anc], Wa, between(prevT, Tf, e / EASE)); X[n:] = Qa + np.einsum('nij,nj->ni', Ra, d_in)   # inner shell rides the deep tissue
        for it in range(NIT):
            r, Fm = rhs(X); X[:n] = lu.solve(r[:n] - Kfp @ X[n:])
    prevT = Tf
    Fm = np.einsum('tki,tkj->tij', X[TT], G); J_ = np.linalg.det(Fm)
    m = measure(X[:n]); out_rows.append(dict(frame=f, skin=m, fat_inverted=float((J_ <= 0).mean()), fat_volume=float((J_ * vol).sum() / vol.sum()), fat_J_p1_p99=[float(np.percentile(J_, 1)), float(np.percentile(J_, 99))]))
    if f % 15 == 0: keep.append(X[:n].astype(np.float32))
    if f % 30 == 0 or REST_ONLY: print('SF frame %3d: skin p99 x%.2f, flipped %.2f%% | fat inverted %.2f%%, volume %.3f, J p1..p99 %.2f..%.2f  (%.0f s)' % (
        f, m['p99'], 100 * m['flipped'], 100 * out_rows[-1]['fat_inverted'], out_rows[-1]['fat_volume'], *out_rows[-1]['fat_J_p1_p99'], time.time() - t0), flush=True)
summ = dict(skin_vertices=n, fat_tets=int(len(TT)), fat_depth_median_mm=1000 * float(np.median(h)), frame_step=STEP, iterations=NIT, seconds=round(time.time() - t0),
            skin_p99_max=max(r['skin']['p99'] for r in out_rows), skin_flipped_max=max(r['skin']['flipped'] for r in out_rows),
            fat_inverted_max=max(r['fat_inverted'] for r in out_rows), fat_volume_range=[min(r['fat_volume'] for r in out_rows), max(r['fat_volume'] for r in out_rows)])
json.dump(dict(summary=summ, frames=out_rows), open(os.path.join(OUT, 'skin_fat.json'), 'w'), indent=1)
np.savez_compressed(os.path.join(OUT, 'skin_fat.npz'), skin=np.array(keep), faces=F, rest=P0.astype(np.float32), inner=Q0.astype(np.float32), depth=h.astype(np.float32))
print('SF SUMMARY ' + json.dumps(summ))
