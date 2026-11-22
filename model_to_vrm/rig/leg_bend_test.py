"""One-leg rig audit (handout experiment T-rig): drive a slow hip and knee bend through the fitted anatomy and report what
breaks, before any tissue solver exists.
  E:\\splat\\.venv\\Scripts\\python.exe leg_bend_test.py <fits dir> <model dir> [l|r]
  bones     pelvis fixed; thigh bone turns about the hip; shin and calf bone, kneecap and foot turn with it and about the knee
  joints    hip centre = centre of the sphere fitted to the thigh bone's head (its vertices facing the hip bone, nearer
            than the Otsu split of the log distances); knee axis = the side-to-side direction of the thigh bone's lower
            end (its vertices facing the shin bone): the direction of largest extent across the bone's long axis, through
            their middle; the hip bends about a parallel axis through its centre
  motion    a prescribed test clip, not a fitted quantity: hip flexion and knee flexion rise together to the angles
            given by HIP_DEG and KNEE_DEG (defaults 60 and 90), in FRAMES steps
  muscles   every muscle vertex moves by the bone shares muscle_carry.py gave it, blended as dual quaternions (the
            existing kinematic rig)
  skin      the leg's own skin (its flesh piece) bound once: each skin vertex takes the shares of the nearest muscle or
            bone vertex at rest
Checks per frame: attachment drift (muscle vertices that lie on a bone, against that bone), muscle points inside bone,
closed-muscle volume against rest, skin edge stretch and flipped skin triangles. Also the audit at rest: which muscles
cross the hip or knee, the bones each lies on, and whether the atlas has its origin and end areas.
Writes <fits dir>/leg_bend_<side>.json, leg_bend_<side>.npz (frames) and leg_bend_<side>.jpg."""
import sys, os, re, json, numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'anatomy'))
from solids import bone_solid, sample, closed_volumes
FD, MD = sys.argv[1:3]; sd = sys.argv[3] if len(sys.argv) > 3 else 'l'; S = '.' + sd
HIP_DEG = float(os.environ.get('HIP_DEG', 60)); KNEE_DEG = float(os.environ.get('KNEE_DEG', 90)); FRAMES = int(os.environ.get('FRAMES', 7))
def log(*a): print('LB', *a, flush=True)
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']; FM, FB = R['F_muscle'], R['F_bone']; contact = R['contact'].astype(bool)
W = np.load(os.path.join(FD, 'atlas_warp.npz')); names = [str(x) for x in W['names']]; V0 = W['V0'].astype(np.float64); nid = {n: i for i, n in enumerate(names)}
Wt = csr_matrix((R['W_data'], R['W_indices'], R['W_indptr'])); wb = [int(b) for b in R['bones']]; col = {b: k for k, b in enumerate(wb)}; muscles = [int(m) for m in R['muscles']]
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
FOOT = r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform'
hipb, fem, tib, fib, pat = (nid['Hip bone' + S], nid['Femur' + S], nid['Tibia' + S], nid['Fibula' + S], nid['Patella' + S])
foot = [i for i, n in enumerate(names) if n.endswith(S) and re.search(FOOT, n) and i in col]
GROUP = {fem: 'thigh', tib: 'shank', fib: 'shank', pat: 'shank', **{b: 'shank' for b in foot}}        # everything else: fixed (pelvis and the rest of the body)
# --- joints from the fitted bones
Pf, Ph, Ptb = V[OB == fem], V[OB == hipb], V[OB == tib]
dh = cKDTree(Ph).query(Pf)[0]; head = Pf[np.log(dh + 1e-9) < otsu(np.log(dh + 1e-9))]
A = np.c_[2 * head, np.ones(len(head))]; sol = np.linalg.lstsq(A, (head ** 2).sum(1), rcond=None)[0]; hip_c = sol[:3]; hip_r = float(np.sqrt(sol[3] + hip_c @ hip_c))
dk = cKDTree(Ptb).query(Pf)[0]; cond = Pf[np.log(dk + 1e-9) < otsu(np.log(dk + 1e-9))]; long_ = np.linalg.svd(Pf - Pf.mean(0), full_matrices=False)[2][0]
Q = (cond - cond.mean(0)); Q = Q - np.outer(Q @ long_, long_); ax = np.linalg.svd(Q, full_matrices=False)[2][0]; ax = ax if ax[0] > 0 else -ax; knee_c = cond.mean(0)
log('hip centre %s (head sphere radius %.1f mm from %d vertices); knee axis %s through %s (from %d condyle vertices)' % (np.round(hip_c, 3).tolist(), 1000 * hip_r, len(head), np.round(ax, 3).tolist(), np.round(knee_c, 3).tolist(), len(cond)))
def about(c, axis, ang):                                            # rigid: x -> Rm (x - c) + c
    Rm = Rot.from_rotvec(axis * ang).as_matrix(); return Rm, c - Rm @ c
def compose(a, b): return a[0] @ b[0], a[0] @ b[1] + a[1]          # a after b
def transforms(u):
    """bone -> (R, t) at clip position u in [0, 1]. Model axes: x to her left, y to her back, z up; flexing the hip
    brings the thigh forward (a negative turn about the left-pointing axis), flexing the knee takes the shin back."""
    Th = about(hip_c, ax, -np.radians(HIP_DEG) * u); Tk = compose(Th, about(knee_c, ax, np.radians(KNEE_DEG) * u)); I = (np.eye(3), np.zeros(3))
    return {b: (Th if GROUP.get(b) == 'thigh' else Tk if GROUP.get(b) == 'shank' else I) for b in wb}
def dq_blend(P, Wrow, T):
    """dual-quaternion blend (Kavan et al. 2008) of rigid transforms T (per column of Wrow) at points P."""
    q = np.array([Rot.from_matrix(T[b][0]).as_quat() for b in wb]); t = np.array([T[b][1] for b in wb])              # xyzw
    q[q @ q[0] < 0] *= -1; qd = 0.5 * np.c_[t[:, 0] * q[:, 3] + t[:, 1] * q[:, 2] - t[:, 2] * q[:, 1], -t[:, 0] * q[:, 2] + t[:, 1] * q[:, 3] + t[:, 2] * q[:, 0], t[:, 0] * q[:, 1] - t[:, 1] * q[:, 0] + t[:, 2] * q[:, 3], -(t * q[:, :3]).sum(1)]
    br, bd = Wrow @ q, Wrow @ qd; n = np.linalg.norm(br, axis=1, keepdims=True); br, bd = br / n, bd / n
    Rm = Rot.from_quat(br).as_matrix(); w, x, y, z = br[:, 3], br[:, 0], br[:, 1], br[:, 2]; dw, dx, dy, dz = bd[:, 3], bd[:, 0], bd[:, 1], bd[:, 2]
    tt = 2 * np.c_[-dw * x + dx * w - dy * z + dz * y, -dw * y + dx * z + dy * w - dz * x, -dw * z - dx * y + dy * x + dz * w]
    return np.einsum('nij,nj->ni', Rm, P) + tt
# --- what moves: muscle vertices with a share on a moving bone, the moving bones themselves, the leg's skin
mov_cols = np.array([col[b] for b in GROUP if b in col]); mi = R['mverts']; has = np.asarray(Wt[mi][:, mov_cols].sum(1)).ravel() > 0; mv = mi[has]; Wm = Wt[mv]
bone_v = {b: np.flatnonzero(OB == b) for b in [hipb] + list(GROUP)}
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']; ISL = np.load(os.path.join(MD, 'flesh', 'islands.npz'))['island']
leg_isl = int(np.bincount(ISL[cKDTree(MV).query(V[OB == fem])[1]][ISL[cKDTree(MV).query(V[OB == fem])[1]] >= 0]).argmax()); sk = np.flatnonzero(ISL == leg_isl); SF = MF[(ISL[MF] == leg_isl).all(1)]
carried = np.r_[mi, np.concatenate(list(bone_v.values()))]; rows = []
for b, ii in bone_v.items():                                        # a bone vertex belongs wholly to its own bone
    pass
Wb = csr_matrix((np.ones(sum(len(v) for v in bone_v.values())), (np.arange(sum(len(v) for v in bone_v.values())), np.concatenate([np.full(len(v), col[b]) for b, v in bone_v.items()]))), shape=(sum(len(v) for v in bone_v.values()), len(wb)))
from scipy.sparse import vstack
side_cols = np.array([col[b] for b in [hipb] + list(GROUP) if b in col]); own = np.asarray(Wt[mi][:, side_cols].sum(1)).ravel() > 0.5       # tissue of this leg: most of its bone share on this side's bones (the other leg, pressed against it, must not hold this leg's skin)
own_all = np.r_[own, np.ones(Wb.shape[0], bool)]; Wc = vstack([Wt[mi], Wb]).tocsr()[own_all]; carried = carried[own_all]; nn = cKDTree(V[carried]).query(MV[sk])[1]; Ws = Wc[nn]; bind_d = np.linalg.norm(V[carried][nn] - MV[sk], axis=1)
log('moving: %d muscle vertices, %d bones; skin patch (flesh piece %d): %d vertices bound to the nearest tissue vertex (median %.1f mm away, max %.1f)' % (len(mv), len(GROUP), leg_isl, len(sk), 1000 * np.median(bind_d), 1000 * bind_d.max()))
# --- audit at rest: the muscles that cross the hip or the knee
seg_of = lambda b: 'pelvis' if b == hipb else GROUP.get(b)
ZA = np.load(os.environ.get('ATLAS', r'E:\figures\models\zanatomy\export\atlas.npz'), allow_pickle=True); zn = {str(n) for n in ZA['names']}
cross = []; mset = set(muscles)
for m in muscles:
    if not names[m].endswith(S): continue
    vi = np.flatnonzero((OB == m) & contact)
    if not len(vi): continue
    top = np.asarray(Wt[vi].argmax(1)).ravel(); segs = {seg_of(wb[c]) for c in np.unique(top)} - {None}
    if len(segs) >= 2:
        base = names[m][:-2]; cross.append(dict(name=names[m], segments=sorted(segs), on_bones=sorted({names[wb[c]] for c in np.unique(top)}), origin_area=(base + '.o' + sd) in zn, end_area=(base + '.e' + sd) in zn, vertices=int((OB == m).sum()), on_bone_vertices=int(len(vi))))
log('muscle items crossing the hip or the knee on this side: %d; with both attachment areas in the atlas: %d; with neither: %d' % (len(cross), sum(c['origin_area'] and c['end_area'] for c in cross), sum(not c['origin_area'] and not c['end_area'] for c in cross)))
cm = [nid[c['name']] for c in cross]; cmv = np.flatnonzero(np.isin(OB, cm)); Wcm = Wt[cmv]
# rest solids of the bones, queried in each bone's own rest frame
hstep = 2 * float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1))); SOL = {}
for b in (hipb, fem, tib, fib, pat):
    Fb = FB[OB[FB[:, 0]] == b]; sdb, lob, _ = bone_solid(V, Fb, hstep); SOL[b] = (sdb, lob)
def in_bone(X, T):
    worst = np.full(len(X), -np.inf)
    for b, (sdb, lob) in SOL.items(): Rm, t = T[b]; worst = np.maximum(worst, sample(sdb, lob, hstep, (X - t) @ Rm))    # back to that bone's rest frame
    return worst
ofm = OB[FM[:, 0]]; cv_idx = np.flatnonzero(np.isin(OB, cm) & contact); cv_top = np.asarray(Wt[cv_idx].argmax(1)).ravel(); Wcv = Wt[cv_idx]
E = np.r_[SF[:, [0, 1]], SF[:, [1, 2]], SF[:, [2, 0]]]; loc = np.zeros(len(MV), int); loc[sk] = np.arange(len(sk)); e0 = np.linalg.norm(MV[E[:, 0]] - MV[E[:, 1]], axis=1); n0 = np.cross(MV[SF[:, 1]] - MV[SF[:, 0]], MV[SF[:, 2]] - MV[SF[:, 0]])
frames = []; rows = []; keep = {}
for k, u in enumerate(np.linspace(0, 1, FRAMES)):
    T = transforms(u); Xm = dq_blend(V[cmv], Wcm, T); Xs = dq_blend(MV[sk], Ws, T); Xc = dq_blend(V[cv_idx], Wcv, T)
    rigid = np.array([T[wb[c]][0] @ p + T[wb[c]][1] for c, p in zip(cv_top, V[cv_idx])]); drift = np.linalg.norm(Xc - rigid, axis=1)        # a vertex on a bone should move exactly as that bone
    ib = in_bone(Xm, T); Vf = V.copy(); Vf[cmv] = Xm; vol = closed_volumes(V, FM, ofm, cm, V, Vf); vr = np.array([b / a for a, b in vol.values() if a > 0])
    e1 = np.linalg.norm(Xs[loc[E[:, 0]]] - Xs[loc[E[:, 1]]], axis=1); st = e1[e0 > 0] / e0[e0 > 0]; n1 = np.cross(Xs[loc[SF[:, 1]]] - Xs[loc[SF[:, 0]]], Xs[loc[SF[:, 2]]] - Xs[loc[SF[:, 0]]]); flip = float(((n0 * n1).sum(1) < 0).mean())
    rows.append(dict(u=float(u), hip_deg=HIP_DEG * u, knee_deg=KNEE_DEG * u, attachment_drift_mm=dict(median=1000 * float(np.median(drift)), max=1000 * float(drift.max())), muscle_in_bone=dict(any=float((ib > 0).mean()), over_one_cell=float((ib > hstep).mean())),
                     closed_muscle_volume_ratio=dict(median=float(np.median(vr)), min=float(vr.min()), max=float(vr.max()), n=int(len(vr))), skin_edge=dict(p1=float(np.percentile(st, 1)), median=float(np.median(st)), p99=float(np.percentile(st, 99))), skin_triangles_flipped=flip))
    log('frame %d: hip %.0f deg, knee %.0f deg | on-bone vertices off their bone %.2f mm median (max %.1f) | muscle inside bone %.1f%% (over a cell %.1f%%) | closed-muscle volume x%.2f median (x%.2f..x%.2f, %d muscles) | skin edges x%.2f..x%.2f, %.2f%% triangles flipped' % (
        k, HIP_DEG * u, KNEE_DEG * u, rows[-1]['attachment_drift_mm']['median'], rows[-1]['attachment_drift_mm']['max'], 100 * rows[-1]['muscle_in_bone']['any'], 100 * rows[-1]['muscle_in_bone']['over_one_cell'], np.median(vr), vr.min(), vr.max(), len(vr), rows[-1]['skin_edge']['p1'], rows[-1]['skin_edge']['p99'], 100 * flip))
    Xb = {b: V[ii] @ T[b][0].T + T[b][1] for b, ii in bone_v.items()}; frames.append((Xm, Xs, np.vstack(list(Xb.values()))))
    if k == FRAMES - 1:
        names_c = [c['name'] for c in cross]; vl = sorted(((b / a, names[m]) for m, (a, b) in vol.items() if a > 0)); log('   closed muscles that shrink most at full bend: ' + '; '.join('%s x%.2f' % (n, r) for r, n in vl[:5])); log('   ... and that swell most: ' + '; '.join('%s x%.2f' % (n, r) for r, n in vl[-5:]))
        ed = np.linalg.norm(Xm[:, None].repeat(1, 1)[:, 0] - V[cmv], axis=1); far = sorted(((float(np.percentile(ed[OB[cmv] == m], 99)) if (OB[cmv] == m).any() else 0, names[m]) for m in cm))[-5:]; log('   moved farthest: ' + '; '.join('%s %.0f mm' % (n, 1000 * d) for d, n in far))
        top_s = np.asarray(Ws.argmax(1)).ravel(); segs = np.array([seg_of(wb[c]) or 'other' for c in top_s]); log('   skin bound to: ' + ', '.join('%s %d%%' % (g, round(100 * (segs == g).mean())) for g in ('pelvis', 'thigh', 'shank', 'other')))
rep = dict(side=sd, hip_centre=hip_c.tolist(), hip_head_radius_mm=1000 * hip_r, knee_axis=ax.tolist(), knee_point=knee_c.tolist(), clip=dict(hip_deg=HIP_DEG, knee_deg=KNEE_DEG, frames=FRAMES), crossing=cross, frames=rows, skin_binding_mm=dict(median=1000 * float(np.median(bind_d)), max=1000 * float(bind_d.max())), grid_mm=1000 * hstep)
json.dump(rep, open(os.path.join(FD, 'leg_bend_%s.json' % sd), 'w'), indent=1)
np.savez_compressed(os.path.join(FD, 'leg_bend_%s.npz' % sd), muscle=np.array([f[0] for f in frames], np.float32), skin=np.array([f[1] for f in frames], np.float32), bone=np.array([f[2] for f in frames], np.float32), skin_faces=loc[SF], muscle_ids=cmv)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pick = [0, FRAMES // 2, FRAMES - 1]; fig, axs = plt.subplots(1, 3, figsize=(15, 8))
for a_, k in zip(axs, pick):
    Xm, Xs, Xb = frames[k]; a_.scatter(Xs[:, 1], Xs[:, 2], s=0.15, c='#e3b58f', alpha=0.35); a_.scatter(Xm[::3, 1], Xm[::3, 2], s=0.1, c='#b5433c', alpha=0.5); a_.scatter(Xb[::2, 1], Xb[::2, 2], s=0.15, c='#222222')
    a_.set_aspect('equal'); a_.invert_xaxis(); a_.set_title('hip %.0f deg, knee %.0f deg (seen from her left; front is left)' % (rows[k]['hip_deg'], rows[k]['knee_deg']), fontsize=10); a_.axis('off')
plt.savefig(os.path.join(FD, 'leg_bend_%s.jpg' % sd), dpi=95, bbox_inches='tight'); log('done')
