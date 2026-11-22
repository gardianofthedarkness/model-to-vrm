"""A jointed bone fit for one intricate part (a hand, a foot): the atlas bones of the part placed in the model from the
named detector points (agents/point_names.py), each bone moving as a rigid piece of its own size, bones held together at
their joints, and no bone outside the surface. The detected points of a left and a right part are pooled first, as far
as the figure's skin is its own mirror image there (symmetry.py).
  E:\\splat\\.venv\\Scripts\\python.exe part_bone_fit.py <fits dir> <model dir> <region>      region: hand_r hand_l foot_r foot_l
Stage 1  one size and pose for the whole part (Umeyama 1991): atlas positions of the named points -> detected points.
Stage 2  per bone b, 7 numbers on top of stage 1: turn r_b, shift t_b, size s_b (about the bone's own centre c_b)
           T_b(x) = c_b + exp(s_b) Rot(r_b) (x - c_b) + t_b
         least squares (scipy 'trf'), every scale from the data:
  joints   every two touching bones (bone_gaps.npz: the measured list) share one point, the middle of their contact:
           T_a(m) = T_b(m). One point per joint, so the bones may turn about it freely (a ball joint).   / sigma_c
  points   a named joint's detected place, carried by each of its bones in the part: T_b(a) = x.            / sigma_j
           free bone ends and bone-under-skin points (fingertips, toe tips, heel) likewise, with their own   / sigma_e
           scale, since soft tissue of unknown depth covers them.
  size     s_a - s_b for touching bones: neighbours are alike in size                                         / sigma_s
  turn     r_a - r_b for touching bones: a joint bends, but not without cause (no free spin about a bone's axis) / sigma_r
  toes     (feet, when toe_lines.py has run) each toe's tip is a free bone end of that toe's last bone, and the joints
           between that toe's bones lie on the toe's centre line (free to slide along it).                    / sigma_e, sigma_j
  inside   every bone hull point's depth outside the solid model of the part (as in skull_line_fit.py), as a penalty
           whose scale is halved until the worst point is inside to within one grid cell.
Scales: sigma_c = the skeleton fit's joint gap; sigma_j, sigma_e re-estimated from their own residuals after each solve
(1.4826 x median absolute value; sigma_j never below the detector's own spread across views). sigma_s and sigma_r are
measured once from the stretches - two named points on one bone give that bone's length and direction against the
atlas: sigma_s = the spread of the lengths less what the detector's spread explains, sigma_r = the spread of the bends.
Writes <fits dir>/<region>_bones.npz (V: all atlas vertices with the part's bones placed; V_stage1; V_before = the body
skeleton fit; F; J_det) and <region>_bones.json (the checks)."""
import sys, os, re, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symmetry
from scipy import ndimage
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
FD, MD, REG = sys.argv[1:4]; sd = '.' + REG[-1]; kind = REG[:4]; t00 = time.time()
def log(*a): print('PB', *a, flush=True)
W = np.load(os.path.join(FD, 'atlas_warp.npz')); V0 = W['V0'].astype(np.float64); OB = W['obj']; names = [str(x) for x in W['names']]; FB = W['F_bone']; nid = {n: i for i, n in enumerate(names)}; sysn = [str(x) for x in W['system']]
SK = np.load(os.path.join(FD, 'skeleton_fit.npz')); Vsk = SK['V_rigid'].astype(np.float64); SKJ = json.load(open(os.path.join(FD, 'skeleton_fit.json')))
NM = json.load(open(os.path.join(MD, 'lines', 'points_named.json'), encoding='utf-8')); G = np.load(os.path.join(FD, 'bone_gaps.npz'))
# the part's bones: by the atlas's own names (a dictionary lookup: the hand's and the foot's bones)
PAT = {'hand': r'of hand|metacarpal|Scaphoid|Lunate|Triquetr|Pisiform|Trapezi|Capitate|Hamate', 'foot': r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform'}[kind]
bones = [i for i, n in enumerate(names) if n.endswith(sd) and re.search(PAT, n) and sysn[i] == 'bone' and 'cartilage' not in n.lower() and (OB == i).any()]; B = len(bones); bi = {b: k for k, b in enumerate(bones)}
C = np.array([V0[OB == b].mean(0) for b in bones])
# detected points of the part, by the names used in points_named.json
if kind == 'hand':
    Z = np.load(os.path.join(MD, 'parts_zoom', REG, 'points3d.npz')); keys = ['hand wrist'] + ['%s %d' % (f, k) for f in ('thumb', 'index finger', 'middle finger', 'ring finger', 'little finger') for k in (1, 2, 3, 4)]
    DET = dict(zip(keys, symmetry.hand_points(MD, sd[1], log=log))); spread = float(np.nanmedian(Z['rms']))
else:
    Z = dict(X=symmetry.body_joints(MD, log=log)); o = 0 if sd == '.l' else 1; DET = {'ankle': Z['X'][15 + o], 'big toe': Z['X'][17 + 3 * o], 'small toe': Z['X'][18 + 3 * o], 'heel': Z['X'][19 + 3 * o]}   # COCO-WholeBody order
    spread = None
P = []                                                              # (name, kind, atlas place, detected place, bones of the part that carry it)
for k, x in DET.items():
    e = NM['points'].get(k + sd)
    if e is None or e['atlas'] is None or not np.isfinite(x).all(): continue
    P.append((k, e['kind'], np.array(e['atlas']), np.array(x, float), [bi[nid[n]] for n in e['bones'] if nid[n] in bi]))
TOE = None; tf = os.path.join(MD, 'parts_zoom', REG, 'toes.npz')
if kind == 'foot' and os.path.exists(tf):                           # the toes (toe_lines.py): numbered first to fifth, as the atlas numbers them
    TOE = symmetry.toes(MD, sd[1], log=log); ORD = ['first', 'second', 'third', 'fourth', 'fifth']; P = [p for p in P if p[0] not in ('big toe', 'small toe')]; thr_ = float(G['threshold']); touch = {tuple(sorted(map(int, pr))) for pr, g in zip(G['pairs'], G['rel_gap']) if g < thr_}
    for k in range(5):
        d_ = nid['Distal phalanx of %s finger of foot%s' % (ORD[k], sd)]; nb = [b for b in bones if b != d_ and tuple(sorted((b, d_))) in touch and 'phalanx' in names[b]]
        Pa, Pb = V0[OB == d_], V0[OB == nb[0]]; dd, ii_ = cKDTree(Pb).query(Pa); kk = np.argsort(dd)[:max(3, len(Pa) // 100)]; jmid = (Pa[kk].mean(0) + Pb[ii_[kk]].mean(0)) / 2
        far = Pa[np.argsort(np.linalg.norm(Pa - jmid, axis=1))[-max(3, len(Pa) // 100):]].mean(0); P.append(('toe %d tip' % (k + 1), 'bone_end', far, TOE['tip'][k].astype(float), [bi[d_]]))
A0 = np.array([p[2] for p in P]); X = np.array([p[3] for p in P]); isj = np.array([p[1] == 'joint' for p in P])
def umeyama(A, Bx):
    ma, mb = A.mean(0), Bx.mean(0); U, D_, Vt = np.linalg.svd((Bx - mb).T @ (A - ma) / len(A)); Sg = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R_ = U @ Sg @ Vt
    s_ = np.trace(np.diag(D_) @ Sg) / ((A - ma) ** 2).sum(1).mean(); return s_, R_, mb - s_ * R_ @ ma
s1, R1, t1 = umeyama(A0, X); W2A = lambda Y: (Y - t1) @ R1 / s1; A2W = lambda Y: s1 * Y @ R1.T + t1; Xa = W2A(X); r1 = np.linalg.norm(A0 - Xa, axis=1) * s1
log('%s: %d bones, %d named points (%d joints, %d ends); stage 1 size %.3f, points left: joints %.1f mm, ends %.1f mm median' % (REG, B, len(P), isj.sum(), (~isj).sum(), s1, 1000 * np.median(r1[isj]), 1000 * np.median(r1[~isj]) if (~isj).any() else 0))
# --- joints of the part: the measured touching pairs, one shared point each
jn = os.path.join(FD, 'joints_named.json'); NOJ = {tuple(sorted((nid[e_['a']], nid[e_['b']]))) for e_ in json.load(open(jn, encoding='utf-8')) if not e_['joint']} if os.path.exists(jn) else set()   # touching, but no joint between them (agents/joint_names.py)
thr = float(G['threshold']); JT = []
for (a, b), g in zip(G['pairs'], G['rel_gap']):
    if g < thr and int(a) in bi and int(b) in bi and tuple(sorted((int(a), int(b)))) not in NOJ:
        Pa, Pb = V0[OB == a], V0[OB == b]; d, i = cKDTree(Pb).query(Pa); k = np.argsort(d)[:max(3, len(Pa) // 100)]; JT.append((bi[int(a)], bi[int(b)], (Pa[k].mean(0) + Pb[i[k]].mean(0)) / 2))
ja = np.array([j[0] for j in JT]); jb = np.array([j[1] for j in JT]); jm = np.array([j[2] for j in JT])
tl_j, tl_c, tl_a = np.zeros(0, int), np.zeros((0, 3)), np.zeros((0, 3))
if TOE is not None:                                                 # joints between the phalanges of toe k lie on toe k's centre line
    rows_ = [(j, k) for j in range(len(JT)) for k in range(5) if all(('phalanx of %s finger of foot' % ORD[k]) in names[bones[b]] for b in (ja[j], jb[j]))]
    tl_j = np.array([j for j, _ in rows_]); tl_c = W2A(TOE['centre'][[k for _, k in rows_]].astype(float)); tl_a = TOE['axis'][[k for _, k in rows_]].astype(float) @ R1; log('toes: 5 tips, %d toe joints held to their toe centre lines' % len(rows_))
log('joints inside the part: %d (ball joints at the contact middles)' % len(JT))
# --- solid model of the part (surface in the part's box -> grid one surface edge fine; faces the surface runs through are capped; enclosed space filled)
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']
box = json.load(open(os.path.join(MD, 'parts_zoom', 'boxes.json')))[REG]; hstep = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1)))
glo = np.array(box['centre']) - np.array(box['half']) - 4 * hstep; ghi = np.array(box['centre']) + np.array(box['half']) + 4 * hstep; gn = np.ceil((ghi - glo) / hstep).astype(int) + 1
fin = ((MV[MF] > glo) & (MV[MF] < ghi)).all(2).any(1); T = MV[MF[fin]]; nsub = np.clip(np.ceil(np.maximum(np.linalg.norm(T[:, 1] - T[:, 0], axis=1), np.linalg.norm(T[:, 2] - T[:, 0], axis=1)) / (hstep / 2)).astype(int), 1, 64); occ = np.zeros(gn, bool)
for n_ in np.unique(nsub):
    Tn = T[nsub == n_]; u, v = np.meshgrid(np.arange(n_ + 1) / n_, np.arange(n_ + 1) / n_); m_ = (u + v) <= 1 + 1e-9; u, v = u[m_], v[m_]
    Pn = (Tn[:, None, 0] * (1 - u - v)[None, :, None] + Tn[:, None, 1] * u[None, :, None] + Tn[:, None, 2] * v[None, :, None]).reshape(-1, 3); ii = np.rint((Pn - glo) / hstep).astype(int)
    ok_ = ((ii >= 0) & (ii < gn)).all(1); occ[ii[ok_, 0], ii[ok_, 1], ii[ok_, 2]] = True
faces = [(ax, end) for ax in range(3) for end in (0, -1) if np.take(occ, end, axis=ax).any()]          # box faces the surface runs through (the wrist, the leg)
def capped(a):
    a = a.copy()
    for ax, end in faces: sl = [slice(None)] * 3; sl[ax] = end; a[tuple(sl)] = True
    return a
area = sum(int(np.prod(np.delete(gn, ax))) for ax, _ in faces)
for seal in range(1, 33):
    thick = ndimage.binary_dilation(occ, iterations=seal); fill = ndimage.binary_fill_holes(capped(thick))
    if fill.sum() - area - thick.sum() > thick.sum(): break
else: print('WARNING part_bone_fit.py: the solid did not close within 33 sealing steps; the widest sealing is used and the result may be wrong', flush=True)
solid = ndimage.binary_erosion(fill, iterations=seal, border_value=1) | occ
for ax, end in faces: sl = [slice(None)] * 3; s2 = [slice(None)] * 3; sl[ax] = end; s2[ax] = 1 if end == 0 else -2; solid[tuple(sl)] = solid[tuple(s2)]
sdf = (ndimage.distance_transform_edt(solid) - ndimage.distance_transform_edt(~solid)) * hstep
def depth(Pw): return ndimage.map_coordinates(sdf, ((Pw - glo) / hstep).T, order=1, mode='nearest')    # + inside
log('solid: grid %s at %.2f mm; open faces capped %s; gaps sealed by %d cells; volume %.0f cm3' % (tuple(int(g) for g in gn), 1000 * hstep, faces, seal, 1e6 * solid.sum() * hstep ** 3))
hull, hown = [], []
for b in bones:
    ii = np.flatnonzero(OB == b)
    try: ii = ii[np.unique(ConvexHull(V0[ii]).vertices)] if len(ii) >= 8 else ii
    except Exception: pass
    hull.append(ii); hown.append(np.full(len(ii), bi[b]))
hull = np.concatenate(hull); hown = np.concatenate(hown); H0 = V0[hull]
# --- stage 2
def TF(x, pts, own):                                               # atlas frame
    q = x.reshape(B, 7); Rm = Rot.from_rotvec(q[own, :3]).as_matrix(); return C[own] + np.exp(q[own, 6])[:, None] * np.einsum('nij,nj->ni', Rm, pts - C[own]) + q[own, 3:6]
drow = [(k, b) for k, p in enumerate(P) for b in p[4]]; dk = np.array([k for k, _ in drow]); db = np.array([b for _, b in drow]); dj = isj[dk]
sig_c = SKJ['joint_gap_mm']['median'] / 1000 / s1; floor = (spread if spread else 0.0) / s1
sig_j = max(1.4826 * float(np.median(r1[isj])) / s1, floor, sig_c); sig_e = max(1.4826 * float(np.median(r1[~isj])) / s1, sig_j) if (~isj).any() else sig_j
seg = [np.log(np.linalg.norm(Xa[i] - Xa[j]) / np.linalg.norm(A0[i] - A0[j])) for i in range(len(P)) for j in range(i + 1, len(P)) if set(P[i][4]) & set(P[j][4])]   # two points on one bone: its length against the atlas
pairs_on = [(i, j) for i in range(len(P)) for j in range(i + 1, len(P)) if set(P[i][4]) & set(P[j][4])]
if len(seg) > 2:                                                    # true spread = observed spread less what the detector's own spread explains (empirical Bayes)
    obs = 1.4826 * float(np.median(np.abs(np.array(seg) - np.median(seg)))); noise = float(np.median([np.sqrt(2) * spread / np.linalg.norm(X[i] - X[j]) for i, j in pairs_on])) if spread else 0.0
    sig_s = float(np.sqrt(max(obs ** 2 - noise ** 2, 0))) or obs / np.sqrt(len(seg)); cosb = [np.clip((Xa[j] - Xa[i]) @ (A0[j] - A0[i]) / np.linalg.norm(Xa[j] - Xa[i]) / np.linalg.norm(A0[j] - A0[i]), -1, 1) for i, j in pairs_on]
    sig_r = 1.4826 * float(np.median(np.arccos(cosb))); log('stretches (two named points on one bone): %d; length spread seen %.3f, detector explains %.3f -> size scale %.3f; bend from the atlas pose median %.0f deg -> turn scale %.0f deg' % (len(seg), obs, noise, sig_s, np.degrees(np.median(np.arccos(cosb))), np.degrees(sig_r)))
else:                                                               # no stretch in this part (the foot's four points): the body skeleton fit's size spread; turn scale found from the fit's own residuals
    sig_s = float(SKJ['scales_final']['s']); import glob
    tr = [json.load(open(f))['scales'].get('turn_deg') for f in glob.glob(os.path.join(FD, '*_bones.json')) if json.load(open(f)).get('stretches')]
    sig_r = float(np.radians(np.median(tr))) if tr else None; log('no stretches in this part: size scale %.3f from the body skeleton fit; turn scale %s' % (sig_s, '%.0f deg, as measured on the parts that have stretches (%d)' % (np.degrees(sig_r), len(tr)) if sig_r else 'from the fit itself'))
vce_r = sig_r is None; sig_r = sig_r or 1.0
x = np.zeros(7 * B); sig_in = sig_j * s1; hist = []
from scipy.sparse import lil_matrix
def cols(b): return 7 * b + np.arange(7)
rows_b = [[b] for b in np.repeat(db, 3)] + [[b] for b in np.repeat(ja[tl_j], 3)] + [[a, b] for a, b in zip(np.repeat(ja, 3), np.repeat(jb, 3))] + [[a, b] for a, b in zip(ja, jb)] + [[a, b] for a, b in zip(np.repeat(ja, 3), np.repeat(jb, 3))] + [[b] for b in hown]
SP = lil_matrix((len(rows_b), 7 * B), dtype=np.int8)
for r_, bb in enumerate(rows_b):
    for b in bb: SP[r_, cols(b)] = 1
def resid(x_, sig_in_):
    Dp = TF(x_, A0[dk], db) - Xa[dk]; w = np.where(dj, 1 / sig_j, 1 / sig_e)[:, None]
    q_ = x_.reshape(B, 7); dl_ = TF(x_, jm[tl_j], ja[tl_j]) - tl_c; dl_ = dl_ - (dl_ * tl_a).sum(1)[:, None] * tl_a
    return np.r_[(Dp * w).ravel(), (dl_ / sig_j).ravel(), ((TF(x_, jm, ja) - TF(x_, jm, jb)) / sig_c).ravel(), (q_[ja, 6] - q_[jb, 6]) / sig_s, ((q_[ja, :3] - q_[jb, :3]) / sig_r).ravel(), np.maximum(0, -depth(A2W(TF(x_, H0, hown)))) / sig_in_]
for outer in range(16):
    x = least_squares(resid, x, args=(sig_in,), method='trf', loss='linear', jac_sparsity=SP, max_nfev=200).x
    Dp = np.linalg.norm(TF(x, A0[dk], db) - Xa[dk], axis=1); dp = depth(A2W(TF(x, H0, hown))); worst = float(max(0, -dp.min())); gapj = np.linalg.norm(TF(x, jm, ja) - TF(x, jm, jb), axis=1) * s1
    sig_j = max(1.4826 * float(np.median(Dp[dj])), floor, sig_c); sig_e = max(1.4826 * float(np.median(Dp[~dj])), sig_j) if (~dj).any() else sig_j
    qq = x.reshape(B, 7); dr = np.linalg.norm(qq[ja, :3] - qq[jb, :3], axis=1)
    if vce_r: sig_r = max(1.4826 * float(np.median(dr)), np.radians(1))
    hist.append(dict(round=outer, penalty_mm=1000 * sig_in, outside_worst_mm=1000 * worst, outside_share=float((dp < 0).mean()), joints_mm=1000 * s1 * float(np.median(Dp[dj])), joint_gap_mm=1000 * float(np.median(gapj)), turn_between_neighbours_deg=float(np.degrees(np.median(dr)))))
    log('   round %d: inside penalty %.3f mm, worst outside %.2f mm (%.1f%% of hull points), named joints %.1f mm, joint gaps %.2f mm, neighbours turned apart by median %.0f deg (%.0f s)' % (outer, 1000 * sig_in, 1000 * worst, 100 * (dp < 0).mean(), hist[-1]['joints_mm'], hist[-1]['joint_gap_mm'], np.degrees(np.median(dr)), time.time() - t00))
    if worst < hstep and outer > 0: break
    sig_in /= 2
else: log('WARNING: the inside penalty was halved 16 times and the worst point is still %.2f mm outside (one grid cell is %.2f mm); the last round is used' % (1000 * worst, 1000 * hstep))
# --- results and checks
m = np.isin(OB, bones); own_all = np.array([bi.get(int(b), 0) for b in OB[m]]); V1 = Vsk.copy(); V1[m] = A2W(V0[m]); V2 = Vsk.copy(); V2[m] = A2W(TF(x, V0[m], own_all)); q = x.reshape(B, 7)
out_ = lambda Vx: (100 * float((depth(Vx[m]) < 0).mean()), 1000 * float(max(0, -depth(Vx[m]).min())))
e2 = np.linalg.norm(A2W(TF(x, A0[dk], db)) - X[dk], axis=1) * 1000; ang = np.degrees(np.linalg.norm(q[ja, :3] - q[jb, :3], axis=1)); o = np.argsort(-ang)
rep = dict(region=REG, bones=B, points=len(P), stretches=len(seg), toes=TOE is not None, stage1=dict(size=float(s1), joints_mm=1000 * float(np.median(r1[isj])), ends_mm=1000 * float(np.median(r1[~isj])) if (~isj).any() else None),
           stage2=dict(joints_mm=float(np.median(e2[dj])), joints_max_mm=float(e2[dj].max()), ends_mm=float(np.median(e2[~dj])) if (~dj).any() else None, joint_gap_mm=dict(median=1000 * float(np.median(gapj)), max=1000 * float(gapj.max())),
                       bone_size=dict(median=float(s1 * np.exp(np.median(q[:, 6]))), min=float(s1 * np.exp(q[:, 6].min())), max=float(s1 * np.exp(q[:, 6].max())), smallest=names[bones[int(q[:, 6].argmin())]], largest=names[bones[int(q[:, 6].argmax())]]),
                       turn_deg=dict(median=float(np.median(ang)), largest=[['%s / %s' % (names[bones[ja[i]]], names[bones[jb[i]]]), float(ang[i])] for i in o[:5]])),
           outside=dict(before=out_(Vsk), stage1=out_(V1), stage2=out_(V2)), rounds=hist, scales=dict(joint_gap_mm=1000 * sig_c * s1, joints_mm=1000 * sig_j * s1, ends_mm=1000 * sig_e * s1, size=sig_s, turn_deg=float(np.degrees(sig_r)), detector_spread_mm=1000 * spread if spread else None), seconds=time.time() - t00)
log('named joints: stage 1 %.1f mm -> stage 2 %.1f mm (max %.1f); ends %s -> %s mm; joint gaps %.2f mm median, %.2f max' % (rep['stage1']['joints_mm'], rep['stage2']['joints_mm'], rep['stage2']['joints_max_mm'], '%.1f' % rep['stage1']['ends_mm'] if rep['stage1']['ends_mm'] else '-',
    '%.1f' % rep['stage2']['ends_mm'] if rep['stage2']['ends_mm'] else '-', rep['stage2']['joint_gap_mm']['median'], rep['stage2']['joint_gap_mm']['max']))
log('bone outside the part (share of bone vertices, worst): body skeleton fit %.1f%% / %.1f mm; stage 1 %.1f%% / %.1f mm; stage 2 %.1f%% / %.1f mm' % (*rep['outside']['before'], *rep['outside']['stage1'], *rep['outside']['stage2']))
log('bone size: median %.3f (min %.3f %s, max %.3f %s); joints bent most: %s' % (rep['stage2']['bone_size']['median'], rep['stage2']['bone_size']['min'], rep['stage2']['bone_size']['smallest'], rep['stage2']['bone_size']['max'], rep['stage2']['bone_size']['largest'],
    '; '.join('%s %.0f deg' % (n, a) for n, a in rep['stage2']['turn_deg']['largest'][:3])))
np.savez(os.path.join(FD, REG + '_bones.npz'), V=V2, V_stage1=V1, V_before=Vsk, F=FB[m[FB].all(1)], J_det=X, params=q, bones=np.array(bones), s1=s1, R1=R1, t1=t1)
json.dump(rep, open(os.path.join(FD, REG + '_bones.json'), 'w'), indent=1); log('done in %.0f s' % (time.time() - t00))
