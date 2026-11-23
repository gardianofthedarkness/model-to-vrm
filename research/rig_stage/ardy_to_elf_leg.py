"""Cheap test: an ARDY (SOMA) clip mapped onto the elf's fitted left leg, and what that motion does to its bones and muscles.
  E:\\splat\\.venv\\Scripts\\python.exe ardy_to_elf_leg.py <fits dir> <out dir> <clip.npz> [<clip.npz> ...]

Mapping (pelvis-relative; the pelvis is held still):
  hip   ARDY's thigh turn relative to its hips (global rotations, from its neutral T-pose) is moved into the elf's axes and
        applied as a ball joint about the centre of the sphere fitted to the femur head (as model_to_vrm/rig/leg_bend_test.py);
        the rest offset between ARDY's neutral thigh and the elf's fitted femur is removed as in motion/soma_drive.py
  knee  ARDY's shin turn relative to its thigh is split into the part about its main axis (the knee's own hinge, found as the
        first principal direction of the clip's knee rotation vectors) and the rest; the hinge angle is applied about the
        elf's fitted knee axis, the rest is reported as what an anatomical hinge cannot follow
Measured per frame:
  hip flexion, abduction and axial rotation of the femur (swing-twist split of the hip turn in the elf's pelvis axes)
  knee flexion and ARDY's off-hinge remainder
  bone contact: share of femur surface points inside the hip bone's solid (impingement), and the shin against the femur
  muscle chords: for each muscle crossing the hip or knee (from leg_bend_l.json), the straight distance between the centroids
        of its attachment patches on the proximal and distal bone groups (contact vertices, by their largest bone share);
        length / rest length.  A chord is a lower bound on the path where a muscle wraps over bone.
Writes ardy_to_elf_leg.json and ardy_to_elf_leg.png (curves only, no pictures of the body)."""
import sys, os, json, numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'model_to_vrm', 'anatomy'))
from solids import bone_solid, sample
FD, OUT, clips = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']; FB = R['F_bone']; contact = R['contact'].astype(bool)
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
Wt = csr_matrix((R['W_data'], R['W_indices'], R['W_indptr'])); wb = [int(b) for b in R['bones']]
S = '.l'; hipb, fem, tib, fib, pat = (nid[n + S] for n in ('Hip bone', 'Femur', 'Tibia', 'Fibula', 'Patella'))
group = {hipb: 'pelvis', fem: 'thigh', tib: 'shank', fib: 'shank', pat: 'shank'}
# --- joints from the fitted bones (same construction as leg_bend_test.py)
Pf, Ph, Ptb = V[OB == fem], V[OB == hipb], V[OB == tib]
dh = cKDTree(Ph).query(Pf)[0]; head = Pf[np.log(dh + 1e-9) < otsu(np.log(dh + 1e-9))]
A = np.c_[2 * head, np.ones(len(head))]; sol = np.linalg.lstsq(A, (head ** 2).sum(1), rcond=None)[0]; hip_c = sol[:3]
long_ = np.linalg.svd(Pf - Pf.mean(0), full_matrices=False)[2][0]; long_ = long_ if long_[2] < 0 else -long_          # hip -> knee
dk = cKDTree(Ptb).query(Pf)[0]; cond = Pf[np.log(dk + 1e-9) < otsu(np.log(dk + 1e-9))]
Q = cond - cond.mean(0); Q = Q - np.outer(Q @ long_, long_); kax = np.linalg.svd(Q, full_matrices=False)[2][0]; kax = kax if kax[0] > 0 else -kax; knee_c = cond.mean(0)
# elf model axes: x to her left, y to her back, z up.  ARDY: x to the left, y up, z forward.
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
snames = open(r'E:\figures\ardy\soma77_names.txt').read().split('\n'); si = {n: i for i, n in enumerate(snames)}
N = np.load(r'E:\figures\ardy\soma77_neutral.npy') @ M.T
def turn(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b); v = np.cross(a, b); c = a @ b
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K / (1 + c)
rest_off = float(np.degrees(np.arccos(np.clip((N[si['LeftShin']] - N[si['LeftLeg']]) @ long_ / np.linalg.norm(N[si['LeftShin']] - N[si['LeftLeg']]), -1, 1))))   # ARDY neutral thigh vs elf femur
# --- solids for contact tests, grid from the bone meshes' own edge length
def edges(F): return np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
hstep = 2 * float(np.median(edges(FB[np.isin(OB[FB[:, 0]], [hipb, fem, tib])])))
SOL = {b: bone_solid(V, FB[OB[FB[:, 0]] == b], hstep)[:2] for b in (hipb, fem)}
def inside(X, b, Rm=np.eye(3), t=np.zeros(3)): sdb, lob = SOL[b]; return sample(sdb, lob, hstep, (X - t) @ Rm) > 0
# --- muscle attachment patches
cross = json.load(open(os.path.join(FD, 'leg_bend_l.json')))['crossing']; order = {'pelvis': 0, 'thigh': 1, 'shank': 2}
chords = []
for c in cross:
    m = nid[c['name']]; vi = np.flatnonzero((OB == m) & contact)
    if not len(vi): continue
    top = np.array([wb[k] for k in np.asarray(Wt[vi].argmax(1)).ravel()]); g = np.array([group.get(b) for b in top])
    segs = sorted({x for x in g if x}, key=order.get)
    if len(segs) < 2 or any(w in c['name'].lower() for w in ('bursa', 'fascia', 'ligament')): continue   # muscles only
    chords.append(dict(name=c['name'], a=segs[0], b=segs[-1], pa=V[vi[g == segs[0]]].mean(0), pb=V[vi[g == segs[-1]]].mean(0)))
def swing_twist(Rm, axis):
    q = Rot.from_matrix(Rm).as_quat(); p = (q[:3] @ axis) * axis; tw = np.r_[p, q[3]]; tw /= np.linalg.norm(tw)
    return Rot.from_quat(q) * Rot.from_quat(tw).inv(), Rot.from_quat(tw)
REST = dict(femur_in_pelvis=float(inside(Pf[::4], hipb).mean()), tibia_in_femur=float(inside(Ptb[::4], fem).mean()))   # overlap already in the fit at rest
print('rest overlap', REST)
res = {}
for cp in clips:
    C = np.load(cp); G = C['global_rot_mats']; T = len(G); nm = os.path.splitext(os.path.basename(cp))[0]
    rel_h = np.einsum('tji,tjk->tik', G[:, si['Hips']], G[:, si['LeftLeg']]); rel_k = np.einsum('tji,tjk->tik', G[:, si['LeftLeg']], G[:, si['LeftShin']])
    rv = Rot.from_matrix(rel_k).as_rotvec(); kax_s = np.linalg.svd(rv, full_matrices=False)[2][0]; kang = rv @ kax_s
    off = np.degrees(np.linalg.norm(rv - np.outer(kang, kax_s), axis=1))
    s_elf = np.sign((M @ kax_s) @ kax)                                   # same hinge direction sense in the elf's axes
    rows = []; L0 = None
    for t in range(T):
        D = M @ rel_h[t] @ M.T                                           # hip turn from the T-pose, in the elf's axes
        Rk = Rot.from_rotvec(kax * s_elf * kang[t]).as_matrix()
        Th = (D, hip_c - D @ hip_c); Tk = (D @ Rk, D @ (knee_c - Rk @ knee_c) + Th[1])
        TF = {'pelvis': (np.eye(3), np.zeros(3)), 'thigh': Th, 'shank': Tk}
        sw, tw = swing_twist(D, long_); sv = sw.as_rotvec()
        flex = -np.degrees(sv @ np.array([1., 0, 0])); abd = -np.degrees(sv @ np.array([0, 1., 0])); rot = np.degrees(tw.as_rotvec() @ long_)
        Xf = Pf[::4] @ D.T + Th[1]; imp = float(inside(Xf, hipb).mean())
        Xt = Ptb[::4] @ Tk[0].T + Tk[1]; kimp = float(inside(Xt, fem, D, Th[1]).mean())
        L = np.array([np.linalg.norm((TF[c['b']][0] @ c['pb'] + TF[c['b']][1]) - (TF[c['a']][0] @ c['pa'] + TF[c['a']][1])) for c in chords])
        if L0 is None: L0 = np.array([np.linalg.norm(c['pb'] - c['pa']) for c in chords])
        rows.append(dict(t=t, hip_flex=flex, hip_abd=abd, hip_rot=rot, knee=float(np.degrees(s_elf * kang[t])), knee_off_hinge=float(off[t]), femur_in_pelvis=imp, tibia_in_femur=kimp, chord=L / L0))
    ch = np.array([r['chord'] for r in rows])
    res[nm] = dict(frames=T, fps=float(C['fps']),
                   hip_flex_deg=[round(min(r['hip_flex'] for r in rows), 1), round(max(r['hip_flex'] for r in rows), 1)],
                   hip_abd_deg=[round(min(r['hip_abd'] for r in rows), 1), round(max(r['hip_abd'] for r in rows), 1)],
                   hip_rot_deg=[round(min(r['hip_rot'] for r in rows), 1), round(max(r['hip_rot'] for r in rows), 1)],
                   knee_deg=[round(min(r['knee'] for r in rows), 1), round(max(r['knee'] for r in rows), 1)], knee_off_hinge_max_deg=round(float(off.max()), 2),
                   femur_points_in_pelvis_max=round(max(r['femur_in_pelvis'] for r in rows), 4), tibia_points_in_femur_max=round(max(r['tibia_in_femur'] for r in rows), 4),
                   chord_ratio_range=dict(sorted([(c['name'], [round(float(ch[:, i].min()), 3), round(float(ch[:, i].max()), 3)]) for i, c in enumerate(chords)], key=lambda kv: -(kv[1][1] - kv[1][0]))[:12]))
    res[nm]['_rows'] = rows
    print(nm, json.dumps({k: v for k, v in res[nm].items() if k not in ('_rows', 'chord_ratio_range')}))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(clips), 3, figsize=(15, 3.6 * len(clips)), squeeze=False)
for i, (nm, r) in enumerate(res.items()):
    rows = r['_rows']; t = np.arange(len(rows)) / r['fps']
    a = axs[i, 0]
    for k, lab in (('hip_flex', 'hip flexion'), ('hip_abd', 'hip abduction'), ('hip_rot', 'hip rotation'), ('knee', 'knee flexion')): a.plot(t, [x[k] for x in rows], label=lab)
    a.set_title('%s: joint angles mapped onto the elf (deg)' % nm, fontsize=9); a.legend(fontsize=7); a.grid(alpha=.3)
    a = axs[i, 1]; a.plot(t, [100 * x['femur_in_pelvis'] for x in rows], label='femur points inside hip bone (%)'); a.plot(t, [100 * x['tibia_in_femur'] for x in rows], label='tibia points inside femur (%)')
    a.plot(t, [x['knee_off_hinge'] for x in rows], label="ARDY knee off the hinge (deg)"); a.set_title('bone contact and hinge remainder', fontsize=9); a.legend(fontsize=7); a.grid(alpha=.3)
    a = axs[i, 2]; ch = np.array([x['chord'] for x in rows]); top = np.argsort(-(ch.max(0) - ch.min(0)))[:8]
    for j in top: a.plot(t, ch[:, j], label=chords[j]['name'][:28])
    a.axhline(1, c='k', lw=.5); a.set_title('muscle chord length / rest (8 that change most)', fontsize=9); a.legend(fontsize=6); a.grid(alpha=.3)
for a in axs[-1]: a.set_xlabel('time (s)')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'ardy_to_elf_leg.png'), dpi=100)
json.dump(dict(rest_overlap=REST, rest_thigh_offset_deg=rest_off, hip_centre=hip_c.tolist(), knee_axis=kax.tolist(), grid_mm=1000 * hstep, chords=len(chords), clips={k: {kk: vv for kk, vv in v.items() if kk != '_rows'} for k, v in res.items()}),
          open(os.path.join(OUT, 'ardy_to_elf_leg.json'), 'w'), indent=1)
print('wrote', OUT)
