"""Calibrated mapping of ARDY (SOMA) clips onto the elf's fitted left leg: joint angles from anatomical frames built from
geometry on both sides, checked against the joint ranges of the Rajagopal 2016 OpenSim model.
  E:\\splat\\.venv\\Scripts\\python.exe ardy_to_elf_leg_calibrated.py <fits dir> <out dir> <Rajagopal2016.osim> <clip.npz> [...]

Frames (columns X anterior, Y up / proximal, Z to the subject's right; OpenSim's convention):
  pelvis  elf: the model axes at rest (it stands upright facing -y).  ARDY: its hips' turn from its neutral pose applied to
          the same axes (its neutral stands upright facing +z).
  thigh   Y = knee centre -> hip centre (mechanical axis); Z = the knee hinge axis, pointing right, made orthogonal to Y.
          elf: hip centre from the sphere fitted to the femur head, knee centre and axis from the condyles facing the tibia
          (as model_to_vrm/rig/leg_bend_test.py).  ARDY: its LeftLeg and LeftShin joint positions, and its knee's hinge axis
          (first principal direction of the clip's knee rotation vectors, in the thigh's frame) carried by the thigh's turn.
  hip angles = intrinsic Z-X-Y Euler angles of pelvis^T * thigh (OpenSim hip: flexion, adduction, rotation; left side
          signs mirrored as in the model), measured for ARDY; the elf's femur is then placed so that its own angles equal
          ARDY's.  The knee angle is the angle between the thigh's mechanical axis and the shin (knee -> ankle) about the
          hinge; the elf's shin turns about its fitted axis by ARDY's knee angle minus the elf's rest knee angle.
No T-pose is assumed to equal the elf's rest: each side's angles come from its own geometry.
Writes ardy_to_elf_leg_calibrated.json/.png; muscle chords and bone contact as in ardy_to_elf_leg.py."""
import sys, os, re, json, math, numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, 'model_to_vrm', 'anatomy'))
from solids import bone_solid, sample
FD, OUT, OSIM, clips = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
os.makedirs(OUT, exist_ok=True)
# --- joint ranges from the published model (degrees)
osim = open(OSIM, encoding='utf-8').read(); RANGE = {}
for m in re.finditer(r'<Coordinate name="([^"]+)">(.*?)</Coordinate>', osim, re.S):
    r = re.search(r'<range>\s*([-\d.e]+)\s+([-\d.e]+)\s*</range>', m.group(2))
    if r: RANGE[m.group(1)] = (math.degrees(float(r.group(1))), math.degrees(float(r.group(2))))
LIM = {'hip_flex': RANGE['hip_flexion_l'], 'hip_add': RANGE['hip_adduction_l'], 'hip_rot': RANGE['hip_rotation_l'], 'knee': RANGE['knee_angle_l']}
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']; FB = R['F_bone']; contact = R['contact'].astype(bool)
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
Wt = csr_matrix((R['W_data'], R['W_indices'], R['W_indptr'])); wb = [int(b) for b in R['bones']]
S = '.l'; hipb, fem, tib, fib, pat = (nid[n + S] for n in ('Hip bone', 'Femur', 'Tibia', 'Fibula', 'Patella'))
talus = nid['Talus' + S]
group = {hipb: 'pelvis', fem: 'thigh', tib: 'shank', fib: 'shank', pat: 'shank'}
Pf, Ph, Ptb, Pta = V[OB == fem], V[OB == hipb], V[OB == tib], V[OB == talus]
def facing(P, Q):
    d = cKDTree(Q).query(P)[0]; return P[np.log(d + 1e-9) < otsu(np.log(d + 1e-9))]
head = facing(Pf, Ph); A = np.c_[2 * head, np.ones(len(head))]; sol = np.linalg.lstsq(A, (head ** 2).sum(1), rcond=None)[0]; hip_c = sol[:3]
long_ = np.linalg.svd(Pf - Pf.mean(0), full_matrices=False)[2][0]
cond = facing(Pf, Ptb); Q = cond - cond.mean(0); Q = Q - np.outer(Q @ long_, long_); kax = np.linalg.svd(Q, full_matrices=False)[2][0]; knee_c = cond.mean(0)
ank_c = facing(Ptb, Pta).mean(0)                                             # tibia surface facing the talus
RIGHT = np.array([-1., 0, 0]); kax = kax if kax @ RIGHT > 0 else -kax       # elf axes: x to her left, y back, z up
P_e = np.c_[[0, -1., 0], [0, 0, 1.], RIGHT]                                  # pelvis frame columns X anterior, Y up, Z right
def frame(Yv, Zv):
    Y = Yv / np.linalg.norm(Yv); Z = Zv - (Zv @ Y) * Y; Z /= np.linalg.norm(Z); return np.c_[np.cross(Y, Z), Y, Z]
T_e = frame(hip_c - knee_c, kax)
def knee_angle(T, shin):                                                     # flexion: shin swings back about Z (right)
    s = shin / np.linalg.norm(shin); return math.degrees(math.atan2(-(np.cross(-T[:, 1], s) @ T[:, 2]), -T[:, 1] @ s))
def hip_angles(Pm, Tm):
    z, x, y = Rot.from_matrix(Pm.T @ Tm).as_euler('ZXY', degrees=True)
    return z, -x, -y                                                         # left side: adduction and internal rotation mirrored
rest_hip = hip_angles(P_e, T_e); rest_knee = knee_angle(T_e, ank_c - knee_c)
print('elf rest: hip flex %.1f add %.1f rot %.1f, knee %.1f deg' % (*rest_hip, rest_knee))
# --- ARDY side
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
snames = open(r'E:\figures\ardy\soma77_names.txt').read().split('\n'); si = {n: i for i, n in enumerate(snames)}
# --- solids and muscle chords (as ardy_to_elf_leg.py)
def edges(F): return np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1)
hstep = 2 * float(np.median(edges(FB[np.isin(OB[FB[:, 0]], [hipb, fem, tib])])))
SOL = {b: bone_solid(V, FB[OB[FB[:, 0]] == b], hstep)[:2] for b in (hipb, fem)}
def inside(X, b, Rm=np.eye(3), t=np.zeros(3)): sdb, lob = SOL[b]; return sample(sdb, lob, hstep, (X - t) @ Rm) > 0
REST = dict(femur_in_pelvis=float(inside(Pf[::4], hipb).mean()), tibia_in_femur=float(inside(Ptb[::4], fem).mean()))
cross = json.load(open(os.path.join(FD, 'leg_bend_l.json')))['crossing']; order = {'pelvis': 0, 'thigh': 1, 'shank': 2}; chords = []
for c in cross:
    m = nid[c['name']]; vi = np.flatnonzero((OB == m) & contact)
    if not len(vi): continue
    top = np.array([wb[k] for k in np.asarray(Wt[vi].argmax(1)).ravel()]); g = np.array([group.get(b) for b in top]); segs = sorted({x for x in g if x}, key=order.get)
    if len(segs) < 2 or any(w in c['name'].lower() for w in ('bursa', 'fascia', 'ligament')): continue
    chords.append(dict(name=c['name'], a=segs[0], b=segs[-1], pa=V[vi[g == segs[0]]].mean(0), pb=V[vi[g == segs[-1]]].mean(0)))
L0 = np.array([np.linalg.norm(c['pb'] - c['pa']) for c in chords])
res = {}
for cp in clips:
    C = np.load(cp); G = C['global_rot_mats']; J = C['posed_joints'] @ M.T; T = len(G); nm = os.path.splitext(os.path.basename(cp))[0]
    rel_k = np.einsum('tji,tjk->tik', G[:, si['LeftLeg']], G[:, si['LeftShin']]); rv = Rot.from_matrix(rel_k).as_rotvec()
    kax_s = np.linalg.svd(rv, full_matrices=False)[2][0]                      # hinge axis in the thigh's own (neutral) frame
    rows = []; XF = []
    for t in range(T):
        Pa = M @ G[t, si['Hips']] @ M.T @ P_e
        hz = M @ G[t, si['LeftLeg']] @ kax_s; hz = hz if hz @ (Pa[:, 2]) > 0 else -hz
        Ta = frame(J[t, si['LeftLeg']] - J[t, si['LeftShin']], hz)
        flex, add, rot = hip_angles(Pa, Ta); kn = knee_angle(Ta, J[t, si['LeftFoot']] - J[t, si['LeftShin']])
        # place the elf's femur so its own hip angles equal ARDY's: thigh frame = P_e * (ARDY's pelvis^T thigh)
        Te_t = P_e @ (Pa.T @ Ta); D = Te_t @ T_e.T
        Rk = Rot.from_rotvec(T_e[:, 2] * -math.radians(kn - rest_knee)).as_matrix()      # flexion turns the shin back about Z
        Th = (D, hip_c - D @ hip_c); Tk = (D @ Rk, D @ (knee_c - Rk @ knee_c) + Th[1]); TF = {'pelvis': (np.eye(3), np.zeros(3)), 'thigh': Th, 'shank': Tk}
        # check: the elf's own angles after placement
        ef = hip_angles(P_e, D @ T_e); ek = knee_angle(D @ T_e, Tk[0] @ (ank_c - knee_c))
        imp = float(inside(Pf[::4] @ D.T + Th[1], hipb).mean()); kimp = float(inside(Ptb[::4] @ Tk[0].T + Tk[1], fem, D, Th[1]).mean())
        L = np.array([np.linalg.norm((TF[c['b']][0] @ c['pb'] + TF[c['b']][1]) - (TF[c['a']][0] @ c['pa'] + TF[c['a']][1])) for c in chords])
        XF.append(np.r_[D.ravel(), Th[1], Tk[0].ravel(), Tk[1]])
        rows.append(dict(hip_flex=flex, hip_add=add, hip_rot=rot, knee=kn, elf_check=[ef[0] - flex, ef[1] - add, ef[2] - rot, ek - kn],
                         femur_in_pelvis=imp - REST['femur_in_pelvis'], tibia_in_femur=kimp - REST['tibia_in_femur'], chord=L / L0))
    np.savez(os.path.join(OUT, nm + '_leg_transforms.npz'), thigh_R=np.array([x[:9].reshape(3, 3) for x in XF]), thigh_t=np.array([x[9:12] for x in XF]),
             shank_R=np.array([x[12:21].reshape(3, 3) for x in XF]), shank_t=np.array([x[21:24] for x in XF]), fps=float(C['fps']),
             **{k: np.array([r[k] for r in rows]) for k in ('hip_flex', 'hip_add', 'hip_rot', 'knee')})   # OpenSim coordinates, degrees
    a = {k: np.array([r[k] for r in rows]) for k in ('hip_flex', 'hip_add', 'hip_rot', 'knee')}; ch = np.array([r['chord'] for r in rows])
    over = {k: dict(range=[round(float(v.min()), 1), round(float(v.max()), 1)], limit=[round(LIM[k][0], 1), round(LIM[k][1], 1)],
                    frames_outside=int(((v < LIM[k][0]) | (v > LIM[k][1])).sum()), worst_excess_deg=round(float(max(0, LIM[k][0] - v.min(), v.max() - LIM[k][1])), 1)) for k, v in a.items()}
    res[nm] = dict(frames=T, fps=float(C['fps']), angles=over, mapping_check_max_deg=round(float(np.abs([r['elf_check'] for r in rows]).max()), 3),
                   femur_in_pelvis_above_rest_max=round(max(r['femur_in_pelvis'] for r in rows), 4), tibia_in_femur_above_rest_max=round(max(r['tibia_in_femur'] for r in rows), 4),
                   chord_ratio_range=dict(sorted([(c['name'], [round(float(ch[:, i].min()), 3), round(float(ch[:, i].max()), 3)]) for i, c in enumerate(chords)], key=lambda kv: -(kv[1][1] - kv[1][0]))[:10]),
                   _rows=rows)
    print(nm, json.dumps({k: v for k, v in res[nm].items() if k not in ('_rows', 'chord_ratio_range')}))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axs = plt.subplots(len(clips), 2, figsize=(13, 3.6 * len(clips)), squeeze=False)
col = dict(hip_flex='C0', hip_add='C1', hip_rot='C2', knee='C3')
for i, (nm, r) in enumerate(res.items()):
    rows = r['_rows']; t = np.arange(len(rows)) / r['fps']; a = axs[i, 0]
    for k, lab in (('hip_flex', 'hip flexion'), ('hip_add', 'hip adduction'), ('hip_rot', 'hip internal rotation'), ('knee', 'knee flexion')):
        a.plot(t, [x[k] for x in rows], col[k], label=lab); a.axhspan(*LIM[k], color=col[k], alpha=0.04)
        for lim in LIM[k]: a.axhline(lim, color=col[k], lw=0.5, ls=':')
    a.set_title('%s: calibrated joint angles (dotted: Rajagopal 2016 ranges)' % nm, fontsize=9); a.legend(fontsize=7); a.grid(alpha=.3)
    a = axs[i, 1]; ch = np.array([x['chord'] for x in rows]); top = np.argsort(-(ch.max(0) - ch.min(0)))[:8]
    for j in top: a.plot(t, ch[:, j], label=chords[j]['name'][:28])
    a.axhline(1, c='k', lw=.5); a.set_title('muscle chord length / rest (8 that change most)', fontsize=9); a.legend(fontsize=6); a.grid(alpha=.3)
for a in axs[-1]: a.set_xlabel('time (s)')
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'ardy_to_elf_leg_calibrated.png'), dpi=100)
json.dump(dict(limits_source='Rajagopal 2016 (opensim-models, Rajagopal2016.osim), left-side coordinates', limits=LIM,
               elf_rest=dict(hip=[round(x, 1) for x in rest_hip], knee=round(rest_knee, 1)), rest_overlap=REST, chords=len(chords),
               clips={k: {kk: vv for kk, vv in v.items() if kk != '_rows'} for k, v in res.items()}), open(os.path.join(OUT, 'ardy_to_elf_leg_calibrated.json'), 'w'), indent=1)
print('wrote', OUT)
