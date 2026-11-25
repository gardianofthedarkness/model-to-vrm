"""R0 (runtime-plan.md): muscle effort from motion alone, for ANY ARDY clip (walking, jogging, jumping, flight phases).
MyoSuite's myoLegs (80 Hill-type muscles, Rajagopal 2016 derived, Apache-2.0) driven by the clip, solved with MyoSuite's own
inverse dynamics and static optimisation (minimum sum of squared activations, OpenSim-style reserves, ground reactions
estimated from the contacts).  Generalises myo_sit_activations.py: the root is free and follows ARDY, the ankles follow ARDY.

Kinematics, all from the clip or from the models, nothing tuned:
  * hip flexion/adduction/rotation and knee flexion of both legs from ARDY, the geometric frames of
    ardy_to_elf_leg_calibrated.py (OpenSim intrinsic Z-X-Y hip angles, left side mirrored);
  * ankle flexion = the change of the shin-to-foot angle (LeftShin->LeftFoot vs LeftFoot->LeftToeBase) from ARDY's own
    neutral pose (soma77_neutral.npy), added to the model's standing ankle;
  * pelvis orientation = ARDY's hips rotation (heading kept); trunk = ARDY's chest relative to the hips through one ball joint
    at L5/S1 (no muscles there: its torque is a reserve);
  * root position = the standing root plus ARDY's hips displacement scaled by the ratio of leg lengths (myoLegs hip-knee-ankle
    over ARDY's), then corrected vertically frame by frame: where ARDY labels a foot planted (heel, toe base or toe tip),
    the lowest planted foot touches the floor; through flight the correction is interpolated between its neighbours
    (MyoSuite's own advice: put the stance feet on the floor rather than widen the contact reach);
  * the floor: an endless plane at the height of the model's ground platform under its standing feet (the platform is
    finite and a jog leaves it);
  * coordinates low-passed at 6 Hz before differentiation (Winter), as myo_sit_activations.py.
Activation dynamics (R2): the model's own first-order muscle dynamics (MuJoCo, tau_act 10 ms / tau_deact 40 ms in myoLegs):
the drive that would follow the static-optimisation activation is capped to [0, 1] and played forward, so activation cannot
jump faster than a muscle can.
Writes <out>.npz: activation (T x muscles, after activation dynamics; activation_static before), names, qpos, fps, fiber_norm (MuJoCo normalised fibre length per frame), contact
forces, reserves, joint torques; and <out>.json (summary).
  E:\\splat\\.venv-myo\\Scripts\\python.exe r0_clip_activations.py <ARDY clip npz> <soma77 dir> <out npz>"""
import sys, os, json, math, warnings, numpy as np, mujoco, myo_sim
from scipy.spatial.transform import Rotation as Rot
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.static_optimization import static_optimization, StaticOptimizationConfig
warnings.filterwarnings('ignore')
CLIP, SOMA, OUT = sys.argv[1:4]
C = np.load(CLIP); G = C['global_rot_mats']; fps = float(C['fps']); NT = len(G)
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float); J = C['posed_joints'] @ M.T
sn = open(os.path.join(SOMA, 'soma77_names.txt')).read().split('\n'); si = {n: i for i, n in enumerate(sn)}
J0 = np.load(os.path.join(SOMA, 'soma77_neutral.npy')).reshape(-1, 3) @ M.T
RIGHT = np.array([-1., 0, 0]); P_e = np.c_[[0, -1., 0], [0, 0, 1.], RIGHT]
def frame(Yv, Zv):
    Y = Yv / np.linalg.norm(Yv); Z = Zv - (Zv @ Y) * Y; Z /= np.linalg.norm(Z); return np.c_[np.cross(Y, Z), Y, Z]
def knee_angle(T, shin):
    s = shin / np.linalg.norm(shin); return math.degrees(math.atan2(-(np.cross(-T[:, 1], s) @ T[:, 2]), -T[:, 1] @ s))
def foot_angle(Jx, S):
    s = Jx[si[S + 'Foot']] - Jx[si[S + 'Shin']]; f = Jx[si[S + 'ToeBase']] - Jx[si[S + 'Foot']]
    return math.acos(np.clip(s @ f / np.linalg.norm(s) / np.linalg.norm(f), -1, 1))
ang = {}
for side, S in (('l', 'Left'), ('r', 'Right')):
    rel = np.einsum('tji,tjk->tik', G[:, si[S + 'Leg']], G[:, si[S + 'Shin']]); kax = np.linalg.svd(Rot.from_matrix(rel).as_rotvec(), full_matrices=False)[2][0]
    rows = []; a0 = foot_angle(J0, S)
    for t in range(NT):
        Pa = M @ G[t, si['Hips']] @ M.T @ P_e; hz = M @ G[t, si[S + 'Leg']] @ kax; hz = hz if hz @ Pa[:, 2] > 0 else -hz
        Ta = frame(J[t, si[S + 'Leg']] - J[t, si[S + 'Shin']], hz); z, x, y = Rot.from_matrix(Pa.T @ Ta).as_euler('ZXY', degrees=True)
        if side == 'l': x, y = -x, -y
        rows.append((z, x, y, knee_angle(Ta, J[t, si[S + 'Foot']] - J[t, si[S + 'Shin']]), math.degrees(a0 - foot_angle(J[t], S))))
    ang[side] = np.radians(np.array(rows))
Rh = np.array([M @ G[t, si['Hips']] @ M.T for t in range(NT)])
Rtrunk = np.array([Rh[t].T @ M @ G[t, si['Chest']] @ M.T for t in range(NT)])
def build(floor_z=None):
    spec = myo_sim.load_spec('myolegs')
    jb = spec.body('lumbar5').add_joint(); jb.type = mujoco.mjtJoint.mjJNT_BALL; jb.name = 'trunk_ball'
    for g in spec.geoms:
        if g.parent.name in ('tibia_l', 'tibia_r', 'femur_l', 'femur_r'): g.contype = 0; g.conaffinity = 0
        if g.parent.name == 'world':
            g.margin = StaticOptimizationConfig().contact_distance
            if floor_z is not None: g.contype = 0; g.conaffinity = 0     # the model's ground is a finite platform; an endless floor replaces it
    if floor_z is not None:
        g = spec.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.size = [0, 0, 1]; g.pos = [0, 0, floor_z]; g.name = 'r0_floor'
        g.margin = StaticOptimizationConfig().contact_distance
    m = spec.compile(); return m, mujoco.MjData(m)
m, d = build(); jid = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n); cpl = JointCouplings.from_model(m)
mb = myo_sim.load('myolegs')[0]; key = m.qpos0.copy()
for j in range(mb.njnt):
    n = mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_JOINT, j); a, k = mb.jnt_qposadr[j], {0: 7, 1: 4}.get(int(mb.jnt_type[j]), 1); key[jid(n):jid(n) + k] = mb.key_qpos[0][a:a + k]
d.qpos[:] = key; mujoco.mj_kinematics(m, d); S_sac = d.xmat[bid('sacrum')].reshape(3, 3).copy()
leg_myo = np.linalg.norm(d.xpos[bid('femur_l')] - d.xpos[bid('tibia_l')]) + np.linalg.norm(d.xpos[bid('tibia_l')] - d.xpos[bid('talus_l')])
leg_ardy = np.linalg.norm(J0[si['LeftLeg']] - J0[si['LeftShin']]) + np.linalg.norm(J0[si['LeftShin']] - J0[si['LeftFoot']]); scale = leg_myo / leg_ardy
foot_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] in [bid(n) for n in ('calcn_l', 'calcn_r', 'toes_l', 'toes_r')] and m.geom_contype[i]]
d.qpos[:] = key; mujoco.mj_forward(m, d)
ground = min((i for i in range(m.ngeom) if m.geom_bodyid[i] == 0 and (m.geom_contype[i] or m.geom_conaffinity[i])),
             key=lambda i: min(abs(mujoco.mj_geomDistance(m, d, i, j, 1.0, None)) for j in foot_geoms))
fi = min(foot_geoms, key=lambda j: mujoco.mj_geomDistance(m, d, ground, j, 1.0, None)); ft = np.zeros(6)
mujoco.mj_geomDistance(m, d, ground, fi, 1.0, ft); floor_z = float(ft[2])          # nearest point on the platform under the standing feet
m, d = build(floor_z); foot_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] in [bid(n) for n in ('calcn_l', 'calcn_r', 'toes_l', 'toes_r')] and m.geom_contype[i]]
ground = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'r0_floor'); cpl = JointCouplings.from_model(m)
def pose(t):
    q = key.copy(); q[:3] = key[:3] + scale * (J[t, si['Hips']] - J[0, si['Hips']])
    q[3:7] = np.roll(Rot.from_matrix(Rh[t]).as_quat(), 1); jt = jid('trunk_ball'); q[jt:jt + 4] = np.roll(Rot.from_matrix(S_sac.T @ Rtrunk[t] @ S_sac).as_quat(), 1)
    for s in 'lr':
        a = ang[s][t]; q[jid('hip_flexion_' + s)], q[jid('hip_adduction_' + s)], q[jid('hip_rotation_' + s)], q[jid('knee_angle_' + s)] = a[:4]
        j = jid('ankle_angle_' + s); lo, hi = m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'ankle_angle_' + s)]; q[j] = np.clip(key[j] + a[4], lo, hi)
    q2 = q[None].copy(); project_joint_couplings(cpl, q2); return q2[0]
Q = np.array([pose(t) for t in range(NT)])
fcl = C['foot_contacts'].astype(bool); FOOTG = {'l': [i for i in foot_geoms if m.geom_bodyid[i] in (bid('calcn_l'), bid('toes_l'))],
                                            'r': [i for i in foot_geoms if m.geom_bodyid[i] in (bid('calcn_r'), bid('toes_r'))]}
planted = {'l': fcl[:, :3].any(1), 'r': fcl[:, 3:].any(1)}                       # ARDY: heel, toe base, toe tip per foot (SOMA77)
fc = planted['l'] | planted['r']
def gap(q, side): d.qpos[:] = q; mujoco.mj_forward(m, d); return min(mujoco.mj_geomDistance(m, d, ground, i, 1.0, None) for i in FOOTG[side])
# per frame: the lowest planted foot touches the floor; flight frames blend between the neighbouring corrections
corr = np.full(NT, np.nan)
for t in range(NT):
    g = [gap(Q[t], s_) for s_ in 'lr' if planted[s_][t]]
    if g: corr[t] = min(g)
ok = np.isfinite(corr); dzs = np.interp(np.arange(NT), np.flatnonzero(ok), corr[ok]) if ok.any() else np.zeros(NT); Q[:, 2] -= dzs; dz = float(np.median(dzs))
print('MY %d frames at %.0f fps; leg scale %.3f; vertical correction median %.1f mm, range %.1f..%.1f mm; contact frames %d; flight frames %d'
      % (NT, fps, scale, 1e3 * dz, 1e3 * dzs.min(), 1e3 * dzs.max(), fc.sum(), (~fc).sum()), flush=True)
kin = clip_kinematics(m, Q, 1 / fps, lowpass_hz=6.0)
# ground reaction from the motion itself (R2; Ren, Jones & Howard 2008: whole-body inverse dynamics from kinematics only).
# The free root's required generalized force IS the external wrench the motion needs.  It is shared among point forces at
# the heel and toe of each foot ARDY labels planted (heel label -> heel point, toe labels -> toe point), each inside the
# floor's friction pyramid (the foot shapes' own friction), by non-negative least squares on force and moment; flight
# frames get none.  What stays unexplained is the clip's own dynamic inconsistency and is reported, not hidden.
from scipy.optimize import nnls
from myosuite.biomechanics.inverse_dynamics import inverse_dynamics
TAU = inverse_dynamics(m, kin.qpos, kin.qvel, kin.qacc)
gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
PTS = {('l', 'heel'): gid('foot_col1_l'), ('l', 'toe'): gid('bofoot_col1_l'), ('r', 'heel'): gid('foot_col1_r'), ('r', 'toe'): gid('bofoot_col1_r')}
CARRY_B = {g_: bid('calcn_' + sd) for (sd, _), g_ in PTS.items()}   # the forefoot pad (metatarsal heads) belongs to the foot segment, not the toes
mu = float(m.geom_friction[gid('foot_col1_l'), 0]); E4 = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]]); E4 /= np.linalg.norm(E4, axis=1, keepdims=True)
EXT = np.zeros_like(TAU); GRF = np.zeros((len(TAU), 3)); unexpl = np.zeros(len(TAU)); dd = mujoco.MjData(m); fcl_k = fcl[:len(TAU)]
for t in range(len(TAU)):
    dd.qpos[:] = kin.qpos[t]; mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
    Fw = TAU[t, :3]; Rr = dd.xmat[bid('pelvis')].reshape(3, 3) if False else Rot.from_quat(np.roll(kin.qpos[t, 3:7], -1)).as_matrix(); Mw = Rr @ TAU[t, 3:6]; p = kin.qpos[t, :3]
    act = []
    for side, cols in (('l', (0, 1, 2)), ('r', (3, 4, 5))):
        if fcl_k[t, cols[0]]: act.append(PTS[(side, 'heel')])
        if fcl_k[t, cols[1]] or fcl_k[t, cols[2]]: act.append(PTS[(side, 'toe')])
    if not act: unexpl[t] = np.linalg.norm(Fw); continue
    cols_ = []
    for g_ in act:
        c = dd.geom_xpos[g_]
        for e in E4: cols_.append(np.r_[e, np.cross(c - p, e)])
    A_ = np.array(cols_).T; Lr = float(np.linalg.norm(dd.geom_xpos[act[0]] - p))         # moment rows scaled by the lever length
    W = np.r_[np.ones(3), np.ones(3) / Lr]; f, _ = nnls(A_ * W[:, None], np.r_[Fw, Mw] * W)
    k = 0
    for g_ in act:
        F_ = (E4 * f[k:k + 4, None]).sum(0); k += 4; GRF[t] += F_
        qf = np.zeros(m.nv); mujoco.mj_applyFT(m, dd, F_, np.zeros(3), dd.geom_xpos[g_].copy(), CARRY_B[g_], qf); EXT[t] += qf
    unexpl[t] = np.linalg.norm(Fw - GRF[t])
res = static_optimization(m, kin.qpos, kin.qvel, kin.qacc, config=StaticOptimizationConfig(contact_forces=False), external_qfrc=EXT)
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
FN = np.zeros((len(kin.qpos), m.nu))
lr, rg = m.actuator_lengthrange, m.actuator_gainprm[:, :2]                       # MuJoCo muscle: length range -> fibre range
for t, q in enumerate(kin.qpos):
    d.qpos[:] = q; mujoco.mj_forward(m, d); FN[t] = rg[:, 0] + (d.actuator_length - lr[:, 0]) / (lr[:, 1] - lr[:, 0]) * (rg[:, 1] - rg[:, 0])
d.qpos[:] = key; mujoco.mj_forward(m, d); FN_key = rg[:, 0] + (d.actuator_length - lr[:, 0]) / (lr[:, 1] - lr[:, 0]) * (rg[:, 1] - rg[:, 0])
bw = float(m.body_subtreemass[1]) * 9.81; cf = res.contact_force
feet = GRF[:, 2]
dof_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[i]) for i in range(m.nv)]
# activation dynamics: the model's own (MuJoCo muscle, tau_act / tau_deact from actuator_dynprm).  The drive u each muscle would
# need to follow the static-optimisation activation, u = a + tau(u, a) da/dt, is capped to [0, 1] and played forward at 1 kHz,
# so only changes no muscle can make within a frame are smoothed.
A_req = res.activation; dt_f = 1 / fps; ta_ = np.arange(len(A_req)) * dt_f; dA = np.gradient(A_req, dt_f, axis=0)
tA, tD = m.actuator_dynprm[:, 0], m.actuator_dynprm[:, 1]
tau = np.where(dA > 0, tA * (0.5 + 1.5 * A_req), tD / (0.5 + 1.5 * A_req)); U = np.clip(A_req + tau * dA, 0, 1)
h = 1e-3; tt_ = np.arange(0, ta_[-1] + 1e-9, h); Ui = np.stack([np.interp(tt_, ta_, U[:, k]) for k in range(m.nu)], 1)
a = A_req[0].copy(); Adyn = np.zeros_like(A_req); fr = 0
for i, t in enumerate(tt_):
    if fr < len(ta_) and t >= ta_[fr] - 1e-12: Adyn[fr] = a; fr += 1
    a = np.clip(a + h * np.array([mujoco.mju_muscleDynamics(Ui[i, k], a[k], m.actuator_dynprm[k, :3]) for k in range(m.nu)]), 0, 1)
np.savez_compressed(OUT, activation=Adyn, activation_static=A_req, drive=U, ankle_l=ang['l'][:, 4], names=np.array(names), qpos=kin.qpos, fps=fps, fiber_norm=FN, fiber_norm_standing=FN_key, feet_N=feet,
                    reserve=res.reserve, grf=GRF, unexplained_N=unexpl, joint_torque=res.joint_torque, dof_names=np.array(dof_names), status=res.status, body_weight_N=bw, foot_contact=fcl)
A = Adyn; top = np.argsort(-A.max(0))[:14]
summary = dict(frames=NT, fps=fps, unexplained_force_bw=dict(stance_median=float(np.median(unexpl[fc[:len(unexpl)]]) / (float(m.body_subtreemass[1]) * 9.81)), flight_median=float(np.median(unexpl[~fc[:len(unexpl)]]) / (float(m.body_subtreemass[1]) * 9.81)) if (~fc).any() else None), tau_act_deact_s=[float(tA[0]), float(tD[0])], frames_where_drive_capped=int(((A_req + tau * dA > 1) | (A_req + tau * dA < 0)).any(1).sum()), solved=int((res.status == 'solved').sum()), leg_scale=scale, vertical_shift_m=dz, body_weight_N=bw,
               feet_load_bw=[float(feet.min() / bw), float(feet.max() / bw)], peak_activation={names[i]: round(float(A[:, i].max()), 3) for i in top},
               largest_reserve_Nm={dof_names[i]: round(float(np.abs(res.reserve[:, i]).max()), 1) for i in np.argsort(-np.abs(res.reserve).max(0))[:6]})
json.dump(summary, open(OUT.replace('.npz', '.json'), 'w'), indent=1); print(json.dumps(summary, indent=1))
