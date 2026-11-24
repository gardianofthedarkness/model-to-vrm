"""Which leg muscles fire, and how hard, while ARDY's character sits down: MyoSuite's myoLegs (80 Hill-type muscles,
Rajagopal 2016 derived, Apache-2.0) driven by the ARDY clip, solved with MyoSuite's own inverse dynamics and static
optimisation (minimum sum of squared activations, OpenSim-style reserves, ground and chair reactions estimated).

Kinematics, all from the clip or from constraints, nothing tuned:
  * hip flexion/adduction/rotation and knee flexion of BOTH legs from ARDY, the same geometric frames as
    ardy_to_elf_leg_calibrated.py (OpenSim intrinsic Z-X-Y hip angles, left side mirrored);
  * pelvis orientation = ARDY's hips rotation without her turn about the vertical (elf axes = MuJoCo world axes:
    x her left, y back, z up); only the frames where ARDY's foot_contacts say both feet are planted (she first turns
    round and steps); the trunk follows ARDY's chest through one added ball joint at L5/S1 (no muscles: its torque is
    reported as a reserve, the work of the back muscles this model lacks);
  * ankle = the angle that keeps each foot flat (toe-heel height difference as in the model's standing keyframe);
  * root position = the one that keeps both heels where the standing keyframe has them (feet planted), then lowered
    until the lowest foot collision shape touches the ground;
  * a chair (world box) whose top is the lowest point the pelvis collision shapes reach in the clip, under their
    footprint at that frame; without it the seated frames have nothing to sit on.  Contacts follow ARDY's labels: feet
    on the floor, pelvis on the seat; the shin and thigh collision shapes are switched off.
  * coordinates low-passed at 6 Hz before differentiation (Winter, Biomechanics and Motor Control of Human Movement).
  E:\\splat\\.venv-myo\\Scripts\\python.exe myo_sit_activations.py <ARDY clip npz> <soma77_names.txt> <out npz>"""
import sys, json, math, warnings, numpy as np, mujoco, myo_sim
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import brentq
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.static_optimization import static_optimization, StaticOptimizationConfig
warnings.filterwarnings('ignore')
CLIP, NAMES, OUT = sys.argv[1:4]
C = np.load(CLIP); G = C['global_rot_mats']; fps = float(C['fps']); NT = len(G)
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float); J = C['posed_joints'] @ M.T
sn = open(NAMES).read().split('\n'); si = {n: i for i, n in enumerate(sn)}
RIGHT = np.array([-1., 0, 0]); P_e = np.c_[[0, -1., 0], [0, 0, 1.], RIGHT]      # pelvis frame columns: anterior, up, right
def frame(Yv, Zv):
    Y = Yv / np.linalg.norm(Yv); Z = Zv - (Zv @ Y) * Y; Z /= np.linalg.norm(Z); return np.c_[np.cross(Y, Z), Y, Z]
def knee_angle(T, shin):
    s = shin / np.linalg.norm(shin); return math.degrees(math.atan2(-(np.cross(-T[:, 1], s) @ T[:, 2]), -T[:, 1] @ s))
ang = {}
for side, S in (('l', 'Left'), ('r', 'Right')):
    rel = np.einsum('tji,tjk->tik', G[:, si[S + 'Leg']], G[:, si[S + 'Shin']]); kax = np.linalg.svd(Rot.from_matrix(rel).as_rotvec(), full_matrices=False)[2][0]
    rows = []
    for t in range(NT):
        Pa = M @ G[t, si['Hips']] @ M.T @ P_e; hz = M @ G[t, si[S + 'Leg']] @ kax; hz = hz if hz @ Pa[:, 2] > 0 else -hz
        Ta = frame(J[t, si[S + 'Leg']] - J[t, si[S + 'Shin']], hz); z, x, y = Rot.from_matrix(Pa.T @ Ta).as_euler('ZXY', degrees=True)
        if side == 'l': x, y = -x, -y
        rows.append((z, x, y, knee_angle(Ta, J[t, si[S + 'Foot']] - J[t, si[S + 'Shin']])))
    ang[side] = np.radians(np.array(rows))
def no_heading(R):                    # remove the turn about the vertical: she faces -y throughout, feet planted
    f = R @ [0, -1., 0]; return Rot.from_euler('z', -math.atan2(f[0], -f[1])).as_matrix() @ R
Rh = np.array([M @ G[t, si['Hips']] @ M.T for t in range(NT)]); Rpel = np.array([no_heading(R) for R in Rh])
Rtrunk = np.array([Rh[t].T @ M @ G[t, si['Chest']] @ M.T for t in range(NT)])     # chest relative to pelvis (whole spine bend)
fc = C['foot_contacts'].astype(bool); T0 = int(np.flatnonzero(~fc.all(1)).max()) + 1 if (~fc.all(1)).any() else 0   # both feet planted from here on (ARDY's labels)
print('MY frames %d..%d: both feet planted per ARDY foot_contacts (before that she turns and steps)' % (T0, NT - 1), flush=True)
def build(chair=None):
    spec = myo_sim.load_spec('myolegs')
    jb = spec.body('lumbar5').add_joint(); jb.type = mujoco.mjtJoint.mjJNT_BALL; jb.name = 'trunk_ball'   # the scaffold's spine is rigid: one ball joint at L5/S1 carries ARDY's spine bend
    if chair is not None:
        g = spec.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = chair[1]; g.pos = chair[0]; g.name = 'chair'
    for g in spec.geoms:                 # contacts as in ARDY's labels: feet on the floor, pelvis on the seat; shins and thighs touch nothing
        if g.parent.name in ('tibia_l', 'tibia_r', 'femur_l', 'femur_r'): g.contype = 0; g.conaffinity = 0
        if g.parent.name == 'world': g.margin = StaticOptimizationConfig().contact_distance   # MuJoCo lists a contact only inside the margin
    m = spec.compile(); return m, mujoco.MjData(m)
m, d = build(); jid = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]
bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
cpl = JointCouplings.from_model(m)
mb = myo_sim.load('myolegs')[0]; key = m.qpos0.copy()          # the 'standing' posture (load_spec has no keyframes), copied joint by joint
for j in range(mb.njnt):
    n = mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_JOINT, j); a, k = mb.jnt_qposadr[j], {0: 7, 1: 4}.get(int(mb.jnt_type[j]), 1); key[jid(n):jid(n) + k] = mb.key_qpos[0][a:a + k]
d.qpos[:] = key; mujoco.mj_kinematics(m, d); S_sac = d.xmat[bid('sacrum')].reshape(3, 3).copy()
heel0 = (d.xpos[bid('calcn_l')] + d.xpos[bid('calcn_r')]) / 2
dz0 = {s: d.xpos[bid('toes_' + s)][2] - d.xpos[bid('calcn_' + s)][2] for s in 'lr'}
foot_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] in [bid(n) for n in ('calcn_l', 'calcn_r', 'toes_l', 'toes_r')] and m.geom_contype[i]]
d.qpos[:] = key; mujoco.mj_forward(m, d)            # ground = the world shape the standing feet touch ('floor' is a backdrop 0.4 m down)
ground = min((i for i in range(m.ngeom) if m.geom_bodyid[i] == 0 and (m.geom_contype[i] or m.geom_conaffinity[i])),
             key=lambda i: min(abs(mujoco.mj_geomDistance(m, d, i, j, 1.0, None)) for j in foot_geoms))
def fk(q): d.qpos[:] = q; mujoco.mj_kinematics(m, d)
def pose(t):
    q = key.copy(); q[3:7] = np.roll(Rot.from_matrix(Rpel[t]).as_quat(), 1); jt = jid('trunk_ball'); q[jt:jt + 4] = np.roll(Rot.from_matrix(S_sac.T @ Rtrunk[t] @ S_sac).as_quat(), 1)
    for s in 'lr':
        a = ang[s][t]; q[jid('hip_flexion_' + s)], q[jid('hip_adduction_' + s)], q[jid('hip_rotation_' + s)], q[jid('knee_angle_' + s)] = a
    q2 = q[None].copy(); project_joint_couplings(cpl, q2); q = q2[0]
    for s in 'lr':                       # ankle keeps the foot flat
        j = jid('ankle_angle_' + s); lo, hi = m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'ankle_angle_' + s)]
        def g(x): q[j] = x; fk(q); return d.xpos[bid('toes_' + s)][2] - d.xpos[bid('calcn_' + s)][2] - dz0[s]
        try: q[j] = brentq(g, lo, hi)
        except ValueError: q[j] = lo if abs(g(lo)) < abs(g(hi)) else hi
    fk(q); q[:3] += heel0 - (d.xpos[bid('calcn_l')] + d.xpos[bid('calcn_r')]) / 2
    d.qpos[:] = q; mujoco.mj_forward(m, d); q[2] -= min(mujoco.mj_geomDistance(m, d, ground, i, 1.0, None) for i in foot_geoms)   # lowest foot shape on the ground
    return q
Q = np.array([pose(t) for t in range(T0, NT)])
# chair: lowest point of pelvis + thigh collision shapes over the clip
low_geoms = [i for i in range(m.ngeom) if m.geom_bodyid[i] == bid('pelvis') and (m.geom_contype[i] or m.geom_conaffinity[i])]
def lowest(q):
    fk(q); mujoco.mj_kinematics(m, d); zs = []
    for i in low_geoms:
        c, R, s = d.geom_xpos[i], d.geom_xmat[i].reshape(3, 3), m.geom_size[i]
        if m.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX: zs.append(c[2] - np.abs(R[2]) @ s)
        elif m.geom_type[i] == mujoco.mjtGeom.mjGEOM_ELLIPSOID: zs.append(c[2] - np.linalg.norm(R[2] * s))
        else: zs.append(c[2] - abs(R[2, 2]) * s[1] - s[0])        # capsule: half-length along local z, radius s[0]
    return min(zs), np.array([d.geom_xpos[i][:2] for i in low_geoms]), np.array([m.geom_rbound[i] for i in low_geoms])
lows = [lowest(q)[0] for q in Q]; print("MY lowest pelvis/thigh point every 10 frames (mm):", [round(1000 * x) for x in lows[::10]], flush=True); ts = int(np.argmin(lows));  z_seat, xy, rb = lowest(Q[ts])
lo_xy, hi_xy = (xy - rb[:, None]).min(0), (xy + rb[:, None]).max(0)
chair = ((*(lo_xy + hi_xy) / 2, z_seat - 0.05), ((hi_xy - lo_xy) / 2).tolist() + [0.05])
print('MY chair top %.3f m at window frame %d (pelvis root height there %.3f m; standing %.3f m)' % (z_seat, ts, Q[ts][2], key[2]), flush=True)
m, d = build(chair)
kin = clip_kinematics(m, Q, 1 / fps, lowpass_hz=6.0)
res = static_optimization(m, kin.qpos, kin.qvel, kin.qacc, config=StaticOptimizationConfig(contact_forces=True))
names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
bw = float(m.body_subtreemass[1] if m.nbody > 1 else 0) * 9.81
cf = res.contact_force; feet = sum(cf[:, bid(n), 2] for n in ('calcn_l', 'calcn_r', 'toes_l', 'toes_r')); seat = cf[:, bid('pelvis'), 2]; other = -cf[:, 0, 2] - feet - seat
dof_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[i]) for i in range(m.nv)]
np.savez_compressed(OUT, first_frame=T0, activation=res.activation, names=np.array(names), qpos=kin.qpos, fps=fps, feet_N=feet, seat_N=seat, other_contact_N=other, reserve=res.reserve,
                    joint_torque=res.joint_torque, dof_names=np.array(dof_names), status=res.status, body_weight_N=bw, chair=np.r_[chair[0], chair[1]])
A = res.activation; top = np.argsort(-A.max(0))[:14]
summary = dict(frames=NT, first_planted_frame=T0, fps=fps, solved=int((res.status == 'solved').sum()), body_weight_N=bw, chair_top_m=float(z_seat), seated_frame=ts,
               feet_load_bw=[float(feet.min() / bw), float(feet.max() / bw)], seat_load_bw=[float(seat.min() / bw), float(seat.max() / bw)], other_contact_bw_max=float(np.abs(other).max() / bw),
               peak_activation={names[i]: round(float(A[:, i].max()), 3) for i in top},
               recfem_l=[round(float(x), 3) for x in A[::15, names.index('recfem_l')]], trunk_ball_reserve_Nm=round(float(np.abs(res.reserve[:, [i for i, n in enumerate(dof_names) if n == 'trunk_ball']]).max()), 1),
               largest_reserve_Nm={dof_names[i]: round(float(np.abs(res.reserve[:, i]).max()), 1) for i in np.argsort(-np.abs(res.reserve).max(0))[:6]})
json.dump(summary, open(OUT.replace('.npz', '.json'), 'w'), indent=1); print(json.dumps(summary, indent=1))
