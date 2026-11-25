"""R2 core, shared by the ARDY pipeline (r0_clip_activations.py) and the validation against measured running data
(r2_validate.py), so the test runs exactly the code the pipeline uses.

grf_from_motion   ground reaction from the motion itself (Ren, Jones & Howard 2008: whole-body inverse dynamics from
                  kinematics only).  The free root's required generalized force IS the external wrench the motion needs.
                  It is shared among point forces at the heel and toe of each planted foot (labels: heel -> heel point, toe
                  labels -> toe point), each inside the floor's friction pyramid (the foot shapes' own friction), by
                  non-negative least squares on force and moment; frames with no planted foot get none.  The forefoot pad
                  (metatarsal heads) is carried by the foot segment, not the toes.  What stays unexplained is reported.
muscle_effort     MyoSuite static optimisation with that ground reaction as a known load, then the model's own activation
                  dynamics (MuJoCo muscle, tau_act / tau_deact from actuator_dynprm): the drive each muscle would need to
                  follow the static-optimisation activation, u = a + tau(u, a) da/dt, is capped to [0, 1] and played forward
                  at 1 kHz, so only changes no muscle can make within a frame are smoothed."""
import numpy as np, mujoco
from scipy.optimize import nnls
from scipy.spatial.transform import Rotation as Rot
from myosuite.biomechanics.inverse_dynamics import inverse_dynamics
from myosuite.biomechanics.static_optimization import static_optimization, StaticOptimizationConfig


def grf_from_motion(m, kin, labels):
    """labels: (T, 6) bool, ARDY order [L heel, L toe base, L toe tip, R heel, R toe base, R toe tip].
    Returns external generalized forces (T, nv), ground reaction (T, 3) world, unexplained force norm (T,)."""
    bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
    gid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, n)
    TAU = inverse_dynamics(m, kin.qpos, kin.qvel, kin.qacc)
    PTS = {('l', 'heel'): gid('foot_col1_l'), ('l', 'toe'): gid('bofoot_col1_l'), ('r', 'heel'): gid('foot_col1_r'), ('r', 'toe'): gid('bofoot_col1_r')}
    CARRY_B = {g_: bid('calcn_' + sd) for (sd, _), g_ in PTS.items()}
    mu = float(m.geom_friction[gid('foot_col1_l'), 0]); E4 = np.array([[mu, 0, 1], [-mu, 0, 1], [0, mu, 1], [0, -mu, 1]]); E4 /= np.linalg.norm(E4, axis=1, keepdims=True)
    EXT = np.zeros_like(TAU); GRF = np.zeros((len(TAU), 3)); unexpl = np.zeros(len(TAU)); dd = mujoco.MjData(m)
    for t in range(len(TAU)):
        dd.qpos[:] = kin.qpos[t]; mujoco.mj_kinematics(m, dd); mujoco.mj_comPos(m, dd)
        Fw = TAU[t, :3]; Rr = Rot.from_quat(np.roll(kin.qpos[t, 3:7], -1)).as_matrix(); Mw = Rr @ TAU[t, 3:6]; p = kin.qpos[t, :3]
        act = []
        for side, cols in (('l', (0, 1, 2)), ('r', (3, 4, 5))):
            if labels[t, cols[0]]: act.append(PTS[(side, 'heel')])
            if labels[t, cols[1]] or labels[t, cols[2]]: act.append(PTS[(side, 'toe')])
        if not act: unexpl[t] = np.linalg.norm(Fw); continue
        cols_ = []
        for g_ in act:
            c = dd.geom_xpos[g_]
            for e in E4: cols_.append(np.r_[e, np.cross(c - p, e)])
        A_ = np.array(cols_).T; Lr = float(np.linalg.norm(dd.geom_xpos[act[0]] - p))      # moment rows scaled by the lever length
        W = np.r_[np.ones(3), np.ones(3) / Lr]; f, _ = nnls(A_ * W[:, None], np.r_[Fw, Mw] * W)
        k = 0
        for g_ in act:
            F_ = (E4 * f[k:k + 4, None]).sum(0); k += 4; GRF[t] += F_
            qf = np.zeros(m.nv); mujoco.mj_applyFT(m, dd, F_, np.zeros(3), dd.geom_xpos[g_].copy(), CARRY_B[g_], qf); EXT[t] += qf
        unexpl[t] = np.linalg.norm(Fw - GRF[t])
    return EXT, GRF, unexpl


def activation_dynamics(m, A_req, fps):
    """the model's own first-order muscle dynamics applied to the static-optimisation activation; returns (Adyn, drive)"""
    dt_f = 1 / fps; ta_ = np.arange(len(A_req)) * dt_f; dA = np.gradient(A_req, dt_f, axis=0)
    tA, tD = m.actuator_dynprm[:, 0], m.actuator_dynprm[:, 1]
    tau = np.where(dA > 0, tA * (0.5 + 1.5 * A_req), tD / (0.5 + 1.5 * A_req)); U = np.clip(A_req + tau * dA, 0, 1)
    h = 1e-3; tt_ = np.arange(0, ta_[-1] + 1e-9, h); Ui = np.stack([np.interp(tt_, ta_, U[:, k]) for k in range(m.nu)], 1)
    a = A_req[0].copy(); Adyn = np.zeros_like(A_req); fr = 0
    for i, t in enumerate(tt_):
        if fr < len(ta_) and t >= ta_[fr] - 1e-12: Adyn[fr] = a; fr += 1
        a = np.clip(a + h * np.array([mujoco.mju_muscleDynamics(Ui[i, k], a[k], m.actuator_dynprm[k, :3]) for k in range(m.nu)]), 0, 1)
    while fr < len(ta_): Adyn[fr] = a; fr += 1
    return Adyn, U


def muscle_effort(m, kin, labels, fps):
    """full R2 effort step: ground reaction, static optimisation with it as a known load, activation dynamics"""
    EXT, GRF, unexpl = grf_from_motion(m, kin, labels)
    res = static_optimization(m, kin.qpos, kin.qvel, kin.qacc, config=StaticOptimizationConfig(contact_forces=False), external_qfrc=EXT)
    Adyn, U = activation_dynamics(m, res.activation, fps)
    return dict(res=res, GRF=GRF, unexplained=unexpl, activation=Adyn, drive=U)
