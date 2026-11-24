"""Same bones, different load: the left leg in ONE pose, either carrying the whole body (single-leg stance) or hanging free
(the other foot carries everything).  MyoSuite myoLegs, the pose of myo_sit_activations.py's first frame (ARDY sit clip,
both feet planted, knee ~20 deg; root already lowered onto the ground).  The two cases differ only in which foot may push:
  loaded   only the LEFT foot's collision shapes may touch the ground
  hanging  only the RIGHT foot's
Static pose (zero velocity and acceleration), static optimisation as in myo_sit_activations.py.  Balance: the body's centre
of mass is not over a single foot in this pose, so the root residual (MyoSuite's reserve on the free joint) carries the
moment a real person would remove by shifting over the stance foot; it is reported, not hidden.
  E:\\splat\\.venv-myo\\Scripts\\python.exe myo_single_leg.py <myo_sit.npz> <out json>"""
import sys, json, warnings, numpy as np, mujoco, myo_sim
from myosuite.biomechanics.static_optimization import static_optimization, StaticOptimizationConfig
warnings.filterwarnings('ignore')
D = np.load(sys.argv[1]); OUT = sys.argv[2]; q0 = D['qpos'][0]; T0 = int(D['first_frame'])
def build(stance):
    spec = myo_sim.load_spec('myolegs')
    jb = spec.body('lumbar5').add_joint(); jb.type = mujoco.mjtJoint.mjJNT_BALL; jb.name = 'trunk_ball'
    for g in spec.geoms:
        if g.parent.name in ('tibia_l', 'tibia_r', 'femur_l', 'femur_r'): g.contype = 0; g.conaffinity = 0
        if g.parent.name in ('calcn_l', 'toes_l', 'calcn_r', 'toes_r') and not g.parent.name.endswith('_' + stance): g.contype = 0; g.conaffinity = 0
        if g.parent.name == 'world': g.margin = StaticOptimizationConfig().contact_distance
    m = spec.compile(); return m
MAP = {'Rectus femoris muscle.l': 'recfem_l', 'Vastus lateralis muscle.l': 'vaslat_l', 'Vastus medialis muscle.l': 'vasmed_l', 'Long head of biceps femoris.l': 'bflh_l',
       'Semitendinosus muscle.l': 'semiten_l', 'Semimembranosus muscle.l': 'semimem_l', 'Gracilis muscle.l': 'grac_l', 'Sartorius muscle.l': 'sart_l'}
out = dict(clip_frame=T0, pose_note='myo_sit.npz qpos[0] = ARDY sit clip frame %d' % T0, cases={})
for case, stance in (('loaded', 'l'), ('hanging', 'r')):
    m = build(stance); n = 5; Q = np.repeat(q0[None], n, 0); Z = np.zeros((n, m.nv))
    res = static_optimization(m, Q, Z, Z, config=StaticOptimizationConfig(contact_forces=True))
    names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
    bid = lambda s: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, s); bw = float(m.body_subtreemass[1]) * 9.81; cf = res.contact_force[0]
    dof = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.dof_jntid[i]) for i in range(m.nv)]; rv = res.reserve[0]
    A = res.activation[0]; top = np.argsort(-A)[:12]
    out['cases'][case] = dict(status=str(res.status[0]), left_foot_bw=float(sum(cf[bid(s), 2] for s in ('calcn_l', 'toes_l')) / bw), right_foot_bw=float(sum(cf[bid(s), 2] for s in ('calcn_r', 'toes_r')) / bw),
                              activation={k: round(float(A[names.index(v)]), 4) for k, v in MAP.items()}, all_left_top={names[i]: round(float(A[i]), 3) for i in top if names[i].endswith('_l')},
                              largest_reserves_Nm_or_N={dof[i]: round(float(rv[i]), 1) for i in np.argsort(-np.abs(rv))[:6]})
    c = out['cases'][case]; print('ML %-8s %s: left foot %.2f bw, right foot %.2f bw | reserves %s' % (case, c['status'], c['left_foot_bw'], c['right_foot_bw'], c['largest_reserves_Nm_or_N']), flush=True)
    print('ML   thigh muscles:', c['activation'], flush=True)
json.dump(out, open(OUT, 'w'), indent=1); print('wrote', OUT)
