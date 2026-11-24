"""Pictures of myo_sit_activations.py: MyoSuite's myoLegs posed by the ARDY 'sit' clip at a few frames, muscles coloured by
their static-optimisation activation (MuJoCo's own activation colouring), plus activation curves of the left leg's main
muscles and the estimated foot and seat loads.
  E:\\splat\\.venv-myo\\Scripts\\python.exe myo_sit_picture.py <myo_sit.npz> <out png>"""
import sys, warnings, numpy as np, mujoco, myo_sim, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
D = np.load(sys.argv[1]); OUT = sys.argv[2]; Q, A, names, fps = D['qpos'], D['activation'], [str(x) for x in D['names']], float(D['fps'])
spec = myo_sim.load_spec('myolegs'); ch = D['chair']; T0 = int(D['first_frame'])
jb = spec.body('lumbar5').add_joint(); jb.type = mujoco.mjtJoint.mjJNT_BALL; jb.name = 'trunk_ball'   # as in myo_sit_activations.py
g = spec.worldbody.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.pos = ch[:3]; g.size = ch[3:]; g.rgba = [0.55, 0.45, 0.35, 1]
m = spec.compile(); d = mujoco.MjData(m)
picks = [int(x) for x in np.linspace(0, len(Q) - 1, 5).round()]
r = mujoco.Renderer(m, 480, 360); opt = mujoco.MjvOption(); opt.flags[mujoco.mjtVisFlag.mjVIS_ACTIVATION] = True
cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0.0, 0.55]; cam.distance = 2.3; cam.azimuth = 180 + 60; cam.elevation = -8
imgs = []
for t in picks:
    d.qpos[:] = Q[t]; d.act[:] = A[t] if m.na == m.nu else 0; d.ctrl[:] = A[t]; mujoco.mj_forward(m, d)
    r.update_scene(d, cam, opt); imgs.append(r.render().copy())
fig = plt.figure(figsize=(16, 9)); gs = fig.add_gridspec(2, len(picks), height_ratios=[1.15, 1])
for j, (t, im) in enumerate(zip(picks, imgs)):
    ax = fig.add_subplot(gs[0, j]); ax.imshow(im); ax.axis('off'); ax.set_title('t = %.1f s' % ((t + T0) / fps), fontsize=10)
ax = fig.add_subplot(gs[1, :3]); tt = (np.arange(len(A)) + T0) / fps
for n in ('recfem_l', 'vaslat_l', 'vasmed_l', 'vasint_l', 'glmax1_l', 'bflh_l', 'semiten_l', 'tibant_l', 'iliacus_l', 'soleus_l'):
    ax.plot(tt, A[:, names.index(n)], lw=2.2 if n == 'recfem_l' else 1.2, label=n, color='k' if n == 'recfem_l' else None)
for t in picks: ax.axvline((t + T0) / fps, color='0.85', lw=0.8, zorder=0)
ax.set_xlabel('time (s)'); ax.set_ylabel('activation'); ax.set_ylim(0, 1); ax.legend(ncol=5, fontsize=8, loc='upper left'); ax.set_title('Left-leg activations (static optimisation, MyoSuite)', fontsize=10)
ax = fig.add_subplot(gs[1, 3:]); bw = float(D['body_weight_N'])
ax.plot(tt, D['feet_N'] / bw, label='feet'); ax.plot(tt, D['seat_N'] / bw, label='chair'); ax.set_xlabel('time (s)'); ax.set_ylabel('vertical load (body weights)'); ax.legend(fontsize=8)
ax.set_title('Estimated ground and chair reactions', fontsize=10)
fig.suptitle('ARDY "sit" (from the moment both feet are planted) on MyoSuite myoLegs, 80 Hill muscles coloured by activation (red = active)', fontsize=12)
fig.tight_layout(); fig.savefig(OUT, dpi=80); print('wrote', OUT)
