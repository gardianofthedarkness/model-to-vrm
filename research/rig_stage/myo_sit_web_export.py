"""Exports myo_sit_activations.py's result for the interactive web viewer: bone meshes (once, in their body frames), each
body's pose per frame, every muscle's path per frame (MuJoCo's own tendon wrapping), activations, foot and chair loads.
  E:\\splat\\.venv-myo\\Scripts\\python.exe myo_sit_web_export.py <myo_sit.npz> <out json>"""
import sys, json, base64, warnings, numpy as np, mujoco, myo_sim
warnings.filterwarnings('ignore')
D = np.load(sys.argv[1]); OUT = sys.argv[2]; Q, A = D['qpos'], D['activation']; fps = float(D['fps']); T0 = int(D['first_frame']); ch = D['chair']
spec = myo_sim.load_spec('myolegs')
jb = spec.body('lumbar5').add_joint(); jb.type = mujoco.mjtJoint.mjJNT_BALL; jb.name = 'trunk_ball'   # as in myo_sit_activations.py
m = spec.compile(); d = mujoco.MjData(m); d.qpos[:] = Q[0]; mujoco.mj_kinematics(m, d)
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
meshes = []
for g in range(m.ngeom):
    if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH or m.geom_bodyid[g] == 0: continue   # bones only (world meshes are the room)
    b, mid = int(m.geom_bodyid[g]), int(m.geom_dataid[g]); Rb, Rg = d.xmat[b].reshape(3, 3), d.geom_xmat[g].reshape(3, 3)
    V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]] @ (Rb.T @ Rg).T + Rb.T @ (d.geom_xpos[g] - d.xpos[b])
    Fc = m.mesh_face[m.mesh_faceadr[mid]:m.mesh_faceadr[mid] + m.mesh_facenum[mid]]
    meshes.append(dict(body=b, v=b64(V.astype(np.float32)), f=b64(Fc.astype(np.uint32))))
bodies, paths = [], []
for t in range(len(Q)):
    d.qpos[:] = Q[t]; mujoco.mj_forward(m, d)
    bodies.append(np.c_[d.xpos, d.xquat].round(5).tolist())
    fr = []
    for k in range(m.ntendon):
        a, n = d.ten_wrapadr[k], d.ten_wrapnum[k]; fr.append(d.wrap_xpos[a:a + n].round(4).tolist())
    paths.append(fr)
names = [m.tendon(k).name for k in range(m.ntendon)]; act_names = [str(x) for x in D['names']]
order = [act_names.index(n.replace('_tendon', '')) if n.replace('_tendon', '') in act_names else -1 for n in names]
bw = float(D['body_weight_N'])
out = dict(fps=fps, t0=T0 / fps, names=act_names, tendon_actuator=order, width=m.tendon_width.round(4).tolist(),
           activation=A.round(3).tolist(), feet_bw=(D['feet_N'] / bw).round(3).tolist(), seat_bw=(D['seat_N'] / bw).round(3).tolist(),
           chair=ch.round(4).tolist(), meshes=meshes, bodies=bodies, paths=paths, body_names=[m.body(i).name for i in range(m.nbody)])
json.dump(out, open(OUT, 'w'), separators=(',', ':')); print('wrote', OUT, 'tendons', m.ntendon, 'matched', sum(o >= 0 for o in order))
