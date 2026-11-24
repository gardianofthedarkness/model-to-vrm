"""One muscle tube in NVIDIA Newton's VBD solver with bone contact and activation (the "heat-shrink tube").
Builds on newton_muscle.py (same tube, glue, material, pin placement).

Bone contact: hip bone, femur and shank bones (tibia, fibula, patella) are kinematic Newton bodies carrying their atlas
meshes, placed each substep with the calibrated ARDY transforms.  Particle radius = the smallest rest gap between a free
muscle node and any bone (so nothing touches at rest); contact stiffness = the stiffness of one element,
(lambda + 2 mu) * h; frictionless (fascia slides on bone).

Activation (active strain, Taber & Perucchio 2000; Nardinocchi & Teresi 2007): F = Fe Fa with
    Fa = (1 - g) f(x)f + (1 - g)^(-1/2) (I - f(x)f),   f = fibre direction (harmonic field), det Fa = 1,
so an active muscle's own rest shape shortens along its fibres and thickens across them; g = 0 gives back the passive
tube.  It is put into Newton by rewriting each tet's rest pose inv(Fa Dm) every substep.  g = a * g_max with
g_max = 1 - (1.27 um / 2.0 um) * lambda_ofl: the sarcomere length where active force vanishes over the start of the
force plateau (Gordon, Huxley & Julian 1966), at Blemker's optimal fibre stretch lambda_ofl = 1.07.
Activation a follows first-order activation dynamics da/dt = (u - a) / tau, tau = 10 ms rising, 40 ms falling
(OpenSim Millard2012 defaults), from an excitation u(t).

  E:\\splat\\.venv-sim\\Scripts\\python.exe newton_muscle_active.py <mesh npz> <leg_transforms.npz> <bones fits dir> <out npz>
env: CONTACT=1 bones collide | HOLD=f hold the bones at clip frame f (isometric test) instead of playing the clip |
     EXC="t0:u0,t1:u1,..." excitation steps in seconds (default 0) | NU Poisson ratio (default 0.45) |
     NSUB, NIT substeps / iterations (10, 10) | DUR seconds for HOLD runs | SAVE_EVERY frames between saved surfaces |
     STATIC=1 zero all velocities after every substep (quasi-static: a sequence of equilibria, no inertia)"""
import sys, os, time, json, numpy as np, warp as wp, newton
from scipy.spatial.transform import Rotation as Rot, Slerp
wp.config.quiet = True
M = np.load(sys.argv[1]); TR = np.load(sys.argv[2]); FD = sys.argv[3]; OUT = sys.argv[4]
env = os.environ.get
NSUB, NIT = int(env('NSUB', 10)), int(env('NIT', 10))
X0, T, glue, phi, srf, fib, h = M['X'], M['T'], M['glue'], M['phi'], M['srf'], M['fib'], float(M['h'])
G_, RHO, NU = 500.0, 1060.0, float(env('NU', 0.45)); mu = G_; lam = 2 * mu * NU / (1 - 2 * NU)
LOFL = 1.07; GMAX = 1 - (1.27 / 2.0) * LOFL; TAU_A, TAU_D = 0.010, 0.040
fps = float(TR['fps']); NT = len(TR['thigh_R']); K0 = int(round(fps))
HOLD = env('HOLD'); HOLD = None if HOLD is None else int(HOLD)
NF = int(round(float(env('DUR', 3.0)) * fps)) if HOLD is not None else min(NT, int(env('MAXF', NT)))
EXC = sorted((float(a), float(b)) for a, b in (p.split(':') for p in env('EXC', '0:0').split(',')))
def excitation(t): return [u for s, u in EXC if s <= t][-1] if t >= EXC[0][0] else 0.0
def seg_tf(sg, f):
    if sg == 0: return np.eye(3), np.zeros(3)
    return (TR['thigh_R'][f], TR['thigh_t'][f]) if sg == 1 else (TR['shank_R'][f], TR['shank_t'][f])
def seg_at(sg, step):
    if step >= K0: return seg_tf(sg, HOLD if HOLD is not None else step - K0)
    Rb, tb = seg_tf(sg, HOLD or 0); u = step / K0; r = Slerp([0, 1], Rot.from_matrix(np.stack([np.eye(3), Rb])))([u]).as_matrix()[0]; return r, u * tb
pins = np.flatnonzero(glue >= 0); free = glue < 0
def targets(step):
    out = np.empty((len(pins), 3))
    for sg in (0, 1, 2):
        m = glue[pins] == sg
        if m.any(): Rb, tb = seg_at(sg, step); out[m] = X0[pins[m]] @ Rb.T + tb
    return out
# ---- bones
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB, FB = R['V'].astype(np.float32), R['obj'], R['F_bone']
nm = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]
BONES = {0: ['Hip bone.l'], 1: ['Femur.l'], 2: ['Tibia.l', 'Fibula.l', 'Patella.l']}
bone_mesh = {}
for sg, names in BONES.items():
    fb = FB[np.isin(OB[FB[:, 0]], [nm.index(n) for n in names])]; u, inv = np.unique(fb, return_inverse=True)
    bone_mesh[sg] = (V[u], inv.reshape(-1, 3).astype(np.int32))
@wp.kernel
def sdist(mid: wp.uint64, P: wp.array(dtype=wp.vec3), d: wp.array(dtype=float)):
    i = wp.tid(); r = wp.mesh_query_point_sign_winding_number(mid, P[i], 1.0)
    if r.result:
        d[i] = r.sign * wp.length(P[i] - wp.mesh_eval_position(mid, r.face, r.u, r.v))
wmesh = {sg: wp.Mesh(points=wp.array(v, dtype=wp.vec3), indices=wp.array(f.ravel(), dtype=int), support_winding_number=True) for sg, (v, f) in bone_mesh.items()}
def bone_dist(q, step_or_frame, is_frame=True):   # signed distance (m) of each node to the nearest bone, bones posed
    best = np.full(len(q), np.inf)
    for sg, m in wmesh.items():
        Rb, tb = seg_tf(sg, step_or_frame) if is_frame else seg_at(sg, step_or_frame)
        d = wp.full(len(q), 1e9, dtype=float); wp.launch(sdist, len(q), [m.id, wp.array(((q - tb) @ Rb).astype(np.float32), dtype=wp.vec3), d])
        best = np.minimum(best, d.numpy())
    return best
d_rest = bone_dist(X0, 0, is_frame=False)
CONTACT = env('CONTACT') == '1'; STATIC = env('STATIC') == '1'
E_ = np.concatenate([T[:, [i, j]] for i in range(4) for j in range(i + 1, 4)]); nxt_glue = np.zeros(len(X0), bool)
for c in (0, 1): nxt_glue[E_[glue[E_[:, c]] >= 0, 1 - c]] = True
away = free & ~nxt_glue                   # free nodes not sharing an edge with a glued node (those sit on the bone by construction)
radius = float(np.percentile(d_rest[away], 1)); KE = (lam + 2 * mu) * h
print('NA rest gap free nodes to bone: min %.2f mm (min; particle radius = 1st percentile, so 1%% start touching); contact ke %.1f N/m; g_max %.3f' % (1000 * radius, KE, GMAX), flush=True)
# ---- model
b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, 0.0)
b.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=X0.tolist(), indices=T.ravel().tolist(),
                density=RHO, k_mu=mu, k_lambda=lam, k_damp=0.0, particle_radius=max(radius, 1e-5))
for i in pins: b.particle_mass[i] = 0.0
body = {}
if CONTACT:
    cfg = newton.ModelBuilder.ShapeConfig(density=0.0, ke=KE, kd=0.0, kf=0.0, mu=0.0)
    for sg, (v, f) in bone_mesh.items():
        body[sg] = b.add_body(xform=wp.transform_identity(), is_kinematic=True, label='bone%d' % sg)
        b.add_shape_mesh(body[sg], mesh=newton.Mesh(v, f.ravel(), compute_inertia=False, is_solid=True), cfg=cfg)
b.color(); model = b.finalize()
if CONTACT: model.soft_contact_ke = KE; model.soft_contact_kd = 0.0; model.soft_contact_kf = 0.0; model.soft_contact_mu = 0.0
sol = newton.solvers.SolverVBD(model=model, iterations=NIT, particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0, s1, ctl = model.state(), model.state(), model.control(); dt = 1.0 / fps / NSUB
pipe = newton.CollisionPipeline(model) if CONTACT else None; contacts = pipe.contacts() if CONTACT else None
def place_bones(st, step):
    if not CONTACT: return
    bq = st.body_q.numpy()
    for sg, bi in body.items():
        Rb, tb = seg_at(sg, step); bq[bi, :3] = tb; bq[bi, 3:] = Rot.from_matrix(Rb).as_quat()
    st.body_q.assign(bq.astype(np.float32)); st.body_qd.zero_()
# ---- active strain
Dm0 = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2); v0 = np.abs(np.linalg.det(Dm0)) / 6; Dinv0 = np.linalg.inv(Dm0)
gphi = np.einsum('tji,tj->ti', Dinv0, np.stack([phi[T[:, 1]] - phi[T[:, 0]], phi[T[:, 2]] - phi[T[:, 0]], phi[T[:, 3]] - phi[T[:, 0]]], 1))
ok = np.linalg.norm(fib, axis=1) > 0; fibv = np.where(ok[:, None], fib, gphi)          # fibre = harmonic-field gradient where the stored one is missing
nrm = np.linalg.norm(fibv, axis=1); good = nrm > 1e-12; ff = np.zeros_like(fibv); ff[good] = fibv[good] / nrm[good, None]
from scipy.spatial import cKDTree; cen = X0[T].mean(1)
if (~good).any(): ff[~good] = ff[good][cKDTree(cen[good]).query(cen[~good])[1]]          # tets with a flat field (inside the glue) take the nearest tet's fibre
print('NA fibres: %d of %d tets taken from the field gradient (%d from the nearest tet); free nodes next to glue %d (rest gap p50 %.2f mm)' % ((~ok).sum(), len(ok), (~good).sum(), (free & nxt_glue).sum(), 1000 * np.median(d_rest[free & nxt_glue])), flush=True)
FF = ff[:, :, None] * ff[:, None, :]; I3 = np.eye(3)[None]
print('NA rest gap of free nodes away from glue, p0/p1/p5/p50 mm:', np.round(1000 * np.percentile(d_rest[away], [0, 1, 5, 50]), 2), flush=True)
if env('PROBE') == '1': sys.exit()
def set_activation(a):
    g = a * GMAX; Fa = (1 - g) * FF + (1 - g) ** -0.5 * (I3 - FF)
    model.tet_poses.assign(np.linalg.inv(Fa @ Dm0).astype(np.float32))
# ---- belly measure: surface nodes in the middle band, mean distance to their centroid (thickness), and middle length
band = np.flatnonzero((np.abs(phi - 0.5) < 0.05) & free); sband = np.intersect1d(band, srf[srf >= 0])
def belly_radius(q): c = q[sband].mean(0); return float(np.linalg.norm(q[sband] - c, axis=1).mean())
r_rest = belly_radius(X0); mid = band
prev = X0[pins].copy(); rows = []; keep = []; kept = []; a = 0.0; t0 = time.time(); SAVE = int(env('SAVE_EVERY', 15))
for step in range(K0 + NF):
    nxt = targets(step)
    for k in range(NSUB):
        tsim = (step - K0 + (k + 1) / NSUB) / fps
        u = excitation(tsim) if step >= K0 else 0.0; a += dt * (u - a) / (TAU_A if u > a else TAU_D); set_activation(a)
        q = s0.particle_q.numpy(); q[pins] = prev + (nxt - prev) * (k + 1) / NSUB; s0.particle_q.assign(q.astype(np.float32))
        place_bones(s0, step); s0.clear_forces()
        if CONTACT: pipe.collide(s0, contacts)
        sol.step(s0, s1, ctl, contacts, dt); s0, s1 = s1, s0
        if STATIC: s0.particle_qd.zero_()                # quasi-static: drop inertia, walk through equilibria
    prev = nxt; f = step - K0
    if f < 0: continue
    q = s0.particle_q.numpy().astype(np.float64)
    if not np.isfinite(q).all(): print('NA frame %d: not finite, stopping' % f, flush=True); break
    Ds = np.stack([q[T[:, 1]] - q[T[:, 0]], q[T[:, 2]] - q[T[:, 0]], q[T[:, 3]] - q[T[:, 0]]], 2); J = np.linalg.det(Ds @ Dinv0)
    fr = HOLD if HOLD is not None else f; dist = bone_dist(q, fr)
    lam_f = np.linalg.norm(np.einsum('tij,tj->ti', Ds @ Dinv0, ff), axis=1)
    rows.append(dict(frame=f, t=f / fps, excitation=excitation(f / fps), activation=a, volume_ratio=float((J * v0).sum() / v0.sum()), inverted=float((J <= 0).mean()),
                     belly_thickness_ratio=belly_radius(q) / r_rest, fibre_stretch_p5_50_95=[float(x) for x in np.percentile(lam_f, [5, 50, 95])],
                     inside_bone_nodes=int((dist[free] < 0).sum()), deepest_mm=float(-1000 * min(0, dist[free].min()))))
    if f % SAVE == 0 or f == NF - 1:
        keep.append(np.where(srf[:, None] >= 0, q[np.maximum(srf, 0)], np.nan).astype(np.float32)); kept.append(f)
        r = rows[-1]; print('NA frame %3d t %.2f s: u %.0f a %.2f | volume x%.3f inverted %.2f%% | belly thickness x%.3f | fibre stretch p5/50/95 %.2f/%.2f/%.2f | inside bone %d nodes, deepest %.1f mm (%.0f s)' % (
            f, r['t'], r['excitation'], r['activation'], r['volume_ratio'], 100 * r['inverted'], r['belly_thickness_ratio'], *r['fibre_stretch_p5_50_95'], r['inside_bone_nodes'], r['deepest_mm'], time.time() - t0), flush=True)
np.savez_compressed(OUT, surf=np.array(keep), F=M['F'], rest=M['rest'].astype(np.float32), frames=np.array(kept), hold=-1 if HOLD is None else HOLD)
json.dump(dict(material=dict(G=G_, nu=NU, density=RHO), contact=dict(on=CONTACT, radius_m=radius, ke=KE), g_max=GMAX, tau=[TAU_A, TAU_D], excitation=EXC,
               substeps=NSUB, iterations=NIT, frames=rows), open(OUT.replace('.npz', '.json'), 'w'), indent=1)
print('NA done in %.0f s' % (time.time() - t0))
