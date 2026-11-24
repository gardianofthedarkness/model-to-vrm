"""One muscle tube in NVIDIA Newton's VBD solver (neo-Hookean tets), ends glued to the bones, driven by the calibrated ARDY
clip.  Same tube and glue as muscle_sim.py (its MESH_ONLY output); Newton has no fibre direction, so the muscle is the
isotropic part of the cited model: shear modulus G = 500 Pa, bulk modulus K = 1 MPa (Blemker's model as in FEBio's example),
density 1060 kg/m^3 (muscle, textbook).  No gravity (the rest shape already stands under it).
  E:\\splat\\.venv-sim\\Scripts\\python.exe newton_muscle.py <mesh npz> <leg_transforms.npz> <out npz> [substeps] [iterations]
Glued nodes are zero-mass particles placed each substep (the placement checked in newton_pin_test.py), moving rigidly with
their bone segment; the clip is eased in from rest over one second.  Reports per frame: volume ratio, inverted tets,
how far the muscle's middle moved, and the share of glued-to-free edges stretched beyond twice their rest length."""
import sys, time, json, numpy as np, warp as wp, newton
from scipy.spatial.transform import Rotation as Rot, Slerp
M = np.load(sys.argv[1]); TR = np.load(sys.argv[2]); OUT = sys.argv[3]
NSUB = int(sys.argv[4]) if len(sys.argv) > 4 else 10; NIT = int(sys.argv[5]) if len(sys.argv) > 5 else 10   # Newton example defaults
X0, T, glue, phi, srf = M['X'], M['T'], M['glue'], M['phi'], M['srf']
G_, K_, RHO = 500.0, 1e6, 1060.0; mu = G_; lam = K_ - 2 * G_ / 3
import os
if os.environ.get("NU"): NU = float(os.environ["NU"]); lam = 2 * mu * NU / (1 - 2 * NU); K_ = lam + 2 * mu / 3   # Poisson ratio instead of the penalty bulk modulus
fps = float(TR['fps']); NT = len(TR['thigh_R']); K0 = int(round(fps))
def seg_tf(sg, f):
    if sg == 0: return np.eye(3), np.zeros(3)
    return (TR['thigh_R'][f], TR['thigh_t'][f]) if sg == 1 else (TR['shank_R'][f], TR['shank_t'][f])
def seg_at(sg, step):          # step < K0: eased from rest towards frame 0; then clip frame step - K0
    if step >= K0: return seg_tf(sg, step - K0)
    Rb, tb = seg_tf(sg, 0); u = step / K0; r = Slerp([0, 1], Rot.from_matrix(np.stack([np.eye(3), Rb])))([u]).as_matrix()[0]; return r, u * tb
pins = np.flatnonzero(glue >= 0)
def targets(step):
    out = np.empty((len(pins), 3))
    for sg in (0, 1, 2):
        m = glue[pins] == sg
        if m.any(): Rb, tb = seg_at(sg, step); out[m] = X0[pins[m]] @ Rb.T + tb
    return out
b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, 0.0)
b.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=X0.tolist(), indices=T.ravel().tolist(),
                density=RHO, k_mu=mu, k_lambda=lam, k_damp=0.0)
for i in pins: b.particle_mass[i] = 0.0
b.color(); model = b.finalize(); sol = newton.solvers.SolverVBD(model=model, iterations=NIT, particle_enable_self_contact=False, particle_enable_tile_solve=False)
s0, s1, ctl = model.state(), model.state(), model.control(); dt = 1.0 / fps / NSUB
Dm0 = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2); v0 = np.abs(np.linalg.det(Dm0)) / 6; Dinv0 = np.linalg.inv(Dm0)
mid = np.flatnonzero(np.abs(phi - 0.5) < 0.05)
E_ = np.concatenate([T[:, [a, c]] for a in range(4) for c in range(a + 1, 4)]); cross = E_[(glue[E_[:, 0]] >= 0) != (glue[E_[:, 1]] >= 0)]
l0 = np.linalg.norm(X0[cross[:, 0]] - X0[cross[:, 1]], axis=1)
prev = X0[pins].copy(); rows = []; keep = []; kept_frames = []; t0 = time.time()
MAXF = int(__import__("os").environ.get("MAXF", NT))
for step in range(K0 + min(NT, MAXF)):
    nxt = targets(step)
    for k in range(NSUB):
        q = s0.particle_q.numpy(); q[pins] = prev + (nxt - prev) * (k + 1) / NSUB; s0.particle_q.assign(q.astype(np.float32))
        s0.clear_forces(); sol.step(s0, s1, ctl, None, dt); s0, s1 = s1, s0
    prev = nxt; f = step - K0
    if f < 0: continue
    q = s0.particle_q.numpy().astype(np.float64)
    if not np.isfinite(q).all(): print('NM frame %d: not finite, stopping' % f, flush=True); break
    Ds = np.stack([q[T[:, 1]] - q[T[:, 0]], q[T[:, 2]] - q[T[:, 0]], q[T[:, 3]] - q[T[:, 0]]], 2); J = np.linalg.det(Ds @ Dinv0)
    lc = np.linalg.norm(q[cross[:, 0]] - q[cross[:, 1]], axis=1) / l0
    Rb, tb = seg_tf(1, f); thigh_mid = X0[mid] @ Rb.T + tb                       # where the middle would be if it rode the femur
    rows.append(dict(frame=f, volume_ratio=float((J * v0).sum() / v0.sum()), inverted=float((J <= 0).mean()), J_p1_p99=[float(np.percentile(J, 1)), float(np.percentile(J, 99))],
                     middle_moved_mm=1000 * float(np.linalg.norm(q[mid] - X0[mid], axis=1).mean()), middle_vs_femur_mm=1000 * float(np.linalg.norm(q[mid] - thigh_mid, axis=1).mean()),
                     glue_edges_over_2x=float((lc > 2).mean())))
    if f % 15 == 0:
        keep.append(np.where(srf[:, None] >= 0, q[np.maximum(srf, 0)], np.nan).astype(np.float32)); kept_frames.append(f)
        r = rows[-1]; print('NM frame %3d: volume x%.3f, inverted %.2f%%, J p1..p99 %.2f..%.2f | muscle middle moved %.0f mm (%.0f mm from riding the femur) | glue edges > 2x %.1f%%  (%.0f s)' % (
            f, r['volume_ratio'], 100 * r['inverted'], *r['J_p1_p99'], r['middle_moved_mm'], r['middle_vs_femur_mm'], 100 * r['glue_edges_over_2x'], time.time() - t0), flush=True)
np.savez_compressed(OUT, surf=np.array(keep), F=M['F'], rest=M['rest'].astype(np.float32), frames=np.array(kept_frames))
json.dump(dict(material=dict(G=G_, K=K_, density=RHO), substeps=NSUB, iterations=NIT, frames=rows), open(OUT.replace('.npz', '.json'), 'w'), indent=1)
print('NM done in %.0f s' % (time.time() - t0))
