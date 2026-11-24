"""Minimal check of driving pinned (zero-mass) particles in Newton's VBD solver: a block of tets, one end pinned and moved
sideways; reports how far the pinned and the free particles actually went, for two ways of placing the pins."""
import numpy as np, warp as wp, newton
from scipy.spatial import Delaunay
g = np.stack(np.meshgrid(np.linspace(0, 0.1, 5), np.linspace(0, 0.02, 3), np.linspace(0, 0.02, 3), indexing='ij'), -1).reshape(-1, 3)
T = Delaunay(g).simplices; X0 = g
pin = np.flatnonzero(X0[:, 0] > 0.099); fixed = np.flatnonzero(X0[:, 0] < 0.001); pins = np.r_[pin, fixed]
def run(mode):
    b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, 0.0)
    b.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=X0.tolist(), indices=T.ravel().tolist(), density=1000.0, k_mu=1e4, k_lambda=1e5, k_damp=0.0)
    for i in pins: b.particle_mass[i] = 0.0
    b.color(); m = b.finalize(); sol = newton.solvers.SolverVBD(model=m, iterations=10)
    s0, s1, c = m.state(), m.state(), m.control(); dt = 1 / 300
    for step in range(300):
        q = s0.particle_q.numpy(); tgt = X0[pin] + np.array([0, 0.03 * min(1, step / 100), 0]); q[pin] = tgt; q[fixed] = X0[fixed]
        s0.particle_q.assign(q.astype(np.float32))
        if mode == 'both': q1 = s1.particle_q.numpy(); q1[pin] = tgt; q1[fixed] = X0[fixed]; s1.particle_q.assign(q1.astype(np.float32))
        s0.clear_forces(); sol.step(s0, s1, c, None, dt); s0, s1 = s1, s0
    q = s0.particle_q.numpy(); mid = np.flatnonzero(np.abs(X0[:, 0] - 0.05) < 0.001)
    print('%-12s pinned end moved %.1f mm (target 30); middle moved %.1f mm (expect ~15)' % (mode, 1000 * np.linalg.norm(q[pin] - X0[pin], axis=1).mean(), 1000 * np.linalg.norm(q[mid] - X0[mid], axis=1).mean()))
run('input only'); run('both')
