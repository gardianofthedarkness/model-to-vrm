"""Sign test of the active-strain muscle (newton_muscle_active.py) on a straight block, before any anatomy.

The block has the rectus tube's own length and belly cross-section area, its tet edge h, material (G 500 Pa, nu 0.45, 1060
kg/m^3), solver settings (Newton VBD, 10 substeps x 10 iterations at 30 fps) and the same active strain
    Fa = (1 - g) f(x)f + (1 - g)^(-1/2) (I - f(x)f),   g = a * g_max,  f = block axis,
put into Newton the same way (rest poses inv(Fa Dm) rewritten every substep).  No gravity, no contact.

  case free   one end face fixed, the other free.  Expected at activation a: length x (1 - g), cross-section sides
              x (1 - g)^(-1/2) away from the fixed face, straight.
  case fixed  both end faces fixed at rest length (isometric).  Expected: stays straight, and the fibre stress is TENSILE
              (the muscle pulls on its attachments).  A compressive fibre stress here would mean the sign convention
              is wrong and would explain buckling.
Stress read-out: Cauchy fibre stress f.sigma.f of the elastic part Fe = F Fa^-1 under Newton's stable neo-Hookean
energy (Smith, de Goes & Kim 2018: Psi = mu/2 (I_C - 3) + lambda/2 (J - alpha)^2, alpha = 1 + mu/lambda).
  E:\\splat\\.venv-sim\\Scripts\\python.exe muscle_block_test.py <rectus mesh npz> <out json>"""
import sys, json, time, numpy as np, warp as wp, newton
wp.config.quiet = True
M = np.load(sys.argv[1]); OUT = sys.argv[2]
X0r, Tr, phi, h = M['X'], M['T'], M['phi'], float(M['h'])
# ---- the rectus's own size: length along its origin->insertion field, belly cross-section area (mid band, convex hull)
from scipy.spatial import ConvexHull
ends = [X0r[phi < 0.02].mean(0), X0r[phi > 0.98].mean(0)]; L = float(np.linalg.norm(ends[1] - ends[0])); ax = (ends[1] - ends[0]) / L
band = X0r[np.abs(phi - 0.5) < 0.05]; e1 = np.cross(ax, [0, 0, 1.0]); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
area = ConvexHull(np.c_[band @ e1, band @ e2]).volume; W = float(np.sqrt(area))     # square of the same area
nx, nw = max(2, int(round(L / h))), max(2, int(round(W / h)))
print('MB rectus chord length %.1f mm, belly area %.0f mm^2 -> block %.1f x %.1f x %.1f mm, grid %d x %d x %d (h %.1f mm)' % (1000 * L, 1e6 * area, 1000 * L, 1000 * W, 1000 * W, nx, nw, nw, 1000 * h), flush=True)
# ---- regular grid, 6 tets per cell (Kuhn split), every tet with positive signed volume (Newton drops the others)
gx, gy, gz = np.meshgrid(np.linspace(0, L, nx + 1), np.linspace(-W / 2, W / 2, nw + 1), np.linspace(-W / 2, W / 2, nw + 1), indexing='ij')
X0 = np.c_[gx.ravel(), gy.ravel(), gz.ravel()]; idx = lambda i, j, k: (i * (nw + 1) + j) * (nw + 1) + k
T = []
for i in range(nx):
    for j in range(nw):
        for k in range(nw):
            c = [idx(i + a, j + b, k + d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]   # c[4a+2b+d]
            for p in ([0, 4, 6, 7], [0, 4, 5, 7], [0, 2, 6, 7], [0, 2, 3, 7], [0, 1, 5, 7], [0, 1, 3, 7]): T.append([c[q] for q in p])
T = np.array(T); Dm = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2)
neg = np.linalg.det(Dm) < 0; T[neg] = T[neg][:, [0, 1, 3, 2]]
Dm0 = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2); Dinv0 = np.linalg.inv(Dm0); v0 = np.linalg.det(Dm0) / 6
assert (v0 > 0).all()
G_, RHO, NU = 500.0, 1060.0, 0.45; mu = G_; lam = 2 * mu * NU / (1 - 2 * NU); alpha = 1 + mu / lam
LOFL = 1.07; GMAX = 1 - (1.27 / 2.0) * LOFL; TAU_A, TAU_D = 0.010, 0.040; fps, NSUB, NIT = 30.0, 10, 10; dt = 1 / fps / NSUB
f = np.array([1.0, 0, 0]); FF = np.outer(f, f)
import os; STATIC = os.environ.get('STATIC') == '1'
x0end, x1end = X0[:, 0] < 1e-9, X0[:, 0] > L - 1e-9
mid = np.abs(X0[:, 0] - (nx // 2) * L / nx) < 1e-9 * L + 1e-12   # mid cross-section nodes (grid slice nx // 2)
def run(case, a_target, secs=3.0):
    pins = np.flatnonzero(x0end | (x1end if case == 'fixed' else False))
    b = newton.ModelBuilder(); b.gravity = (0.0, 0.0, 0.0)
    b.add_soft_mesh(pos=wp.vec3(0, 0, 0), rot=wp.quat_identity(), scale=1.0, vel=wp.vec3(0, 0, 0), vertices=X0.tolist(), indices=T.ravel().tolist(),
                    density=RHO, k_mu=mu, k_lambda=lam, k_damp=0.0)
    for i in pins: b.particle_mass[i] = 0.0
    b.color(); model = b.finalize(); assert model.tet_count == len(T), (model.tet_count, len(T))
    sol = newton.solvers.SolverVBD(model=model, iterations=NIT, particle_enable_self_contact=False, particle_enable_tile_solve=False)
    s0, s1, ctl = model.state(), model.state(), model.control(); a = 0.0; hist = []
    for step in range(int(secs * fps)):
        for k in range(NSUB):
            a += dt * (a_target - a) / (TAU_A if a_target > a else TAU_D); g = a * GMAX
            Fa = (1 - g) * FF + (1 - g) ** -0.5 * (np.eye(3) - FF); model.tet_poses.assign(np.linalg.inv(Fa @ Dm0).astype(np.float32))
            s0.clear_forces(); sol.step(s0, s1, ctl, None, dt); s0, s1 = s1, s0
            if STATIC: s0.particle_qd.zero_()            # quasi-static: drop inertia, walk through equilibria
        q = s0.particle_q.numpy().astype(np.float64)
        Fe = np.stack([q[T[:, 1]] - q[T[:, 0]], q[T[:, 2]] - q[T[:, 0]], q[T[:, 3]] - q[T[:, 0]]], 2) @ Dinv0 @ np.linalg.inv(Fa)
        J = np.linalg.det(Fe); P = mu * Fe + lam * (J - alpha)[:, None, None] * J[:, None, None] * np.linalg.inv(Fe).transpose(0, 2, 1)
        sig = np.einsum('tij,tkj->tik', P, Fe) / J[:, None, None]; s_ff = np.einsum('i,tij,j->t', f, sig, f)
        # centreline: centroid of each grid slice; bend = its largest distance from the line through the two end centroids
        slc = np.round(X0[:, 0] / (L / nx)).astype(int); cen = np.array([q[slc == i].mean(0) for i in range(nx + 1)])
        u = (cen[-1] - cen[0]) / np.linalg.norm(cen[-1] - cen[0]); off = cen - cen[0]; bend = np.linalg.norm(off - np.outer(off @ u, u), axis=1).max()
        ym = q[mid]; Fall = np.stack([q[T[:, 1]] - q[T[:, 0]], q[T[:, 2]] - q[T[:, 0]], q[T[:, 3]] - q[T[:, 0]]], 2) @ Dinv0
        hist.append(dict(t=(step + 1) / fps, a=a, length_ratio=float((q[x1end, 0].mean() - q[x0end, 0].mean()) / L),
                         mid_width_ratio=float(np.ptp(ym[:, 1]) / W), mid_height_ratio=float(np.ptp(ym[:, 2]) / W), volume_ratio=float((np.linalg.det(Fall) * v0).sum() / v0.sum()),
                         bend_mm=1000 * bend, fibre_stress_Pa_p5_50_95=[float(x) for x in np.percentile(s_ff, [5, 50, 95])], inverted=int((np.linalg.det(Fall) <= 0).sum())))
    last = hist[-int(fps):]                                  # last second, averaged (no damping, so it may oscillate)
    avg = {k: float(np.mean([r[k] for r in last])) for k in ('length_ratio', 'mid_width_ratio', 'mid_height_ratio', 'volume_ratio', 'bend_mm')}
    avg['fibre_stress_Pa_median'] = float(np.mean([r['fibre_stress_Pa_p5_50_95'][1] for r in last])); avg['swing_length_ratio'] = float(np.ptp([r['length_ratio'] for r in last]))
    g = a_target * GMAX; avg['expected_length_ratio_if_free'] = 1 - g; avg['expected_width_ratio_if_free'] = (1 - g) ** -0.5
    print('MB %-5s a %.2f: length x%.3f (free expectation x%.3f, swing %.3f), mid width x%.3f height x%.3f (expect x%.3f), volume x%.3f, bend %.2f mm, fibre stress median %+.1f Pa, inverted %d'
          % (case, a_target, avg['length_ratio'], 1 - g, avg['swing_length_ratio'], avg['mid_width_ratio'], avg['mid_height_ratio'], (1 - g) ** -0.5, avg['volume_ratio'], avg['bend_mm'], avg['fibre_stress_Pa_median'], hist[-1]['inverted']), flush=True)
    return dict(case=case, activation=a_target, last_second=avg, history=hist)
t0 = time.time(); res = [run(c, a) for c in os.environ.get('CASES', 'free,fixed').split(',') for a in (0.16, 0.5, 1.0)]
json.dump(dict(static=STATIC, block=dict(L=L, W=W, h=h, nx=nx, nw=nw, tets=len(T)), material=dict(G=G_, nu=NU, rho=RHO), g_max=GMAX, runs=res), open(OUT, 'w'), indent=1)
print('MB done in %.0f s' % (time.time() - t0))
