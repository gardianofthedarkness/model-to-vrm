"""PINN toy for tissue mechanics (made-up numbers, no scan): a ring of soft tissue (inner radius a = bone, outer
radius b = skin), plane strain, linear elastic, nearly incompressible. The bone is fixed. Energy form of a PINN
('deep energy method'): the network gives the displacement, multiplied by (r - a) so the bone condition holds exactly,
and is trained to minimise potential energy = strain energy - work of the loads, by Monte Carlo over the ring.
Case 1: uniform pressure p on the skin (a tight garment). Exact Lame solution: u_r = A r + B / r.
Case 2: gravity (body force) sideways, as when a person scanned lying down stands up. No closed form: reported
against a fine finite-difference-free check = the same PINN at 4x the samples and width (self-convergence).
Assumed: Young's modulus 1 (unit), Poisson ratio 0.45; lengths in units of b."""
import time, json, numpy as np, torch
torch.manual_seed(0); dev = 'cuda' if torch.cuda.is_available() else 'cpu'
a, b, E, nu = 0.3, 1.0, 1.0, 0.45
mu = E / (2 * (1 + nu)); lam = E * nu / ((1 + nu) * (1 - 2 * nu))

def net(width):
    return torch.nn.Sequential(torch.nn.Linear(2, width), torch.nn.Tanh(), torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 2)).to(dev)

def disp(model, x):
    r = x.norm(dim=1, keepdim=True); return (r - a) * model(x)                      # zero on the bone, by construction

def energy(model, n, p=0.0, g=(0.0, 0.0)):
    # interior samples, uniform in area
    r = torch.sqrt(torch.rand(n, 1, device=dev) * (b * b - a * a) + a * a); th = torch.rand(n, 1, device=dev) * 2 * np.pi
    x = torch.cat([r * torch.cos(th), r * torch.sin(th)], 1).requires_grad_(True); u = disp(model, x)
    gx = torch.autograd.grad(u[:, 0].sum(), x, create_graph=True)[0]; gy = torch.autograd.grad(u[:, 1].sum(), x, create_graph=True)[0]
    exx, eyy, exy = gx[:, 0], gy[:, 1], 0.5 * (gx[:, 1] + gy[:, 0]); tr = exx + eyy
    w = 0.5 * lam * tr ** 2 + mu * (exx ** 2 + eyy ** 2 + 2 * exy ** 2)
    area = np.pi * (b * b - a * a); U = area * w.mean()
    Wb = area * (u[:, 0] * g[0] + u[:, 1] * g[1]).mean()                              # body force work
    tb = torch.rand(n // 4, 1, device=dev) * 2 * np.pi; xb = torch.cat([b * torch.cos(tb), b * torch.sin(tb)], 1); ub = disp(model, xb)
    nrm = xb / b; Wp = 2 * np.pi * b * (-p * (ub * nrm).sum(1)).mean()                # pressure pushes inward
    return U - Wb - Wp

def solve(width=64, n=8000, steps=4000, **load):
    m = net(width); opt = torch.optim.Adam(m.parameters(), 2e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); t = time.time()
    for it in range(steps):
        opt.zero_grad(); L = energy(m, n, **load); L.backward(); opt.step(); sched.step()
    return m, time.time() - t, float(L)

out = {}
# ---- case 1: pressure, exact solution
p = 0.05; A = -p / (2 * (lam + mu) + 2 * mu * a * a / (b * b)); B = -A * a * a
m, secs, L = solve(p=p)
rr = torch.linspace(a, b, 50, device=dev)[:, None]; errs = []; prof = []
for ang in np.linspace(0, 2 * np.pi, 16, endpoint=False):
    x = torch.cat([rr * np.cos(ang), rr * np.sin(ang)], 1); u = disp(m, x).detach()
    ur = (u * (x / rr)).sum(1).cpu().numpy(); ex = (A * rr + B / rr)[:, 0].cpu().numpy(); errs.append(np.abs(ur - ex)); prof.append(ur)
ex = (A * rr + B / rr)[:, 0].cpu().numpy(); err = np.concatenate(errs)
out['pressure'] = dict(seconds=round(secs, 1), exact_skin_displacement=float(ex[-1]), pinn_skin_displacement=float(np.mean([q[-1] for q in prof])),
                       max_abs_error=float(err.max()), error_relative_to_max_displacement=float(err.max() / np.abs(ex).max()), mean_relative_error=float(err.mean() / np.abs(ex).max()))
print('pressure', out['pressure'], flush=True)
# ---- case 2: gravity, self-convergence
g = (0.0, -0.05)
m1, s1, _ = solve(g=g); m2, s2, _ = solve(width=128, n=32000, steps=6000, g=g)
x = torch.cat([r_ * torch.cat([torch.cos(t_), torch.sin(t_)], 1) for r_, t_ in [(torch.sqrt(torch.rand(20000, 1, device=dev) * (b * b - a * a) + a * a), torch.rand(20000, 1, device=dev) * 2 * np.pi)]])
u1, u2 = disp(m1, x).detach(), disp(m2, x).detach(); d = (u1 - u2).norm(dim=1); mag = u2.norm(dim=1)
out['gravity'] = dict(seconds_small=round(s1, 1), seconds_large=round(s2, 1), max_displacement=float(mag.max()), small_vs_large_mean_diff_relative=float(d.mean() / mag.max()), small_vs_large_max_diff_relative=float(d.max() / mag.max()))
print('gravity', out['gravity'], flush=True)
# ---- figure
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
f, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ax[0].plot(rr.cpu().numpy()[:, 0], ex, 'k', lw=2, label='exact (Lame)')
for q in prof[::4]: ax[0].plot(rr.cpu().numpy()[:, 0], q, '--', lw=1)
ax[0].set_title('Garment pressure: radial displacement'); ax[0].set_xlabel('radius (bone at %.1f, skin at 1)' % a); ax[0].set_ylabel('u_r'); ax[0].legend(['exact (Lame)', 'PINN, 4 directions'])
xs = x.cpu().numpy(); us = u2.cpu().numpy(); k = np.random.default_rng(0).choice(len(xs), 700, replace=False)
ax[1].quiver(xs[k, 0], xs[k, 1], us[k, 0], us[k, 1], np.linalg.norm(us[k], axis=1), cmap='viridis', scale=None); ax[1].add_patch(plt.Circle((0, 0), a, color='lightgray')); ax[1].set_aspect('equal'); ax[1].set_title('Gravity: tissue sags round the fixed bone')
sc = ax[2].scatter(xs[k, 0], xs[k, 1], c=(d / mag.max()).cpu().numpy()[k], s=8, cmap='magma'); ax[2].set_aspect('equal'); ax[2].set_title('Gravity: small vs large PINN (relative difference)'); f.colorbar(sc, ax=ax[2])
f.tight_layout(); f.savefig('pinn_ring.png', dpi=120); json.dump(out, open('pinn_ring.json', 'w'), indent=1); print('done')
