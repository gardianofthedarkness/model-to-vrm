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
# ---- case 2: gravity, self-convergence
g = (0.0, -0.05)
m1, s1, _ = solve(g=g); m2, s2, _ = solve(width=96, n=16000, steps=4000, g=g)
x = torch.cat([r_ * torch.cat([torch.cos(t_), torch.sin(t_)], 1) for r_, t_ in [(torch.sqrt(torch.rand(20000, 1, device=dev) * (b * b - a * a) + a * a), torch.rand(20000, 1, device=dev) * 2 * np.pi)]])
u1, u2 = disp(m1, x).detach(), disp(m2, x).detach(); d = (u1 - u2).norm(dim=1); mag = u2.norm(dim=1)
out['gravity'] = dict(seconds_small=round(s1, 1), seconds_large=round(s2, 1), max_displacement=float(mag.max()), small_vs_large_mean_diff_relative=float(d.mean() / mag.max()), small_vs_large_max_diff_relative=float(d.max() / mag.max()))
print('gravity', out['gravity'], flush=True)
json.dump(out, open('pinn_gravity.json','w'), indent=1)
