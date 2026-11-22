"""FLAME fitted to the twin's face (T-pose): python flame_fit.py <twin_full.npz> <face68_px.npy> <flame_lmk.npz> <out.npz>
  landmarks  the twin's 68 face points (rtmlib on the twin's front face render: x = (px-300)/4500, z = 0.865-(py-300)/4500,
             the front-most face point under each) against FLAME's same 68 points (the detector on FLAME's own render)
  surface    every twin face point (part 9) to its nearest FLAME point (eyeballs excluded)
  prior      FLAME's shape and expression coefficients are unit Gaussians
  objective  MAP: (sum of squared distances) / (2 sigma^2) + |coefficients|^2 / 2, sigma = the twin's own surface spacing
             (median edge), so the data and the prior are on one footing without a chosen weight
  stages     size, turn and place by the landmarks first; then everything (shape, expression, jaw) together"""
import sys, numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, __file__.replace('\\', '/').rsplit('/', 1)[0]); from flame_model import load, Flame, rodr
T, L68, LM, out = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]
V, F, P = T['V'], T['F'], T['part']; face = np.flatnonzero(P == 9)
lm = np.c_[(L68[:, 0] - 300) / 4500.0, 0.865 - (L68[:, 1] - 300) / 4500.0]
def front_point(x, z):
    m = face[(np.abs(V[face, 0] - x) < 0.0015) & (np.abs(V[face, 2] - z) < 0.0015)]; return V[m[np.argmin(V[m, 1])]] if len(m) else None
tw = [front_point(*p) for p in lm]; ids = LM['ids']; ok = [i for i in range(68) if ids[i] >= 0 and tw[i] is not None]
tgt_l = torch.tensor(np.array([tw[i] for i in ok]), dtype=torch.float32, device='cuda'); fl_l = torch.tensor(ids[ok], device='cuda')
Ef = F[np.isin(F, face).all(1)]; sig = np.median(np.linalg.norm(V[Ef[:, 0]] - V[Ef[:, 1]], axis=1))
print('FF landmark pairs %d; face points %d; sigma (twin surface spacing) %.5f' % (len(ok), len(face), sig))
d = load(); m = Flame(d); eye = (d['w'][:, 3] > 0.99) | (d['w'][:, 4] > 0.99); keep = torch.tensor(np.flatnonzero(~eye), device='cuda')
rng = np.random.default_rng(0); fs = torch.tensor(V[rng.choice(face, min(8000, len(face)), replace=False)], dtype=torch.float32, device='cuda')
beta = torch.zeros(300, device='cuda', requires_grad=True); psi = torch.zeros(100, device='cuda', requires_grad=True); jaw = torch.zeros(3, device='cuda', requires_grad=True)
v0, _ = m(beta.detach(), psi.detach()); a0 = tgt_l.mean(0); b0 = v0[fl_l].mean(0)
sc0 = float((tgt_l - a0).norm(dim=1).mean() / (v0[fl_l] - b0).norm(dim=1).mean())
ls = torch.tensor(np.log(sc0), device='cuda', requires_grad=True); rot = torch.zeros(3, device='cuda', requires_grad=True); tr = (a0 - sc0 * b0).clone().requires_grad_(True)
def place(v): return torch.exp(ls) * v @ rodr(rot).T + tr
def energy(full):
    v, J = m(beta, psi, jaw=jaw); w = place(v)
    e_l = ((w[fl_l] - tgt_l) ** 2).sum(); n_l = len(ok)
    if not full: return e_l / n_l, w, J
    wk = w[keep]; dmin = torch.cdist(fs, wk).min(1).values; e_s = (dmin ** 2).sum()
    E = (e_l + e_s) / (2 * sig ** 2) + 0.5 * ((beta ** 2).sum() + (psi ** 2).sum())
    return E, w, J
opt = torch.optim.Adam([ls, rot, tr], lr=0.01)
for it in range(400): e, _, _ = energy(False); opt.zero_grad(); e.backward(); opt.step()
print('FF stage 1 (size, turn, place): landmark rms %.4f; scale %.3f' % (float(e) ** 0.5, float(torch.exp(ls))), flush=True)
opt = torch.optim.Adam([ls, rot, tr, beta, psi, jaw], lr=0.005)
for it in range(1500):
    e, w, J = energy(True); opt.zero_grad(); e.backward(); opt.step()
    if it % 500 == 0: print('FF stage 2 step %d: energy %.1f' % (it, float(e)), flush=True)
with torch.no_grad():
    e, w, J = energy(True); wn = w.cpu().numpy(); Jw = (torch.exp(ls) * J @ rodr(rot).T + tr).cpu().numpy()
lmd = np.linalg.norm(wn[ids[ok]] - np.array([tw[i] for i in ok]), axis=1); sd = cKDTree(wn[~eye]).query(V[face])[0]
eyes = [np.flatnonzero(d['w'][:, k] > 0.99) for k in (3, 4)]; ec = [wn[e].mean(0) for e in eyes]; er = [np.linalg.norm(wn[e] - wn[e].mean(0), axis=1).mean() for e in eyes]
print('FF done: landmarks median %.4f, max %.4f; face surface to FLAME median %.4f, 95%% %.4f; |beta| %.1f |psi| %.1f; jaw %.1f deg' % (
    np.median(lmd), lmd.max(), np.median(sd), np.percentile(sd, 95), float(beta.norm()), float(psi.norm()), np.degrees(float(jaw.norm()))))
print('FF eyeballs: centres %s, radii %s' % ([c.round(4).tolist() for c in ec], np.round(er, 4).tolist()))
np.savez(out, V=wn, F=d['f'], J=Jw, eye_centres=np.array(ec), eye_radii=np.array(er), beta=beta.detach().cpu().numpy(), psi=psi.detach().cpu().numpy(),
         jaw=jaw.detach().cpu().numpy(), scale=float(torch.exp(ls)), rot=rot.detach().cpu().numpy(), trans=tr.detach().cpu().numpy())
