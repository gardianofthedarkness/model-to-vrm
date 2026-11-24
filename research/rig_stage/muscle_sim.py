"""One muscle as a closed tube with glued ends, passive, driven by the calibrated ARDY clip: quasi-static energy minimum per
frame with torch autodiff (L-BFGS).  Formulation and constants are cited, not tuned:
  E:\\splat\\.venv\\Scripts\\python.exe muscle_sim.py <fits dir> <muscle_rest.npz> <leg_transforms.npz> <out dir> "<muscle>" [frame step]

  mesh      rest tube surface (muscle_rest.py) + interior grid at spacing h = volume / area, Delaunay, tets kept whose centre
            is inside (offset to the nearest surface vertex against its outward normal)
  fibres    harmonic field phi on the tet mesh, phi = 0 on origin nodes, 1 on insertion nodes (Choi & Blemker 2013, used by
            Saito et al. 2015); fibre a = grad phi / |grad phi| per tet
  energy    per tet, reference volume v:  W = G/2 (I1~ - 3) + W_f(lambda~) + K/2 (ln J)^2
            I1~ = J^(-2/3) tr C;  lambda~ = sqrt(a . C~ a), C~ = J^(-2/3) C
            W_f = sigma_max P1 [ (exp(P2 (x - 1)) - 1) / P2 - (x - 1) ] for x = lambda~ / lambda_ofl > 1, else 0
              (its derivative is Blemker et al. 2005's passive fibre stress, J Biomech 38:657; the exponential branch only)
            constants: FEBio "muscle material" example of Blemker's model: G = 500 Pa, P1 = 0.05, P2 = 6.6, sigma_max = 3e5 Pa,
            lambda_ofl = 1.07, K = 1e6 Pa
  glue      nodes within h of the origin / insertion patch follow that bone segment rigidly (Dirichlet, as Teran et al. 2005)
  contact   bone penetration penalty K v_node (d / h)^2 for nodes inside a leg bone (signed distance d from the bone solids,
            each queried in its bone's moving frame); weight from the bulk modulus and the element size
Writes muscle_sim_<muscle>.json (per frame: volume ratio, inverted tets, fibre stretch percentiles, bone penetration,
belly cross-section area ratio) and .npz (surface per saved frame)."""
import sys, os, re, json, time, numpy as np, torch
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, 'model_to_vrm', 'anatomy'))
from solids import bone_solid, sample
FD, RESTF, TRF, OUT, MUS = sys.argv[1:6]; STEP = int(sys.argv[6]) if len(sys.argv) > 6 else 15
os.makedirs(OUT, exist_ok=True)
DEV = torch.device('cuda' if torch.cuda.is_available() and os.environ.get('CPU') != '1' else 'cpu')
DT = torch.float32 if DEV.type == 'cuda' else torch.float64; torch.set_default_dtype(DT)   # the GTX 1650 is slow in float64
tt = lambda a: torch.as_tensor(np.asarray(a), dtype=DT, device=DEV)
G_, P1, P2, SMAX, LOFL, LSTAR, K_ = 500.0, 0.05, 6.6, 3e5, 1.07, 1.4, 1e6
XS = LSTAR / LOFL                                                                   # Blemker's linear branch starts at lambda*
WXS = SMAX * P1 * ((np.exp(P2 * (XS - 1)) - 1) / P2 - (XS - 1)); DWXS = SMAX * P1 * (np.exp(P2 * (XS - 1)) - 1)
D = np.load(RESTF); names = [str(x) for x in D['names']]; k = names.index(MUS)
Ps, Fs, oi, ii, so, si = (D['%d_%s' % (k, x)] for x in ('P', 'F', 'origin', 'insertion', 'seg_o', 'seg_i')); so, si = int(so), int(si)
vol_s = abs(np.einsum('ij,ij->i', Ps[Fs[:, 0]], np.cross(Ps[Fs[:, 1]], Ps[Fs[:, 2]])).sum()) / 6
area = np.linalg.norm(np.cross(Ps[Fs[:, 1]] - Ps[Fs[:, 0]], Ps[Fs[:, 2]] - Ps[Fs[:, 0]]), axis=1).sum() / 2; h = vol_s / area
vn = np.zeros_like(Ps); fn = np.cross(Ps[Fs[:, 1]] - Ps[Fs[:, 0]], Ps[Fs[:, 2]] - Ps[Fs[:, 0]])
for j in range(3): np.add.at(vn, Fs[:, j], fn)
vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30
if np.einsum('ij,ij->', Ps - Ps.mean(0), vn) < 0: vn = -vn
kdv = cKDTree(Ps)
def inside(Q): d_, j_ = kdv.query(Q); return np.einsum('ij,ij->i', Q - Ps[j_], vn[j_]) < 0
lo, hi = Ps.min(0), Ps.max(0); g = np.stack(np.meshgrid(*[np.arange(a + h / 2, b, h) for a, b in zip(lo, hi)], indexing='ij'), -1).reshape(-1, 3)
g = g[inside(g)]; g = g[kdv.query(g)[0] > h / 2]
X0 = np.r_[Ps, g]; T = Delaunay(X0).simplices.astype(np.int64); T = T[inside(X0[T].mean(1))]
Dm = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2); dt = np.linalg.det(Dm)
T[dt < 0] = T[dt < 0][:, [0, 2, 1, 3]]; Dm = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2)
vt = np.abs(np.linalg.det(Dm)) / 6; ok = vt > 1e-3 * np.median(vt); T, Dm, vt = T[ok], Dm[ok], vt[ok]; Dinv = np.linalg.inv(Dm)
from scipy.sparse.csgraph import connected_components
Eg = np.concatenate([T[:, [a_, b_]] for a_ in range(4) for b_ in range(a_ + 1, 4)])
nc_, comp_ = connected_components(coo_matrix((np.ones(len(Eg)), (Eg[:, 0], Eg[:, 1])), shape=(len(X0), len(X0))), directed=False)
main = np.bincount(comp_[T[:, 0]]).argmax(); T = T[comp_[T[:, 0]] == main]; Dm = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2)
vt = np.abs(np.linalg.det(Dm)) / 6; Dinv = np.linalg.inv(Dm)
used = np.unique(T); remap = np.full(len(X0), -1); remap[used] = np.arange(len(used)); X0 = X0[used]; T = remap[T]   # nodes in no tet removed; one connected piece
srf = remap[:len(Ps)]                                                               # surface vertex -> node (-1 if dropped)
n = len(X0); print('SIM device %s %s' % (DEV, DT), flush=True); print('SIM %s: tube %.1f cm3, h %.2f mm, %d nodes, %d tets (tet volume sum %.1f cm3)' % (MUS, 1e6 * vol_s, 1000 * h, n, len(T), 1e6 * vt.sum()), flush=True)
# glue
gd_o = cKDTree(Ps[oi]).query(X0)[0]; gd_i = cKDTree(Ps[ii]).query(X0)[0]
glue = np.full(n, -1); glue[gd_o <= h] = so; glue[(gd_i <= h) & (gd_i < gd_o)] = si
free = np.flatnonzero(glue < 0); print('SIM glued nodes: origin %d, insertion %d; free %d' % ((glue == so).sum(), (glue == si).sum(), len(free)), flush=True)
# fibres: harmonic phi on the tet mesh (cotangent-free: P1 FEM stiffness)
Gt = np.concatenate([-Dinv.sum(1, keepdims=True), Dinv], 1)                       # 4 x 3 gradients of the hat functions
Ke = np.einsum('tik,tjk->tij', Gt, Gt) * vt[:, None, None]
Kg = coo_matrix((Ke.ravel(), (np.repeat(T, 4, 1).ravel(), np.tile(T, (1, 4)).ravel())), shape=(n, n)).tocsr()
bc = np.flatnonzero(glue >= 0); phi = np.zeros(n); phi[glue == si] = 1.0; fr = np.flatnonzero(glue < 0)
phi[fr] = spsolve(Kg[fr][:, fr].tocsc(), -(Kg[fr][:, bc] @ phi[bc]))
ga = np.einsum('tk,tkj->tj', phi[T], Gt); fib = ga / (np.linalg.norm(ga, axis=1, keepdims=True) + 1e-30)
if os.environ.get('MESH_ONLY') == '1':   # hand the tube to another solver (newton_muscle.py)
    tag_ = re.sub(r'[^A-Za-z0-9]+', '_', MUS).strip('_')
    np.savez(os.path.join(OUT, 'mesh_%s.npz' % tag_), X=X0, T=T, glue=glue, phi=phi, fib=fib, srf=srf, F=Fs, rest=Ps, h=h, seg_o=so, seg_i=si)
    print('SIM mesh saved'); sys.exit(0)
# bones: solids for contact, moving with their segment
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']; FB = R['F_bone']
nm = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {x: i for i, x in enumerate(nm)}
bseg = {'Hip bone': 0, 'Femur': 1, 'Tibia': 2, 'Fibula': 2, 'Patella': 2}
hb = 2 * float(np.median(np.linalg.norm(V[FB[:, 0]] - V[FB[:, 1]], axis=1)))
SOL = {}
for b_, sg in bseg.items():
    sd, lo_, _ = bone_solid(V, FB[OB[FB[:, 0]] == nid[b_ + '.l']], hb); SOL[b_] = (tt(sd), np.asarray(lo_, np.float64), sg)
def sdf(sd, lo_, Pb):     # trilinear sample of the bone's signed grid (positive inside), differentiable in Pb
    q = (Pb - tt(lo_)) / hb; i0 = torch.floor(q).long(); f = q - i0
    shp = torch.tensor(sd.shape, device=DEV) - 2; i0 = torch.maximum(torch.minimum(i0, shp), torch.zeros(3, dtype=torch.long, device=DEV)); out = 0
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                w = (f[:, 0] if dx else 1 - f[:, 0]) * (f[:, 1] if dy else 1 - f[:, 1]) * (f[:, 2] if dz else 1 - f[:, 2])
                out = out + w * sd[i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz]
    inb = ((q >= 0) & (q <= torch.tensor(sd.shape, device=DEV) - 1)).all(1); return torch.where(inb, out, torch.full_like(out, -1.0))
TR = np.load(TRF); NT = len(TR['thigh_R'])
def seg_tf(sg, f):
    if sg == 0: return np.eye(3), np.zeros(3)
    return (TR['thigh_R'][f], TR['thigh_t'][f]) if sg == 1 else (TR['shank_R'][f], TR['shank_t'][f])
Tt = torch.as_tensor(T, device=DEV); Dinv_t = tt(Dinv); vt_t = tt(vt); a_t = tt(fib)
vnode = np.zeros(len(X0)); np.add.at(vnode, T.ravel(), np.repeat(vt / 4, 4)); vnode_t = tt(vnode)
def energy(X, f):
    Ds = torch.stack([X[Tt[:, 1]] - X[Tt[:, 0]], X[Tt[:, 2]] - X[Tt[:, 0]], X[Tt[:, 3]] - X[Tt[:, 0]]], 2); Fm = Ds @ Dinv_t
    J = torch.linalg.det(Fm); C = Fm.transpose(1, 2) @ Fm; Jc = torch.clamp(J, min=1e-6)
    I1 = Jc ** (-2 / 3) * C.diagonal(dim1=1, dim2=2).sum(1); lam = torch.sqrt(torch.clamp(Jc ** (-2 / 3) * torch.einsum('ti,tij,tj->t', a_t, C, a_t), min=1e-12))
    x = lam / LOFL; xc = torch.clamp(x, min=1.0, max=XS)
    Wf = SMAX * P1 * ((torch.exp(P2 * (xc - 1)) - 1) / P2 - (xc - 1)) + torch.where(x > XS, DWXS * (x - XS), torch.zeros_like(x))   # exponential to lambda*, then linear (C1)
    W = G_ / 2 * (I1 - 3) + Wf + K_ / 2 * torch.log(Jc) ** 2 + torch.where(J < 1e-6, K_ * (1e-6 - J) ** 2 * 1e6, torch.zeros_like(J))
    E = (W * vt_t).sum()
    for b_, (sd, lo_, sg) in SOL.items():
        Rb, tb = seg_tf(sg, f); Pb = (X - tt(tb)) @ tt(Rb)                              # into the bone's rest frame
        d = torch.clamp(sdf(sd, lo_, Pb), min=0); E = E + (K_ * vnode_t * (d / h) ** 2).sum()
    return E, J, lam
gl = np.flatnonzero(glue >= 0)
ESC = K_ * float(vt.sum())                                                          # energy unit: bulk modulus x muscle volume
def glued(f):
    out = np.empty((len(gl), 3))
    for sg in (0, 1, 2):
        m_ = glue[gl] == sg
        if m_.any(): Rb, tb = seg_tf(sg, f); out[m_] = X0[gl[m_]] @ Rb.T + tb
    return out
# belly cross-section: nodes near phi = 1/2
belly = np.flatnonzero(np.abs(phi - 0.5) < np.percentile(np.abs(phi - 0.5), 5)); ax0 = np.linalg.svd(X0 - X0.mean(0))[2][0]
def xsec(X):
    P_ = X[belly]; P_ = P_ - P_.mean(0); a_ = np.linalg.svd(X - X.mean(0))[2][0]; Q_ = P_ - np.outer(P_ @ a_, a_); return float(np.linalg.norm(Q_, axis=1).mean() ** 2)
xs0 = xsec(X0)
def guess(f):     # free nodes: blend of the origin's and the insertion's rigid motion by the fibre coordinate phi
    Ro, to = seg_tf(so, f); Ri, ti = seg_tf(si, f); return (1 - phi[:, None]) * (X0 @ Ro.T + to) + phi[:, None] * (X0 @ Ri.T + ti)
from scipy.spatial.transform import Rotation as Rot_, Slerp
SEG_TF0 = seg_tf
def seg_tf_u(sg, f, u):   # part way from rest to frame f
    Rb, tb = SEG_TF0(sg, f); r = Slerp([0, 1], Rot_.from_matrix(np.stack([np.eye(3), Rb])))([u]).as_matrix()[0]; return r, u * tb
X = tt(X0.copy()); rows = []; keep = []; t0 = time.time(); frames = list(range(0, NT, STEP)); prev = 0
for f in frames:
    for e in ((0.5, 1.0) if f else tuple((j + 1) / 8 for j in range(8))):
        if f == 0:                                                                     # ease in from rest
            _tf = seg_tf; seg_tf = lambda sg, ff, u=e: seg_tf_u(sg, 0, u); fe = 0
        else: fe = int(round(prev + (f - prev) * e))
        if f == 0 and e == 1 / 8: X[free] = tt(guess(fe)[free])
        Xg = tt(glued(fe)); xf = (X[free] / h).clone().requires_grad_(True)            # dimensionless: element-size units
        opt = torch.optim.LBFGS([xf], lr=1, max_iter=200, history_size=20, line_search_fn='strong_wolfe', tolerance_grad=1e-9 if DT == torch.float64 else 1e-6, tolerance_change=1e-12 if DT == torch.float64 else 1e-9)
        def closure():
            opt.zero_grad(); Xa = X.clone(); Xa[free] = xf * h; Xa[gl] = Xg; E, _, _ = energy(Xa, fe); E = E / ESC; E.backward(); return E
        opt.step(closure)
        with torch.no_grad(): X[free] = xf.detach() * h; X[gl] = Xg
        if f == 0: seg_tf = _tf
    prev = f
    with torch.no_grad():
        E, J, lam = energy(X, f); Xn = X.cpu().numpy().astype(np.float64); pen = 0.0
        for b_, (sd, lo_, sg) in SOL.items():
            Rb, tb = seg_tf(sg, f); pen = max(pen, float(sdf(sd, lo_, (X - tt(tb)) @ tt(Rb)).max()))
        Jn, ln_ = J.cpu().numpy(), lam.cpu().numpy()
        rows.append(dict(frame=f, energy_J=float(E), volume_ratio=float((Jn * vt).sum() / vt.sum()), inverted=float((Jn <= 0).mean()),
                         fibre_stretch_p5_p50_p95=[float(x) for x in np.percentile(ln_, [5, 50, 95])], bone_penetration_mm=1000 * pen, belly_xsec_ratio=xsec(Xn) / xs0))
    if f % 15 == 0: keep.append(np.where(srf[:, None] >= 0, Xn[np.maximum(srf, 0)], np.nan).astype(np.float32))
    print('SIM frame %3d: volume x%.3f, inverted %.2f%%, fibre stretch p5/50/95 %.2f/%.2f/%.2f, bone penetration %.1f mm, belly cross-section x%.2f  (%.0f s)' % (
        f, rows[-1]['volume_ratio'], 100 * rows[-1]['inverted'], *rows[-1]['fibre_stretch_p5_p50_p95'], rows[-1]['bone_penetration_mm'], rows[-1]['belly_xsec_ratio'], time.time() - t0), flush=True)
tag = re.sub(r'[^A-Za-z0-9]+', '_', MUS).strip('_')
json.dump(dict(muscle=MUS, nodes=n, tets=int(len(T)), h_mm=1000 * h, frames=rows), open(os.path.join(OUT, 'muscle_sim_%s.json' % tag), 'w'), indent=1)
np.savez_compressed(os.path.join(OUT, 'muscle_sim_%s.npz' % tag), surf=np.array(keep), F=Fs, rest=Ps.astype(np.float32), frames=np.array([f for f in frames if f % 15 == 0]))
print('SIM done')
