"""Option 4, step 2: the coarse-warped atlas (atlas_warp.py) fitted to every corner of the model, by small convex problems:
  E:\\splat\\.venv\\Scripts\\python.exe skin_fit.py <model dir>
  skin      the atlas's named skin regions, welded into one surface (shared vertices merged); unknown: a displacement u
            per skin vertex
  data      per skin vertex its nearest model point whose surface faces the same way (normals agree), so a thigh pressed
            on the belly does not take the belly; then, by the model's material there:
              flesh  on the surface: Huber point-to-plane, knee at the residual scale (median flesh distance)
              cloth / hair / prop  not outside it (one-sided, squared, same scale), and near the fitted MHR body
                     (the body under the clothes; Huber point-to-plane, knee at the MHR-to-atlas spread there)
  smooth    graph Laplacian of u over the welded skin, scale = the median edge length (u differs from its neighbours'
            mean by about an edge at most)
  strands   one patch per segment (the skin vertices nearest that segment's axis); each patch is a small convex problem
            (cvxpy + Clarabel) with its neighbours' u held; patches in turn (block coordinate descent on the one convex
            problem), correspondences refreshed between rounds; until no vertex moves more than the residual scale
  inside    muscle vertices: the displacement of the nearest skin vertices, times depth fraction d_bone / (d_bone + d_skin)
            (full at the skin, nothing at the bone); bones keep the coarse warp (they follow the skeleton)
  writes    anatomy/atlas_fit.npz: V (fitted), V_coarse, u_skin, skin vertex map, patch per skin vertex, every scale used,
            per-round log; anatomy/skin_fit_log.json"""
import sys, os, json, time, numpy as np, trimesh, cvxpy as cp
from scipy.spatial import cKDTree
from scipy import sparse
R = sys.argv[1]; j = lambda *p: os.path.join(R, *p)
W = np.load(j('anatomy', 'atlas_warp.npz')); V = W['V'].astype(np.float64); F = W['F']; OB = W['obj']; names = [str(x) for x in W['names']]; sysn = [str(x) for x in W['system']]
vsys = np.array(sysn)[OB]
# --- weld the skin regions into one surface
sk = np.flatnonzero(vsys == 'skin'); Fs = W['F_skin']
key = np.round(W['V0'][sk].astype(np.float64), 6); _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True); inv = inv.ravel()
loc = -np.ones(len(V), np.int64); loc[sk] = inv; nS = inv.max() + 1
SV = np.zeros((nS, 3)); np.add.at(SV, inv, V[sk]); SV /= np.bincount(inv)[:, None]
SF = loc[Fs]; SF = SF[(SF[:, 0] != SF[:, 1]) & (SF[:, 1] != SF[:, 2]) & (SF[:, 0] != SF[:, 2])]
mS = trimesh.Trimesh(SV, SF, process=False); E = mS.edges_unique; el = float(np.median(mS.edges_unique_length))
print('SF skin: %d atlas vertices welded to %d, %d faces, median edge %.4f m' % (len(sk), nS, len(SF), el), flush=True)
Lap = sparse.csr_matrix((np.ones(2 * len(E)), (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])), shape=(nS, nS))
deg = np.asarray(Lap.sum(1)).ravel(); Lmat = sparse.eye(nS) - sparse.diags(1 / np.maximum(deg, 1)) @ Lap      # u - mean of neighbours
# --- the model: surface samples with material and normals; the MHR body under the clothes
MV = np.load(j('parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(j('parts', 'orig_normals.npz'))['F']
mat = np.load(j('contacts', 'material.npz'))['material']; MN = [str(x) for x in np.load(j('contacts', 'material.npz'))['materials']]; FL = MN.index('flesh')
mm = trimesh.Trimesh(MV, MF, process=False); MN_ = mm.vertex_normals; mk = cKDTree(MV)
H = np.load(j('body', 'mhr_fit.npz')); hm = trimesh.Trimesh(H['V'].astype(np.float64), H['F'], process=False); HV, HN = np.asarray(hm.vertices), hm.vertex_normals; hk = cKDTree(HV)
# --- patches: each skin vertex to its nearest segment axis (as atlas_warp did)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segments import ORDER
segs = [s for s in ORDER if os.path.exists(j('anatomy', s + '_fit.npz'))]
AX = [(np.load(j('anatomy', s + '_fit.npz'))['J0'], np.load(j('anatomy', s + '_fit.npz'))['J1']) for s in segs]
def seg_d(P, a, b):
    d = b - a; u = np.clip(((P - a) @ d) / (d @ d), 0, 1); return np.linalg.norm(P - a - u[:, None] * d, axis=1)
patch = np.array([seg_d(SV, a, b) for a, b in AX]).argmin(0)
print('SF patches: %s' % ', '.join('%s %d' % (s, (patch == k).sum()) for k, s in enumerate(segs)), flush=True)
def corr(X):
    """per skin vertex: model target (point, normal, material) with agreeing normals; MHR point and normal"""
    Nx = trimesh.Trimesh(X, SF, process=False).vertex_normals; d, i = mk.query(X, k=8)
    ok = (MN_[i] * Nx[:, None]).sum(2) > 0; first = np.where(ok.any(1), ok.argmax(1), -1)
    ti = np.where(first >= 0, i[np.arange(len(X)), np.maximum(first, 0)], -1)
    _, hi = hk.query(X); return ti, Nx, hi
u = np.zeros((nS, 3)); log = []; t0 = time.time()
ti, Nx, hi = corr(SV)
valid = ti >= 0; tm = np.where(valid, mat[np.maximum(ti, 0)], -1); isF = valid & (tm == FL); isC = valid & (tm != FL)
dF = np.abs(((SV - MV[np.maximum(ti, 0)]) * MN_[np.maximum(ti, 0)]).sum(1))
sr = float(np.median(dF[isF])) if isF.any() else el                  # residual scale: median distance to the seen flesh
dH = np.abs(((SV - HV[hi]) * HN[hi]).sum(1)); sh = float(np.median(np.abs(dH - np.median(dH)))) + 1e-9   # MHR: the spread of its offsets
print('SF residual scale %.4f m (median distance to seen flesh, %d skin vertices on flesh, %d on cloth/hair/prop); MHR spread %.4f m' % (sr, isF.sum(), isC.sum(), sh), flush=True)
for rnd in range(len(segs) * 4):                                   # rounds of patch sweeps; stop when nothing moves more than sr
    X = SV + u; ti, Nx, hi = corr(X); valid = ti >= 0; tm = np.where(valid, mat[np.maximum(ti, 0)], -1)
    isF = valid & (tm == FL); isC = valid & (tm != FL); maxmove = 0.0
    for k in range(len(segs)):
        P = np.flatnonzero(patch == k)
        if not len(P): continue
        gl = -np.ones(nS, np.int64); gl[P] = np.arange(len(P)); Uv = cp.Variable((len(P), 3)); terms = []
        Xp = X[P]; tp = np.maximum(ti[P], 0); n_t = MN_[tp]; y = MV[tp]
        f = isF[P]
        if f.any(): terms.append(cp.sum(cp.huber(cp.sum(cp.multiply(Xp[f] - u[P][f] + Uv[f] - y[f], n_t[f]), axis=1) / sr, 1)))   # x0 + u_new on the flesh
        c = isC[P]
        if c.any():
            terms.append(cp.sum_squares(cp.pos(cp.sum(cp.multiply(Xp[c] - u[P][c] + Uv[c] - y[c], n_t[c]), axis=1)) / sr))         # not outside the cloth
            hp = hi[P][c]; terms.append(cp.sum(cp.huber(cp.sum(cp.multiply(SV[P][c] + Uv[c] - HV[hp], HN[hp]), axis=1) / sh, 1)))  # near the body under it
        Lp = Lmat[P]; inner = Lp[:, P]; outer = Lp @ np.where((gl < 0)[:, None], u, 0)          # neighbours outside the patch held
        terms.append(cp.sum_squares((inner @ Uv + outer) / el))
        pr = cp.Problem(cp.Minimize(cp.sum(terms))); pr.solve(solver=cp.CLARABEL)
        if Uv.value is None: continue
        mv = float(np.abs(Uv.value - u[P]).max()); maxmove = max(maxmove, mv); u[P] = Uv.value
    X = SV + u; dd = np.abs(((X - MV[np.maximum(ti, 0)]) * MN_[np.maximum(ti, 0)]).sum(1))
    log.append(dict(round=rnd, max_move=maxmove, flesh_median=float(np.median(dd[isF])) if isF.any() else None, flesh_p95=float(np.percentile(dd[isF], 95)) if isF.any() else None, seconds=time.time() - t0))
    print('SF round %d: max move %.4f m; flesh distance median %.4f p95 %.4f m (%.0f s)' % (rnd, maxmove, log[-1]['flesh_median'] or 0, log[-1]['flesh_p95'] or 0, time.time() - t0), flush=True)
    if maxmove < sr: break
# --- the inside follows: muscles by depth fraction, bones keep the coarse warp
Vfit = V.copy(); Vfit[sk] = (SV + u)[inv]
mu = np.flatnonzero(vsys == 'muscle'); bo = np.flatnonzero(vsys == 'bone')
dsk, isk = cKDTree(SV).query(V[mu], k=4); w = 1 / (dsk + 1e-9); w /= w.sum(1, keepdims=True); um = (u[isk] * w[..., None]).sum(1)
dbo = cKDTree(V[bo]).query(V[mu])[0]; frac = dbo / (dbo + dsk[:, 0] + 1e-12); Vfit[mu] = V[mu] + um * frac[:, None]
print('SF muscles: moved by depth fraction, median %.2f; displacement median %.4f m' % (np.median(frac), np.median(np.linalg.norm(um * frac[:, None], axis=1))), flush=True)
np.savez(j('anatomy', 'atlas_fit.npz'), V=Vfit.astype(np.float32), V_coarse=V.astype(np.float32), F=F, F_skin=W['F_skin'], F_muscle=W['F_muscle'], F_bone=W['F_bone'], obj=OB, names=W['names'], system=W['system'],
         u_skin=u.astype(np.float32), skin_weld=inv.astype(np.int32), skin_index=sk.astype(np.int32), skin_patch=patch.astype(np.int16), patch_segments=np.array(segs),
         scale_residual=sr, scale_mhr=sh, scale_edge=el, muscle_depth_fraction=frac.astype(np.float32))
json.dump(dict(scales=dict(residual=sr, mhr_spread=sh, edge=el), rounds=log, patches=segs), open(j('anatomy', 'skin_fit_log.json'), 'w'), indent=1)
print('SF done')
