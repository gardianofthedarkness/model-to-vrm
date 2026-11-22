"""Muscles, step 3: no muscle nearer the skin than its share in the atlas allows; each muscle moved as a whole.
  E:\\splat\\.venv\\Scripts\\python.exe muscle_depth.py <fits dir> <model dir>
Carried by the bones (muscle_carry.py), the muscles have the size of her frame - their closed meshes fill about the
share of the body they fill in people. What differs from person to person is mostly the fat over them. So:
  breasts, hips   where her skin lies further out than the carried muscles reach, the muscles stay and the room is fat
  thin places     where a muscle comes nearer the skin than in the atlas, it goes deeper. Measured as a proportion:
                    rho = (distance to the nearest bone) / (that + depth under the skin), per muscle vertex in the atlas
                    phi(x) = the same ratio in the figure (placed bones, the figure's solid, on the solid's grid)
                  a vertex with phi > rho has a target: along the gradient of phi to where phi = rho (Newton steps, none
                  longer than one grid cell). The atlas's own fat layer is thus the least share of fat anywhere.
  as a whole      per muscle, the smoothest displacement of its mesh (least || L u ||^2, graph Laplacian, vertices on a
                  bone fixed) that brings the vertices whose target is more than a grid cell away to within one cell
                  of it (root mean square):
                  the weight on the targets is raised from zero-like until that holds - the grid is the resolution
                  at which phi is known, so fitting the targets closer would fit the grid's noise (lumps: tried first)
  repeated        targets are taken again from the new places until the share of vertices too near the skin stops falling
  under hair      where the nearest surface of the model is hair or a prop the skin is not seen: no target there
Writes <fits dir>/muscle_depth.npz (as muscle_inside.npz, plus rho) and muscle_depth.json."""
import sys, os, json, time, numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import splu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solids import inner_solid, bone_solid, solid_of, sdf_of, sample, closed_volumes
FD, MD = sys.argv[1:3]; t00 = time.time()
def log(*a): print('MD', *a, flush=True)
C = np.load(os.path.join(FD, 'muscle_inside.npz')); V = C['V'].astype(np.float64); FM, FB, OB = C['F_muscle'], C['F_bone'], C['obj']; mi = C['mverts']; contact = C['contact'][mi]; muscles = [int(m) for m in C['muscles']]
W = np.load(os.path.join(FD, 'atlas_warp.npz')); V0 = W['V0'].astype(np.float64); FS = W['F_skin']; names = [str(x) for x in W['names']]; s1 = float(np.load(os.path.join(FD, 'body_assembly.npz'))['s1'])
_bv = np.unique(FB); sdf, glo, hstep, body, MV, MF, _vout, _used = inner_solid(MD, V[_bv], OB[_bv], FD, V); log('her body: the solid less %.0f cm3 that is hair or hanging cloth, not her body (limb pieces %s)' % (1e6 * _vout, _used)); MT = np.load(os.path.join(MD, 'contacts', 'material.npz'), allow_pickle=True); mat = np.array([str(x) for x in MT['materials']])[MT['material']]
# --- atlas: rho
bv = np.unique(FB); db_a = cKDTree(V0[bv]).query(V0[mi])[0]; step_a = hstep / s1                    # the same grid, in the atlas's size
occ, lo_a = solid_of(V0, FS, step_a)
for seal in range(1, 17):
    thick = ndimage.binary_dilation(occ, iterations=seal); fill = ndimage.binary_fill_holes(thick)
    if fill.sum() - thick.sum() > thick.sum(): break
else: print('WARNING muscle_depth.py: the solid did not close within 17 sealing steps; the widest sealing is used and the result may be wrong', flush=True)
sk_a = sdf_of(ndimage.binary_erosion(fill, iterations=seal) | occ, step_a); ds_a = np.maximum(sample(sk_a, lo_a, step_a, V0[mi]), 0)
rho = np.where(db_a + ds_a > 0, db_a / np.maximum(db_a + ds_a, 1e-12), 0.0)
log('atlas: rho median %.2f (10%%..90%%: %.2f..%.2f)' % (np.median(rho), *np.percentile(rho, [10, 90])))
# --- figure: phi on the body's grid
sb, lob, _ = bone_solid(V, FB, hstep); ins = np.argwhere(ndimage.binary_dilation(body)); Pc = glo + ins * hstep
db = cKDTree(V[bv]).query(Pc)[0]; db[sample(sb, lob, hstep, Pc) > 0] = 0; ds = np.maximum(sdf[tuple(ins.T)], 0)
phi = np.ones(sdf.shape, np.float32); phi[tuple(ins.T)] = np.where(db + ds > 0, db / np.maximum(db + ds, 1e-12), 0); G = np.stack(np.gradient(phi, hstep), -1)
def f(P): return sample(phi, glo, hstep, P)
def g(P): return np.stack([sample(G[..., k], glo, hstep, P) for k in range(3)], -1)
mtree = cKDTree(MV)
X = V[mi].copy(); X0 = X.copy(); unseen = np.isin(mat[mtree.query(X)[1]], ('hair', 'prop')); mv = ~contact & ~unseen
log('%d muscle vertices: %d on a bone (fixed), %d under hair or a prop (no target)' % (len(X), contact.sum(), (unseen & ~contact).sum()))
def too_near(P): return (f(P) - rho > 0) & mv
def far_share(P): T_, a_ = targets(P); return float((np.linalg.norm(T_ - P, axis=1) > hstep).sum() / mv.sum())
def targets(P):
    """for the vertices too near the skin: where phi = rho along the gradient of phi"""
    T = P.copy(); act = np.flatnonzero(too_near(P)); last = np.inf
    for it in range(200):
        r = rho[act] - f(T[act]); k = r < 0
        if not k.any(): break
        gr = g(T[act][k]); st = (r[k] / np.maximum((gr ** 2).sum(1), 1e-12))[:, None] * gr; ln = np.linalg.norm(st, axis=1); st *= np.minimum(1, hstep / np.maximum(ln, 1e-12))[:, None]
        Tn = T[act]; Tn[k] += st; T[act] = Tn; m = float(-r[k].sum())
        if m >= last: break
        last = m
    return T, act
# per muscle: welded mesh, Laplacian (built once)
om = OB[mi]; pos = -np.ones(len(V), np.int64); pos[mi] = np.arange(len(mi)); mfirst = OB[FM[:, 0]]; MESH = {}
for m in muscles:
    vi = np.flatnonzero(om == m); F = pos[FM[mfirst == m]]; loc = -np.ones(len(mi), np.int64); loc[vi] = np.arange(len(vi)); F = loc[F]
    _, wid = np.unique(np.round(V0[mi][vi], 6), axis=0, return_inverse=True); wid = wid.ravel(); n = wid.max() + 1; rep = np.zeros(n, np.int64); rep[wid] = np.arange(len(vi))
    Fw = wid[F]; E = np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]; E = E[E[:, 0] != E[:, 1]]
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(np.float64); deg = np.asarray(A.sum(1)).ravel()
    L = (identity(n) - diags(1 / np.maximum(deg, 1)) @ A).tocsr(); fix = np.zeros(n, bool); fix[wid[contact[vi]]] = True; fr = np.flatnonzero(~fix)
    if len(fr): Lf = L[:, fr]; MESH[m] = (vi, wid, rep, fr, (Lf.T @ Lf).tocsc())
eps = float(np.finfo(np.float32).eps); hist = [far_share(X)]; log('too near the skin before: %.1f%% of the free vertices at all, %.1f%% by more than one grid cell' % (100 * too_near(X).sum() / mv.sum(), 100 * hist[0]))
for rnd in range(20):
    T, act = targets(X); far = np.linalg.norm(T - X, axis=1) > hstep; isact = np.zeros(len(X), bool); isact[act] = True; isact &= far; Xn = X.copy(); ws = []   # a target nearer than one cell is within what the grid can tell
    for m, (vi, wid, rep, fr, LtL) in MESH.items():
        a = isact[vi][rep][fr]
        if not a.any(): continue
        D = (T[vi] - X[vi])[rep][fr]; Aa = diags(a.astype(float))
        def solve(w): U = splu((LtL + w * Aa + eps * identity(len(fr))).tocsc()).solve(w * (Aa @ D)); return U, float(np.sqrt(((U[a] - D[a]) ** 2).sum(1).mean()))
        w = 1.0; U, r = solve(w)                                                        # the smoothest that fits: weight in steps of 16 down while it still fits, or up until it does
        if r <= hstep:
            for _ in range(8):
                U2, r2 = solve(w / 16)
                if r2 > hstep: break
                w, U, r = w / 16, U2, r2
        else:
            for _ in range(8):
                w *= 16; U, r = solve(w)
                if r <= hstep: break
        Uw = np.zeros((len(rep), 3)); Uw[fr] = U; Xn[vi] = X[vi] + Uw[wid]; ws.append(w)
    share = far_share(Xn)
    log('  round %d: %d muscles moved (weights %.3g..%.3g), too near the skin by more than a cell %.2f%% -> %.2f%% (%.0f s)' % (rnd + 1, len(ws), min(ws) if ws else 0, max(ws) if ws else 0, 100 * hist[-1], 100 * share, time.time() - t00))
    if share >= hist[-1]: break
    X = Xn; hist.append(share)
miss = f(X) - rho; moved = np.linalg.norm(X - X0, axis=1); V[mi] = X
# --- checks
dep0 = sample(sdf, glo, hstep, X0); dep = sample(sdf, glo, hstep, X); inb0 = sample(sb, lob, hstep, X0); inb = sample(sb, lob, hstep, X)
V_in = C['V'].astype(np.float64); cv = closed_volumes(V0, FM, OB[FM[:, 0]], muscles, V_in, V); v0 = sum(v[0] for v in cv.values()); v1 = sum(v[1] for v in cv.values()); per = []
for m in muscles:
    k = om == m; a, b = cv.get(m, (float('nan'), float('nan')))
    per.append(dict(name=names[m], closed=m in cv, volume_before_cm3=1e6 * a, volume_cm3=1e6 * b, rho=float(np.median(rho[k])), moved_mm=1000 * float(np.median(moved[k])), moved_max_mm=1000 * float(moved[k].max())))
e0 = np.linalg.norm(V_in[FM[:, 0]] - V_in[FM[:, 1]], axis=1); e1 = np.linalg.norm(V[FM[:, 0]] - V[FM[:, 1]], axis=1); st_ = e1[e0 > 0] / e0[e0 > 0]
# the fat layer this leaves: every model vertex of flesh or cloth to the nearest muscle or bone, against people's layer (fat_layer.py)
lean = cKDTree(np.r_[X, V[bv]]); seen = np.isin(mat, ('flesh', 'cloth')); left = lean.query(MV[seen])[0]; fat_cmp = {}
ff = os.path.join(FD, 'fat_layer.npz')
if os.path.exists(ff):
    pf = np.load(ff)['fat'][seen]; fat_cmp = dict(people_median_mm=1000 * float(np.median(pf)), ratio_median=float(np.median(left / np.maximum(pf, 1e-9))), rank_correlation=float(np.corrcoef(np.argsort(np.argsort(left)), np.argsort(np.argsort(pf)))[0, 1]))
rep_ = dict(rounds=len(hist) - 1, too_near=dict(before=hist[0], after=hist[-1]), moved_mm=dict(median=1000 * float(np.median(moved[mv])), p90=1000 * float(np.percentile(moved[mv], 90)), max=1000 * float(moved.max())),
            outside=dict(before=float((dep0 < 0).mean()), after=float((dep < 0).mean()), after_over_one_cell=float((dep < -hstep).mean())), in_bone=dict(before=float((inb0 > hstep).mean()), after=float((inb > hstep).mean())),
            volume_cm3=dict(body=1e6 * float(body.sum() * hstep ** 3), muscles_before=1e6 * v0, muscles=1e6 * v1), edge_change=dict(p1=float(np.percentile(st_, 1)), median=float(np.median(st_)), p99=float(np.percentile(st_, 99))),
            left_under_skin_mm=dict(median=1000 * float(np.median(left)), p10=1000 * float(np.percentile(left, 10)), p90=1000 * float(np.percentile(left, 90))), against_people=fat_cmp, grid_mm=1000 * hstep, per_muscle=per, seconds=time.time() - t00)
json.dump(rep_, open(os.path.join(FD, 'muscle_depth.json'), 'w'), indent=1)
np.savez(os.path.join(FD, 'muscle_depth.npz'), **{k: C[k] for k in C.files if k not in ('V', 'depth', 'in_bone')}, V=V.astype(np.float32), depth=dep.astype(np.float32), in_bone=inb.astype(np.float32), rho=rho.astype(np.float32), left_under_skin=left.astype(np.float32), seen=seen)
log('after %d rounds: too near the skin by more than a cell %.2f%% -> %.2f%%; moved %.1f mm median, %.1f at 90%%, %.1f at most' % (len(hist) - 1, 100 * hist[0], 100 * hist[-1], rep_['moved_mm']['median'], rep_['moved_mm']['p90'], rep_['moved_mm']['max']))
log('outside the body: %.2f%% -> %.2f%% (over one cell %.2f%%); inside bone over one cell: %.1f%% -> %.1f%%' % (100 * rep_['outside']['before'], 100 * rep_['outside']['after'], 100 * rep_['outside']['after_over_one_cell'], 100 * rep_['in_bone']['before'], 100 * rep_['in_bone']['after']))
log('muscle volume (closed meshes): %.0f -> %.0f cm3; body %.0f cm3; mesh edges x%.2f..x%.2f (1%%..99%%)' % (rep_['volume_cm3']['muscles_before'], rep_['volume_cm3']['muscles'], rep_['volume_cm3']['body'], rep_['edge_change']['p1'], rep_['edge_change']['p99']))
log('left between skin (flesh, cloth) and the nearest muscle or bone: median %.1f mm (10%%..90%%: %.1f..%.1f)%s' % (rep_['left_under_skin_mm']['median'], rep_['left_under_skin_mm']['p10'], rep_['left_under_skin_mm']['p90'], '; people\'s fat layer by height %.1f mm, ratio %.2f, rank correlation %.2f' % (fat_cmp['people_median_mm'], fat_cmp['ratio_median'], fat_cmp['rank_correlation']) if fat_cmp else ''))
for p_ in sorted(per, key=lambda p_: -p_['moved_mm'])[:10]: log('  moved most: %-44s %.1f mm median, %.1f at most' % (p_['name'], p_['moved_mm'], p_['moved_max_mm']))
log('done in %.0f s' % (time.time() - t00))
