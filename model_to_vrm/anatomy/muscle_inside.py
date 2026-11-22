"""Muscles, step 2: the carried muscles (muscle_carry.py) brought inside the body, each keeping its shape.
  E:\\splat\\.venv\\Scripts\\python.exe muscle_inside.py <fits dir> <model dir>
Per muscle with vertices outside the body's solid, Laplacian mesh editing (Sorkine et al. 2004): find positions X with
  shape    the mesh's own Laplacian coordinates stay as carried:      || L X - L X_carried ||^2
  bones    vertices lying on a bone (muscle_carry.py) do not move
  inside   every vertex outside the solid goes to the nearest point of the surface (along the solid's depth gradient)
           and is then held where it got to, as a penalty whose weight doubles each round until the worst vertex is
           inside to within one grid cell
The body is her solid without what hangs beside her limbs (solids.inner_solid: a robe behind the legs is not her body).
  muscle_inside.py <fits dir> <model dir> <source> <result> skin [ct data dir]      the repair stage, run last on any earlier result
          (muscle_depth -> muscle_repair): the same, with the surface moved in by the thickness of skin, so that
          nothing is left in or outside the skin.
Writes <fits dir>/muscle_inside.npz (V, as muscle_carry.npz) and muscle_inside.json, or <result>.npz / .json."""
import sys, os, re, json, time, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix, diags, identity
from scipy.sparse.linalg import splu
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solids import inner_solid, bone_solid, sample
FD, MD = sys.argv[1:3]; SRC = sys.argv[3] if len(sys.argv) > 3 else 'muscle_carry'; DST = sys.argv[4] if len(sys.argv) > 4 else 'muscle_inside'; REPAIR = len(sys.argv) > 5 and sys.argv[5] == 'skin'; t00 = time.time()
def log(*a): print('MI', *a, flush=True)
C = np.load(os.path.join(FD, SRC + '.npz')); V = C['V'].astype(np.float64); FM, FB, OB = C['F_muscle'], C['F_bone'], C['obj']; muscles = [int(m) for m in C['muscles']]
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]
onbone = C['contact']                                              # vertices lying on a bone
_bv = np.unique(FB); sdf, glo, hstep, body, MV, MF, _vout, _used = inner_solid(MD, V[_bv], OB[_bv], FD, V); log('her body: the solid less %.0f cm3 that is hair or hanging cloth, not her body (limb pieces %s)' % (1e6 * _vout, _used))
# repair: muscle and bone lie under the skin, not in it. How thin the covering can be is measured, not set: the thinnest
# tissue the CT study found over bone anywhere on the face (the 5th percentile at its thinnest site, 388 adults:
# tissue_depth_ct/summary.json, the same bound the skull fit uses under bare skin), at the figure's size against an adult.
CT = sys.argv[6] if len(sys.argv) > 6 else os.environ.get('CT_DEPTH', r'E:igures\models	issue_depth_ct')
margin = float(json.load(open(os.path.join(CT, 'summary.json')))['thinnest_p5_mm']) / 1000 * float(np.load(os.path.join(FD, 'body_assembly.npz'))['s1']) if REPAIR else 0.0
if REPAIR: log('repair: nothing nearer the surface than the skin is thick, %.2f mm on this figure' % (1000 * margin))
G = np.stack(np.gradient(sdf, hstep), -1)
def depth(P): return sample(sdf, glo, hstep, P) - margin
def grad(P): g = np.stack([sample(G[..., k], glo, hstep, P) for k in range(3)], -1); return g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-12)
V0c = V.copy(); loc = -np.ones(len(V), np.int64); hist = []; per = []
def bring_in(m, Fall, fixmask):
    """one object's mesh brought inside, its Laplacian coordinates kept, the vertices in fixmask held; returns its report."""
    vi = np.flatnonzero(OB == m); d0 = depth(V[vi])
    if not len(vi) or d0.min() > -hstep: return None
    loc[vi] = np.arange(len(vi)); F = loc[Fall]
    _, wid = np.unique(np.round(V[vi], 7), axis=0, return_inverse=True); wid = wid.ravel(); n = wid.max() + 1; rep = np.zeros(n, np.int64); rep[wid] = np.arange(len(vi))
    Fw = wid[F]; E = np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]]; E = E[E[:, 0] != E[:, 1]]
    A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)).tocsr(); A = ((A + A.T) > 0).astype(np.float64); deg = np.asarray(A.sum(1)).ravel()
    L = (identity(n) - diags(1 / np.maximum(deg, 1)) @ A).tocsr(); X0 = V[vi][rep]; fix = np.zeros(n, bool); fix[wid[fixmask[vi]]] = True; fr = np.flatnonzero(~fix); fx = np.flatnonzero(fix)
    if not len(fr): return None
    Lf = L[:, fr]; LtL = (Lf.T @ Lf).tocsc(); rhs0 = Lf.T @ (L @ X0 - (L[:, fx] @ X0[fx] if len(fx) else 0)); X = X0.copy(); w = 1.0; held = np.zeros(len(fr), bool); eps = np.finfo(np.float32).eps
    for rnd in range(40):
        d = depth(X[fr]); out = d < 0
        if d.min() > -hstep: break
        T = np.where(out[:, None], X[fr] + (-d)[:, None] * grad(X[fr]), np.where(held[:, None], X[fr], X0[fr])); held |= out   # a vertex once brought in stays held where it got to, or it springs back out
        D = diags(held.astype(float) * w + eps)                                                                 # eps: a mesh piece with nothing holding it keeps its place
        X[fr] = splu((LtL + D).tocsc()).solve(rhs0 + D @ T); w *= 2
    if d.min() <= -hstep: log('WARNING: %s is still %.1f mm outside after 40 rounds of doubling the hold; the last round is used' % (names[m], -1000 * d.min()))
    V[vi] = X[wid]; d1 = depth(V[vi]); mv = np.linalg.norm(V[vi] - V0c[vi], axis=1)
    return dict(name=names[m], rounds=rnd, outside_before=float((d0 < 0).mean()), worst_before_mm=1000 * float(-d0.min()), outside_after=float((d1 < 0).mean()), worst_after_mm=1000 * float(max(0, -d1.min())), moved_median_mm=1000 * float(np.median(mv)), moved_max_mm=1000 * float(mv.max()))
# repair only: a bone that stands out of the body is brought in first, as little as needed. It stands out by p at most,
# so only what lies less than p under the surface has to give way: every vertex deeper than p is held. The muscle
# vertices lying on that bone then move as the nearest point of the bone moved.
bper = []
if REPAIR:
    bfirst = OB[FB[:, 0]]; bone_ids = np.unique(bfirst); Vb0 = V.copy()
    for b in bone_ids:
        vb = np.flatnonzero(OB == b); db_ = depth(V[vb])
        if db_.min() > -hstep: continue
        fm = np.zeros(len(V), bool); fm[vb[db_ > -db_.min()]] = True; r_ = bring_in(int(b), FB[bfirst == b], fm)
        if r_: bper.append(r_)
    bvv = np.unique(FB); mvd = np.linalg.norm(V[bvv] - Vb0[bvv], axis=1) > 0
    if mvd.any():
        cv = np.flatnonzero(onbone); j_ = cKDTree(Vb0[bvv]).query(V[cv])[1]; V[cv] += V[bvv[j_]] - Vb0[bvv[j_]]
    log('repair: %d bones brought in%s' % (len(bper), ''.join('; %s %.1f -> %.1f mm' % (r_['name'], r_['worst_before_mm'], r_['worst_after_mm']) for r_ in sorted(bper, key=lambda r_: -r_['worst_before_mm'])[:8])))
mfirst = OB[FM[:, 0]]
for mi, m in enumerate(muscles):
    r_ = bring_in(m, FM[mfirst == m], onbone)
    if r_ is None: continue
    per.append(r_)
    if len(per) % 25 == 0: log('  %d muscles brought in (%.0f s)' % (len(per), time.time() - t00))
mi_ = C['mverts']; dep0 = depth(V0c[mi_]); dep = depth(V[mi_]); sb, lob, _ = bone_solid(V, FB, hstep); inb0 = sample(sb, lob, hstep, V0c[mi_]); inb = sample(sb, lob, hstep, V[mi_])
e0 = np.linalg.norm(V0c[FM[:, 0]] - V0c[FM[:, 1]], axis=1); e1 = np.linalg.norm(V[FM[:, 0]] - V[FM[:, 1]], axis=1); st = e1[e0 > 0] / e0[e0 > 0]
rep_ = dict(muscles_changed=len(per), outside=dict(before=float((dep0 < 0).mean()), after=float((dep < 0).mean()), before_over_one_cell=float((dep0 < -hstep).mean()), after_over_one_cell=float((dep < -hstep).mean()), worst_before_mm=1000 * float(-dep0.min()), worst_after_mm=1000 * float(max(0, -dep.min()))),
            in_bone=dict(before=float((inb0 > hstep).mean()), after=float((inb > hstep).mean())), edge_change=dict(p1=float(np.percentile(st, 1)), median=float(np.median(st)), p99=float(np.percentile(st, 99))), grid_mm=1000 * hstep, per_muscle=per, per_bone=bper, seconds=time.time() - t00)
json.dump(rep_, open(os.path.join(FD, DST + '.json'), 'w'), indent=1)
NOSE = [i for i, n_ in enumerate(names) if re.search(r'nasal.*cartilage|alar cartilage', n_, re.I)]      # the nose's cartilages are left out of the repaired result for now (they stand out of her small nose)
FBo = FB[~np.isin(OB[FB[:, 0]], NOSE)] if REPAIR else FB
if REPAIR: log('left out: %d nose cartilages (%s)' % (len(NOSE), ', '.join(names[i] for i in NOSE)))
np.savez(os.path.join(FD, DST + '.npz'), **{k: C[k] for k in C.files if k not in ('V', 'depth', 'in_bone', 'F_bone')}, F_bone=FBo, V=V.astype(np.float32), depth=dep.astype(np.float32), in_bone=inb.astype(np.float32))
log('%d muscles changed. Outside the body: %.1f%% -> %.1f%% of muscle vertices (by more than one %.1f mm cell: %.1f%% -> %.2f%%); worst %.1f -> %.1f mm' % (len(per), 100 * rep_['outside']['before'], 100 * rep_['outside']['after'], 1000 * hstep, 100 * rep_['outside']['before_over_one_cell'], 100 * rep_['outside']['after_over_one_cell'], rep_['outside']['worst_before_mm'], rep_['outside']['worst_after_mm']))
log('inside bone by more than one cell: %.1f%% -> %.1f%%; mesh edges x%.2f..x%.2f (1%%..99%%)' % (100 * rep_['in_bone']['before'], 100 * rep_['in_bone']['after'], rep_['edge_change']['p1'], rep_['edge_change']['p99']))
for p_ in sorted(per, key=lambda p_: -p_['worst_after_mm'])[:8]: log('  still furthest out: %-44s worst %.1f mm (was %.1f), %d rounds, moved up to %.0f mm' % (p_['name'], p_['worst_after_mm'], p_['worst_before_mm'], p_['rounds'], p_['moved_max_mm']))
log('done in %.0f s' % (time.time() - t00))
