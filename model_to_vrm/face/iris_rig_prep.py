"""Everything the rigged twin needs for movable irises: python iris_rig_prep.py <twin_corrected_nr.npz> <ab_free.npz> <setting_iris.npz> <out.npz>
  body     the corrected, coloured twin (full detail); skin weights from the nearest rig point of the same part
  iris     per eye piece (parts 23, 25): seen from the front, its outline and frame (as iris_move.py); the iris is the
           ellipse at the eye's centre with the setting picture's relative size (SAM 3); the eye's triangles inside it are
           copied as the iris mesh, lifted off the eye surface by one surface spacing (the eye mesh's median edge) so it
           shows in front of the eye white
  eyeball  a virtual eyeball behind the iris: radius = the eye's half-width, centre that far behind the eye surface at the
           eye's centre; the eye bone sits there (not bounded by the skull)
  limits   the largest turn left/right/up/down for which the whole iris ellipse still lies inside the outline (the iris
           centre's place is R sin(angle) along that axis)"""
import sys, numpy as np, cv2
from scipy.spatial import cKDTree
T, fit, SI, out = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]
V, F, P, C = T['V'], T['F'], T['part'], T['C'].copy(); C0 = T['C']                     # C0: the colours before the eye white is painted in
W0, pt, V0 = fit['W'], fit['part_t'], fit['V0']; W = np.zeros((len(V), W0.shape[1]), np.float32)
for p in np.unique(P):
    mf = np.flatnonzero(P == p); mr = np.flatnonzero(pt == p)
    if len(mr) == 0: mr = np.arange(len(V0))
    W[mf] = W0[mr][cKDTree(V0[mr]).query(V[mf])[1]]
def frame(Q):
    c = Q.mean(0); _, _, Vt = np.linalg.svd(Q - c, full_matrices=False); au = Vt[0] * np.sign(Vt[0][0] or 1); av = np.array([-au[1], au[0]])
    pu, pv = (Q - c) @ au, (Q - c) @ av; o = c + au * (pu.max() + pu.min()) / 2 + av * (pv.max() + pv.min()) / 2
    return o, au, av, (pu.max() - pu.min()) / 2, (pv.max() - pv.min()) / 2
# the iris's size relative to its eye, from SAM 3 on the setting picture (in picture pixels: y grows down)
rel = []
for e, ir in zip(SI['eyes'], SI['irises']):
    o, au, av, hu, hv = frame(np.argwhere(e)[:, ::-1].astype(float)); Q = np.argwhere(ir)[:, ::-1].astype(float); qc = Q.mean(0)
    rel.append((np.ptp((Q - qc) @ au) / 2 / hu, np.ptp((Q - qc) @ av) / 2 / hv))
res = {}
for p, (su, sv) in zip((23, 25), rel):                                                   # part 23 is at -x: the character's right eye
    ids = np.flatnonzero(P == p); X = V[ids]; q = X[:, [0, 2]]                          # front view: x right, z up
    o, au, av, hu, hv = frame(q); hull = cv2.convexHull(q.astype(np.float32))
    a, b = su * hu, sv * hv
    inside = lambda c: (((q - c) @ au / a) ** 2 + ((q - c) @ av / b) ** 2) <= 1
    sel = inside(o); fsel = F[np.isin(F, ids[sel]).all(1)]
    eF = F[np.isin(F, ids).all(1)]; h = np.median(np.linalg.norm(V[eF[:, 0]] - V[eF[:, 1]], axis=1))
    # the virtual eyeball: a sphere fitted to the eye's front surface around the iris (inside the ellipse grown by half),
    # front-most points only (each the nearest to the viewer in its spot), so turning about it slides the iris along the eye
    near = (((q - o) @ au / (1.5 * a)) ** 2 + ((q - o) @ av / (1.5 * b)) ** 2) <= 1; Xn = X[near]
    cell = np.floor(Xn[:, [0, 2]] / h).astype(int); key = cell[:, 0] * 100000 + cell[:, 1]; order = np.lexsort((Xn[:, 1], key)); first = np.r_[True, key[order][1:] != key[order][:-1]]
    Xs = Xn[order][first]; sp = np.linalg.lstsq(np.c_[2 * Xs, np.ones(len(Xs))], (Xs ** 2).sum(1), rcond=None)[0]; centre = sp[:3]; R = float(np.sqrt(sp[3] + centre @ centre))
    if centre[1] < Xs[:, 1].mean(): centre = Xs.mean(0) + (Xs.mean(0) - centre)                # the sphere must lie behind the surface
    R_proj = R                                                                               # the iris slides R sin(angle) across the eye
    ring = lambda c: c + np.stack([np.cos(t) * a * au + np.sin(t) * b * av for t in np.linspace(0, 2 * np.pi, 72)])
    fits = lambda c: all(cv2.pointPolygonTest(hull, (float(x), float(z)), False) >= 0 for x, z in ring(c))
    lim = {}
    for name, d in (('left', au * -1), ('right', au), ('up', av), ('down', -av)):           # au points to +x (the character's left)
        best = 0.0
        for ang in np.arange(0.5, 89.5, 0.5):
            if fits(o + d * R_proj * np.sin(np.radians(ang))): best = ang
            else: break
        lim[name] = best
    # the iris mesh: its own copy of the selected triangles, lifted toward the viewer by one surface spacing
    uid = np.unique(fsel); remap = -np.ones(len(V), int); remap[uid] = np.arange(len(uid))
    nrm = np.zeros((len(uid), 3)); nrm[:, 1] = -1.0
    Lc = cv2.cvtColor((np.clip(C[ids], 0, 1)[None] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[0][:, 0].astype(float)
    def otsu(v):
        v = np.sort(np.asarray(v, float)); n_ = len(v); cs_ = np.cumsum(v); i_ = np.arange(1, n_); m0, m1 = cs_[:-1] / i_, (cs_[-1] - cs_[:-1]) / (n_ - i_)
        return v[np.argmax(i_ * (n_ - i_) * (m0 - m1) ** 2)]
    white = np.median(C[ids][~sel & (Lc > otsu(Lc[~sel]))], 0); C[ids[sel]] = white
    res[p] = dict(iris_V=(V[uid] + nrm * h).astype(np.float32), iris_F=remap[fsel], iris_C=C0[uid], centre=centre, radius=R, limits=lim, lift=h)
    print('IR eye part %d: eyeball fitted to %d front points; iris %d points, %d triangles; eyeball centre %s radius %.4f; limits %s; lift %.5f' % (p, len(Xs), len(uid), len(fsel), centre.round(4), R, lim, h))
np.savez(out, V=V.astype(np.float32), F=F, C=C, W=W, names=fit['names'], parent=fit['parent'], head=fit['head0'], tail=fit['tail0'],
         **{'%s_%d' % (k, p): np.asarray(v) for p, d in res.items() for k, v in d.items() if k != 'limits'},
         **{'limits_%d' % p: np.array([d['limits'][k] for k in ('left', 'right', 'up', 'down')]) for p, d in res.items()})
print('IR saved')
