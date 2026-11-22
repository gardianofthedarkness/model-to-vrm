"""The skull placed under a face by matching named face lines between the atlas head and the model, points free to slide
along their line; one size and pose for the whole skull (real skulls differ mostly in size; a stretch per axis fitted
detector noise):
  E:\\splat\\.venv\\Scripts\\python.exe skull_line_fit.py <fits dir> <model dir> <atlas head dir>
  points    the 68 face points, on BOTH heads the same way: detected in the front view and cast onto the front-most
            surface under each pixel (the atlas's many-view 3D points were up to 7 mm off its own front view, so the
            many-view combination is not used here)
  lines     the runs that follow bone: jaw outline and both eyebrows (named by a language model against the atlas's
            bone names, agents/landmark_lines.py); each a spline through its points, sampled at the model surface's
            median edge. Left out of the fit and only reported: the nose (cartilage, not bone, below its short bony
            root), the eye and lip outlines (soft tissue).
  fit       similarity atlas -> model: start = closed form (Umeyama) on the lines' points; then repeat: every sample of
            an atlas line is matched to the nearest place on the model's line of the same name, and every sample of the
            model's line to the nearest place on the placed atlas line (both ways, so neither end can run off), closed
            form again; until the size changes by less than the detector's own resolution (a pixel over the head size)
  bone      only real bone counts as skull: the atlas files the nose, ear and septum cartilages under 'bone', and drawn
            with the skull they made a bony nose that no skull has (a skull ends at the short nasal bones; below them is
            the open nasal aperture). Cartilages ride with the skull but are kept, checked and drawn apart.
  inside    strict rule: no bone outside the head surface. After the line fit the same matches are solved again with
            every bone hull point's distance outside the surface as a penalty, its scale halved each round until the
            worst point is inside to within one surface edge (the surface's own resolution). Inside / outside is read
            from a solid model of the head (its surface drawn into a grid, the neck capped, enclosed space filled).
  checks    depth to bone under each line against the atlas's (scaled), which teeth lie under the lips, the gap to the
            first neck bone, bone and cartilage outside the surface - for the slice route, lines only, lines + inside.
Writes <fits dir>/skull_lines.npz (V for all atlas vertices: head bones and cartilages placed; F_head bone faces,
F_cartilage; lines) and skull_lines.json."""
import sys, os, re, json, numpy as np, cv2
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segments import SEG
from rtmlib import RTMPose
FD, MD, AH = sys.argv[1:4]; log = lambda *a: print('SF', *a, flush=True)
det = RTMPose(os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx'), model_input_size=(192, 256), backend='onnxruntime', device='cpu')
def front_points(D, mode):
    """68 face points: detected in the front view, each cast onto the front-most surface vertex within one median edge of its ray"""
    V = np.load(os.path.join(D, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); F = np.load(os.path.join(D, 'parts', 'orig_normals.npz'))['F']; hv = os.path.join(D, 'parts_zoom', 'head_vids.npy')
    H = V[np.load(hv)]; inh = np.zeros(len(V), bool); inh[np.load(hv)] = True; Fh = F[inh[F].all(1)]; edge = float(np.median(np.linalg.norm(V[Fh[:, 0]] - V[Fh[:, 1]], axis=1)))
    zd = os.path.join(D, 'parts_zoom', 'head'); c = json.load(open(os.path.join(zd, 'cameras.json')))[mode + '_e+00_a000']; im = cv2.imread(os.path.join(zd, mode + '_e+00_a000.png')); ys, xs = np.where(im.min(2) < 250)
    k, s = det(im, bboxes=[[xs.min(), ys.min(), xs.max(), ys.max()]]); k = k[0][23:91]; M = np.array(c['matrix_inv'])[:3]; q = H @ M[:, :3].T + M[:, 3]; t2 = cKDTree(q[:, :2])
    uv = np.c_[(k[:, 0] / c['res'] - 0.5) * c['scale'], (0.5 - k[:, 1] / c['res']) * c['scale']]; X = np.full((68, 3), np.nan)
    for p in range(68):
        ii = t2.query_ball_point(uv[p], edge)
        if not ii: ii = [int(t2.query(uv[p])[1])]                   # a point just off the outline (jaw against the background): the nearest surface vertex in the view
        X[p] = H[ii[int(np.argmax(q[ii, 2]))]]
    return X, H, edge, c['scale'] / c['res']
XA, HA, eA, pxA = front_points(AH, 'clay'); XE, HE, eE, pxE = front_points(MD, 'tex')
import face_lines; FL_ = face_lines.load(MD)
RUNS = {n: l['points'] for n, l in FL_.items() if l['under_skin'] and not l['cartilage']}   # the lines the naming step says lie on bone (a line over cartilage - the nose - does not place the skull)
SOFT = {'nose bridge': range(27, 31), 'nose base': range(31, 36), 'outer lip outline': list(range(48, 60)) + [48]}   # reported only
def line(P, step): L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()); tck, _ = splprep(P.T, s=0, k=min(3, len(P) - 1)); return np.array(splev(np.linspace(0, 1, max(8, int(L / step) + 1)), tck)).T
LA = {n: line(XA[list(r)], eA) for n, r in RUNS.items()}; LE = {n: line(XE[list(r)], eE) for n, r in RUNS.items()}
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, S_, Vt = np.linalg.svd(Yc.T @ Xc / len(X)); D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ D @ Vt
    s = np.trace(np.diag(S_) @ D) / (Xc ** 2).sum(1).mean(); return R, s, my - s * R @ mx
def nearest_on(P, Q):                                               # nearest place on polyline Q for each point of P
    a, b = Q[:-1], Q[1:]; d = b - a; u = np.clip(((P[:, None] - a) * d).sum(2) / ((d * d).sum(1) + 1e-30), 0, 1); C = a + u[..., None] * d; j = np.linalg.norm(P[:, None] - C, axis=2).argmin(1); return C[np.arange(len(P)), j]
ix = [i for r in RUNS.values() for i in r]; R, s, t = umeyama(XA[ix], XE[ix]); res0 = np.linalg.norm(XA[ix] @ (s * R).T + t - XE[ix], axis=1); s_start = s
tol = pxE / float(np.ptp(HE, 0).max())                              # a pixel over the head's size: the relative resolution of a detected point
for it in range(200):
    src, dst = [], []
    for n in RUNS:
        TA = LA[n] @ (s * R).T + t; src.append(LA[n]); dst.append(nearest_on(TA, LE[n]))                       # atlas samples -> the model's line
        near = nearest_on(LE[n], TA); src.append((near - t) @ R / s); dst.append(LE[n])                        # model samples <- the placed atlas line (taken back to atlas space)
    Rn, sn, tn = umeyama(np.vstack(src), np.vstack(dst)); ds = abs(sn / s - 1); R, s, t = Rn, sn, tn
    if ds < tol: break
dl = {n: float(np.median(np.linalg.norm(LA[n] @ (s * R).T + t - nearest_on(LA[n] @ (s * R).T + t, LE[n]), axis=1))) for n in RUNS}
log('points matched one to one: size %.3f, residual median %.1f mm; lines with sliding: size %.3f after %d rounds; line-to-line distance median per line (mm): %s' % (
    s_start, 1000 * np.median(res0), s, it + 1, ', '.join('%s %.1f' % (n, 1000 * v) for n, v in dl.items())))
# --- the head's bones (real bone only) and cartilages (nose, ear: not bone - they ride with the skull but are no part of it)
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
W = np.load(os.path.join(FD, 'atlas_warp.npz')); V0 = W['V0'].astype(np.float64); OB = W['obj']; names = [str(x) for x in W['names']]; sysn = np.array([str(x) for x in W['system']]); FB = W['F_bone']
K = np.load(os.path.join(FD, 'skeleton_fit.npz'), allow_pickle=True); Vfit = K['V_rigid'].astype(np.float64)
hall = [i for i, n in enumerate(names) if sysn[i] == 'bone' and (re.search(SEG['head']['bones'], n) or re.search('cartilage|Incus|Malleus|Stapes|Sinus|concha', n)) and not re.search('rib|Thyroid|Cricoid|Costal|Arytenoid|Epiglot|trachea', n, re.I)]
hc = [i for i in hall if 'cartilage' in names[i].lower()]; hb = [i for i in hall if i not in hc]; hm = np.isin(OB, hb); cm = np.isin(OB, hc); base = lambda n: re.sub(r'\.[lr]$', '', n); ob = OB[hm]
log('head: %d bones, %d cartilages kept apart (%s)' % (len(hb), len(hc), ', '.join(sorted({base(names[i]) for i in hc}))))
# inside / outside from a solid model of the head, not from the nearest surface normal (that called points deep in the
# head "outside" next to the ears): the model's surface inside the head's box is drawn into a grid one surface edge
# fine, the box's bottom is capped (the neck is cut there), enclosed space is filled, and the signed distance to the
# solid's boundary is read from the grid (+ inside)
from scipy import ndimage
MVa = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MFa = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']; hstep = eE
glo = HE.min(0) - 4 * hstep; ghi = HE.max(0) + 4 * hstep; gn = np.ceil((ghi - glo) / hstep).astype(int) + 1
fin = ((MVa[MFa] > glo) & (MVa[MFa] < ghi)).all(2).any(1); T = MVa[MFa[fin]]; nsub = np.clip(np.ceil(np.linalg.norm(T[:, 1] - T[:, 0], axis=1).clip(np.linalg.norm(T[:, 2] - T[:, 0], axis=1)) / (hstep / 2)).astype(int), 1, 64)
occ = np.zeros(gn, bool)
for n_ in np.unique(nsub):                                          # every triangle sampled finer than half a grid cell
    Tn = T[nsub == n_]; u, v = np.meshgrid(np.arange(n_ + 1) / n_, np.arange(n_ + 1) / n_); m_ = (u + v) <= 1 + 1e-9; u, v = u[m_], v[m_]
    Pn = (Tn[:, None, 0] * (1 - u - v)[None, :, None] + Tn[:, None, 1] * u[None, :, None] + Tn[:, None, 2] * v[None, :, None]).reshape(-1, 3); ii = np.rint((Pn - glo) / hstep).astype(int)
    ok_ = ((ii >= 0) & (ii < gn)).all(1); occ[ii[ok_, 0], ii[ok_, 1], ii[ok_, 2]] = True
for seal in range(1, 33):                                           # the surface has small gaps (hair against skin): thicken it by the fewest cells that
    thick = ndimage.binary_dilation(occ, iterations=seal); cap = thick.copy(); cap[:, :, 0] = True; fill = ndimage.binary_fill_holes(cap)   # make it enclose a space larger than the
    if fill.sum() - gn[0] * gn[1] - thick.sum() > thick.sum(): break                                                                          # thickened shell itself, fill, then
else: print('WARNING skull_line_fit.py: the solid did not close within 33 sealing steps; the widest sealing is used and the result may be wrong', flush=True)
solid = ndimage.binary_erosion(fill, iterations=seal, border_value=1) | occ; solid[:, :, 0] = solid[:, :, 1]                                   # take the thickening off again
sdf = (ndimage.distance_transform_edt(solid) - ndimage.distance_transform_edt(~solid)) * hstep
log('head solid: grid %s at %.2f mm; gaps sealed by %d cells (%.1f mm); solid volume %.0f cm3 (surface cells alone %.0f cm3)' % (tuple(int(g) for g in gn), 1000 * hstep, seal, 1000 * seal * hstep, 1e6 * solid.sum() * hstep ** 3, 1e6 * occ.sum() * hstep ** 3))
def depth(P): return ndimage.map_coordinates(sdf, ((P - glo) / hstep).T, order=1, mode='nearest')   # + inside the head (skin, or hair where hair covers it)
hull = []
for b in hb:                                                        # each bone's extent: its convex-hull vertices
    ii = np.flatnonzero(OB == b)
    try: hull.append(ii[np.unique(ConvexHull(V0[ii]).vertices)] if len(ii) >= 8 else ii)
    except Exception: hull.append(ii)
hull = np.concatenate(hull); R_l, s_l, t_l = R.copy(), float(s), t.copy()
# --- strict rule: no bone outside the surface. Same line matches, plus every bone hull point's depth outside the surface
# as a penalty whose scale is halved until the worst point is inside to within the surface's own resolution (one edge)
def place(x, P): return P @ (np.exp(x[3]) * (Rot.from_rotvec(x[:3]).as_matrix() @ R_l)).T + x[4:7]
x = np.r_[0, 0, 0, np.log(s_l), t_l]; sig_l = max(float(np.median(list(dl.values()))), eE); sig_in = sig_l; hist = []
for outer in range(16):
    Rc = Rot.from_rotvec(x[:3]).as_matrix() @ R_l; sc = float(np.exp(x[3])); tc = x[4:7]; src, dst = [], []
    for n in RUNS:
        TA = LA[n] @ (sc * Rc).T + tc; src.append(LA[n]); dst.append(nearest_on(TA, LE[n])); near = nearest_on(LE[n], TA); src.append((near - tc) @ Rc / sc); dst.append(LE[n])
    S_, D_ = np.vstack(src), np.vstack(dst)
    f = lambda x_: np.r_[((place(x_, S_) - D_) / sig_l).ravel(), np.maximum(0, -depth(place(x_, V0[hull]))) / sig_in]
    x = least_squares(f, x, method='trf', loss='linear').x; dp = depth(place(x, V0[hull])); worst = float(max(0, -dp.min())); hist.append((sig_in, worst, float(np.exp(x[3]))))
    log('   inside round %d: penalty scale %.4f mm, worst outside %.2f mm, outside share %.2f%%, size %.3f' % (outer, 1000 * sig_in, 1000 * worst, 100 * (dp < 0).mean(), np.exp(x[3])))
    if worst < eE: break
    sig_in /= 2
else: log('WARNING: the inside penalty was halved 16 times and the worst point is still %.2f mm outside; the last round is used' % (1000 * worst))
R = Rot.from_rotvec(x[:3]).as_matrix() @ R_l; s = float(np.exp(x[3])); t = x[4:7]
log('strict inside: %d rounds, penalty scale %.2f -> %.4f mm; worst bone point outside %.2f mm (surface edge %.2f mm); size %.3f -> %.3f; skull moved by median %.1f mm' % (len(hist), 1000 * sig_l, 1000 * sig_in, 1000 * worst, 1000 * eE,
    s_l, s, 1000 * np.median(np.linalg.norm(V0[hull] @ (s * R).T + t - (V0[hull] @ (s_l * R_l).T + t_l), axis=1))))
both = hm | cm; TR = {'lines only': (R_l, s_l, t_l), 'lines + bones inside': (R, s, t)}; PL = {'slice route': Vfit}
for lab, (R_, s_, t_) in TR.items(): Vx = Vfit.copy(); Vx[both] = V0[both] @ (s_ * R_).T + t_; PL[lab] = Vx
ALL = {**RUNS, **SOFT}; rep = {'size': s, 'size_lines_only': s_l, 'size_points_only': float(s_start), 'inside_rounds': len(hist), 'lines': {}}; ta = cKDTree(V0[hm])
o_, c1 = OB == names.index('Occipital bone'), OB == names.index('Atlas (C1)'); g0 = cKDTree(V0[o_]).query(V0[c1])[0]; nr = g0 < np.percentile(g0, 10)
for lab, Vx in PL.items():
    tb = cKDTree(Vx[hm]); rows = {}
    for n, r in ALL.items():
        Le = line(XE[list(r)], eE); La = line(XA[list(r)], eA); d, k = tb.query(Le); da = ta.query(La)[0]; u, c = np.unique([base(names[b]) for b in ob[k]], return_counts=True); o = np.argsort(-c)
        rows[n] = {'depth_mm': 1000 * float(np.median(d)), 'atlas_scaled_mm': 1000 * float(s * np.median(da)), 'under': {u[j]: float(c[j] / c.sum()) for j in o[:4]}}
    dep = depth(Vx[hm]); gap = cKDTree(Vx[o_]).query(Vfit[c1])[0][nr]; dpc = depth(Vx[cm]); ld = None
    if lab in TR:
        R_, s_, t_ = TR[lab]; ld = {}
        for n in RUNS: TA = LA[n] @ (s_ * R_).T + t_; ld[n] = 1000 * float(np.median(np.linalg.norm(TA - nearest_on(TA, LE[n]), axis=1)))
    rep['lines'][lab] = rows; rep[lab] = {'outside_surface_share': float((dep < 0).mean()), 'outside_worst_mm': 1000 * float(max(0, -dep.min())), 'cartilage_outside_share': float((dpc < 0).mean()), 'neck_gap_mm': 1000 * float(np.median(gap)),
                                          'neck_gap_atlas_scaled_mm': 1000 * float(s * np.median(g0[nr])), 'line_distance_mm': ld}
    log('%s: depth to bone under each line, hers / atlas scaled (mm): %s' % (lab, '; '.join('%s %.1f / %.1f' % (n, v['depth_mm'], v['atlas_scaled_mm']) for n, v in rows.items())))
    log('   bone outside the head surface %.2f%% (worst %.1f mm); cartilage outside %.1f%%; gap to the first neck bone %.1f mm (atlas scaled %.1f); under her lips: %s%s' % (100 * rep[lab]['outside_surface_share'], rep[lab]['outside_worst_mm'],
        100 * rep[lab]['cartilage_outside_share'], rep[lab]['neck_gap_mm'], rep[lab]['neck_gap_atlas_scaled_mm'], ', '.join('%s %.0f%%' % (a, 100 * b) for a, b in rows['outer lip outline']['under'].items()),
        '' if ld is None else '; line-to-line distance (mm): ' + ', '.join('%s %.1f' % (n, v) for n, v in ld.items())))
np.savez(os.path.join(FD, 'skull_lines.npz'), V=PL['lines + bones inside'].astype(np.float32), V_lines_only=PL['lines only'].astype(np.float32), V_fit=Vfit.astype(np.float32), F_head=FB[hm[FB].all(1)], F_cartilage=FB[cm[FB].all(1)],
         R=R, s=s, t=t, points_model=XE, points_atlas=XA, sdf=sdf.astype(np.float32), sdf_origin=glo, sdf_step=hstep, **{'line_model_' + n.replace(' ', '_'): v for n, v in LE.items()}, **{'line_atlas_placed_' + n.replace(' ', '_'): v @ (s * R).T + t for n, v in LA.items()})
json.dump(rep, open(os.path.join(FD, 'skull_lines.json'), 'w'), indent=1)
