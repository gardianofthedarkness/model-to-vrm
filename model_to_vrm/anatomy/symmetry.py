"""The figure's own left-right symmetry, measured on its skin, and what follows from it for the bone fits.
A body is built the same on both sides; a detector looking at the left side and at the right side makes two independent
errors. Where the skin is its own mirror image, a point found on one side is therefore also a measurement of its partner
on the other side. How far that holds is measured, place by place - nothing is assumed:
  plane     the mirror plane: the skin reflected (x -> -x) and laid back onto itself by rigid ICP (Besl and McKay 1992,
            trimesh.registration.icp, no scaling); the reflection that results has the plane's normal and offset
  residual  per skin vertex, the distance from its mirror image to the skin: small where the figure is symmetric, large
            where it is not (an arm held differently, a strand of hair, a robe's fold)
  local     at a set of points, the median residual of the flesh vertices nearest each point (their Voronoi cells)
  pool      a left point x_l and a right point x_r with the detector's spread sigma (from their own disagreement after
            mirroring, median / sqrt 2) and the local residual m: each side's best place is the weighted mean of its own
            point (weight 1 / sigma^2) and the partner's mirror image (weight 1 / (sigma^2 + m^2): the partner's error
            plus how far the figure itself departs from symmetry there). m is the larger of the skin's residual and
            what the pair's own disagreement leaves after two detector errors, sqrt(|M x_l - x_r|^2 - 2 sigma^2): two
            hands posed differently disagree by more than the detector can explain. A symmetric place gets the mean of
            both; a place where the figure is not symmetric keeps its own point.
Used by part_bone_fit.py (hands, feet) and body_assembly.py. Cached in <model dir>/parts/symmetry.npz."""
import os, numpy as np
from scipy.spatial import cKDTree
def figure_mirror(MD):
    f = os.path.join(MD, 'parts', 'symmetry.npz'); MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64)
    if os.path.exists(f) and len(np.load(f)['resid']) == len(MV): Z = np.load(f); n, d, resid = Z['n'], float(Z['d']), Z['resid'].astype(np.float64)
    else:
        import trimesh.registration as reg
        M = np.diag([-1.0, 1, 1]); T, _, _ = reg.icp(MV @ M, MV, scale=False, reflection=False, max_iterations=100)       # every vertex; rigid only
        A = T[:3, :3] @ M; w, v = np.linalg.eig(A); n = np.real(v[:, np.argmin(np.real(w))]); n /= np.linalg.norm(n); n = n if n[0] > 0 else -n; d = float(T[:3, 3] @ n) / 2
        resid = cKDTree(MV).query(MV - 2 * ((MV @ n) - d)[:, None] * n)[0]; np.savez(f, n=n, d=d, resid=resid.astype(np.float32))
    isf = os.path.join(MD, 'flesh', 'islands.npz'); flesh = np.load(isf)['island'] >= 0 if os.path.exists(isf) else np.ones(len(MV), bool)
    return dict(n=n, d=d, resid=resid, MV=MV, flesh=flesh if flesh.any() else np.ones(len(MV), bool))
def reflect(S, P): return P - 2 * ((P @ S['n']) - S['d'])[..., None] * S['n']
def reflect_dir(S, A): return A - 2 * (A @ S['n'])[..., None] * S['n']
def local(S, pts):
    """median mirror residual of the flesh vertices nearest each point; a point that owns none takes the figure's median."""
    P = S['MV'][S['flesh']]; r = S['resid'][S['flesh']]; j = cKDTree(pts).query(P)[1]; out = np.full(len(pts), np.median(r))
    for k in np.unique(j): out[k] = np.median(r[j == k])
    return out
def pool(S, XL, XR, sig=None):
    """left and right points (k, 3), partners row by row -> pooled left, pooled right, detector spread, local residuals.
    sig: the detector's own spread per point where it is known (across views); otherwise the pairs' own disagreement.
    Rows with a missing point (not finite) are returned as they came."""
    XL, XR = np.array(XL, float), np.array(XR, float); ok = np.isfinite(XL).all(1) & np.isfinite(XR).all(1); PL, PR = XL.copy(), XR.copy()
    if not ok.any(): return PL, PR, np.nan, np.zeros(len(XL))
    ML, MR = reflect(S, XL[ok]), reflect(S, XR[ok]); dis = np.linalg.norm(ML - XR[ok], axis=1); sg = np.full(ok.sum(), np.median(dis) / np.sqrt(2)) if sig is None else np.broadcast_to(np.asarray(sig, float), (len(XL),))[ok]
    m = np.zeros(len(XL)); m[ok] = np.sqrt(np.maximum(((local(S, XL[ok]) + local(S, XR[ok])) / 2) ** 2, dis ** 2 - 2 * sg ** 2))       # the pair's disagreement beyond what two detector errors explain is the figure's own
    w = (sg ** 2 / np.maximum(sg ** 2 + m[ok] ** 2, np.finfo(float).tiny))[:, None]; PL[ok] = (XL[ok] + w * MR) / (1 + w); PR[ok] = (XR[ok] + w * ML) / (1 + w)
    return PL, PR, float(np.median(sg)), m
def pool_dirs(S, AL, AR, m, sig):
    """unit directions, partners row by row, pooled with the weights of their places (m, sig from pool)."""
    w = (sig ** 2 / np.maximum(sig ** 2 + m ** 2, np.finfo(float).tiny))[:, None]; ML, MR = reflect_dir(S, AL), reflect_dir(S, AR); MR *= np.sign((MR * AL).sum(1))[:, None]; ML *= np.sign((ML * AR).sum(1))[:, None]
    PL, PR = AL + w * MR, AR + w * ML; return PL / np.linalg.norm(PL, axis=1, keepdims=True), PR / np.linalg.norm(PR, axis=1, keepdims=True)
BODY_L = [5, 7, 9, 11, 13, 15, 17, 18, 19]; BODY_R = [6, 8, 10, 12, 14, 16, 20, 21, 22]      # COCO-WholeBody: shoulder .. ankle, big toe, small toe, heel
def body_joints(MD, S=None, log=None):
    """body/joints3d.npz with its left and right body joints pooled."""
    S = S or figure_mirror(MD); X = np.load(os.path.join(MD, 'body', 'joints3d.npz'))['X'].astype(np.float64).copy(); PL, PR, sig, m = pool(S, X[BODY_L], X[BODY_R])
    if log: log('symmetry: body joints pooled left with right - detector spread %.1f mm, the skin off its mirror image there %.1f mm median (max %.1f); joints moved %.1f mm median' % (1000 * sig, 1000 * np.median(m), 1000 * m.max(), 1000 * np.median(np.linalg.norm(np.r_[PL - X[BODY_L], PR - X[BODY_R]], axis=1))))
    X[BODY_L], X[BODY_R] = PL, PR; return X
def hand_points(MD, sd, S=None, log=None):
    """parts_zoom/hand_<sd>/points3d.npz X with the other hand's points pooled in."""
    S = S or figure_mirror(MD); o = {'l': 'r', 'r': 'l'}[sd]; fo = os.path.join(MD, 'parts_zoom', 'hand_' + o, 'points3d.npz'); X = np.load(os.path.join(MD, 'parts_zoom', 'hand_' + sd, 'points3d.npz'))['X'].astype(np.float64)
    if not os.path.exists(fo): return X
    Zs, Zo = np.load(os.path.join(MD, 'parts_zoom', 'hand_' + sd, 'points3d.npz')), np.load(fo); P, _, sig, m = pool(S, X, Zo['X'].astype(np.float64), sig=np.sqrt((Zs['rms'] ** 2 + Zo['rms'] ** 2) / 2))   # the detector's spread across views
    if log: log('symmetry: hand points pooled with the other hand - spread %.1f mm, skin off its mirror image %.1f mm median (max %.1f); points moved %.1f mm median' % (1000 * sig, 1000 * np.median(m), 1000 * m.max(), 1000 * np.nanmedian(np.linalg.norm(P - X, axis=1))))
    return P
def toes(MD, sd, S=None, log=None):
    """parts_zoom/foot_<sd>/toes.npz tip, centre, axis with the other foot's pooled in."""
    S = S or figure_mirror(MD); o = {'l': 'r', 'r': 'l'}[sd]; fo = os.path.join(MD, 'parts_zoom', 'foot_' + o, 'toes.npz'); T = dict(np.load(os.path.join(MD, 'parts_zoom', 'foot_' + sd, 'toes.npz')))
    if not os.path.exists(fo): return T
    O = np.load(fo); n = len(T['tip']); P, _, sig, m = pool(S, np.r_[T['tip'], T['centre']].astype(float), np.r_[O['tip'], O['centre']].astype(float)); ax, _ = pool_dirs(S, T['axis'].astype(float), O['axis'].astype(float), m[n:], sig)
    if log: log('symmetry: toes pooled with the other foot - spread %.1f mm, skin off its mirror image %.1f mm median; tips moved %.1f mm median' % (1000 * sig, 1000 * np.median(m), 1000 * np.median(np.linalg.norm(P[:n] - T['tip'], axis=1))))
    T['tip'], T['centre'], T['axis'] = P[:n], P[n:], ax; return T
