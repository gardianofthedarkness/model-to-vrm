"""Minimal-area slice planes along a path through a body segment (shared by za_slices.py, fit_slices.py, atlas_warp.py;
the method of hit_slices.py):
  path      a polyline (the atlas's joint axis, or the model's skeleton joints between the segment's ends); level t is the
            fraction of its length
  plane     at each level: the plane through the path point whose cut of the segment's surface (per angle the first surface
            out from the point, gaps filled around the circle) has the least area. Searched over the plane's two tilt
            angles (Powell), starting from the previous level's plane (the first from the path direction); a plane may
            turn from its neighbour by at most atan(level spacing / neighbour's mean radius), so neighbouring planes
            cannot cross inside the body
  frame     e1 toward the roll landmark (projected into the plane), e2 = n x e1
Functions take vertices V, faces F (only the segment's own surface) and return per level p, n, e1, e2, the polar radii,
the area and the tilt from the path direction."""
import numpy as np
from scipy.optimize import minimize
TH = 72
def frame(n, ref):
    e1 = ref - n * (ref @ n); e1 = e1 / np.linalg.norm(e1); return e1, np.cross(n, e1)
def cut(V, F, p, n, e1, e2, ret_face=False):
    """crossing segments (k, 2, 2) in the plane's (e1, e2) coordinates of the triangles F with the plane (p, n)"""
    T = V[F] - p; d = T @ n; s = np.sign(d); cr = s[:, [0, 1, 2]] != s[:, [1, 2, 0]]; k = np.flatnonzero(cr.sum(1) == 2)
    if not len(k): return (np.zeros((0, 2, 2)), k) if ret_face else np.zeros((0, 2, 2))
    P = []
    for e in range(3):
        a, b = e, (e + 1) % 3; w = d[k, a] / (d[k, a] - d[k, b] + 1e-30); q = T[k, a] + w[:, None] * (T[k, b] - T[k, a])
        P.append(np.where(cr[k, e][:, None], q, np.nan))
    P = np.stack(P, 1); ok = ~np.isnan(P[:, :, 0]); o = np.argsort(~ok, axis=1, kind='stable')[:, :2]
    out = np.take_along_axis(P, o[:, :, None], 1); out = np.stack([out @ e1, out @ e2], -1)
    return (out, k) if ret_face else out
def polar_first(segs):
    """per angle bin: the first surface out from the plane's origin (smallest radius); empty bins nan"""
    if not len(segs): return np.full(TH, np.nan)
    u = np.linspace(0, 1, 9)[None, :, None]; Q = (segs[:, :1] + u * (segs[:, 1:] - segs[:, :1])).reshape(-1, 2)
    b = ((np.arctan2(Q[:, 1], Q[:, 0]) + np.pi) / (2 * np.pi) * TH).astype(int) % TH; r = np.hypot(Q[:, 0], Q[:, 1])
    out = np.full(TH, np.inf); np.minimum.at(out, b, r); out[np.isinf(out)] = np.nan; return out
def fill(r):
    ok = np.isfinite(r)
    if ok.sum() < 2: return r
    k = np.arange(TH); return np.interp(k, np.r_[k[ok] - TH, k[ok], k[ok] + TH], np.tile(r[ok], 3))
def area(V, F, p, n, ref):
    e1, e2 = frame(n, ref); r = fill(polar_first(cut(V, F, p, n, e1, e2)))
    return (0.5 * np.nansum(r ** 2) * (2 * np.pi / TH) if np.isfinite(r).any() else np.inf), r
def tilt(n0, ab, ref):
    e1, e2 = frame(n0, ref if abs(ref @ n0) < 0.99 * np.linalg.norm(ref) else np.cross(n0, [0, 0, 1.0]))
    n = n0 + ab[0] * e1 + ab[1] * e2; return n / np.linalg.norm(n)
class Path:
    def __init__(self, pts):
        pts = np.asarray(pts, float); pts = pts[np.r_[True, np.linalg.norm(np.diff(pts, axis=0), axis=1) > 0]]
        self.pts = pts; self.seglen = np.linalg.norm(np.diff(pts, axis=0), axis=1); self.s = np.r_[0, np.cumsum(self.seglen)]; self.L = float(self.s[-1])
    def at(self, t):                                                # point and unit direction at fraction t (held straight past the ends)
        d = t * self.L; k = int(np.clip(np.searchsorted(self.s, d, side='right') - 1, 0, len(self.seglen) - 1))
        u = (self.pts[k + 1] - self.pts[k]) / self.seglen[k]; return self.pts[k] + u * (d - self.s[k]), u
def min_area_planes(V, F, path, ts, ref, log=None):
    """per level t in ts (increasing): p, n, e1, e2, radii (TH), area, tilt from the path direction (degrees)"""
    out = {k: [] for k in ('p', 'n', 'e1', 'e2', 'r', 'area', 'tilt')}; prev = None
    ds = path.L * (np.median(np.diff(ts)) if len(ts) > 1 else 1.0)
    for t in ts:
        p, d = path.at(t); n0 = d if prev is None else prev[0]
        A0, r0 = area(V, F, p, n0, ref); Rm = float(np.nanmean(r0)) if np.isfinite(r0).any() else ds
        lim = ds / (prev[1] if prev is not None else Rm)            # tan of the largest turn from the neighbour
        f = lambda ab: area(V, F, p, tilt(n0, np.clip(ab, -lim, lim), ref), ref)[0]
        res = minimize(f, np.zeros(2), method='Powell', bounds=[(-lim, lim)] * 2) if np.isfinite(A0) else None
        n = tilt(n0, res.x, ref) if res is not None and res.fun < A0 else n0
        A, r = area(V, F, p, n, ref); e1, e2 = frame(n, ref)
        for k_, v_ in zip(out, (p, n, e1, e2, r, A, np.degrees(np.arccos(np.clip(n @ d, -1, 1))))): out[k_].append(v_)
        prev = (n, float(np.nanmean(r)) if np.isfinite(r).any() else Rm)
    out = {k: np.array(v) for k, v in out.items()}
    if log: log('minimal-area planes: %d levels, tilt from the path median %.1f deg, max %.1f deg' % (len(ts), np.median(out['tilt']), out['tilt'].max()))
    return out
def plane_coords(P, pl, k):                                        # world points -> (along normal, e1, e2) of plane k
    d = P - pl['p'][k]; return d @ pl['n'][k], d @ pl['e1'][k], d @ pl['e2'][k]
