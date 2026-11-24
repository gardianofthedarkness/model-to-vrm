"""Does the rectus tube's fibre field want the tube to get LONGER when it contracts?  For each tet, the tube axis is the
origin->insertion harmonic-field gradient; the fibre is the one newton_muscle_active.py uses (stored fibre, else the field
gradient, else the nearest tet's).  Under the active strain Fa (det 1) a unit vector along the axis has preferred stretch
|Fa axis|, which is < 1 only when the fibre is within ~arccos(sqrt(1/3)) = 54.7 deg of the axis (for small g).  Tets that
prefer to lengthen push on fixed attachments, i.e. compress the tube, which a slender soft tube relieves by bending.
  E:\\splat\\.venv-sim\\Scripts\\python.exe rectus_fibre_check.py <rectus mesh npz> [activation]"""
import sys, numpy as np
from scipy.spatial import cKDTree
M = np.load(sys.argv[1]); a = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
X0, T, phi, fib = M['X'], M['T'], M['phi'], M['fib']
GMAX = 1 - (1.27 / 2.0) * 1.07; g = a * GMAX
Dm = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2); Dinv = np.linalg.inv(Dm)
gphi = np.einsum('tji,tj->ti', Dinv, np.stack([phi[T[:, 1]] - phi[T[:, 0]], phi[T[:, 2]] - phi[T[:, 0]], phi[T[:, 3]] - phi[T[:, 0]]], 1))
ng = np.linalg.norm(gphi, axis=1); okg = ng > 1e-12; axis = np.zeros_like(gphi); axis[okg] = gphi[okg] / ng[okg, None]
cen = X0[T].mean(1); axis[~okg] = axis[okg][cKDTree(cen[okg]).query(cen[~okg])[1]]
oks = np.linalg.norm(fib, axis=1) > 0; fv = np.where(oks[:, None], fib, gphi); nf = np.linalg.norm(fv, axis=1); okf = nf > 1e-12
ff = np.zeros_like(fv); ff[okf] = fv[okf] / nf[okf, None]; ff[~okf] = ff[okf][cKDTree(cen[okf]).query(cen[~okf])[1]]
ang = np.degrees(np.arccos(np.clip(np.abs((ff * axis).sum(1)), 0, 1)))
c = (ff * axis).sum(1); stretch = np.sqrt((1 - g) ** 2 * c ** 2 + (1 - g) ** -1 * (1 - c ** 2))
vol = np.abs(np.linalg.det(Dm)) / 6; w = vol / vol.sum()
print('RF %d tets; fibres from stored field %d, from the axis gradient %d' % (len(T), oks.sum(), (~oks).sum()))
print('RF fibre-to-axis angle (volume-weighted) p10/50/90: %.1f / %.1f / %.1f deg; volume share over 54.7 deg: %.1f%%' % (
    *[np.interp(p, np.cumsum(w[np.argsort(ang)]), np.sort(ang)) for p in (0.1, 0.5, 0.9)], 100 * w[ang > 54.7].sum()))
print('RF at activation %.2f: preferred stretch along the axis, volume-weighted mean %.3f; volume share that wants to LENGTHEN: %.1f%%' % (a, (w * stretch).sum(), 100 * w[stretch > 1].sum()))
for lo, hi in ((0, 0.1), (0.1, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)):
    m = (phi[T].mean(1) >= lo) & (phi[T].mean(1) < hi)
    print('RF   along the muscle %.1f-%.1f: angle median %.1f deg, lengthening share %.1f%%' % (lo, hi, np.median(ang[m]), 100 * w[m & (stretch > 1)].sum() / w[m].sum()))
