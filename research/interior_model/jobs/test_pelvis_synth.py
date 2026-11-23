"""Synthetic check of t0a_pelvis_job.measure_plane: an elliptical body with known skin, fat, a lean compartment, two hip
bones and a sacrum, a pocket of bowel gas and a table under the body. All numbers made up; no scan."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t0a_pelvis_job as J
vox = 1.0; n = 360; X, Y = np.meshgrid(np.arange(n) - n / 2, np.arange(n) - n / 2, indexing='ij')   # i -> x (left), j -> y (anterior)
img = np.full((n, n), -1000.0)
def ell(a, b, cx=0, cy=0): return ((X - cx) / a) ** 2 + ((Y - cy) / b) ** 2 <= 1
DERM, FATT = 2.0, 18.0; A, B = 150, 110
body = ell(A, B); img[body] = 40                                   # skin and lean
img[ell(A - DERM, B - DERM)] = -100                                # fat layer starts after the dermis
img[ell(A - DERM - FATT, B - DERM - FATT)] = 50                    # lean compartment
gas = ell(12, 8, 0, 40); img[gas] = -950                           # bowel gas inside the body
img[(Y < -B - 12) & (Y > -B - 20) & (np.abs(X) < 170)] = 300       # table, separated by an air gap
hl = ell(25, 45, 70, -10); hr = ell(25, 45, -70, -10); sac = ell(30, 22, 0, -60)
img[hl | hr | sac] = 700
rng = np.random.default_rng(0); img += rng.normal(0, 8, img.shape)
CT = img[:, :, None].repeat(3, 2); aff = np.eye(4); aff[:3, :3] *= vox
mk = lambda m: m[:, :, None].repeat(3, 2)
muscle = ell(A - DERM - FATT, B - DERM - FATT) & ~ell(A - DERM - FATT - 15, B - DERM - FATT - 15)
labs = dict(hip_left=mk(hl), hip_right=mk(hr), sacrum=mk(sac), gluteus_maximus_left=mk(muscle))
rays, flags = J.measure_plane(CT, labs, aff, 1.0)
print('flags', flags)
def expect(th):                                                    # radius of an ellipse along direction th from its centre
    dx, dy = np.sin(th), np.cos(th); return 1 / np.sqrt((dx / A) ** 2 + (dy / B) ** 2)
c = np.argwhere(J.__dict__.get('_dummy', body) if False else body).mean(0)
err = {'r_skin': [], 'dermis': [], 'd_fat': [], 'd_SF': [], 'd_SM': []}
for o in rays:
    th = 2 * np.pi * o['ray'] / J.NRAY
    if 'r_skin' in o: err['r_skin'].append(o['r_skin'] - expect(th))
    for q, truth in (('dermis', DERM), ('d_fat', FATT), ('d_SF', DERM + FATT), ('d_SM', DERM + FATT)):
        if o.get(q) is not None:
            # along an oblique ray through an ellipse the layer is a little thicker than its normal thickness; compare loosely
            err[q].append(o[q] - truth)
for q, v in err.items():
    v = np.array(v); print('%-7s n %2d  median error %+.2f mm  max |error| %.2f mm' % (q, len(v), np.median(v), np.abs(v).max() if len(v) else np.nan))
print('rays hitting bone', sum('d_SB' in o for o in rays), 'no bone', sum(o.get('no_bone', False) for o in rays), 'fascia clear', sum(bool(o.get('fascia_clear')) for o in rays))
