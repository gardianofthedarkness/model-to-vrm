"""Organ-shape check (2026-11-26): the new labelling keeps names only for eyes and brain, the rest is tissue type. Do the
organs' outlines survive as tissue boundaries? For every expert outline in HaN-Seg (glands, muscles, eyes, mandible, oral
cavity, ...), on the CT's own grid: share of the organ's surface lying within 1 mm of a tissue boundary, and the median
distance, for
  old  hard tissue classes (median filter, thresholds) after nearest-neighbour onto 1 mm  (today's labels)
  new  continuous field: raw CT interpolated onto 1 mm, boundaries at -30 HU (fat|lean), 150 HU (bone), -500 HU (air)
Eyes and brain also carry identity, so for them the identity label's surface counts too. Organs inside uniform muscle
cannot show as tissue boundaries; the table says which. args: <HaN-Seg.zip> <head_v1 dir> <out dir> <case>..."""
import os, sys, json, zipfile, re, numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
zp, hv, out = sys.argv[1:4]; cases = sys.argv[4:]; Z = zipfile.ZipFile(zp); names = Z.namelist(); os.makedirs(out, exist_ok=True)
meta, tasks = {}, {}
for f in os.listdir(hv):
    if f.endswith('.json'): d = json.load(open(os.path.join(hv, f))); meta.update(d['stats']); tasks = d['tasks']
LV = (-30.0, 150.0, -500.0)
def read(n):
    p = os.path.join(out, '_t.nrrd'); open(p, 'wb').write(Z.read(n)); im = sitk.ReadImage(p); os.remove(p); return im
def sample(v, f, n=400000, seed=0):
    if len(f) == 0: return v
    t = v[f]; ar = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1); rng = np.random.default_rng(seed)
    k = rng.choice(len(f), n, p=ar / ar.sum()); u = rng.random((n, 2)); m = u.sum(1) > 1; u[m] = 1 - u[m]
    return t[k, 0] + u[:, :1] * (t[k, 1] - t[k, 0]) + u[:, 1:] * (t[k, 2] - t[k, 0])
def iso(vol, lvl, sp):
    if vol.max() <= lvl or vol.min() >= lvl: return np.zeros((0, 3))
    v, f, _, _ = measure.marching_cubes(vol.astype(np.float32), lvl, spacing=sp); return sample(v, f)
def resample(vol, sp, new, order):
    shape = [int(np.floor(n * s / new)) for n, s in zip(vol.shape, sp)]; ax = [((np.arange(m) + 0.5) * new) / s - 0.5 for m, s in zip(shape, sp)]
    return ndi.map_coordinates(vol, np.meshgrid(*ax, indexing='ij'), order=order, mode='nearest')
rows = {}
for c in cases:
    ct = read([x for x in names if x.endswith('%s/%s_IMG_CT.nrrd' % (c, c))][0]); A = sitk.GetArrayFromImage(ct).transpose(2, 1, 0).astype(np.float32); sp = np.array(ct.GetSpacing())
    oars = sorted({re.search(r'_OAR_(.+)\.seg\.nrrd$', n).group(1) for n in names if '/%s/' % c in n and n.endswith('.seg.nrrd')})
    masks = {}
    for o in oars:
        im = read([x for x in names if x.endswith('%s/%s_OAR_%s.seg.nrrd' % (c, c, o))][0])
        if im.GetNumberOfComponentsPerPixel() > 1: a = sitk.GetArrayFromImage(im).any(-1).astype(np.uint8); j = sitk.GetImageFromArray(a); j.CopyInformation(im); im = j
        m = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(im > 0, sitk.sitkUInt8), ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0)).transpose(2, 1, 0) > 0
        if m.sum() > 30: masks[o] = m
    allm = np.logical_or.reduce(list(masks.values())); ii = np.argwhere(allm); lo = np.maximum(ii.min(0) - [20, 20, 5], 0); hi = np.minimum(ii.max(0) + [20, 20, 5], A.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi)); Ab = A[box]; body = ndi.binary_fill_holes(ndi.median_filter(Ab, 3) > -500); Ab_b = np.where(body, Ab, -1000.0); S = ndi.median_filter(Ab, 3)
    F1 = resample(Ab_b, sp, 1.0, 1); sh1 = 0.5 * 1.0 - 0.5 * sp
    new_b = np.concatenate([iso(F1, l, (1.0, 1.0, 1.0)) for l in LV]) + sh1
    old_b = []
    for l in LV:
        cl = resample((body & (S > l)).astype(np.float32), sp, 1.0, 0); old_b.append(iso(np.pad(cl, 1), 0.5, (1.0, 1.0, 1.0)) - 1.0 + sh1)
    old_b = np.concatenate(old_b); tn, to = cKDTree(new_b), cKDTree(old_b)
    for o, m in masks.items():
        mb = m[box]; ref = iso(mb.astype(np.float32), 0.5, sp)[:60000]
        if len(ref) < 50: continue
        dn = tn.query(ref, workers=-1)[0]; do = to.query(ref, workers=-1)[0]
        rows.setdefault(o, []).append(dict(case=c, ml=round(float(mb.sum() * np.prod(sp)) / 1000, 1), new_within_1mm=round(float((dn <= 1).mean()), 3), old_within_1mm=round(float((do <= 1).mean()), 3),
                                          new_median_mm=round(float(np.median(dn)), 2), old_median_mm=round(float(np.median(do)), 2)))
    print(c, len(masks), 'organs', flush=True)
summ = {o: dict(people=len(v), ml=round(float(np.median([r['ml'] for r in v])), 1), old_within_1mm=round(float(np.median([r['old_within_1mm'] for r in v])), 3), new_within_1mm=round(float(np.median([r['new_within_1mm'] for r in v])), 3),
               old_median_mm=round(float(np.median([r['old_median_mm'] for r in v])), 2), new_median_mm=round(float(np.median([r['new_median_mm'] for r in v])), 2)) for o, v in rows.items()}
json.dump(dict(summary=summ, rows=rows), open(os.path.join(out, 'organ_shape_test.json'), 'w'), indent=1)
for o, s in sorted(summ.items(), key=lambda x: -x[1]['new_within_1mm']): print('%-22s %6.1f ml  within 1 mm: old %.2f new %.2f   median: old %.2f new %.2f mm' % (o, s['ml'], s['old_within_1mm'], s['new_within_1mm'], s['old_median_mm'], s['new_median_mm']))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ks = sorted(summ, key=lambda o: summ[o]['new_within_1mm']); y = np.arange(len(ks))
fig, ax = plt.subplots(figsize=(10, max(4, 0.32 * len(ks))))
ax.barh(y - 0.2, [summ[k]['old_within_1mm'] for k in ks], 0.38, color='#b0b7c3', label='old labels (hard, 1 mm)'); ax.barh(y + 0.2, [summ[k]['new_within_1mm'] for k in ks], 0.38, color='#2a6fdb', label='continuous field (1 mm)')
ax.set_yticks(y); ax.set_yticklabels(ks, fontsize=8); ax.set_xlabel('share of the expert organ surface within 1 mm of a tissue boundary'); ax.set_xlim(0, 1); ax.legend(loc='lower right', frameon=False, fontsize=8)
ax.set_title('Do organ outlines survive as tissue boundaries? (%d heads, median)' % len(cases), fontsize=10); ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout(); plt.savefig(os.path.join(out, 'organ_shape_summary.png'), dpi=90); print('chart saved')
