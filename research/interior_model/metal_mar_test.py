"""Test metal_mar.correct on a HaN-Seg head: the worst slice before / after / streaks removed, and the jaw bone surface
from the raw and the corrected CT (private pictures, real patients). args: <HaN-Seg.zip> <head_v1 dir> <picture dir> <case>"""
import os, sys, json, time, zipfile, numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import measure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metal_mar import correct
zp, hv, pics, c = sys.argv[1:5]; Z = zipfile.ZipFile(zp); names = Z.namelist()
meta, tasks = {}, {}
for f in os.listdir(hv):
    if f.endswith('.json'): d = json.load(open(os.path.join(hv, f))); meta.update(d['stats']); tasks = d['tasks']
inv = {v: int(k) for k, v in tasks['craniofacial_structures'].items()}
n = [x for x in names if x.endswith('%s/%s_IMG_CT.nrrd' % (c, c))][0]; p = os.path.join(pics, '_t.nrrd'); open(p, 'wb').write(Z.read(n)); im = sitk.ReadImage(p); os.remove(p)
A = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32); sp = np.array(im.GetSpacing()); o = np.array(meta[c]['offset_in_scan'])
T = np.load(os.path.join(hv, c + '.npz'))['craniofacial_structures']; lab = np.zeros(A.shape, np.uint8); s = T.shape; lab[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] = T
jaw = np.isin(lab, [inv['mandible'], inv['teeth_upper'], inv['teeth_lower']]); jj = np.argwhere(jaw); lo, hi = np.maximum(jj.min(0) - [30, 30, 3], 0), np.minimum(jj.max(0) + [30, 30, 3], A.shape)
box = tuple(slice(a, b) for a, b in zip(lo, hi)); Ab = A[box]; top = float(A.max())
metal = Ab >= top - 1e-3; lb, nl = ndi.label(metal); sz = np.bincount(lb.ravel())[1:]; metal = np.isin(lb, np.flatnonzero(sz >= 8) + 1)
t0 = time.time(); C, streak = correct(Ab, metal); dt = time.time() - t0
zs = np.flatnonzero(metal.any((0, 1))); zw = int(zs[np.argmax(metal[:, :, zs].sum((0, 1)))])
# mark, do not correct: soft-tissue voxels whose streak exceeds the reconstruction noise far from the metal become unknown;
# teeth and jaw keep the labelling model's identity
far = ndi.distance_transform_edt(~metal, sampling=sp) > 40; aff = np.zeros(Ab.shape, bool); aff[:, :, zs] = True
noise = float(np.percentile(streak[far & aff], 99)) if (far & aff).any() else 100.0
jawb = jaw[box]; corrupt = (streak > noise) | metal; unknown = corrupt & ~jawb                     # model label only where the CT is corrupted
jawfix = corrupt & jawb
info0 = dict(noise_hu=round(noise, 1), unknown_ml=round(float(unknown.sum() * np.prod(sp)) / 1000, 2))
info = dict(case=c, **info0, metal_ml=round(float(metal.sum() * np.prod(sp)) / 1000, 2), slices=int(len(zs)), seconds=round(dt, 1),
            streak_over_100hu_ml=round(float((streak > 100).sum() * np.prod(sp)) / 1000, 2), streak_over_300hu_ml=round(float((streak > 300).sum() * np.prod(sp)) / 1000, 2))
print(json.dumps(info), flush=True)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
fig = plt.figure(figsize=(20, 12))
for k, (img, t) in enumerate(((Ab[:, :, zw], 'raw CT, slice with most metal'), (streak[:, :, zw], 'streak map (|change|, HU)'), (np.where(unknown[:, :, zw], 800.0, 0.0) + np.where(jawb[:, :, zw], 300.0, 0.0), 'marked unknown (bright) / tooth-jaw label (dim)'))):
    ax = fig.add_subplot(2, 3, k + 1)
    if k == 0: ax.imshow(np.clip(img.T, -300, 1500), cmap='gray', origin='lower')
    else: h = ax.imshow(np.clip(img.T, 0, 800), cmap='magma', origin='lower'); plt.colorbar(h, ax=ax, fraction=0.04)
    ax.set_title(t, fontsize=10); ax.axis('off')
body = ndi.binary_fill_holes(ndi.median_filter(Ab, 3) > -500)
for k, (vol, t) in enumerate(((Ab, 'jaw bone surface, raw CT'), (Ab, 'raw CT, streak voxels unknown, teeth/jaw from the model label'))):
    F = np.where(body, vol, -1000.0)
    if k == 1: F = np.where(unknown, -1000.0, F); F = np.where(jawfix, 400.0, F)
    v, f, _, _ = measure.marching_cubes(F, 150.0, spacing=sp, step_size=1); tri = v[f]
    if k == 1:
        ci = np.clip(np.round(tri.mean(1) / sp).astype(int), 0, np.array(Ab.shape) - 1); tri = tri[~unknown[ci[:, 0], ci[:, 1], ci[:, 2]]]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9; L = np.array([0.4, -0.7, 0.6]); L /= np.linalg.norm(L)
    a3 = fig.add_subplot(2, 2, 3 + k, projection='3d'); a3.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(np.array([0.93, 0.89, 0.8])[None] * (0.25 + 0.75 * np.abs(nrm @ L))[:, None], 0, 1), edgecolor='none'))
    ext = np.array(Ab.shape) * sp; a3.set_xlim(0, ext[0]); a3.set_ylim(0, ext[1]); a3.set_zlim(0, ext[2]); a3.set_box_aspect(ext); a3.view_init(elev=5, azim=-90); a3.axis('off'); a3.set_title(t, fontsize=10)
plt.tight_layout(); plt.savefig(os.path.join(pics, 'metal_mark_%s.png' % c), dpi=70); print('picture saved')
# close-up of the tooth rows, large
tj = np.argwhere(np.isin(lab[box], [inv['teeth_upper'], inv['teeth_lower']])) * sp; tl, th_ = tj.min(0) - 6, tj.max(0) + 6
fig = plt.figure(figsize=(20, 9))
for k, t in enumerate(('teeth, raw CT', 'teeth, streak voxels unknown, teeth from the model label')):
    F = np.where(body, Ab, -1000.0)
    if k == 1: F = np.where(unknown, -1000.0, F); F = np.where(jawfix, 400.0, F)
    v, f, _, _ = measure.marching_cubes(F, 150.0, spacing=sp, step_size=1); tri = v[f]; keep = ((tri > tl) & (tri < th_)).all((1, 2)); tri = tri[keep]
    if k == 1:
        ci = np.clip(np.round(tri.mean(1) / sp).astype(int), 0, np.array(Ab.shape) - 1); tri = tri[~unknown[ci[:, 0], ci[:, 1], ci[:, 2]]]
    nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9; L = np.array([0.4, -0.7, 0.6]); L /= np.linalg.norm(L)
    a3 = fig.add_subplot(1, 2, 1 + k, projection='3d'); a3.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(np.array([0.93, 0.89, 0.8])[None] * (0.25 + 0.75 * np.abs(nrm @ L))[:, None], 0, 1), edgecolor='none'))
    a3.set_xlim(tl[0], th_[0]); a3.set_ylim(tl[1], th_[1]); a3.set_zlim(tl[2], th_[2]); a3.set_box_aspect(th_ - tl); a3.view_init(elev=8, azim=-90); a3.axis('off'); a3.set_title(t, fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(pics, 'metal_teeth_%s.png' % c), dpi=75); print('teeth picture saved')
json.dump(info, open(os.path.join(pics, 'metal_mar_%s.json' % c), 'w'), indent=1)
