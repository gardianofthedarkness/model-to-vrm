"""Test metal_artifacts.detect on HaN-Seg heads: how much is marked, where, and how the jaw surface looks with the marked
zone removed (private pictures, real patients). args: <HaN-Seg.zip> <head_v1 dir> <picture dir> <case>..."""
import os, sys, json, zipfile, numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import measure
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metal_artifacts import detect
zp, hv, pics = sys.argv[1:4]; cases = sys.argv[4:]; Z = zipfile.ZipFile(zp); names = Z.namelist()
meta, tasks = {}, {}
for f in os.listdir(hv):
    if f.endswith('.json'): d = json.load(open(os.path.join(hv, f))); meta.update(d['stats']); tasks = d['tasks']
inv = {v: int(k) for k, v in tasks['craniofacial_structures'].items()}
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
out = {}
for c in cases:
    n = [x for x in names if x.endswith('%s/%s_IMG_CT.nrrd' % (c, c))][0]; p = os.path.join(pics, '_t.nrrd'); open(p, 'wb').write(Z.read(n)); im = sitk.ReadImage(p); os.remove(p)
    A = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32); sp = np.array(im.GetSpacing()); o = np.array(meta[c]['offset_in_scan'])
    T = np.load(os.path.join(hv, c + '.npz'))['craniofacial_structures']; lab = np.zeros(A.shape, np.uint8); s = T.shape; lab[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] = T
    body = ndi.binary_fill_holes(ndi.median_filter(A, 3) > -500)
    mask, info = detect(A, body)
    keep_bone = np.isin(lab, [inv['teeth_upper'], inv['teeth_lower'], inv['mandible']])
    info.update(artifact_ml=round(float(mask.sum() * np.prod(sp)) / 1000, 1), unknown_ml=round(float((mask & ~keep_bone).sum() * np.prod(sp)) / 1000, 1), body_ml=round(float(body.sum() * np.prod(sp)) / 1000))
    out[c] = info; print(c, info, flush=True)
    # pictures: the worst slice with the mask, and the jaw bone surface before / after removing the marked zone
    md = np.isin(lab, [inv['mandible'], inv['teeth_upper'], inv['teeth_lower']]); jj = np.argwhere(md); lo, hi = np.maximum(jj.min(0) - [15, 15, 3], 0), np.minimum(jj.max(0) + [15, 15, 3], A.shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi)); Ab, Mb, Bb, Kb = A[box], mask[box], body[box], keep_bone[box]
    zw = int(np.argmax(Mb.sum((0, 1))))
    fig = plt.figure(figsize=(18, 6.5))
    ax = fig.add_subplot(1, 3, 1); ax.imshow(np.clip(Ab[:, :, zw].T, -300, 1500), cmap='gray', origin='lower'); ov = np.zeros(Ab[:, :, zw].T.shape + (4,)); ov[Mb[:, :, zw].T & ~Kb[:, :, zw].T] = (1, 0.2, 0.2, 0.45); ov[Mb[:, :, zw].T & Kb[:, :, zw].T] = (1, 0.8, 0.1, 0.45)
    ax.imshow(ov, origin='lower'); ax.set_title('slice with most metal: red = marked unknown,\norange = marked but kept as tooth/jaw (model label)', fontsize=9); ax.axis('off')
    for k, (title, drop) in enumerate((('bone surface, raw CT (with metal streaks)', None), ('bone surface, marked zone removed', Mb & ~Kb))):
        F = np.where(Bb, Ab, -1000.0)
        if drop is not None: F = np.where(drop, -1000.0, F); F = np.where(Mb & Kb, np.maximum(F, 400.0), F)    # inside teeth and jaw the model label decides: bone
        v, f, _, _ = measure.marching_cubes(F, 150.0, spacing=sp, step_size=1); tri = v[f]
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9; L = np.array([0.4, -0.7, 0.6]); L /= np.linalg.norm(L)
        a3 = fig.add_subplot(1, 3, 2 + k, projection='3d'); a3.add_collection3d(Poly3DCollection(tri, facecolors=np.clip(np.array([0.93, 0.89, 0.8])[None] * (0.25 + 0.75 * np.abs(nrm @ L))[:, None], 0, 1), edgecolor='none'))
        ext = np.array(Ab.shape) * sp; a3.set_xlim(0, ext[0]); a3.set_ylim(0, ext[1]); a3.set_zlim(0, ext[2]); a3.set_box_aspect(ext); a3.view_init(elev=5, azim=-90); a3.axis('off'); a3.set_title(title, fontsize=9)
    plt.tight_layout(); plt.savefig(os.path.join(pics, 'metal_%s.png' % c), dpi=75); plt.close()
json.dump(out, open(os.path.join(pics, 'metal_test.json'), 'w'), indent=1)
