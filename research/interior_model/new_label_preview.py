"""Preview of the new training input (plan v8 14.5) for one HaN-Seg head, against the old labels. Private (real patient).
New input, on a 1 mm grid (scanner axes here; the pipeline uses the head frame):
  field     raw CT brightness interpolated (no median filter, no hard classes)
  tissue    read from the field at its HU levels: air < -500, fat -190..-30 (under the skin = fat connected to the outside),
            lean -30..150, bone > 150; drawn at 4x finer display resolution from the interpolated field, so boundaries fall
            between voxels
  identity  eyes (labelling model), brain (inside the skull, filled from bone)
  unknown   outside the scan; dental metal and its streaks (projection-interpolation streak map above the noise far from
            the metal); filled teeth included
Old labels: median filter, hard classes, nearest-neighbour onto 1 mm.
args: <HaN-Seg.zip> <head_v1 dir> <picture dir> <case>"""
import os, sys, json, zipfile, numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metal_mar import correct
zp, hv, pics, c = sys.argv[1:5]; Z = zipfile.ZipFile(zp); names = Z.namelist()
meta, tasks = {}, {}
for f in os.listdir(hv):
    if f.endswith('.json'): d = json.load(open(os.path.join(hv, f))); meta.update(d['stats']); tasks = d['tasks']
inv = {t: {v: int(k) for k, v in m.items()} for t, m in tasks.items()}
n = [x for x in names if x.endswith('%s/%s_IMG_CT.nrrd' % (c, c))][0]; p = os.path.join(pics, '_t.nrrd'); open(p, 'wb').write(Z.read(n)); im = sitk.ReadImage(p); os.remove(p)
A = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32); sp = np.array(im.GetSpacing()); o = np.array(meta[c]['offset_in_scan'])
def tsm(task, names_):
    full = np.zeros(A.shape, bool); T = np.load(os.path.join(hv, c + '.npz'))[task]; t = np.isin(T, [inv[task][x] for x in names_]); s = T.shape; full[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] = t; return full
eyes = tsm('head_glands_cavities', ['eye_left', 'eye_right', 'eye_lens_left', 'eye_lens_right']); jaw = tsm('craniofacial_structures', ['mandible', 'teeth_upper', 'teeth_lower'])
key = eyes | jaw; ii = np.argwhere(key); lo = np.maximum(ii.min(0) - [45, 45, 12], 0); hi = np.minimum(ii.max(0) + [45, 45, 40], A.shape); box = tuple(slice(a, b) for a, b in zip(lo, hi))
Ab, Eb, Jb = A[box], eyes[box], jaw[box]; body = ndi.binary_fill_holes(ndi.median_filter(Ab, 3) > -500)
# unknown: metal and streaks
top = float(A.max()); metal = Ab >= top - 1e-3; lb, _ = ndi.label(metal); sz = np.bincount(lb.ravel())[1:]; metal = np.isin(lb, np.flatnonzero(sz >= 8) + 1)
_, streak = correct(Ab, metal); zs = np.flatnonzero(metal.any((0, 1))); aff = np.zeros(Ab.shape, bool); aff[:, :, zs] = True
far = ndi.distance_transform_edt(~metal, sampling=sp) > 40; noise = float(np.percentile(streak[far & aff], 99)) if (far & aff).any() else 1e9
unknown = ((streak > noise) | metal) & body
# bright streak residue: bone pieces that exist only in metal-affected slices and do not join bone that continues into clean slices
bl, bn = ndi.label(Ab > 150); clean_lab = np.unique(bl[:, :, ~np.isin(np.arange(Ab.shape[2]), zs)]); only_aff = (bl > 0) & ~np.isin(bl, clean_lab) & ~jaw[box]
unknown |= only_aff & body
known_bone = ndi.binary_dilation(tsm('craniofacial_structures', ['skull', 'mandible', 'teeth_upper', 'teeth_lower'])[box], iterations=2)
unknown |= aff & (Ab > 150) & ~known_bone & body                                                # in metal-affected slices, bone-bright voxels outside any bone the labelling model knows
# brain: inside the skull
bone = ndi.binary_dilation(Ab > 300, iterations=1); cav = np.zeros_like(bone)
for z in range(bone.shape[2]): cav[:, :, z] = ndi.binary_fill_holes(bone[:, :, z]) & ~bone[:, :, z]
cl, cn = ndi.label(cav); brain = (cl == (np.bincount(cl.ravel())[1:].argmax() + 1)) if cn else cav
skull = tsm('craniofacial_structures', ['skull'])[box]; inside = np.zeros_like(skull)
for z in range(skull.shape[2]): inside[:, :, z] = ndi.binary_fill_holes(ndi.binary_closing(skull[:, :, z], iterations=3)) & ~skull[:, :, z]
brain = brain | (inside & (Ab > 0) & (Ab < 80) & body & ~Eb)                                       # soft tissue inside the model's skull (the pipeline will use the model's brain label)
def onto(vol, new, order):
    shape = [int(np.floor(n * s / new)) for n, s in zip(vol.shape, sp)]; ax = [((np.arange(m) + 0.5) * new) / s - 0.5 for m, s in zip(shape, sp)]
    return ndi.map_coordinates(vol.astype(np.float32), np.meshgrid(*ax, indexing='ij'), order=order, mode='nearest')
F = onto(np.where(body, Ab, -1000.0), 1.0, 1)                                              # the new stored field (1 mm)
U = onto(unknown, 1.0, 0) > 0.5; E = onto(Eb, 1.0, 1) > 0.5; B = onto(brain, 1.0, 0) > 0.5; outside = onto(body, 1.0, 0) < 0.5
S = ndi.median_filter(Ab, 3); old = np.zeros(Ab.shape, np.uint8); fat = body & (S >= -190) & (S <= -30); fl, _ = ndi.label(fat | ~body); oid = np.bincount(fl[~body].ravel()).argmax()
old[body & (S > -30) & (S <= 150)] = 3; old[body & (S < -500)] = 4; old[body & (S > 150)] = 5; old[fat] = 2; old[fat & (fl == oid)] = 1
OLD = onto(old, 1.0, 0).astype(np.uint8)
COL = {0: (1, 1, 1), 1: (0.98, 0.84, 0.45), 2: (0.90, 0.68, 0.30), 3: (0.66, 0.40, 0.38), 4: (0.35, 0.62, 0.90), 5: (0.94, 0.91, 0.82), 6: (0.25, 0.85, 0.35), 7: (0.82, 0.62, 0.72), 8: (0.55, 0.55, 0.55)}
def classes(f2, up, e2, b2, u2, o2):
    """tissue from a field slice drawn at `up` times finer resolution (linear interpolation of the field)"""
    fz = ndi.zoom(f2, up, order=1); L = np.zeros(fz.shape, np.uint8)
    body2 = ndi.zoom(~o2.astype(bool), up, order=0)
    al, _ = ndi.label((fz < -500) | ~body2); outer = np.isin(al, np.unique(al[~body2])) & (al > 0); body2 = body2 & ~outer   # air joined to the outside is outside (the skin edge)
    fat = body2 & (fz >= -190) & (fz <= -30); fl, _ = ndi.label(fat | ~body2); oid = np.bincount(fl[~body2].ravel()).argmax() if (~body2).any() else -1
    L[body2 & (fz > -30) & (fz <= 150)] = 3; L[body2 & (fz < -500)] = 4; L[body2 & (fz > 150)] = 5; L[fat] = 2; L[fat & (fl == oid)] = 1
    L[ndi.zoom(b2.astype(np.uint8), up, order=0) > 0] = 7; L[ndi.zoom(e2.astype(np.uint8), up, order=0) > 0] = 6; L[ndi.zoom(u2.astype(np.uint8), up, order=0) > 0] = 8
    return L
def rgb(L): return np.array([COL[k] for k in range(9)])[L]
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ex = np.argwhere(E); cz = int(ex[:, 2].mean()) if len(ex) else F.shape[2] // 2; cx = int(np.argwhere(onto(Jb, 1.0, 0) > 0.5)[:, 0].mean()); zt = int(np.argwhere(U)[:, 2].mean()) if U.any() else cz
views = [('top-down through the eyes', lambda V: V[:, :, cz]), ('side view through the middle', lambda V: V[cx]), ('top-down through the filled teeth', lambda V: V[:, :, zt])]
fig, axs = plt.subplots(3, 3, figsize=(18, 17))
for r, (nm, f) in enumerate(views):
    axs[r, 0].imshow(np.clip(f(F).T, -300, 600), cmap='gray', origin='lower'); axs[r, 0].set_title('new: CT field (1 mm), ' + nm, fontsize=10)
    axs[r, 1].imshow(rgb(classes(f(F), 4, f(E), f(B), f(U), f(outside))).transpose(1, 0, 2), origin='lower'); axs[r, 1].set_title('new: tissue from the field + identity + unknown', fontsize=10)
    axs[r, 2].imshow(rgb(f(OLD)).transpose(1, 0, 2), origin='lower'); axs[r, 2].set_title('old: hard labels (1 mm)', fontsize=10)
    for a in axs[r]: a.axis('off')
fig.legend(handles=[Patch(color=COL[k], label=l) for k, l in ((1, 'fat under the skin'), (2, 'other fat'), (3, 'lean (muscle, glands)'), (4, 'air'), (5, 'bone'), (6, 'eye'), (7, 'brain'), (8, 'unknown (metal, streaks)'))],
           loc='lower center', ncol=8, frameon=False, fontsize=10)
fig.suptitle('%s: new training input against the old labels (private, real patient)' % c, fontsize=13)
plt.tight_layout(rect=(0, 0.03, 1, 0.97)); plt.savefig(os.path.join(pics, 'new_input_v3_%s.png' % c), dpi=60)
print(json.dumps(dict(case=c, unknown_ml=round(float(U.sum()) / 1000, 1), eyes_ml=round(float(E.sum()) / 1000, 1), brain_ml=round(float(B.sum()) / 1000), noise_hu=round(noise, 1))))
