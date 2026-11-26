"""Precision test (user idea, 2026-11-26): how much boundary accuracy does the labelling pipeline lose, and does snapping
labels to image edges recover it? HaN-Seg CT (about 0.5 mm in-plane) with expert outlines, locally, numbers and private
pictures only.
A. CT tissue surfaces. Exact: marching cubes on the raw CT at its own resolution at the tissue's HU level (fat|lean -30,
   bone 150). Current: 3-voxel median filter, hard per-voxel classes, nearest-neighbour onto a 1 mm grid (as the head
   frame). Error: distance from the current class boundary (voxel faces) to the exact surface, mm.
B. Eyes and mandible against the expert outlines (surface at native resolution). Three versions:
   ts      TotalSegmentator's label (computed at 1.5 mm, returned on the CT grid)
   ts_1mm  the same after nearest-neighbour onto 1 mm (the current pipeline)
   snapped random walker on the CT inside a band around the TotalSegmentator label (seeds: label eroded 2 voxels = inside,
           beyond the label dilated 4 voxels = outside), surface at its 50% probability (sub-voxel)
args: <HaN-Seg.zip> <head_v1 dir> <out dir> <case>..."""
import os, sys, json, zipfile, numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
from skimage.segmentation import random_walker
zp, hv, out = sys.argv[1:4]; cases = sys.argv[4:]; os.makedirs(out, exist_ok=True); Z = zipfile.ZipFile(zp); names = Z.namelist()
meta = {}; tasks = {}
for f in os.listdir(hv):
    if f.endswith('.json'): d = json.load(open(os.path.join(hv, f))); meta.update(d['stats']); tasks = d['tasks']
inv = {t: {v: int(k) for k, v in m.items()} for t, m in tasks.items()}
def find(c, suf): return [n for n in names if n.endswith('%s/%s_%s' % (c, c, suf))][0]
def read(n):
    p = os.path.join(out, '_tmp.nrrd'); open(p, 'wb').write(Z.read(n)); im = sitk.ReadImage(p); os.remove(p); return im
def arr(im): return sitk.GetArrayFromImage(im).transpose(2, 1, 0)                             # (x, y, z) as the stored outputs
def surf_pts(vol, level, sp, step=1):
    if vol.max() <= level or vol.min() >= level: return np.zeros((0, 3))
    v, f, _, _ = measure.marching_cubes(vol.astype(np.float32), level, spacing=sp, step_size=step); return v
def face_pts(mask, sp):
    """centres of the faces between a mask and its outside, mm"""
    pts = []
    for ax in range(3):
        a = np.moveaxis(mask, ax, 0); d = a[1:] != a[:-1]; ii = np.argwhere(d).astype(float); ii[:, 0] += 1.0
        order = [ax] + [k for k in range(3) if k != ax]; p = np.zeros_like(ii)
        for j, k in enumerate(order): p[:, k] = ii[:, j]
        pts.append(p * np.array(sp))
    return np.concatenate(pts)
def dist(a, b):
    """distances both ways between point sets a and b (mm): median, 90th percentile"""
    if len(a) < 10 or len(b) < 10: return None
    a = a[:: max(1, len(a) // 60000)]; b = b[:: max(1, len(b) // 60000)]
    d = np.concatenate([cKDTree(b).query(a, workers=-1)[0], cKDTree(a).query(b, workers=-1)[0]])
    return dict(median_mm=round(float(np.median(d)), 2), p90_mm=round(float(np.percentile(d, 90)), 2))
def to_1mm(mask, sp):
    """nearest-neighbour onto a 1 mm grid (the head frame's spacing) and back to mm points of its boundary"""
    z = [s / 1.0 for s in sp]; m1 = ndi.zoom(mask.astype(np.uint8), z, order=0) > 0; return face_pts(m1, (1.0, 1.0, 1.0)), m1
res = {}
for c in cases:
    ct = read(find(c, 'IMG_CT.nrrd')); A = arr(ct).astype(np.float32); sp = ct.GetSpacing(); m = meta[c]; o = np.array(m['offset_in_scan']); TS = np.load(os.path.join(hv, c + '.npz'))
    def ts_mask(task, nm):
        full = np.zeros(A.shape, bool); t = TS[task] == inv[task][nm]; s = t.shape; full[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] = t; return full
    def expert(sufs):
        mm = np.zeros(A.shape, bool)
        for s_ in sufs:
            try: im = read(find(c, 'OAR_%s.seg.nrrd' % s_)); mm |= arr(sitk.Resample(sitk.Cast(im, sitk.sitkUInt8), ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
            except Exception: pass
        return mm
    r = dict(spacing_mm=[round(x, 3) for x in sp])
    # A. tissue surfaces in a box around the face (eyes to below the mandible)
    md = ts_mask('craniofacial_structures', 'mandible'); ey = ts_mask('head_glands_cavities', 'eye_left') | ts_mask('head_glands_cavities', 'eye_right')
    ii = np.argwhere(md | ey); lo = np.maximum(ii.min(0) - [40, 40, 10], 0); hi = np.minimum(ii.max(0) + [40, 40, 10], A.shape); box = tuple(slice(a, b) for a, b in zip(lo, hi))
    Ab = A[box]; S = ndi.median_filter(Ab, 3)
    body = ndi.binary_fill_holes(S > -500)
    for nm, lvl, cur in (('fat|lean', -30.0, body & (S > -30)), ('bone', 150.0, body & (S > 150))):
        exact = surf_pts(np.where(body, Ab, -1000.0), lvl, sp, step=1)
        p1, _ = to_1mm(cur, sp); pv = face_pts(cur, sp)
        r['A_' + nm] = dict(current_native_vs_exact=dist(pv, exact), current_1mm_vs_exact=dist(p1, exact))
    # B. eyes and mandible against experts
    for nm, ts_, ex_sufs in (('eye_left', ts_mask('head_glands_cavities', 'eye_left'), ('Eye_AL', 'Eye_PL')), ('eye_right', ts_mask('head_glands_cavities', 'eye_right'), ('Eye_AR', 'Eye_PR')),
                             ('mandible', md, ('Bone_Mandible',))):
        E = expert(ex_sufs)
        if E.sum() < 50 or ts_.sum() < 50: continue
        jj = np.argwhere(E | ts_); a, b = np.maximum(jj.min(0) - 12, 0), np.minimum(jj.max(0) + 13, A.shape); bx = tuple(slice(p, q) for p, q in zip(a, b))
        Eb, Tb, Cb = E[bx], ts_[bx], A[bx]
        ref = surf_pts(Eb.astype(np.float32), 0.5, sp)
        seeds = np.zeros(Tb.shape, np.int32); seeds[ndi.binary_erosion(Tb, iterations=2)] = 1; seeds[~ndi.binary_dilation(Tb, iterations=4)] = 2
        data = np.clip((Cb + 200) / 1200.0, 0, 1)                                              # soft tissue to bone range
        prob = random_walker(data, seeds, beta=2000, mode='cg_j', return_full_prob=True, tol=1e-3)[0]
        snapped = surf_pts(prob, 0.5, sp)
        p1, _ = to_1mm(Tb, sp)
        r['B_' + nm] = dict(ts=dist(face_pts(Tb, sp), ref), ts_1mm=dist(p1, ref), snapped=dist(snapped, ref),
                            volume_ml=dict(expert=round(float(Eb.sum() * np.prod(sp)) / 1000, 2), ts=round(float(Tb.sum() * np.prod(sp)) / 1000, 2), snapped=round(float((prob > 0.5).sum() * np.prod(sp)) / 1000, 2)))
        if nm == 'eye_left':                                                                   # private picture: one slice through the eye
            import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
            zc = int(np.argwhere(Eb)[:, 2].mean()); fig, ax = plt.subplots(1, 1, figsize=(7, 7))
            ax.imshow(np.clip(Cb[:, :, zc].T, -200, 200), cmap='gray', origin='lower', aspect=sp[1] / sp[0])
            for msk, col, lab in ((Eb, 'lime', 'expert'), (Tb, 'red', 'TotalSegmentator'), (prob > 0.5, 'cyan', 'snapped')):
                ax.contour(msk[:, :, zc].T.astype(float), [0.5], colors=col, linewidths=1.2)
                ax.plot([], [], color=col, label=lab)
            ax.legend(loc='upper right'); ax.set_title('%s left eye, one slice' % c); ax.axis('off'); plt.tight_layout(); plt.savefig(os.path.join(out, '%s_eye.png' % c), dpi=90); plt.close()
    res[c] = r; print(c, json.dumps(r), flush=True)
json.dump(res, open(os.path.join(out, 'precision_test.json'), 'w'), indent=1)
