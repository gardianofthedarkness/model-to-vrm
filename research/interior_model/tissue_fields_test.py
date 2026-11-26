"""Continuous tissue fields (plan v8 14.5): carry the raw CT brightness into the training grid by interpolation (no median
filter, no hard classes) and take tissue surfaces at their HU levels. Compared with the exact surface on the CT's own grid:
  current  hard classes after a 3-voxel median filter, nearest-neighbour onto 1 mm (today's labels)
  field1   raw HU interpolated (linear) onto 1 mm, surface at the HU level
  field05  the same onto 0.5 mm
Pictures (private, real patients): a side-view slice through the nose with the fat|lean and bone boundaries of 'current'
and 'field1' over the CT; the facial bone surface in 3D from both; a chart of the errors (numbers only).
args: <HaN-Seg.zip> <head_v1 dir> <picture dir> <case>..."""
import os, sys, json, zipfile, numpy as np, SimpleITK as sitk
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage import measure
zp, hv, pics = sys.argv[1:4]; cases = sys.argv[4:]; Z = zipfile.ZipFile(zp); names = Z.namelist(); os.makedirs(pics, exist_ok=True)
meta, tasks = {}, {}
for f in os.listdir(hv):
    if f.endswith('.json'): d = json.load(open(os.path.join(hv, f))); meta.update(d['stats']); tasks = d['tasks']
inv = {t: {v: int(k) for k, v in m.items()} for t, m in tasks.items()}
LEVELS = {'fat|lean': -30.0, 'bone': 150.0}
def read(c, suf):
    n = [x for x in names if x.endswith('%s/%s_%s' % (c, c, suf))][0]; p = os.path.join(pics, '_t.nrrd'); open(p, 'wb').write(Z.read(n)); im = sitk.ReadImage(p); os.remove(p); return im
def iso(vol, lvl, sp, step=1):
    if vol.max() <= lvl or vol.min() >= lvl: return np.zeros((0, 3)), np.zeros((0, 3), int)
    v, f, _, _ = measure.marching_cubes(vol.astype(np.float32), lvl, spacing=sp, step_size=step); return v, f
def faces(mask, sp):
    pts = []
    for ax in range(3):
        a = np.moveaxis(mask, ax, 0); ii = np.argwhere(a[1:] != a[:-1]).astype(float); ii[:, 0] += 1.0
        order = [ax] + [k for k in range(3) if k != ax]; p = np.zeros_like(ii)
        for j, k in enumerate(order): p[:, k] = ii[:, j]
        pts.append(p * np.array(sp))
    return np.concatenate(pts)
def sample(v, f, n=600000, seed=0):
    """points spread evenly over a triangle mesh (area-weighted), so distances measure the surfaces, not where vertices happen to lie"""
    if len(f) == 0: return v
    t = v[f]; ar = 0.5 * np.linalg.norm(np.cross(t[:, 1] - t[:, 0], t[:, 2] - t[:, 0]), axis=1); rng = np.random.default_rng(seed)
    k = rng.choice(len(f), n, p=ar / ar.sum()); u = rng.random((n, 2)); m = u.sum(1) > 1; u[m] = 1 - u[m]
    return t[k, 0] + u[:, :1] * (t[k, 1] - t[k, 0]) + u[:, 1:] * (t[k, 2] - t[k, 0])
def dist(a, b):
    d = np.concatenate([cKDTree(b).query(a, workers=-1)[0], cKDTree(a).query(b, workers=-1)[0]])
    return dict(median_mm=round(float(np.median(d)), 2), p90_mm=round(float(np.percentile(d, 90)), 2))
def resample(vol, sp, new, order):
    """onto a grid of spacing `new` covering the same box, same origin (voxel centres at (i + 0.5) * spacing)"""
    shape = [int(np.floor(n * s / new)) for n, s in zip(vol.shape, sp)]; ax = [((np.arange(m) + 0.5) * new) / s - 0.5 for m, s in zip(shape, sp)]
    g = np.meshgrid(*ax, indexing='ij'); return ndi.map_coordinates(vol, g, order=order, mode='nearest'), shape
res = {}; first = True
for c in cases:
    ct = read(c, 'IMG_CT.nrrd'); A = sitk.GetArrayFromImage(ct).transpose(2, 1, 0).astype(np.float32); sp = np.array(ct.GetSpacing()); m = meta[c]; o = np.array(m['offset_in_scan']); TS = np.load(os.path.join(hv, c + '.npz'))
    def tsm(task, nm):
        full = np.zeros(A.shape, bool); t = TS[task] == inv[task][nm]; s = t.shape; full[o[0]:o[0] + s[0], o[1]:o[1] + s[1], o[2]:o[2] + s[2]] = t; return full
    key = tsm('craniofacial_structures', 'mandible') | tsm('head_glands_cavities', 'eye_left') | tsm('head_glands_cavities', 'eye_right')
    ii = np.argwhere(key); lo = np.maximum(ii.min(0) - [40, 40, 10], 0); hi = np.minimum(ii.max(0) + [40, 40, 10], A.shape); Ab = A[tuple(slice(a, b) for a, b in zip(lo, hi))]
    body = ndi.binary_fill_holes(ndi.median_filter(Ab, 3) > -500); Ab_b = np.where(body, Ab, -1000.0)   # outside the body: air, as the labels have it
    S = ndi.median_filter(Ab, 3); r = {}
    F1, sh1 = resample(Ab_b, sp, 1.0, 1); F05, sh05 = resample(Ab_b, sp, 0.5, 1)
    for nm, lvl in LEVELS.items():
        exact = sample(*iso(Ab_b, lvl, sp)); cur = body & (S > lvl); cur1, _ = resample(cur.astype(np.float32), sp, 1.0, 0)
        f1 = sample(*iso(F1, lvl, (1.0, 1.0, 1.0))); f05 = sample(*iso(F05, lvl, (0.5, 0.5, 0.5)))
        cm = sample(*iso(np.pad(cur1, 1), 0.5, (1.0, 1.0, 1.0))) - 1.0                                   # the hard labels' own (staircase) surface
        # the grids share the box origin: points of the 1 mm and 0.5 mm grids are offset by half their spacing against the CT's, removed here
        shift1 = 0.5 * 1.0 - 0.5 * sp; shift05 = 0.5 * 0.5 - 0.5 * sp
        r[nm] = dict(current=dist(cm + shift1, exact), field1=dist(f1 + shift1, exact), field05=dist(f05 + shift05, exact))
    res[c] = r; print(c, json.dumps(r), flush=True)
    if first and os.environ.get('PICS', '1') == '1':                                              # pictures for the first head
        first = False
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        xm = int(np.argwhere(key[tuple(slice(a, b) for a, b in zip(lo, hi))])[:, 0].mean())         # side view through the middle of the face
        fig, axs = plt.subplots(1, 2, figsize=(15, 8))
        for ax_, (nm, lvl) in zip(axs, LEVELS.items()):
            ax_.imshow(np.clip(Ab[xm].T, -250, 450), cmap='gray', origin='lower', aspect=sp[2] / sp[1])
            cur = (body & (S > lvl))[xm].T.astype(float); ax_.contour(cur, [0.5], colors='red', linewidths=1.0)
            fx = int(round((xm + 0.5) * sp[0] / 1.0 - 0.5)); fs = F1[min(fx, F1.shape[0] - 1)].T
            yy, zz = np.meshgrid((np.arange(fs.shape[1]) + 0.5) / sp[1] - 0.5, (np.arange(fs.shape[0]) + 0.5) * 1.0 / sp[2] - 0.5)
            ax_.contour(yy, zz, fs, [lvl], colors='cyan', linewidths=1.0)
            ax_.plot([], [], color='red', label='current labels (hard, 1 mm)'); ax_.plot([], [], color='cyan', label='continuous field (1 mm)'); ax_.legend(loc='upper left', fontsize=9)
            ax_.set_title('%s boundary (%d HU), side-view slice' % (nm, lvl)); ax_.axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(pics, 'precision_slices_%s.png' % c), dpi=80); plt.close()
        fig = plt.figure(figsize=(15, 8)); md = tsm('craniofacial_structures', 'mandible')[tuple(slice(a, b) for a, b in zip(lo, hi))]
        jj = np.argwhere(md); jl, jh = jj.min(0) * sp - 5, jj.max(0) * sp + 5
        for k, (title, vf) in enumerate((('current labels (hard, 1 mm)', iso(resample((body & (S > 150)).astype(np.float32), sp, 1.0, 0)[0], 0.5, (1.0, 1.0, 1.0))),
                                         ('continuous field (1 mm)', iso(F1, 150.0, (1.0, 1.0, 1.0))))):
            v, f = vf; tri = v[f]; keep = ((tri > jl) & (tri < jh)).all((1, 2)); tri = tri[keep]
            nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
            L = np.array([0.4, -0.7, 0.6]); L /= np.linalg.norm(L); sh = 0.25 + 0.75 * np.abs(nrm @ L)
            ax = fig.add_subplot(1, 2, k + 1, projection='3d'); pc = Poly3DCollection(tri, facecolors=np.clip(np.array([0.93, 0.89, 0.8])[None] * sh[:, None], 0, 1), edgecolor='none'); ax.add_collection3d(pc)
            ax.set_xlim(jl[0], jh[0]); ax.set_ylim(jl[1], jh[1]); ax.set_zlim(jl[2], jh[2]); ax.set_box_aspect(jh - jl); ax.view_init(elev=5, azim=-90); ax.axis('off'); ax.set_title('jaw bone surface: ' + title)
        plt.tight_layout(); plt.savefig(os.path.join(pics, 'precision_bone3d_%s.png' % c), dpi=80); plt.close()
json.dump(res, open(os.path.join(pics, 'tissue_fields_test_v2.json'), 'w'), indent=1)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for ax_, nm in zip(axs, LEVELS):
    x = np.arange(len(res)); w = 0.26
    for i, (k, col, lab) in enumerate((('current', '#b0b7c3', 'current labels'), ('field1', '#2a6fdb', 'continuous field, 1 mm'), ('field05', '#7aa7ec', 'continuous field, 0.5 mm'))):
        v = [res[c][nm][k]['median_mm'] for c in res]; p = [res[c][nm][k]['p90_mm'] for c in res]
        ax_.bar(x + (i - 1) * w, v, w - 0.03, color=col, label=lab); ax_.scatter(x + (i - 1) * w, p, color='#333', marker='_', s=200)
    ax_.set_xticks(x); ax_.set_xticklabels(list(res)); ax_.set_ylabel('distance to the exact surface, mm'); ax_.set_title('%s: median (bar), 90th percentile (dash)' % nm, fontsize=10); ax_.spines[['top', 'right']].set_visible(False)
axs[0].legend(frameon=False, fontsize=8); plt.tight_layout(); plt.savefig(os.path.join(pics, 'precision_summary.png'), dpi=90); print('pictures saved')
