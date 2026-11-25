"""Plan v8, test P4: cover templates from a character's own hair and loose cloth, in the shell frame of each part.

A character is placed exactly as at inference: its bare-skin vertices (loose parts excluded) are fitted to the population's
mean shell by a similarity transform (shell_align.fit, scale free). Then, on the part's voxel window in that frame:
  solid  = inside the full mesh (generalised winding number > 1/2), hair and cloth included
  cover  = solid voxels whose nearest mesh vertex is a loose vertex (hair, hanging cloth)
Output per part: <out>/cover_<part>.npz with cover (bool), solid (bool), the fit (x), and the transform from mesh units to the
frame, for (a) the training preparation, which lays the cover over real people's shells, and (b) inference on this character.
args: <interior_data root> <mesh parts dir> <loose.npz> <first-guess frames dir (elf_face.npz, elf_pelvis.npz)> <align dir> <out dir>"""
import os, sys, json, numpy as np, igl
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shell_align import MeanShell, fit
root, parts, loosef, guess, alignd, out = sys.argv[1:7]; os.makedirs(out, exist_ok=True)
V = np.load(os.path.join(parts, 'orig_colours.npz'))['V'].astype(np.float64); F = np.load(os.path.join(parts, 'orig_normals.npz'))['F'].astype(np.int64); loose = np.load(loosef)['loose']
fi = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pi = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
CFG = dict(face=(1.0, np.array(fi['window_lo_mm']), np.array(fi['window_hi_mm'])), pelvis=(3.0, np.array(pi['window_lo_mm']), np.array(pi['window_hi_mm'])))
res = {}
for part, (mm, lo, hi) in CFG.items():
    g = np.load(os.path.join(guess, 'elf_%s.npz' % part)); o, A, s0 = g['origin'], g['axes'], float(g['scale_mm_per_unit'])
    Vf = (V - o) @ A.T * s0; inw = ((Vf > lo) & (Vf < hi)).all(1)                              # first guess: keypoint frame and size
    mean = MeanShell.load(os.path.join(alignd, '%s_mean.npz' % part))
    skin = Vf[inw & ~loose]; skin = skin[:: max(1, len(skin) // 30000)]
    x, ap = fit(skin, mean); sf = float(np.exp(x[0])); Q = Rotation.from_rotvec(x[1:4]).as_matrix(); b = x[4:7]
    Y = ap(x, Vf)                                                                                # every vertex in the shell frame, mm
    shape = np.round((hi - lo) / mm).astype(int); ax = [lo[i] + mm * (np.arange(shape[i]) + 0.5) for i in range(3)]
    P = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    near = ((P > Y.min(0) - 5) & (P < Y.max(0) + 5)).all(1); solid = np.zeros(len(P), bool)
    back = ((P[near] - b) @ Q / sf) / s0 @ A + o                                                 # frame mm -> mesh units (inverse of the two maps)
    w = np.concatenate([igl.fast_winding_number(V, F, np.ascontiguousarray(q)) for q in np.array_split(back, max(1, len(back) // 2000000))]); solid[np.flatnonzero(near)[w > 0.5]] = True
    tree = cKDTree(Y[inw | ((Y > lo - 20) & (Y < hi + 20)).all(1)]); idx_v = np.flatnonzero(inw | ((Y > lo - 20) & (Y < hi + 20)).all(1))
    cover = np.zeros(len(P), bool); si = np.flatnonzero(solid); _, j = tree.query(P[si], workers=-1); cover[si[loose[idx_v[j]]]] = True
    solid, cover = solid.reshape(shape), cover.reshape(shape)
    np.savez_compressed(os.path.join(out, 'cover_%s.npz' % part), cover=cover, solid=solid, fit=x, guess_origin=o, guess_axes=A, guess_scale=s0, mm=mm, lo=lo)
    res[part] = dict(size_factor_vs_first_guess=round(sf, 4), fit_shift_mm=round(float(np.linalg.norm(b)), 1), fit_turn_deg=round(float(np.degrees(np.linalg.norm(x[1:4]))), 2),
                     solid_ml=round(float(solid.sum() * mm ** 3 / 1000)), cover_ml=round(float(cover.sum() * mm ** 3 / 1000)), cover_share=round(float(cover.sum() / max(1, solid.sum())), 3))
    print(part, res[part], flush=True)
json.dump(res, open(os.path.join(out, 'cover_templates.json'), 'w'), indent=1)
if os.environ.get('PICS'):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(16, 10))
    for r, part in enumerate(CFG):
        z = np.load(os.path.join(out, 'cover_%s.npz' % part)); S, Cv = z['solid'], z['cover']; c = np.round(np.argwhere(S).mean(0)).astype(int)
        for k in range(3):
            sl = [slice(None)] * 3; sl[k] = c[k]; img = np.ones(S[tuple(sl)].shape + (3,)); img[S[tuple(sl)]] = (0.91, 0.79, 0.66); img[Cv[tuple(sl)]] = (0.49, 0.44, 0.84)
            axs[r, k].imshow(img.transpose(1, 0, 2), origin='lower'); axs[r, k].set_title('%s, slice across axis %s: skin-side solid (tan), cover (purple)' % (part, 'xyz'[k]), fontsize=9); axs[r, k].axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(os.environ['PICS'], 'elf_covers.png'), dpi=70); print('picture saved')
