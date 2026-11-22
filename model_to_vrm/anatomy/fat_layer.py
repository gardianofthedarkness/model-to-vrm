"""The fat layer under the figure's skin, from people: HIT's MRI statistics (hit_support_stats.py: per point of the standard
body surface, the median fat-layer thickness of 241 women or 157 men as a share of body height) carried to the model.
  E:\\splat\\.venv\\Scripts\\python.exe fat_layer.py <fits dir with smpl_fit.npz> <model dir> <hit_support.npz> [female|male]
  carry     smpl_fit.npz (mhr_to_smpl.py) is the standard body posed and shaped like the figure. It does not lie on her
            skin well enough to be her body (checked: 17 mm off on flesh), but it says which place of the standard body a
            skin point corresponds to: the nearest standard-body vertices whose surface faces the same way as the model's
            there (so the inner thigh does not take the other leg), the three nearest of them weighted by 1 / distance
  size      share of height x the fitted standard body's height. NOT used to place muscles: on a slim figure it is too
            thick (on the elf 40% of the skin-to-bone depth on the limbs, 26% in the women scanned), and scaling it by
            her own skin-to-bone depth against the atlas's gave a patchy map (tried, dropped). It is the comparison for
            the fat layer the muscle steps leave between the outermost muscles and the skin (muscle_depth.py).
Writes <fits dir>/fat_layer.npz: fat (metres, per model vertex), share, and fat_layer.jpg."""
import sys, os, numpy as np
from scipy.spatial import cKDTree
FD, MD, HS = sys.argv[1:4]; sex = sys.argv[4] if len(sys.argv) > 4 else 'female'
S = np.load(os.path.join(FD, 'smpl_fit.npz'), allow_pickle=True); SV = S['V'].astype(np.float64); SF = S['F']; H = np.load(HS); share = H['fat_%s_med' % sex]
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']
def vnormals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); N = np.zeros_like(V)
    for k in range(3): np.add.at(N, F[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
MN, SN = vnormals(MV, MF), vnormals(SV, SF); ring = int(np.bincount(SF.ravel()).max()) * 3               # search among about three rings of the standard body's mesh
d, j = cKDTree(SV).query(MV, k=ring); ok = np.einsum('nkj,nj->nk', SN[j], MN) > 0; none = ~ok.any(1); ok[none] = True
rank = np.cumsum(ok, 1); use = ok & (rank <= 3); w = use / np.maximum(d, 1e-9); w /= w.sum(1, keepdims=True)
sh = (w * share[j]).sum(1); height = float(np.ptp(SV[:, 2])); by_height = sh * height
fat = by_height
np.savez(os.path.join(FD, 'fat_layer.npz'), fat=fat.astype(np.float32), share=sh.astype(np.float32), height=height, sex=sex)
print('FL %s fat layer on %d model vertices: median %.1f mm, 5%%..95%% %.1f..%.1f mm (standard body height %.3f m); %.1f%% had no vertex facing the same way' % (sex, len(MV), 1000 * np.median(fat), *(1000 * np.percentile(fat, [5, 95])), height, 100 * none.mean()))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 9)); vmax = 1000 * np.percentile(fat, 98)
for a, (sel, key, ti) in zip(ax, [(MN[:, 1] < 0, MV[:, 0], 'front'), (MN[:, 1] > 0, -MV[:, 0], 'back'), (MN[:, 0] > 0, MV[:, 1], 'left side')]):
    s = a.scatter(key[sel], MV[sel, 2], c=1000 * fat[sel], s=0.3, cmap='turbo', vmin=0, vmax=vmax); a.set_aspect('equal'); a.set_title(ti); a.axis('off')
plt.colorbar(s, ax=ax, shrink=0.5, label='fat layer, mm'); plt.savefig(os.path.join(FD, 'fat_layer.jpg'), dpi=110, bbox_inches='tight')
