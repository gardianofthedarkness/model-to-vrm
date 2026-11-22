"""Bone-support heat map: how far under the skin the nearest bone lies, on the atlas and carried to a model.
  E:\\splat\\.venv\\Scripts\\python.exe bone_support_map.py <fits dir with atlas_warp.npz, skeleton_fit.npz> <model dir> <out dir>
  atlas     every atlas skin vertex: distance to the nearest atlas bone surface (Z-Anatomy; its 'bone' system includes the
            nose, ear, rib and larynx cartilages). Small = bone-supported (stiff), large = soft tissue.
  carried   a model vertex takes the value of the nearest warped atlas skin vertex (atlas_warp.npz), as a share of body
            height (a proportion, not millimetres), times the model's height
  own       the model's own surface to its fitted bones (skeleton_fit.npz): includes cloth and hair thickness, and any
            misfit of the bones - a check on the carried map, not a replacement
Writes <out>/support_atlas.glb, support_carried.glb, support_own.glb (vertex colours: red thin .. blue thick, log scale
between the atlas map's 2nd and 98th percentile as shares of height) and support.npz (the values per vertex)."""
import sys, os, numpy as np, trimesh
from scipy.spatial import cKDTree
from matplotlib import colormaps
FD, MD, OD = sys.argv[1:4]; os.makedirs(OD, exist_ok=True)
W = np.load(os.path.join(FD, 'atlas_warp.npz')); V0 = W['V0'].astype(np.float64); Vw = W['V'].astype(np.float64); OB = W['obj']; sysn = np.array([str(x) for x in W['system']]); FS = W['F_skin']
sk = np.unique(FS); isb = sysn[OB] == 'bone'; d_a = np.zeros(len(V0)); d_a[sk] = cKDTree(V0[isb]).query(V0[sk])[0]
Ha = float(np.ptp(V0[sk][:, 2])); ra = d_a[sk] / Ha; lo, hi = np.percentile(ra, [2, 98])
print('BS atlas: height %.3f m; skin-to-bone distance median %.1f mm, 2%%..98%% %.1f..%.1f mm (%.4f..%.4f of height)' % (Ha, 1000 * np.median(d_a[sk]), 1000 * lo * Ha, 1000 * hi * Ha, lo, hi))
cm = colormaps['RdYlBu']
def colours(r): return (cm(np.clip((np.log(np.maximum(r, 1e-9)) - np.log(lo)) / (np.log(hi) - np.log(lo)), 0, 1)) * 255).astype(np.uint8)
def export(V, F, r, name):
    u, inv = np.unique(F, return_inverse=True); P = V[u]; m = trimesh.Trimesh(np.c_[P[:, 0], P[:, 2], -P[:, 1]], inv.reshape(-1, 3), process=False, vertex_colors=colours(r[u])); m.export(os.path.join(OD, name))
export(V0, FS, d_a / Ha, 'support_atlas.glb')
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']; Hm = float(np.ptp(MV[:, 2]))
dd, k = cKDTree(Vw[sk]).query(MV); r_c = ra[k]                       # carried: nearest warped atlas skin vertex
K = np.load(os.path.join(FD, 'skeleton_fit.npz'), allow_pickle=True); Vr = K['V_rigid'].astype(np.float64); r_o = cKDTree(Vr[isb]).query(MV)[0] / Hm
print('BS model: height %.3f m; carried map median %.1f mm; own surface-to-bone median %.1f mm; model vertex to its atlas skin vertex: median %.1f mm, p90 %.1f mm' % (
    Hm, 1000 * Hm * np.median(r_c), 1000 * Hm * np.median(r_o), 1000 * np.median(dd), 1000 * np.percentile(dd, 90)))
export(MV, MF, r_c, 'support_carried.glb'); export(MV, MF, r_o, 'support_own.glb')
np.savez(os.path.join(OD, 'support.npz'), atlas_ids=sk, atlas_share=ra, model_carried_share=r_c, model_own_share=r_o, scale_lo=lo, scale_hi=hi, atlas_height=Ha, model_height=Hm)
