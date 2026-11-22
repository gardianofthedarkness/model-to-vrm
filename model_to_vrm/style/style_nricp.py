"""The bent twin registered onto the original by non-rigid ICP (Amberg et al. 2007, trimesh.registration.nricp_amberg,
its default steps), then carried back to T-pose: python style_nricp.py <ab_free.npz> <twin_rig.npz> <orig_colours.npz> <out.npz>
  weld     the twin's rig copy is welded at coincident points (file seams), so the registration sees one surface
  target   the original's full-detail points (nearest-point matching; no rtree on this machine for face matching)
  unpose   each registered point returns to T-pose by its own turn: x_T = twin_rest + R^T (x_reg - twin_bent)"""
import sys, time, numpy as np, trimesh
from scipy.spatial import cKDTree
fit, rig, oc, out = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]
V0, Vb, W, Rg = fit['V0'], fit['Vbent'], fit['W'], fit['Rg']; F = rig['F']
_, first, inv = np.unique(V0.round(7), axis=0, return_index=True, return_inverse=True); inv = inv.ravel()
Fw = inv[F]; Fw = Fw[(Fw[:, 0] != Fw[:, 1]) & (Fw[:, 1] != Fw[:, 2]) & (Fw[:, 0] != Fw[:, 2])]
src = trimesh.Trimesh(Vb[first], Fw, process=False); tgt = oc['V']
print('NR source %d points (welded from %d), %d faces; target %d points' % (len(src.vertices), len(V0), len(Fw), len(tgt)), flush=True)
t0 = time.time()
reg = trimesh.registration.nricp_amberg(src, tgt, use_faces=False, use_vertex_normals=False)
reg = np.asarray(reg); print('NR registered in %.0f s' % (time.time() - t0), flush=True)
Vreg = reg[inv]                                                                         # back to every rig point
d0 = cKDTree(tgt).query(Vb)[0]; d1 = cKDTree(tgt).query(Vreg)[0]
print('NR twin onto the original: before median %.4f, 95%% %.4f | after median %.4f, 95%% %.4f' % (np.median(d0), np.percentile(d0, 95), np.median(d1), np.percentile(d1, 95)))
M = np.einsum('nj,jab->nab', W, Rg); U, _, Vt = np.linalg.svd(M); bad = np.linalg.det(U @ Vt) < 0; U[bad, :, -1] *= -1; R = U @ Vt
Dsp = np.einsum('nba,nb->na', R, Vreg - Vb)                                           # the correction in T-pose
dn = np.linalg.norm(Dsp, axis=1); print('NR correction in T-pose: median %.4f, 95%% %.4f, max %.4f' % (np.median(dn), np.percentile(dn, 95), dn.max()))
np.savez(out, rig_D=Dsp, Vreg=Vreg)
