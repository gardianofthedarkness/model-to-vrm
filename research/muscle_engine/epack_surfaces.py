"""E-PACK, GPU route, step 2: shared region surfaces from the label grid (epack_grid.py) with VTK's SurfaceNets3D (multi-label
surface nets with constrained smoothing: every interface between two regions is ONE surface carrying both labels, so regions
can never overlap or leave gaps).  Checks every region's surface is closed and manifold, and its volume against the grid.
Writes <out>/surf_thigh_l.npz: P (points), F (triangles), side (the two labels of each triangle; -1 = outside the body).
  E:\\splat\\.venv-sim\\Scripts\\python.exe epack_surfaces.py <grid_thigh_l.npz> <out dir> [smoothing iterations]"""
import sys, os, json, time, numpy as np, vtk
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy
GR, OUT = sys.argv[1:3]; IT = int(sys.argv[3]) if len(sys.argv) > 3 else None
G = np.load(GR); h = float(G['h']); L = np.pad(G['lab'].astype(np.int32) + 1, 1); o = G['origin'] - h; names = [str(s) for s in G['region_names']]   # 0 = outside
t0 = time.time()
img = vtk.vtkImageData(); img.SetDimensions(*L.shape); img.SetSpacing(h, h, h); img.SetOrigin(*(o + 0.5 * h))
arr = numpy_to_vtk(L.ravel(order='F'), deep=True, array_type=vtk.VTK_INT); arr.SetName('lab'); img.GetPointData().SetScalars(arr)
sn = vtk.vtkSurfaceNets3D(); sn.SetInputData(img); sn.SetBackgroundLabel(0); sn.GenerateLabels(len(names), 1, len(names))
sn.SetOutputMeshTypeToTriangles(); sn.SmoothingOn()
if IT is not None: sn.SetNumberOfIterations(IT)
sn.Update(); out = sn.GetOutput()
P = vtk_to_numpy(out.GetPoints().GetData()).astype(np.float64)
F = vtk_to_numpy(out.GetPolys().GetConnectivityArray()).reshape(-1, 3).astype(np.int64)
side = vtk_to_numpy(out.GetCellData().GetArray('BoundaryLabels')).reshape(-1, 2).astype(np.int64) - 1   # back to region ids, -1 outside
dt = time.time() - t0
print('SN surface nets %.1f s: %d points, %d triangles, smoothing iterations %d' % (dt, len(P), len(F), sn.GetNumberOfIterations()), flush=True)
rep = {}
for r, nm in enumerate(names):
    a = side[:, 0] == r; b = side[:, 1] == r; f = np.concatenate([F[a], F[b][:, ::-1]])
    if not len(f): rep[nm] = 'empty'; continue
    e = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), 1); _, c = np.unique(e, axis=0, return_counts=True)
    v = np.einsum('ij,ij->i', P[f[:, 0]], np.cross(P[f[:, 1]], P[f[:, 2]])).sum() / 6 * 1e6
    rep[nm] = dict(triangles=int(len(f)), open_edges=int((c == 1).sum()), nonmanifold_edges=int((c > 2).sum()), volume_cm3=round(abs(v), 3),
                   grid_cm3=round(float((L == r + 1).sum()) * h ** 3 * 1e6, 3), sign=int(np.sign(v)))
bad = {k: v for k, v in rep.items() if not isinstance(v, dict) or v['open_edges'] or v['nonmanifold_edges']}
np.savez_compressed(os.path.join(OUT, 'surf_thigh_l.npz'), P=P, F=F, side=side, region_names=G['region_names'], region_kind=G['region_kind'],
                    source_volume_cm3=G['source_volume_cm3'], h=h)
json.dump(dict(seconds=dt, points=int(len(P)), triangles=int(len(F)), regions=rep), open(os.path.join(OUT, 'surf_thigh_l.json'), 'w'), indent=1)
print('SN regions with open or non-manifold edges: %d of %d' % (len(bad), len(names)))
for k, v in rep.items(): print('SN  %-40s %s' % (k, v))
