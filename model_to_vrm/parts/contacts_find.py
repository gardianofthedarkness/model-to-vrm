"""Blender: for every vertex, cast a ray outward along its normal; if it hits the mesh on a face that faces back toward
the vertex, the two surfaces face each other across a gap (or meet at a fused crease) - a possible contact:
  blender -b -P contacts_find.py -- <model.glb> <out.npz>
Writes per vertex: gap (hit distance, inf if none or the hit faces away), hit_vertex (nearest vertex of the hit face),
in import order (as part_colours.py)."""
import bpy, sys, mathutils, numpy as np
from mathutils.bvhtree import BVHTree
a = sys.argv[sys.argv.index('--') + 1:]; src, out = a[:2]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); Vs, Ns, Fs, off = [], [], [], 0
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    me = o.data; M = np.array(o.matrix_world); co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    n = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('normal', n); n = n.reshape(-1, 3) @ np.linalg.inv(M[:3, :3]); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    Vs.append(co @ M[:3, :3].T + M[:3, 3]); Ns.append(n); Fs += [[off + i for i in p.vertices] for p in me.polygons]; off += len(co)
P = np.vstack(Vs); N = np.vstack(Ns); F = Fs; H = np.ptp(P[:, 2]); bvh = BVHTree.FromPolygons([tuple(x) for x in P.tolist()], F)
gap = np.full(len(P), np.inf, np.float32); hitv = -np.ones(len(P), np.int64); eps = H * 1e-4
for i in range(len(P)):
    p = mathutils.Vector(P[i]); d = mathutils.Vector(N[i]); loc, nrm, fi, dist = bvh.ray_cast(p + d * eps, d)
    if loc is None or nrm.dot(d) >= 0: continue                    # nothing ahead, or the hit face looks away (a back side)
    fv = F[fi]; hv = fv[int(np.argmin([(mathutils.Vector(P[j]) - loc).length for j in fv]))]
    gap[i] = dist + eps; hitv[i] = hv
    if i % 200000 == 0: print('CF', i, flush=True)
np.savez(out, gap=gap, hit_vertex=hitv); print('CF done: %d of %d vertices face another surface' % (np.isfinite(gap).sum(), len(P)))
