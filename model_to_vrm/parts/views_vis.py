"""Blender: which vertices each saved camera sees (ray cast from the vertex toward the camera, nothing in the way):
  blender -b -P views_vis.py -- <model.glb> <views dir> [prefix]   ->  <dir>/vis.npz (names, vis: views x vertices)
Vertex order = objects in import order, vertices in mesh order (the order part_colours.py / part_normals.py write)."""
import bpy, sys, os, json, mathutils, numpy as np
from mathutils.bvhtree import BVHTree
a = sys.argv[sys.argv.index('--') + 1:]; src, D = a[:2]; pre = a[2] if len(a) > 2 else 'tex_'
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); Vs, Fs, off = [], [], 0
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    me = o.data; co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3); M = np.array(o.matrix_world)
    Vs.append(co @ M[:3, :3].T + M[:3, 3]); Fs += [[off + i for i in p.vertices] for p in me.polygons]; off += len(co)
P = np.vstack(Vs); H = np.ptp(P[:, 2]); bvh = BVHTree.FromPolygons([tuple(x) for x in P.tolist()], Fs)
cams = json.load(open(os.path.join(D, 'cameras.json'))); names = sorted(n for n in cams if n.startswith(pre)); vis = []
for n in names:
    Mi = np.array(cams[n]['matrix_inv']); d = mathutils.Vector((-Mi[2, :3]).tolist()).normalized() * -1   # camera looks along its -Z; toward camera = +Z row of the inverse
    d = mathutils.Vector(np.linalg.inv(Mi)[:3, 2].tolist()).normalized(); eps = H * 1e-4; vv = np.zeros(len(P), bool)
    for i, p in enumerate(P):
        vv[i] = bvh.ray_cast(mathutils.Vector(p) + d * eps, d)[0] is None
    vis.append(vv); print('VV', n, round(float(vv.mean()), 3), flush=True)
np.savez(os.path.join(D, 'vis.npz'), names=np.array(names), vis=np.array(vis)); print('VV done')
