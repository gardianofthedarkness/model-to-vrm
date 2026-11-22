"""Inputs the slice fit needs, for an untextured skin-tight body (e.g. the T-pose base body shaped from the H3 views):
  blender -b -P plain_body_inputs.py -- <model.glb> <out dir>
  parts/orig_colours.npz   V (world) in the same point order as parts/part_normals.py (every mesh in import order,
                           no welding), plain grey C: the pipeline's colours step needs a texture and this body has none
  contacts/material.npz    every point 'flesh': the whole surface is the body itself (fitted clothes, no props)"""
import bpy, sys, os, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, R = a
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
Vs = []
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    me = o.data; v = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', v); v = v.reshape(-1, 3)
    M = np.array(o.matrix_world); Vs.append(v @ M[:3, :3].T + M[:3, 3])
V = np.vstack(Vs).astype(np.float64)
np.savez(os.path.join(R, 'parts', 'orig_colours.npz'), V=V, C=np.full((len(V), 3), 0.7))
os.makedirs(os.path.join(R, 'contacts'), exist_ok=True)
np.savez(os.path.join(R, 'contacts', 'material.npz'), material=np.zeros(len(V), np.int64), materials=np.array(['flesh', 'cloth', 'hair', 'prop']))
print('PB wrote %d points, all flesh' % len(V))
