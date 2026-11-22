"""Blender: one part of a model as its own textured model (for looking at it alone: views, SAM 3, vote):
  blender -b -P part_glb.py -- <model.glb> <faces.npy: face centres of the part, world (Z up)> <out.glb>
The part's faces are found by their centres (nearest face centre of the imported mesh); every other face is deleted, the
materials and textures stay. Face centres rather than indices, so the face order of other tools does not matter."""
import bpy, bmesh, sys, numpy as np, mathutils
a = sys.argv[sys.argv.index('--') + 1:]; src, fc_path, out = a
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); want = np.load(fc_path)
kd = mathutils.kdtree.KDTree(len(want))
for i, p in enumerate(want): kd.insert(p, i)
kd.balance(); kept = 0
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    bm = bmesh.new(); bm.from_mesh(o.data); M = o.matrix_world; dele = []
    for f in bm.faces:                                            # a face matches when a wanted centre lies within half its shortest edge
        c = M @ f.calc_center_median(); _, _, d = kd.find(c)
        if d > 0.5 * min(e.calc_length() for e in f.edges) * max(M.to_scale()): dele.append(f)
    kept += len(bm.faces) - len(dele)
    bmesh.ops.delete(bm, geom=dele, context='FACES'); bm.to_mesh(o.data); bm.free()
    if len(o.data.polygons) == 0: bpy.data.objects.remove(o)
bpy.ops.export_scene.gltf(filepath=out, export_format='GLB'); print('PG kept %d faces of %d wanted' % (kept, len(want)))
