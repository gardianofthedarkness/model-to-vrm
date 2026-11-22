"""Blender: contact sides painted by the material the LLM gave them, on the bare (clay) mesh, from four directions:
  blender -b -P show_contact_labels.py -- <model.glb> <contacts dir> <out prefix>
flesh pink, cloth blue, hair brown, prop green, unknown/not a real contact grey-violet; the rest clay grey."""
import bpy, sys, os, json, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, D, pre = a[:3]
Z = np.load(os.path.join(D, 'patches.npz')); pid, side = Z['patch'], Z['side']; L = json.load(open(os.path.join(D, 'labels.json')))
MAT = {'flesh': (1.0, 0.55, 0.6), 'cloth': (0.2, 0.45, 1.0), 'hair': (0.45, 0.28, 0.12), 'prop': (0.2, 0.8, 0.3)}
col = np.tile([0.8, 0.8, 0.8, 1.0], (len(pid), 1))
for k, ans in L.items():
    for s, nm in ((0, 'A'), (1, 'B')):
        m = (pid == int(k)) & (side == s)
        mat = ans.get(nm, {}).get('material', '') if isinstance(ans.get(nm), dict) else ''
        c = MAT.get(mat, (0.6, 0.5, 0.75)) if ans.get('real_contact', True) else (0.6, 0.5, 0.75)
        col[m, :3] = c
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); off = 0; target = [o for o in bpy.data.objects if o.type == 'MESH']
for o in target:
    n = len(o.data.vertices); at = o.data.color_attributes.new('lab', 'FLOAT_COLOR', 'POINT'); at.data.foreach_set('color', col[off:off + n].ravel()); o.data.color_attributes.active_color = at; off += n
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'VERTEX'; sh.show_cavity = True
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in target]); c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0).max() * 1.05)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.ortho_scale = ext; cam.data.clip_end = 100
for vn, az in (('front', 0), ('side', 90), ('back', 180), ('side2', 270)):
    d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0)); cam.location = mathutils.Vector(c.tolist()) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = '%s_%s.png' % (pre, vn); bpy.ops.render.render(write_still=True)
print('SCL done')
