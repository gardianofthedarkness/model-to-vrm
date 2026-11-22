"""Blender (called by render.py): vertex colours per part object in the order part_colours.py read them, flat render."""
import bpy, sys, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, dp, cp, pre = a
D = np.load(dp); full_part = D['full_part'] if 'full_part' in D.files else None; C = np.load(cp)
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
for ob in [o for o in bpy.data.objects if o.type == 'MESH']:
    p = int(ob.name.rsplit('_', 1)[1]); cols = C[full_part == p]; me = ob.data
    if len(cols) != len(me.vertices): print('mismatch', p, len(cols), len(me.vertices)); continue
    ca = me.color_attributes.new('flat', 'FLOAT_COLOR', 'POINT'); lin = np.where(cols <= 0.04045, cols / 12.92, ((cols + 0.055) / 1.055) ** 2.4)
    ca.data.foreach_set('color', np.c_[lin, np.ones(len(lin))].astype(np.float32).ravel()); me.color_attributes.active_color = ca
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1)
sh = sc.display.shading; sh.light = 'FLAT'; sh.color_type = 'VERTEX'; sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 50
for v, d, c, s, res in (('front', (0, -1, 0), (0, 0, 0.49), 1.05, (700, 1100)), ('back', (0, 1, 0), (0, 0, 0.49), 1.05, (700, 1100)), ('head', (0, -1, 0), (0.017, 0, 0.86), 0.2, (700, 1100))):
    sc.render.resolution_x, sc.render.resolution_y = res; cam.data.ortho_scale = s * (1100 / 700 if v == 'head' else 1); d = mathutils.Vector(d); cam.location = mathutils.Vector(c) + d * 10
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); sc.render.filepath = '%s_%s.png' % (pre, v); bpy.ops.render.render(write_still=True)
print('BF done')
