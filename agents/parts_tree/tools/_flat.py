"""Blender (called by render.py): one colour per vertex (import order), flat light, through two saved cameras:
  blender -b -P _flat.py -- <model.glb> <colours.npy> <cameras.json> <front cam> <back cam> <out prefix>"""
import bpy, sys, json, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, cp, cj, fc, bc, pre = a
C = np.load(cp); cams = json.load(open(cj))
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); off = 0
for ob in [o for o in bpy.data.objects if o.type == 'MESH']:
    me = ob.data; n = len(me.vertices); cols = C[off:off + n]; off += n
    lin = np.where(cols <= 0.04045, cols / 12.92, ((cols + 0.055) / 1.055) ** 2.4)
    ca = me.color_attributes.new('flat', 'FLOAT_COLOR', 'POINT'); ca.data.foreach_set('color', np.c_[lin, np.ones(len(lin))].astype(np.float32).ravel()); me.color_attributes.active_color = ca
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1)
sh = sc.display.shading; sh.light = 'FLAT'; sh.color_type = 'VERTEX'; sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 1e4
for tag, n in (('front', fc), ('back', bc)):
    cm = cams[n]; sc.render.resolution_x = sc.render.resolution_y = cm['res']; cam.matrix_world = mathutils.Matrix(cm['matrix_inv']).inverted(); cam.data.ortho_scale = cm['scale']
    sc.render.filepath = '%s_%s.png' % (pre, tag); bpy.ops.render.render(write_still=True)
print('FLAT done')
