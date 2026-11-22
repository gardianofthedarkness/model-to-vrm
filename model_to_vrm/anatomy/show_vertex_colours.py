"""Blender: a .glb with vertex colours (e.g. a heat map) rendered flat from the front, back, side and three-quarter:
  blender -b -P show_vertex_colours.py -- <file.glb> <out prefix>"""
import bpy, sys, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, pre = a[0], a[1]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in obs]); c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0).max() * 1.05)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'FLAT'; sh.color_type = 'VERTEX'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100; cam.data.ortho_scale = ext
for vn, az, el in (('front', 0, 0), ('back', 180, 0), ('side', 90, 0), ('three_quarter', 40, 15)):
    d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
    cam.location = mathutils.Vector(c.tolist()) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); sc.render.filepath = '%s_%s.png' % (pre, vn); bpy.ops.render.render(write_still=True)
print('SVC done')
