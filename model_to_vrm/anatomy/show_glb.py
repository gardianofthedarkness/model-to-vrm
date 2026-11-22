"""Blender: one or more GLB/OBJ files as plain clay, side by side, front / side / back / three-quarter:
  blender -b -P show_glb.py -- <out prefix> <a.glb> [b.glb ...]      -> <prefix>_<view>.png"""
import bpy, sys, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; pre, files = a[0], a[1:]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
clay = bpy.data.materials.new('clay'); clay.diffuse_color = (0.78, 0.78, 0.8, 1)
x0 = 0.0; allp = []
for f in files:
    before = set(bpy.data.objects)
    if f.lower().endswith('.obj'): bpy.ops.wm.obj_import(filepath=f)
    else: bpy.ops.import_scene.gltf(filepath=f)
    new = [o for o in bpy.data.objects if o not in before and o.type == 'MESH']
    P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in new]); lo, hi = P.min(0), P.max(0)
    for o in new:                                                 # stand each on the floor, next to the previous one
        o.data.materials.clear(); o.data.materials.append(clay)
        o.location.x += x0 - lo[0]; o.location.z -= lo[2]
    bpy.context.view_layer.update(); allp.append(np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in new]))
    x0 += (hi[0] - lo[0]) * 1.15
P = np.concatenate(allp)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x, sc.render.resolution_y = 1400, 1000
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_cavity = True
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100
c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0)[[0, 2]].max() * 1.1)
for vn, az, el in (('front', 0, 0), ('side', 90, 0), ('back', 180, 0), ('three_quarter', 35, 12)):
    d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
    cam.location = mathutils.Vector(c.tolist()) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = ext
    sc.render.filepath = '%s_%s.png' % (pre, vn); bpy.ops.render.render(write_still=True)
print('SG done')
