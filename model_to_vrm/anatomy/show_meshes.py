"""Blender: one or more meshes from .npz files (vertices V, faces F) inside the see-through model, front/side/back/3-4:
  blender -b -P show_meshes.py -- <model.glb> <out prefix> <file.npz>:<V key>:<F key>:<r,g,b,a> [...]
  e.g. smpl_fit.npz:V:F:0.2,0.5,0.9,0.6"""
import bpy, sys, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, pre, specs = a[0], a[1], a[2:]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); model = [o for o in bpy.data.objects if o.type == 'MESH']
clay = bpy.data.materials.new('clay'); clay.diffuse_color = (0.8, 0.8, 0.8, 0.25)
for o in model: o.data.materials.clear(); o.data.materials.append(clay)
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in model])
for i, sp in enumerate(specs):
    f, kv, kf, col = sp.rsplit(':', 3); Z = np.load(f, allow_pickle=True); V, F = np.asarray(Z[kv], float), np.asarray(Z[kf], int)
    me = bpy.data.meshes.new('m%d' % i); me.from_pydata(V.tolist(), [], F.tolist()); ob = bpy.data.objects.new('m%d' % i, me)
    bpy.context.scene.collection.objects.link(ob); m = bpy.data.materials.new('c%d' % i); m.diffuse_color = tuple(float(x) for x in col.split(',')); ob.data.materials.append(m)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100
c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0).max() * 1.05)
for vn, az, el in (('front', 0, 0), ('side', 90, 0), ('back', 180, 0), ('three_quarter', 40, 15)):
    d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
    cam.location = mathutils.Vector(c.tolist()) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = ext
    sc.render.filepath = '%s_%s.png' % (pre, vn); bpy.ops.render.render(write_still=True)
print('SM done')
