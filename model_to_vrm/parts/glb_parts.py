"""Blender: each part of a GLB in its own colour, front and back, plus a per-part table (size, centre, height span).
blender -b -P glb_parts.py -- <model.glb> <prefix>"""
import bpy, sys, colorsys, mathutils
a = sys.argv[sys.argv.index('--') + 1:]; src, pre = a[0], a[1]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
M = sorted([o for o in bpy.data.objects if o.type == 'MESH'], key=lambda o: int(o.name.rsplit('_', 1)[1]))
V = [o.matrix_world @ v.co for o in M for v in o.data.vertices]
lo = mathutils.Vector([min(v[k] for v in V) for k in range(3)]); hi = mathutils.Vector([max(v[k] for v in V) for k in range(3)]); d = hi - lo
for i, o in enumerate(M):
    P = [o.matrix_world @ v.co for v in o.data.vertices]; c = sum(P, mathutils.Vector()) / len(P)
    z0, z1 = min(p.z for p in P), max(p.z for p in P); x0, x1 = min(p.x for p in P), max(p.x for p in P)
    print('GP %-14s faces %7d  centre x %+.3f y %+.3f z %.3f  z %.3f..%.3f  x %+.3f..%+.3f' % (o.name, len(o.data.polygons), c.x, c.y, c.z, z0, z1, x0, x1))
    o.color = (*colorsys.hsv_to_rgb((i * 0.618) % 1, 0.7, 0.95), 1)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; HP = 1600; pad = 0.04 * d.z
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'OBJECT'; sc.view_settings.view_transform = 'Standard'
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam
cam.data.type = 'ORTHO'; cam.data.clip_end = 100 * d.length; R = 3 * d.length; c = (lo + hi) / 2
w = max(d.x, d.y) + 2 * pad; h = d.z + 2 * pad; sc.render.resolution_x, sc.render.resolution_y = int(HP * w / h), HP; cam.data.ortho_scale = h
for name, dirv in (('front', (0, -1, 0)), ('back', (0, 1, 0)), ('side', (1, 0, 0))):
    cam.location = c + R * mathutils.Vector(dirv); cam.rotation_euler = (-mathutils.Vector(dirv)).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = '%s_parts_%s.png' % (pre, name); bpy.ops.render.render(write_still=True)
