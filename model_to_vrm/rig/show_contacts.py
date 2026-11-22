"""Blender: contact patches (where two body pieces press together) inside the see-through mesh, front/side/back:
  blender -b -P show_contacts.py -- <model.glb> <seg dir> <out prefix>
Each contact_<A>__<B>.ply from segment.py gets its own strong colour; the legend is printed."""
import bpy, sys, os, math, colorsys, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, sd, pre = a[:3]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); target = [o for o in bpy.context.scene.objects if o.type == 'MESH']
clay = bpy.data.materials.new('clay'); clay.diffuse_color = (0.8, 0.8, 0.8, 1)
for o in target: o.data.materials.clear(); o.data.materials.append(clay)
files = sorted(f for f in os.listdir(sd) if f.startswith('contact_') and f.endswith('.ply'))
for k, f in enumerate(files):
    bpy.ops.wm.ply_import(filepath=os.path.join(sd, f)); o = bpy.context.selected_objects[0]
    m = bpy.data.materials.new(f); m.diffuse_color = (*colorsys.hsv_to_rgb(k / max(len(files), 1), 0.9, 1.0), 1); o.data.materials.clear(); o.data.materials.append(m)
    print('SC legend', f[8:-4].replace('__', ' + '), '#%02x%02x%02x' % tuple(int(255 * c) for c in m.diffuse_color[:3]))
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_xray = True; sh.xray_alpha = 0.3
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in target]); c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0).max() * 1.05)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.ortho_scale = ext; cam.data.clip_end = 100
for vn, az in (('front', 0), ('side', 90), ('back', 180), ('side2', 270)):
    d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0)); cam.location = mathutils.Vector(c.tolist()) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = '%s_%s.png' % (pre, vn); bpy.ops.render.render(write_still=True)
print('SC done')
