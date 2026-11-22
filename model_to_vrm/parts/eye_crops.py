"""Blender: close-ups of both eyes from the 4 front directions (v00, v01, v07, v08 of views_vis.py), cameras saved into
cameras.json beside the other renders: blender -b -P eye_crops.py -- <textured.glb> <eye points .npy> <sam3views dir> <height>
Each crop frames the eye points (both eyes) with a margin of half their extent."""
import bpy, sys, os, json, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, ep, od, H = a[0], a[1], a[2], float(a[3])
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); E = np.load(ep); c = (E.min(0) + E.max(0)) / 2; ext = float(np.ptp(E, 0).max() * 1.5)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; R = 1024; sc.render.resolution_x = sc.render.resolution_y = R
sh = sc.display.shading; sh.light = 'FLAT'; sh.color_type = 'TEXTURE'; sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 40 * H; cam.data.ortho_scale = ext
cams = json.load(open(os.path.join(od, 'cameras.json')))
views = [(az, el) for el in (0, 35) for az in np.arange(8) * 45]
for vi in (0, 1, 7, 8):
    az, el = views[vi]; d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
    cam.location = mathutils.Vector(c.tolist()) + d * 10 * H; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    n = 'v%02d_eyes' % vi; sc.render.filepath = os.path.join(od, n + '.png'); bpy.ops.render.render(write_still=True); bpy.context.view_layer.update()
    cams[n] = {'view': vi, 'band': 'eyes', 'matrix_inv': [list(r) for r in cam.matrix_world.inverted()], 'scale': ext, 'res': R}
json.dump(cams, open(os.path.join(od, 'cameras.json'), 'w')); print('EC done')
