"""Blender: whole-body renders around the model (8 directions, level), textured and clay, with cameras saved:
  blender -b -P body_views.py -- <model.glb> <out dir>
Writes <out>/<tex|clay>_a<az>.png and cameras.json (orthographic: matrix_inv, scale, res)."""
import bpy, sys, os, json, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, od = a[0], a[1]; os.makedirs(od, exist_ok=True)
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']; dg = bpy.context.evaluated_depsgraph_get()
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.evaluated_get(dg).data.vertices]) for o in obs])
c = (P.min(0) + P.max(0)) / 2; H = np.ptp(P[:, 2]); ext = float(np.ptp(P, 0).max() * 1.05)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; R = 1024; sc.render.resolution_x = sc.render.resolution_y = R
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 40 * H; cam.data.ortho_scale = ext
sh = sc.display.shading; cams = {}
textured = any(n.type == 'TEX_IMAGE' and n.image and n.image.size[0] > 0 for o in obs for m in o.data.materials if m and m.node_tree for n in m.node_tree.nodes)
if not textured: print('BV no texture image on the model: the "tex" views are shaded clay renders (an untextured model has only its shape to show)')
for mode in ('tex', 'clay'):
    if mode == 'tex' and textured: sh.light = 'FLAT'; sh.color_type = 'TEXTURE'; sh.show_cavity = False
    else: sh.light = 'STUDIO'; sh.color_type = 'SINGLE'; sh.single_color = (0.8, 0.8, 0.8); sh.show_cavity = True; sh.cavity_type = 'BOTH'
    for az in range(0, 360, 45):
        d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0))
        cam.location = mathutils.Vector(c.tolist()) + d * 10 * H; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); bpy.context.view_layer.update()
        n = '%s_a%03d' % (mode, az); sc.render.filepath = os.path.join(od, n + '.png'); bpy.ops.render.render(write_still=True)
        cams[n] = {'az': az, 'matrix_inv': [list(r) for r in cam.matrix_world.inverted()], 'scale': ext, 'res': R}
json.dump(cams, open(os.path.join(od, 'cameras.json'), 'w')); print('BV done')
