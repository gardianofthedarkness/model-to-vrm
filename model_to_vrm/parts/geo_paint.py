"""Blender: paint labels onto the bare mesh and render them over the clay shading (head views):
  blender -b -P geo_paint.py -- <model.glb> <organs.npz> <views dir> <out.png>
Each head vertex gets its label's colour (unlabelled = clay grey), shown with studio light and cavity so the relief
stays visible; renders the front head and front face cameras of geo_views.py side by side."""
import bpy, sys, os, json, colorsys, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, lp, vd, out = a[:4]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
L = np.load(lp); lab, names, idx = L['label'], list(L['names']), L['idx']
obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']; n_tot = sum(len(o.data.vertices) for o in obs)
col = np.tile([0.8, 0.8, 0.8, 1.0], (n_tot, 1)); pal = {k: colorsys.hsv_to_rgb(k / max(len(names), 1), 0.85, 1.0) for k in range(len(names))}
for i, l in zip(idx, lab):
    if l >= 0: col[i, :3] = pal[l]
off = 0
for o in obs:
    me = o.data; n = len(me.vertices); a_ = me.color_attributes.new('lab', 'FLOAT_COLOR', 'POINT'); a_.data.foreach_set('color', col[off:off + n].ravel()); me.color_attributes.active_color = a_; off += n
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'VERTEX'; sh.show_cavity = True; sh.cavity_type = 'BOTH'; sh.show_specular_highlight = False
cams = json.load(open(os.path.join(vd, 'cameras.json'))); cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'
files = []
for n in ('head_a+00_e00', 'face_a+00_e00', 'face_a+45_e00'):
    cm = cams[n]; cam.matrix_world = mathutils.Matrix(cm['matrix_inv']).inverted(); cam.data.ortho_scale = cm['scale']; cam.data.clip_end = 1e4
    f = out[:-4] + '_' + n + '.png'; sc.render.filepath = f; bpy.ops.render.render(write_still=True); files.append(f)
print('GP done', ' '.join(files)); print('GP legend', ' '.join('%s=%s' % (names[k], '#%02x%02x%02x' % tuple(int(255 * c) for c in pal[k])) for k in range(len(names))))
