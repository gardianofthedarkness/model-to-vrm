"""Blender: look at a model, textured and bare (clay, lit so relief shows), whole body and head:
  blender -b -P look.py -- <model.glb> <out prefix> [front|side|both]
Writes <prefix>_<tex|clay>_<body|head>_<view>.png (1024 px, orthographic). The head frame is the top of the model down
to the narrowest slice below the head's widest slice (the neck), found from the mesh's own width profile."""
import bpy, sys, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, pre = a[0], a[1]; which = a[2] if len(a) > 2 else 'both'
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']; dg = bpy.context.evaluated_depsgraph_get()
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.evaluated_get(dg).data.vertices]) for o in obs])
lo, hi = P.min(0), P.max(0); H = hi[2] - lo[2]
# head: width profile along height (x extent per slice); neck = narrowest slice in the top third
z = P[:, 2]; edges = np.linspace(hi[2] - H / 3, hi[2], 61); w = np.array([np.ptp(P[(z >= e0) & (z < e1), 0]) if ((z >= e0) & (z < e1)).sum() > 10 else np.inf for e0, e1 in zip(edges[:-1], edges[1:])])
wf = np.where(np.isfinite(w), w, 0); top = np.argmax(wf[len(wf) // 2:]) + len(wf) // 2   # widest slice of the head (upper half)
neck = edges[np.argmin(np.where(np.arange(len(w)) < top, w, np.inf))]; HP = P[z >= neck]   # narrowest slice below it
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 40 * H
views = {'front': 0, 'side': 90}
if which != 'both': views = {which: views[which]}
sh = sc.display.shading
for mode in ('tex', 'clay'):
    if mode == 'tex': sh.light = 'FLAT'; sh.color_type = 'TEXTURE'
    else: sh.light = 'STUDIO'; sh.color_type = 'SINGLE'; sh.single_color = (0.8, 0.8, 0.8); sh.show_cavity = True; sh.cavity_type = 'BOTH'; sh.show_specular_highlight = False
    for frame, Q in (('body', P), ('head', HP)):
        c = (Q.min(0) + Q.max(0)) / 2; ext = float(np.ptp(Q, 0).max() * 1.1)
        for vn, az in views.items():
            d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0))
            cam.location = mathutils.Vector(c.tolist()) + d * 10 * H; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = ext
            sc.render.filepath = '%s_%s_%s_%s.png' % (pre, mode, frame, vn); bpy.ops.render.render(write_still=True)
print('LOOK done neck z %.3f of %.3f..%.3f' % (neck, lo[2], hi[2]))
