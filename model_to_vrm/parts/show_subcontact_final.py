"""Blender: the labelled sub-contacts on the bare mesh. Sides of real contacts (two things pressing) in full SAM 3 material
colour (flesh pink, cloth blue, hair brown, prop green), folds of one thing violet, noise grey, the rest a light material
tint; four whole views, then a cut-through section per real flesh contact:
  blender -b -P show_subcontact_final.py -- <model.glb> <contacts dir>      -> <contacts dir>/sub_final/"""
import bpy, sys, os, json, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, D = a[:2]; od = os.path.join(D, 'sub_final'); os.makedirs(od, exist_ok=True)
mat = np.load(os.path.join(D, 'combined.npz'))['material']; S = np.load(os.path.join(D, 'sub.npz')); G = json.load(open(os.path.join(D, 'sub.json')))
L = json.load(open(os.path.join(D, 'sub_labels.json')))
COL = np.array([(1.0, 0.45, 0.5), (0.15, 0.4, 1.0), (0.45, 0.25, 0.08), (0.15, 0.75, 0.25)])
col = np.tile([0.7, 0.7, 0.7, 1.0], (len(mat), 1)); k = mat >= 0; col[k, :3] = COL[mat[k]] * 0.4 + 0.6
kind = {g['sub']: L.get(str(g['sub']), {}).get('kind', 'noise') for g in G}
for order in ('noise', 'fold', 'two_things'):                       # real contacts painted last, so they win shared vertices
    for g in G:
        if kind[g['sub']] != order: continue
        v = S['vert'][S['group'] == g['sub']]
        if order == 'two_things': kv = v[mat[v] >= 0]; col[kv, :3] = COL[mat[kv]]
        else: col[v, :3] = (0.55, 0.3, 0.85) if order == 'fold' else (0.5, 0.5, 0.5)
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); off = 0; target = [o for o in bpy.data.objects if o.type == 'MESH']
for o in target:
    n = len(o.data.vertices); at = o.data.color_attributes.new('lab', 'FLOAT_COLOR', 'POINT'); at.data.foreach_set('color', col[off:off + n].ravel()); o.data.color_attributes.active_color = at; off += n
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'VERTEX'; sh.show_cavity = True
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in target]); c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0).max() * 1.05)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100
def shoot(centre, az, scale, path, clip=0.01):
    d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0)); cam.location = mathutils.Vector(list(centre)) + d * 10
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = scale; cam.data.clip_start = clip; sc.render.filepath = path; bpy.ops.render.render(write_still=True)
for vn, az in (('front', 0), ('side', 90), ('back', 180), ('side2', 270)): shoot(c, az, ext, os.path.join(od, 'all_%s.png' % vn))
sc.render.resolution_x = sc.render.resolution_y = 512
for g in G:
    if kind[g['sub']] == 'two_things' and 'flesh' in g['pair']:
        shoot(g['centre'], g['az'], max(g['extent'] * 1.6, ext / 12), os.path.join(od, 's%03d_section.png' % g['sub']), 10.0)
print('SSF done')
