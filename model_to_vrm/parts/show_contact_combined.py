"""Blender: SAM 3 materials on the bare mesh (light tint), contact vertices of real contacts in full colour, contacts the
LLM called not real in violet; four whole views, then per real patch a close-up and a cut-through section (camera clip
plane through the patch centre, so a contact inside a crease shows both sides):
  blender -b -P show_contact_combined.py -- <model.glb> <contacts dir> <out dir>
flesh pink, cloth blue, hair brown, prop green, unknown grey."""
import bpy, sys, os, json, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, D, od = a[:3]; os.makedirs(od, exist_ok=True)
C = np.load(os.path.join(D, 'combined.npz')); mat = C['material']; pid = np.load(os.path.join(D, 'patches.npz'))['patch']; J = json.load(open(os.path.join(D, 'combined.json')))
COL = np.array([(1.0, 0.45, 0.5), (0.15, 0.4, 1.0), (0.45, 0.25, 0.08), (0.15, 0.75, 0.25)])
real = np.array([r['real_contact'] is not False for r in J])
col = np.tile([0.7, 0.7, 0.7, 1.0], (len(mat), 1)); k = mat >= 0; col[k, :3] = COL[mat[k]] * 0.35 + 0.65
cv = pid >= 0; rv = cv & real[np.maximum(pid, 0)]; col[rv & k, :3] = COL[mat[rv & k]]; col[cv & ~rv, :3] = (0.55, 0.3, 0.85)
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
sc.render.resolution_x = sc.render.resolution_y = 640
for r in J:
    if r['real_contact'] is False: continue
    s = max(r['extent'] * 1.6, ext / 12)
    shoot(r['centre'], r['az'], s, os.path.join(od, 'p%02d_close.png' % r['patch']))
    shoot(r['centre'], r['az'], s, os.path.join(od, 'p%02d_section.png' % r['patch']), clip=10.0)
print('SCC done')
