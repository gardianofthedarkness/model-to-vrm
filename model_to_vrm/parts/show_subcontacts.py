"""Blender: per sub-contact a close-up and a cut-through section (clip plane through its centre), side A magenta, side B
cyan, the rest tinted by SAM 3 material (flesh pink, cloth blue, hair brown, prop green):
  blender -b -P show_subcontacts.py -- <model.glb> <contacts dir>      -> <contacts dir>/sub_renders/s<k>_close|section.png
Reads combined.npz (material), sub.npz and sub.json (contacts_split.py)."""
import bpy, sys, os, json, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, D = a[:2]; od = os.path.join(D, 'sub_renders'); os.makedirs(od, exist_ok=True)
mat = np.load(os.path.join(D, 'combined.npz'))['material']; S = np.load(os.path.join(D, 'sub.npz')); G = json.load(open(os.path.join(D, 'sub.json')))
COL = np.array([(1.0, 0.45, 0.5), (0.15, 0.4, 1.0), (0.45, 0.25, 0.08), (0.15, 0.75, 0.25)])
base = np.tile([0.7, 0.7, 0.7, 1.0], (len(mat), 1)); k = mat >= 0; base[k, :3] = COL[mat[k]] * 0.5 + 0.5
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); target = [o for o in bpy.data.objects if o.type == 'MESH']; ats = []
for o in target: at = o.data.color_attributes.new('lab', 'FLOAT_COLOR', 'POINT'); o.data.color_attributes.active_color = at; ats.append((o, len(o.data.vertices)))
def paint(col):
    off = 0
    for o, n in ats: o.data.color_attributes['lab'].data.foreach_set('color', col[off:off + n].ravel()); o.data.update(); off += n
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 640
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'VERTEX'; sh.show_cavity = True
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in target]); ext = float(np.ptp(P, 0).max() * 1.05)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100
def shoot(centre, az, scale, path, clip):
    d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0)); cam.location = mathutils.Vector(list(centre)) + d * 10
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = scale; cam.data.clip_start = clip; sc.render.filepath = path; bpy.ops.render.render(write_still=True)
for g in G:
    m = S['group'] == g['sub']; col = base.copy(); col[S['vert'][m][S['side'][m] == 0], :3] = (1, 0, 1); col[S['vert'][m][S['side'][m] == 1], :3] = (0, 1, 1); paint(col)
    s = max(g['extent'] * 1.6, ext / 12)
    shoot(g['centre'], g['az'], s, os.path.join(od, 's%03d_close.png' % g['sub']), 0.01)
    shoot(g['centre'], g['az'], s, os.path.join(od, 's%03d_section.png' % g['sub']), 10.0)
    print('SSC', g['sub'], flush=True)
print('SSC done')
