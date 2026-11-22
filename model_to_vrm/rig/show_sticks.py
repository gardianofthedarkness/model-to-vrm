"""Blender: the stick skeleton inside the target mesh (see-through clay), front and side, plus a part-label paint if given:
  blender -b -P show_sticks.py -- <model.glb> <skeleton.npz> <out prefix> [labels.npz (label per vertex, names)]
Sticks blue, joints white; with labels, the mesh is painted one colour per label (opaque, clay light) instead."""
import bpy, sys, math, colorsys, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, sp, pre = a[:3]; lp = a[3] if len(a) > 3 else None
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); target = [o for o in bpy.data.objects if o.type == 'MESH']   # the order every data step uses
S = np.load(sp); heads, par = S['heads'], S['parents']
def mat(n, c): m = bpy.data.materials.new(n); m.diffuse_color = c; return m
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in target]); H = np.ptp(P[:, 2]); r = H / 300
ms, mj = mat('s', (0.1, 0.3, 1, 1)), mat('j', (1, 1, 1, 1)); sticks = []
for i, p in enumerate(par):
    if p >= 0:
        a_, b_ = mathutils.Vector(heads[p]), mathutils.Vector(heads[i]); d = b_ - a_
        if d.length > 1e-6:
            bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, location=(a_ + b_) / 2, vertices=8); o = bpy.context.object
            o.rotation_euler = d.to_track_quat('Z', 'Y').to_euler(); o.data.materials.append(ms); sticks.append(o)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r * 1.8, location=tuple(heads[i]), segments=12, ring_count=6); o = bpy.context.object; o.data.materials.append(mj); sticks.append(o)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.show_cavity = True
if lp:
    L = np.load(lp); lab = L['label']; n = int(lab.max()) + 1; rng = np.random.default_rng(2)
    pal = L['palette'] if 'palette' in L.files else np.array([colorsys.hsv_to_rgb(h, 0.65, 0.95) for h in rng.permutation(n) / max(n, 1)]); col = np.tile([0.85, 0.85, 0.85, 1.0], (len(lab), 1))
    col[lab >= 0, :3] = pal[lab[lab >= 0]]; off = 0
    for o in target:
        k = len(o.data.vertices); at = o.data.color_attributes.new('lab', 'FLOAT_COLOR', 'POINT'); at.data.foreach_set('color', col[off:off + k].ravel()); o.data.color_attributes.active_color = at; off += k
    sh.color_type = 'VERTEX'; sh.show_xray = False
    for o in sticks: o.hide_render = True
else:
    sh.color_type = 'MATERIAL'; sh.show_xray = True; sh.xray_alpha = 0.3; clay = mat('clay', (0.8, 0.8, 0.8, 1))
    for o in target: o.data.materials.clear(); o.data.materials.append(clay)
c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0).max() * 1.05)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.ortho_scale = ext; cam.data.clip_end = 100
for vn, az in (('front', 0), ('side', 90), ('back', 180), ('side2', 270)):
    d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0)); cam.location = mathutils.Vector(c.tolist()) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    sc.render.filepath = '%s_%s.png' % (pre, vn); bpy.ops.render.render(write_still=True)
print('SS done')
