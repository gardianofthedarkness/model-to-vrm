"""Blender: show a fitted body model inside the target mesh:
  blender -b -P show_fit.py -- <model.glb> <fit.npz (V, F, heads, parents)> <out prefix>
Renders <prefix>_<front|side>_<overlay|body|sticks>.png: overlay = target as see-through clay with the fitted body
(orange) inside; body = the fitted body alone; sticks = bones as sticks (parent head to child head) with joint balls
over the body in see-through, the stick skeleton the rig is built from. Keypoints from the fit (kp, X) are drawn
as small spheres: green = triangulated targets, red = the body model's own."""
import bpy, bmesh, sys, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, fp, pre = a[:3]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); target = [o for o in bpy.context.scene.objects if o.type == 'MESH']
Z = np.load(fp); V, F = Z['V'], Z['F']
def mat(name, rgba):
    m = bpy.data.materials.new(name); m.diffuse_color = rgba; return m
me = bpy.data.meshes.new('fit'); me.from_pydata(V.tolist(), [], F.tolist()); body = bpy.data.objects.new('fit', me); bpy.context.collection.objects.link(body)
body.data.materials.append(mat('body', (1.0, 0.55, 0.15, 1)))
for p in body.data.polygons: p.use_smooth = True
sticks = []
def seg(p, q, r, m):
    p, q = mathutils.Vector(p), mathutils.Vector(q); d = q - p
    if d.length < 1e-6: return
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=d.length, location=(p + q) / 2, vertices=8)
    o = bpy.context.object; o.rotation_euler = d.to_track_quat('Z', 'Y').to_euler(); o.data.materials.append(m); sticks.append(o)
def ball(p, r, m):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=tuple(p), segments=12, ring_count=6); o = bpy.context.object; o.data.materials.append(m); sticks.append(o)
H = np.ptp(V[:, 2]); r = H / 250; ms, mj = mat('stick', (0.1, 0.3, 1.0, 1)), mat('joint', (1, 1, 1, 1))
heads, par = Z['heads'], Z['parents']
for i, pi in enumerate(par):
    if pi >= 0: seg(heads[pi], heads[i], r, ms)
    ball(heads[i], r * 1.8, mj)
if 'kp' in Z.files:
    mg, mr = mat('kpX', (0, 0.8, 0, 1)), mat('kpM', (0.9, 0, 0, 1))
    for p in Z['X']: ball(p, r * 3, mg)
    for p in Z['kp']: ball(p, r * 2.5, mr)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_cavity = True
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in target]); c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0).max() * 1.05)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.ortho_scale = ext; cam.data.clip_end = 100
clay = mat('clay', (0.8, 0.8, 0.8, 1))
for o in target: o.data.materials.clear(); o.data.materials.append(clay)
for vn, az in (('front', 0), ('side', 90)):
    d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0)); cam.location = mathutils.Vector(c.tolist()) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
    for mode in ('overlay', 'body', 'sticks'):
        for o in target: o.hide_render = mode != 'overlay'
        body.hide_render = False
        for o in sticks: o.hide_render = mode != 'sticks'
        sh.show_xray = mode in ('overlay', 'sticks'); sh.xray_alpha = 0.35 if mode == 'overlay' else 0.25
        sc.render.filepath = '%s_%s_%s.png' % (pre, vn, mode); bpy.ops.render.render(write_still=True)
print('SF done')
