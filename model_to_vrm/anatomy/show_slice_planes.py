"""Blender: the slicing topology - each critical slice drawn as a see-through square in its own plane (perpendicular to
the segment's straight axis J0 -> J1), the axis (blue), and the model's stick skeleton (bones dark, joints as balls),
inside the see-through model:
  blender -b -P show_slice_planes.py -- <model.glb> <skeleton.npz> <out prefix> <segment_fit.npz> [more fit.npz ...]
  square size = twice the slice's fitted outline radius; colour runs along the slices (first warm, last cool)"""
import bpy, sys, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, skp, pre, fits = a[0], a[1], a[2], a[3:]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); model = [o for o in bpy.data.objects if o.type == 'MESH']
clay = bpy.data.materials.new('clay'); clay.diffuse_color = (0.8, 0.8, 0.8, 0.18)
for o in model: o.data.materials.clear(); o.data.materials.append(clay)
def mat(n, c):
    m = bpy.data.materials.new(n); m.diffuse_color = c; return m
def curve(name, polys, m, bev):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = bev
    for p in polys:
        sp = cu.splines.new('POLY'); sp.points.add(len(p) - 1)
        for i, q in enumerate(p): sp.points[i].co = (*q, 1)
    ob = bpy.data.objects.new(name, cu); bpy.context.scene.collection.objects.link(ob); ob.data.materials.append(m); return ob
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in model]); Hh = float(np.ptp(P[:, 2]))
K = np.load(skp); H, par = K['heads'], K['parents']
curve('bones', [np.array([H[p], H[i]]) for i, p in enumerate(par) if p >= 0], mat('bone', (0.15, 0.15, 0.15, 1)), Hh / 700)
mj = mat('joint', (0.1, 0.1, 0.1, 1))
for i, h in enumerate(H):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=Hh / 180, location=tuple(h)); bpy.context.object.data.materials.append(mj)
boxes = []
for fp in fits:
    Z = np.load(fp, allow_pickle=True); J0, J1, ax, ex, ey, L, t = Z['J0'], Z['J1'], Z['ax'], Z['ex'], Z['ey'], float(Z['L']), Z['t']
    lines, kinds, levs = list(Z['lines']), Z['kinds'], Z['levels']
    curve('axis', [np.array([J0, J1])], mat('axis', (0.1, 0.3, 1, 1)), Hh / 450)
    for k, tk in enumerate(t):
        sk = [np.asarray(l, float) for l, kd, lv in zip(lines, kinds, levs) if kd == 0 and lv == k]
        c0 = J0 + ax * tk * L; r = max(float(np.linalg.norm(s_ - c0, axis=1).max()) for s_ in sk) if sk else L / 4
        u = k / max(1, len(t) - 1); col = (0.95 - 0.7 * u, 0.35 + 0.2 * u, 0.2 + 0.75 * u, 0.28)
        V = [c0 + r * (sx * ex + sy * ey) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        me = bpy.data.meshes.new('pl'); me.from_pydata([v.tolist() for v in V], [], [[0, 1, 2, 3]]); ob = bpy.data.objects.new('pl%d' % k, me)
        bpy.context.scene.collection.objects.link(ob); ob.data.materials.append(mat('pl%d' % k, col))
        curve('edge%d' % k, [np.array(V + V[:1])], mat('e%d' % k, col[:3] + (1,)), Hh / 1500)
        if sk: curve('sk%d' % k, sk, mat('s%d' % k, (0.85, 0.1, 0.1, 1)), Hh / 1200)
    allp = np.vstack([np.asarray(l, float) for l in lines]); boxes.append((allp.min(0), allp.max(0)))
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_xray = False
sc.render.film_transparent = False
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100
lo = np.min([b[0] for b in boxes], 0); hi = np.max([b[1] for b in boxes], 0); c = (lo + hi) / 2; ext = float(np.ptp(np.c_[lo, hi], 1).max() * 1.5)
for vn, az, el in (('front', 0, 0), ('side', 90, 0), ('back', 180, 0), ('three_quarter', 40, 20)):
    d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
    cam.location = mathutils.Vector(c.tolist()) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = ext
    sc.render.filepath = '%s_%s.png' % (pre, vn); bpy.ops.render.render(write_still=True)
print('SP done')
