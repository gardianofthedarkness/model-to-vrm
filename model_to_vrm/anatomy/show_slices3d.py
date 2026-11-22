"""Blender: fitted cross-sections drawn in 3D inside the see-through model (skin outline red, muscles purple, bones tan,
the skeleton line of each segment blue), whole body front/side and a close-up per segment:
  blender -b -P show_slices3d.py -- <model.glb> <out prefix> <segment_fit.npz> [more fit.npz ...]"""
import bpy, sys, os, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, pre, fits = a[0], a[1], a[2:]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); model = [o for o in bpy.data.objects if o.type == 'MESH']
clay = bpy.data.materials.new('clay'); clay.diffuse_color = (0.82, 0.82, 0.82, 1)
for o in model: o.data.materials.clear(); o.data.materials.append(clay)
def curve(name, polys, col, bev):
    cu = bpy.data.curves.new(name, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = bev
    for p in polys:
        sp = cu.splines.new('POLY'); sp.points.add(len(p) - 1)
        for i, q in enumerate(p): sp.points[i].co = (*q, 1)
    ob = bpy.data.objects.new(name, cu); bpy.context.scene.collection.objects.link(ob)
    m = bpy.data.materials.new(name); m.diffuse_color = col; ob.data.materials.append(m); return ob
P = np.concatenate([np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in model]); H = float(np.ptp(P[:, 2]))
boxes = []
for fp in fits:
    Z = np.load(fp, allow_pickle=True); ok = np.array([np.isfinite(np.asarray(l, float)).all() for l in Z['lines']])   # skip broken (non-finite) lines
    lines, kinds = [l for l, o in zip(Z['lines'], ok) if o], Z['kinds'][ok]; nm = fp.replace('\\', '/').split('/')[-1][:-8]
    KINDS = [int(x) for x in os.environ.get('SLICE_KINDS', '0,1,2').split(',')]   # dense whole-body views: skin and bone only fit in memory
    for kind, col, bev in [k for k in ((0, (0.85, 0.1, 0.1, 1), H / 900), (1, (0.55, 0.2, 0.65, 1), H / 2500), (2, (0.85, 0.7, 0.45, 1), H / 1500)) if k[0] in KINDS]:
        ps = [np.asarray(l, float) for l, k in zip(lines, kinds) if k == kind and len(l) > 1]
        if ps: curve('%s_%d' % (nm, kind), ps, col, bev)
    curve(nm + '_axis', [np.array([Z['J0'], Z['J1']])], (0.1, 0.3, 1, 1), H / 600)
    allp = np.vstack([np.asarray(l, float) for l in lines]); boxes.append((nm, np.percentile(allp, 2, 0), np.percentile(allp, 98, 0)))   # stray lines do not set the frame
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_xray = True; sh.xray_alpha = 0.25
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100
def shoot(c, scale, az, el, path):
    d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
    cam.location = mathutils.Vector(list(c)) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = scale
    sc.render.filepath = path; bpy.ops.render.render(write_still=True)
c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0).max() * 1.05)
for vn, az in (('front', 0), ('side', 90), ('back', 180)): shoot(c, ext, az, 0, '%s_body_%s.png' % (pre, vn))
for nm, lo, hi in boxes:
    for vn, az, el in (('a', 30, 20), ('b', 120, 20)): shoot((lo + hi) / 2, float(np.ptp(np.c_[lo, hi], 1).max() * 1.4), az, el, '%s_%s_%s.png' % (pre, nm, vn))
print('S3 done')
