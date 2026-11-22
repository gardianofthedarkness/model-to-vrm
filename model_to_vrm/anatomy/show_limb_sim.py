"""Blender: a simulated limb at rest and after each move - skin see-through pink, muscles red, bones ivory, muscle strands
dark red lines; plus a cut: the tets on one side of a plane through the limb's long axis, each tet face coloured by its
layer (skin pink, sticky layer yellow, muscle red, bone ivory):
  blender -b -P show_limb_sim.py -- <sim.npz> <out prefix>      -> <prefix>_<pose>_<view>.png, <prefix>_cut_<pose>.png
  cut plane  through the tet nodes' centre, spanned by their first two principal axes (the limb's long axis and its
             widest direction), so the cut shows the deepest section"""
import bpy, sys, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; Z = np.load(a[0], allow_pickle=True); pre = a[1]
def mat(n, c):
    m = bpy.data.materials.new(n); m.diffuse_color = c; return m
MS, MM, MB, MSTR = mat('skin', (1.0, 0.72, 0.65, 0.35)), mat('mus', (0.72, 0.18, 0.16, 0.55)), mat('bone', (0.93, 0.89, 0.78, 1)), mat('str', (0.35, 0.02, 0.05, 1))
ML = [mat('L0', (1.0, 0.72, 0.65, 1)), mat('L1', (0.98, 0.86, 0.35, 1)), mat('L2', (0.75, 0.2, 0.18, 1)), mat('L3', (0.93, 0.89, 0.78, 1))]
def mesh(n, V, F, ms, fmat=None):
    me = bpy.data.meshes.new(n); me.from_pydata(np.asarray(V, float).tolist(), [], np.asarray(F).tolist()); ob = bpy.data.objects.new(n, me)
    bpy.context.scene.collection.objects.link(ob)
    for m in (ms if isinstance(ms, list) else [ms]): ob.data.materials.append(m)
    if fmat is not None: me.polygons.foreach_set('material_index', np.asarray(fmat, np.int32))
    return ob
def curves(n, polys, m, bev):
    cu = bpy.data.curves.new(n, 'CURVE'); cu.dimensions = '3D'; cu.bevel_depth = bev
    for p in polys:
        p = np.asarray(p, float)
        if len(p) < 2: continue
        sp = cu.splines.new('POLY'); sp.points.add(len(p) - 1)
        for i, q in enumerate(p): sp.points[i].co = (*q, 1)
    ob = bpy.data.objects.new(n, cu); bpy.context.scene.collection.objects.link(ob); ob.data.materials.append(m); return ob
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 800
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_xray = False
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100
allv = np.vstack([Z['%s_skin' % p] for p in ('rest', 'bend', 'twist')]); c = (allv.min(0) + allv.max(0)) / 2; ext = float(np.ptp(allv, 0).max() * 1.15)
bev = float(np.ptp(Z['rest_skin'], 0).max()) / 1500
def shoot(d, cen, scale, path):
    cam.location = mathutils.Vector(list(cen)) + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = scale
    sc.render.filepath = path; bpy.ops.render.render(write_still=True)
def clear():
    for o in [o for o in bpy.data.objects if o.type in ('MESH', 'CURVE')]: bpy.data.objects.remove(o)
def vdir(az, el):
    return mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
for pose in ('rest', 'bend', 'twist'):
    clear(); mesh('skin', Z[pose + '_skin'], Z['skin_F'], MS)
    for i, (V, F) in enumerate(zip(Z[pose + '_muscles'], Z['mus_F'])): mesh('m%d' % i, V, F, MM)
    for i, (V, F) in enumerate(zip(Z[pose + '_bones'], Z['bone_F'])): mesh('b%d' % i, V, F, MB)
    if pose + '_strands' in Z: curves('strands', list(Z[pose + '_strands']), MSTR, bev)
    for vn, az, el in (('a', 20, 15), ('b', 110, 15)): shoot(vdir(az, el), c, ext, '%s_%s_%s.png' % (pre, pose, vn))
# the cut: tets whose centre lies behind the plane; faces of that half that are not shared inside it
X0, T, layer = Z['X'], Z['T'], Z['layer']; m0 = X0.mean(0); w, e = np.linalg.eigh(np.cov((X0 - m0).T)); nrm = e[:, 0]   # plane normal = least spread
keep = ((X0[T].mean(1) - m0) @ nrm) < 0; Tk, Lk = T[keep], layer[keep]
Fc = np.vstack([Tk[:, [0, 2, 1]], Tk[:, [0, 1, 3]], Tk[:, [0, 3, 2]], Tk[:, [1, 2, 3]]]); Lf = np.tile(Lk, 4)
key = np.sort(Fc, 1); _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True); bnd = cnt[inv.ravel()] == 1
for pose in ('rest', 'bend'):
    clear(); q = Z[pose + '_q']; mesh('cut', q, Fc[bnd], ML, Lf[bnd])
    d = mathutils.Vector(nrm.tolist()); d = d - 0.25 * mathutils.Vector(e[:, 2].tolist()); d.normalize()   # look at the cut face, a little from the side
    shoot(d, (q.min(0) + q.max(0)) / 2, ext, '%s_cut_%s.png' % (pre, pose))
print('SL done')
