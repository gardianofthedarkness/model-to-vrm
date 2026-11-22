"""Blender: each flesh island alone (its faces only, texture colour), from front, side, back, other side:
  blender -b -P show_islands.py -- <real dir> [how many, default 4]     -> <real dir>/flesh/island<k>_<view>.png
Reads parts/orig_colours.npz (V, C), parts/orig_normals.npz (F), flesh/islands.npz."""
import bpy, sys, os, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; R = a[0]; K = int(a[1]) if len(a) > 1 else 4
O = np.load(os.path.join(R, 'parts', 'orig_colours.npz')); V, C = O['V'], O['C']; F = np.load(os.path.join(R, 'parts', 'orig_normals.npz'))['F']
I = np.load(os.path.join(R, 'flesh', 'islands.npz'))['island']; Lc = np.where(C <= 0.04045, C / 12.92, ((C + 0.055) / 1.055) ** 2.4)
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 768
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'VERTEX'; sh.show_cavity = True; sh.show_backface_culling = False
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100
for k in range(K):
    ff = F[(I[F] == k).all(1)]; u, inv = np.unique(ff, return_inverse=True); me = bpy.data.meshes.new('i%d' % k)
    me.from_pydata(V[u].tolist(), [], inv.reshape(-1, 3).tolist()); ob = bpy.data.objects.new('i%d' % k, me); sc.collection.objects.link(ob)
    at = me.color_attributes.new('c', 'FLOAT_COLOR', 'POINT'); at.data.foreach_set('color', np.c_[Lc[u], np.ones(len(u))].ravel()); me.color_attributes.active_color = at
    P = V[u]; c = (P.min(0) + P.max(0)) / 2; ext = float(np.ptp(P, 0).max() * 1.1); cam.data.ortho_scale = ext
    for vn, az in (('front', 0), ('side', 90), ('back', 180), ('side2', 270)):
        d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0)); cam.location = mathutils.Vector(c.tolist()) + d * 10
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); sc.render.filepath = os.path.join(R, 'flesh', 'island%d_%s.png' % (k, vn)); bpy.ops.render.render(write_still=True)
    bpy.data.objects.remove(ob); print('SI island', k, len(u), 'vertices', flush=True)
print('SI done')
