"""Blender: bare-geometry (clay) close-ups of the head from several directions, with cameras and per-point visibility,
so masks found on the renders can be voted onto the mesh's own vertices. No texture is used anywhere.
  blender -b -P geo_views.py -- <model.glb> <out dir>
Head = vertices above the neck (narrowest width slice below the head's widest slice, from the mesh's width profile).
Front = the model's own facing (-Y in Blender after glTF import, glTF's +Z forward); views fan around it.
The face frame is a tighter close-up of the front half of the head box (a viewing frame only, not a part boundary).
Writes <out>/<view>.png, cameras.json, head_points.npz (V: head vertex positions, idx: their index in the full
vertex list, vis: views x points visibility by ray cast toward each camera)."""
import bpy, sys, os, json, math, mathutils, numpy as np
from mathutils.bvhtree import BVHTree
a = sys.argv[sys.argv.index('--') + 1:]; src, od = a[0], a[1]; os.makedirs(od, exist_ok=True)
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']; dg = bpy.context.evaluated_depsgraph_get()
Vs, Fs, off = [], [], 0
for o in obs:
    m = o.evaluated_get(dg).data; v = np.array([o.matrix_world @ x.co for x in m.vertices]); Vs.append(v)
    Fs += [[off + i for i in p.vertices] for p in m.polygons]; off += len(v)
P = np.concatenate(Vs); lo, hi = P.min(0), P.max(0); H = hi[2] - lo[2]
z = P[:, 2]; edges = np.linspace(hi[2] - H / 3, hi[2], 61)
w = np.array([np.ptp(P[(z >= e0) & (z < e1), 0]) if ((z >= e0) & (z < e1)).sum() > 10 else np.inf for e0, e1 in zip(edges[:-1], edges[1:])])
wf = np.where(np.isfinite(w), w, 0); top = np.argmax(wf[len(wf) // 2:]) + len(wf) // 2
neck = edges[np.argmin(np.where(np.arange(len(w)) < top, w, np.inf))]
idx = np.flatnonzero(z >= neck); HP = P[idx]; c = (HP.min(0) + HP.max(0)) / 2; ext = float(np.ptp(HP, 0).max() * 1.05)
bvh = BVHTree.FromPolygons([tuple(x) for x in P.tolist()], Fs, epsilon=0.0)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; R = 1024; sc.render.resolution_x = sc.render.resolution_y = R
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'SINGLE'; sh.single_color = (0.8, 0.8, 0.8); sh.show_cavity = True; sh.cavity_type = 'BOTH'; sh.show_specular_highlight = False
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 40 * H; cam.data.ortho_scale = ext
# views: the front and turns of a quarter of the half-circle each way, level and from above; the face frame is the front
# half of the head box (the part of the head the face occupies), rendered as its own tighter close-up
views = [(az, el) for el in (0, 30) for az in (-45, 0, 45)]
cams, vis = {}, []
for band, cc, ee in (('head', c, ext), ('face', c + np.array([0, -np.ptp(HP[:, 1]) / 4, -np.ptp(HP[:, 2]) / 8]), ext / 2)):
    cam.data.ortho_scale = ee
    for k, (az, el) in enumerate(views):
        d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
        cam.location = mathutils.Vector(cc.tolist()) + d * 10 * H; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        n = '%s_a%+03d_e%02d' % (band, az, el); sc.render.filepath = os.path.join(od, n + '.png'); bpy.ops.render.render(write_still=True); bpy.context.view_layer.update()
        cams[n] = {'view': k, 'band': band, 'matrix_inv': [list(r) for r in cam.matrix_world.inverted()], 'scale': ee, 'res': R}
        if band == 'head':
            eps = H * 1e-4; vv = np.zeros(len(HP), bool)
            for i, p in enumerate(HP):
                hit = bvh.ray_cast(mathutils.Vector(p) + d * eps, d)
                vv[i] = hit[0] is None
            vis.append(vv); print('GV vis', n, vv.mean().round(3), flush=True)
json.dump(cams, open(os.path.join(od, 'cameras.json'), 'w')); np.savez(os.path.join(od, 'head_points.npz'), V=HP, idx=idx, vis=np.array(vis))
print('GV done: %d head points, neck z %.3f' % (len(HP), neck))
