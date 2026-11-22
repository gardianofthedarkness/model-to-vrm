"""Blender: textured renders of the original from the 16 view directions of views_vis.py, each as the whole body and as
close-ups (head, torso, legs), with every camera saved for projecting points:
  blender -b -P views_crops.py -- <textured.glb> <rig points npz> <outdir>
Crops are centred on the body's own bands (head: top 22% of the height... taken from the points of the Tripo parts the part
sheet names hair/face; torso and legs likewise), each framed to its band's extent."""
import bpy, sys, os, json, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, pts, od = a[:3]; os.makedirs(od, exist_ok=True)
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
D = np.load(pts); P = D['V']; part = D['part']; lo, hi = P.min(0), P.max(0); c0 = (lo + hi) / 2; H = hi[2] - lo[2]
S = json.load(open(os.path.join(os.path.dirname(od.rstrip('/\\')), 'parts_ab', 'part_sheet.json')))
group = lambda names: np.isin(part, [int(p) for p, s in S.items() if s.get('name') in names])
bands = {'body': np.ones(len(P), bool), 'head': group({'hair', 'face and head skin'}) & (P[:, 2] > np.percentile(P[group({'face and head skin'}), 2], 1)),
         'torso': group({'cardigan', 'cardigan sleeve', 'collar', 'bow', 'trim', 'hand', 'scrunchie'}), 'legs': group({'leg skin', 'shoe'})}
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; R = 1024; sc.render.resolution_x = sc.render.resolution_y = R
sh = sc.display.shading; sh.light = 'FLAT'; sh.color_type = 'TEXTURE'; sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 40 * H
views = [(az, el) for el in (0, 35) for az in np.arange(8) * 45]; cams = {}
for vi, (az, el) in enumerate(views):
    d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
    for bname, m in bands.items():
        Q = P[m]; c = (Q.min(0) + Q.max(0)) / 2; ext = np.ptp(Q, 0).max() * 1.15
        cam.location = mathutils.Vector(c.tolist()) + d * 10 * H; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = float(ext)
        f = os.path.join(od, 'v%02d_%s.png' % (vi, bname)); sc.render.filepath = f; bpy.ops.render.render(write_still=True); bpy.context.view_layer.update()
        cams['v%02d_%s' % (vi, bname)] = {'view': vi, 'band': bname, 'matrix_inv': [list(r) for r in cam.matrix_world.inverted()], 'scale': float(ext), 'res': R}
json.dump(cams, open(os.path.join(od, 'cameras.json'), 'w')); print('VC done %d renders' % len(cams))
