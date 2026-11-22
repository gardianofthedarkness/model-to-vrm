"""Blender: export the Z-Anatomy atlas (CC BY-SA 4.0, BodyParts3D-based) meshes we need as plain arrays, and render it:
  blender -b -Y <Startup.blend> -P za_export.py -- <out dir> [render]     (-Y: the file's own scripts never run)
  systems  1 skeletal -> 'bone', 4 muscular -> 'muscle', 9 regions of the body -> 'skin' (hair meshes left out),
           2 muscular insertions -> 'insertion' (where each muscle attaches: the ends of its strands)
  arrays   atlas.npz: V (world, metres, Z up), F (triangles), obj (object id per vertex), names, system per object
  renders  only with 'render': atlas_<view>.png (whole atlas; needs ~19 GB RAM, slow)"""
import bpy, sys, os, math, mathutils, numpy as np
od = sys.argv[sys.argv.index('--') + 1]; os.makedirs(od, exist_ok=True)
SYS = {'1': 'bone', '4': 'muscle', '9': 'skin', '2': 'insertion'}
def colls(c, path, out):
    for ch in c.children:
        p = path + [ch.name]; out.append((p, ch)); colls(ch, p, out)
    return out
own = {}
for p, c in colls(bpy.context.scene.collection, [], []):
    k = p[0].split(':')[0].strip()
    if k in SYS:
        for o in c.objects: own.setdefault(o.name, SYS[k])
dg = bpy.context.evaluated_depsgraph_get(); V, F, OB, names, sysn = [], [], [], [], []; off = 0
for o in bpy.data.objects:
    s = own.get(o.name)
    if s is None or o.type != 'MESH' or len(o.data.vertices) < 4 or (s == 'skin' and 'hair' in o.name.lower()): continue
    me = o.evaluated_get(dg).to_mesh(); me.calc_loop_triangles()
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    M = np.array(o.matrix_world); co = co @ M[:3, :3].T + M[:3, 3]
    t = np.empty(len(me.loop_triangles) * 3, np.int64); me.loop_triangles.foreach_get('vertices', t)
    V.append(co); F.append(t.reshape(-1, 3) + off); OB.append(np.full(len(co), len(names))); names.append(o.name); sysn.append(s); off += len(co)
    o.evaluated_get(dg).to_mesh_clear()
V, F, OB = np.vstack(V), np.vstack(F), np.concatenate(OB)
np.savez(os.path.join(od, 'atlas.npz'), V=V, F=F, obj=OB, names=np.array(names), system=np.array(sysn))
print('ZE', {s: sysn.count(s) for s in set(sysn)}, 'objects,', len(V), 'vertices, height %.3f m' % np.ptp(V[:, 2]))
if 'render' not in sys.argv[sys.argv.index('--') + 1:]: print('ZE done'); sys.exit()
# render: keep only exported meshes, colour by system
for o in bpy.data.objects:
    s = own.get(o.name); o.hide_render = not (o.type == 'MESH' and s in ('bone', 'muscle', 'skin') and not (s == 'skin' and 'hair' in o.name.lower()))
    if not o.hide_render:
        o.color = {'bone': (0.93, 0.88, 0.76, 1), 'muscle': (0.75, 0.22, 0.2, 1), 'skin': (0.95, 0.75, 0.68, 0.25)}[s]
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x, sc.render.resolution_y = 768, 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'OBJECT'; sh.show_cavity = True; sh.show_xray = False
c = (V.min(0) + V.max(0)) / 2; ext = float(np.ptp(V, 0).max() * 1.05)
cam = bpy.data.objects.new('cam', bpy.data.cameras.new('cam')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.ortho_scale = ext; cam.data.clip_end = 100
for vn, az in (('front', 0), ('side', 90)):
    d = mathutils.Vector((math.sin(math.radians(az)), -math.cos(math.radians(az)), 0)); cam.location = mathutils.Vector(c.tolist()) + d * 10
    cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); sc.render.filepath = os.path.join(od, 'atlas_%s.png' % vn); bpy.ops.render.render(write_still=True)
print('ZE done')
