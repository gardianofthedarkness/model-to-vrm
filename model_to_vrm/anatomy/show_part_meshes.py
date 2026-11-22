"""Blender: meshes from .npz files inside one see-through body region (close-up; show_meshes.py is the whole body):
  blender -b -P show_part_meshes.py -- <model.glb> <parts_zoom dir> <region> <out prefix> <file.npz>:<V key>:<F key>:<r,g,b,a> [...]
The region alone is shown (boxes.json, <region>_vids.npy from part_boxes.py), see-through; views front, side, three-quarter, top.
Optional points: <file.npz>:<X key>::<r,g,b,a> (empty F key) draws small spheres at the points (nan rows skipped)."""
import bpy, bmesh, sys, os, json, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, D, reg, pre, specs = a[0], a[1], a[2], a[3], a[4:]
box = json.load(open(os.path.join(D, 'boxes.json')))[reg]; keep = np.load(os.path.join(D, reg + '_vids.npy'))
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); model = [o for o in bpy.context.scene.objects if o.type == 'MESH']; off = 0
clay = bpy.data.materials.new('clay'); clay.diffuse_color = (0.8, 0.8, 0.8, 1)
for o in model:
    n = len(o.data.vertices); kk = keep[(keep >= off) & (keep < off + n)] - off; off += n; m = np.zeros(n, bool); m[kk] = True
    bm = bmesh.new(); bm.from_mesh(o.data); bm.verts.ensure_lookup_table(); bmesh.ops.delete(bm, geom=[v for v in bm.verts if not m[v.index]], context='VERTS'); bm.to_mesh(o.data); bm.free()
    o.data.materials.clear(); o.data.materials.append(clay)
ext = float(2 * np.linalg.norm(box['half']) * 1.05)
for i, sp in enumerate(specs):
    f, kv, kf, col = sp.rsplit(':', 3); Z = np.load(f, allow_pickle=True); V = np.asarray(Z[kv], float); mt = bpy.data.materials.new('c%d' % i); mt.diffuse_color = tuple(float(x) for x in col.split(','))
    if kf:
        F = np.asarray(Z[kf], int); u, inv = np.unique(F, return_inverse=True); me = bpy.data.meshes.new('m%d' % i); me.from_pydata(V[u].tolist(), [], inv.reshape(-1, 3).tolist())
        ob = bpy.data.objects.new('m%d' % i, me); bpy.context.scene.collection.objects.link(ob); ob.data.materials.append(mt)
    else:
        for p in V[np.isfinite(V).all(1)]:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=ext / 250, location=p.tolist(), segments=8, ring_count=6); bpy.context.object.data.materials.append(mt)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
sh = sc.display.shading; sh.light = 'STUDIO'; sh.color_type = 'MATERIAL'; sh.show_xray = True; sh.xray_alpha = 0.3
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100; cam.data.ortho_scale = ext
c = mathutils.Vector(box['centre'])
for vn, az, el in (('front', 0, 0), ('side', 90, 0), ('three_quarter', 40, 15), ('top', 0, 89.9)):
    d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
    cam.location = c + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); sc.render.filepath = '%s_%s.png' % (pre, vn); bpy.ops.render.render(write_still=True)
print('SPM done')
