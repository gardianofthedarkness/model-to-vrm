"""Blender: per-point normals and triangles of a many-part GLB, in the same point order as part_colours.py (no welding):
  blender -b -P part_normals.py -- <in.glb> <out.npz>   -> N (world normals), F (triangles, global point ids), part"""
import bpy, sys, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, out = a
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
MESHES = [o for o in bpy.data.objects if o.type == 'MESH']
def pid(o):
    t = o.name.rsplit('_', 1)
    return int(t[1]) if len(t) == 2 and t[1].isdigit() else MESHES.index(o)   # Tripo parts are named tripo_part_N; else the object's order
Ns, Fs, Ps, off = [], [], [], 0
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    me = o.data; me.calc_loop_triangles(); n = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('normal', n); n = n.reshape(-1, 3)
    R = np.array(o.matrix_world)[:3, :3]; n = n @ np.linalg.inv(R); n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    t = np.empty(len(me.loop_triangles) * 3, np.int32); me.loop_triangles.foreach_get('vertices', t)
    Ns.append(n); Fs.append(t.reshape(-1, 3) + off); Ps.append(np.full(len(n), pid(o))); off += len(n)
np.savez(out, N=np.vstack(Ns), F=np.vstack(Fs), part=np.concatenate(Ps)); print('PN %d points, %d triangles' % (off, sum(len(f) for f in Fs)))
