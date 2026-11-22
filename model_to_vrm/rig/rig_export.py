"""Blender: write a GLB's mesh (and its rig, if any) to an .npz for the fitting step.
blender -b -P rig_export.py -- <model.glb> <out.npz>
  V (n,3) points in world space (z up), F (m,3) triangles (all meshes joined), part (n,) the 'part' point attribute if present
  rigged: bone names, parent index, head/tail (world, rest), W (n, bones) skin weights (rows sum to 1)"""
import bpy, sys, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, out = a[0], a[1]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
M = [o for o in bpy.data.objects if o.type == 'MESH']; arm = next((o for o in bpy.data.objects if o.type == 'ARMATURE'), None)
if arm: M = [o for o in M if o.find_armature() == arm]   # only the skinned body (riggers may leave stray meshes, e.g. an icosphere)
names = [b.name for b in arm.data.bones] if arm else []; bi = {n: i for i, n in enumerate(names)}
Vs, Fs, Ps, Ws = [], [], [], []; off = 0
for o in M:
    me = o.data; me.calc_loop_triangles(); mw = o.matrix_world
    V = np.array([tuple(mw @ v.co) for v in me.vertices]); F = np.array([tuple(t.vertices) for t in me.loop_triangles]) + off
    pa = me.attributes.get('_PART') or me.attributes.get('_part') or me.attributes.get('part'); P = np.rint([d.value for d in pa.data]).astype(int) if pa else np.full(len(V), -1)
    if arm:
        gn = {g.index: g.name for g in o.vertex_groups}; W = np.zeros((len(V), len(names)))
        for v in me.vertices:
            for g in v.groups:
                n = gn.get(g.group)
                if n in bi: W[v.index, bi[n]] += g.weight
        Ws.append(W)
    Vs.append(V); Fs.append(F); Ps.append(P); off += len(V)
d = dict(V=np.vstack(Vs), F=np.vstack(Fs), part=np.concatenate(Ps))
if arm:
    W = np.vstack(Ws); s = W.sum(1, keepdims=True); d['W'] = W / np.where(s > 0, s, 1)
    B = arm.data.bones; aw = arm.matrix_world
    d['names'] = np.array(names); d['parent'] = np.array([bi[b.parent.name] if b.parent else -1 for b in B])
    d['head'] = np.array([tuple(aw @ b.head_local) for b in B]); d['tail'] = np.array([tuple(aw @ b.tail_local) for b in B])
np.savez(out, **d); print('RE points %d triangles %d parts %d bones %d' % (len(d['V']), len(d['F']), len(set(d['part'].tolist())), len(names)))
