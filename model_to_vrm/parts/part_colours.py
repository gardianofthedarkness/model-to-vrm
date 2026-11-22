"""Blender: each point's texture colour on a textured many-part GLB: blender -b -P part_colours.py -- <in.glb> <out.npz>
Writes V (world), C (sRGB 0..1, the base-colour texture at the point's UV, averaged over the point's corners) and part
(from the object name tripo_part_N, the numbering rig_input.py uses). The texture is read once, at its own size."""
import bpy, sys, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, out = a
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
MESHES = [o for o in bpy.data.objects if o.type == 'MESH']
def pid(o):
    t = o.name.rsplit('_', 1)
    return int(t[1]) if len(t) == 2 and t[1].isdigit() else MESHES.index(o)   # Tripo parts are named tripo_part_N; else the object's order
cache = {}
def tex(mat):
    for n in mat.node_tree.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            l = n.inputs['Base Color'].links
            if l and l[0].from_node.type == 'TEX_IMAGE': return l[0].from_node.image
Vs, Cs, Ps = [], [], []
for o in [o for o in bpy.data.objects if o.type == 'MESH']:
    me = o.data; img = tex(me.materials[0]) if me.materials else None
    if img is None:                                                 # a part with a flat colour, no texture: its base colour
        co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3); M = np.array(o.matrix_world)
        bc = [n for n in me.materials[0].node_tree.nodes if n.type == 'BSDF_PRINCIPLED'] if me.materials and me.materials[0].node_tree else []
        rgb = np.array(bc[0].inputs['Base Color'].default_value[:3]) if bc else np.array(me.materials[0].diffuse_color[:3]) if me.materials else np.full(3, 0.5)
        rgb = np.where(rgb <= 0.0031308, 12.92 * rgb, 1.055 * np.maximum(rgb, 0) ** (1 / 2.4) - 0.055)   # stored linear; C is sRGB
        Vs.append(co @ M[:3, :3].T + M[:3, 3]); Cs.append(np.tile(rgb, (len(co), 1))); Ps.append(np.full(len(co), pid(o)))
        print('PC %s: no texture, flat colour %s' % (o.name, np.round(rgb, 3))); continue
    if img.name not in cache:
        w, h = img.size; px = np.empty(w * h * 4, np.float32); img.pixels.foreach_get(px); cache[img.name] = (px.reshape(h, w, 4)[..., :3], w, h)
        print('PC texture %s %dx%d' % (img.name, w, h))
    P, w, h = cache[img.name]
    uv = np.empty(len(me.loops) * 2, np.float32); me.uv_layers.active.data.foreach_get('uv', uv); uv = uv.reshape(-1, 2) % 1.0
    lv = np.empty(len(me.loops), np.int32); me.loops.foreach_get('vertex_index', lv)
    col = P[np.clip((uv[:, 1] * h).astype(int), 0, h - 1), np.clip((uv[:, 0] * w).astype(int), 0, w - 1)]
    C = np.zeros((len(me.vertices), 3)); n = np.zeros(len(me.vertices)); np.add.at(C, lv, col); np.add.at(n, lv, 1); C /= np.maximum(n, 1)[:, None]
    co = np.empty(len(me.vertices) * 3, np.float32); me.vertices.foreach_get('co', co); co = co.reshape(-1, 3)
    M = np.array(o.matrix_world); co = co @ M[:3, :3].T + M[:3, 3]
    Vs.append(co); Cs.append(C); Ps.append(np.full(len(co), pid(o)))
C = np.vstack(Cs)                                                   # pixels() gives the stored sRGB values of byte images
np.savez(out, V=np.vstack(Vs), C=C, part=np.concatenate(Ps)); print('PC %d points, %d parts' % (len(C), len(Vs)))
