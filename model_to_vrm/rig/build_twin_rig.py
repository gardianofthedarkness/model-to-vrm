"""Blender: the rigged, coloured twin with movable irises: blender -b -P build_twin_rig.py -- <iris_prep.npz> <out.glb> <test_prefix>
  armature  the twin's bones (names, parents, T-pose heads and tails) plus RightEye / LeftEye under Head, each at its
            virtual eyeball centre pointing forward (-y); the iris meshes ride them (weight 1)
  body      the corrected twin, vertex colours, four strongest skin weights per point
  test      renders the face with the eyes turned to each limit (left, right, up, down) and ahead, from the front"""
import bpy, sys, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, out, pre = a
D = np.load(src); names = [str(n) for n in D['names']]; par = D['parent']; H, T = D['head'], D['tail']
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
def mesh(name, V, F, C):
    md = bpy.data.meshes.new(name); md.from_pydata(V.tolist(), [], F.tolist()); md.update(); ob = bpy.data.objects.new(name, md); bpy.context.scene.collection.objects.link(ob)
    ca = md.color_attributes.new('Col', 'FLOAT_COLOR', 'POINT'); cc = np.c_[np.clip(C, 0, 1), np.ones(len(C))].astype(np.float32).ravel(); ca.data.foreach_set('color', cc)
    mat = bpy.data.materials.new(name); mat.use_nodes = True; nt = mat.node_tree; vc = nt.nodes.new('ShaderNodeVertexColor'); vc.layer_name = 'Col'
    nt.links.new(vc.outputs['Color'], nt.nodes['Principled BSDF'].inputs['Base Color']); md.materials.append(mat); return ob
ad = bpy.data.armatures.new('rig'); arm = bpy.data.objects.new('rig', ad); bpy.context.scene.collection.objects.link(arm)
bpy.context.view_layer.objects.active = arm; bpy.ops.object.mode_set(mode='EDIT'); eb = {}
for j, n in enumerate(names):
    b = ad.edit_bones.new(n); b.head = tuple(H[j]); b.tail = tuple(T[j]) if np.linalg.norm(T[j] - H[j]) > 1e-6 else tuple(H[j] + [0, 0, 0.01]); eb[j] = b
for j in range(len(names)):
    if par[j] >= 0: eb[j].parent = eb[par[j]]
head = [b for n, b in zip(names, eb.values()) if n.endswith('Head')][0]
eyes = {23: 'RightEye', 25: 'LeftEye'}                                                      # part 23 is at -x: the character's right
for p, n in eyes.items():
    c = D['centre_%d' % p]; R = float(D['radius_%d' % p]); b = ad.edit_bones.new(n); b.head = tuple(c); b.tail = tuple(c + [0, -R, 0]); b.parent = head
bpy.ops.object.mode_set(mode='OBJECT')
body = mesh('rikka', D['V'], D['F'], D['C']); Wt = D['W']; G = [body.vertex_groups.new(name=n) for n in names]
top = np.argsort(-Wt, axis=1)[:, :4]; tw = np.take_along_axis(Wt, top, 1); tw /= np.maximum(tw.sum(1, keepdims=True), 1e-9)
for j in range(len(names)):                                                               # one call per bone and weight level
    for k in range(4):
        m = (top[:, k] == j) & (tw[:, k] > 0)
        if m.any():
            idx = np.flatnonzero(m); w = tw[m, k]
            for val in np.unique(np.round(w, 3)): G[j].add(idx[np.round(w, 3) == val].tolist(), float(val), 'REPLACE')
for ob in [body]: mo = ob.modifiers.new('arm', 'ARMATURE'); mo.object = arm; ob.parent = arm
irises = []
for p, n in eyes.items():
    ob = mesh('iris_' + n, D['iris_V_%d' % p], D['iris_F_%d' % p], D['iris_C_%d' % p]); g = ob.vertex_groups.new(name=n); g.add(list(range(len(ob.data.vertices))), 1.0, 'REPLACE')
    ob.shape_key_add(name='Basis')                                                            # VRM 'expression' look-at: lookLeft/Right/Up/Down
    for nm in ('left', 'right', 'up', 'down'):
        key = 'look_%s_%d' % (nm, p)
        if key in D.files: sk = ob.shape_key_add(name='look' + nm.capitalize()); sk.data.foreach_set('co', D[key].ravel())
    mo = ob.modifiers.new('arm', 'ARMATURE'); mo.object = arm; ob.parent = arm; irises.append(ob)
print('BT bones %d (+2 eyes), body points %d, iris points %s' % (len(names), len(D['V']), [len(o.data.vertices) for o in irises]), flush=True)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.export_scene.gltf(filepath=out, export_format='GLB'); print('BT exported', out, flush=True)
# test renders
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x, sc.render.resolution_y = 900, 450
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sh = sc.display.shading; sh.light = 'FLAT'; sh.color_type = 'VERTEX'; sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.ortho_scale = 0.11; cam.data.clip_end = 50
c = mathutils.Vector((0.0, 0, 0.879)); d = mathutils.Vector((0, -1, 0)); cam.location = c + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
bpy.context.view_layer.objects.active = arm; bpy.ops.object.mode_set(mode='POSE')
lim = {p: dict(zip(('left', 'right', 'up', 'down'), D['limits_%d' % p])) for p in eyes}
for tag in ('ahead', 'left', 'right', 'up', 'down'):
    for p, n in eyes.items():
        pb = arm.pose.bones[n]; pb.rotation_mode = 'XYZ'; ang = math.radians(lim[p][tag]) if tag != 'ahead' else 0.0
        # bone frame: y along the bone (forward, -y world); turning about world z moves the gaze sideways, about world x up/down
        bm = arm.data.bones[n].matrix_local.to_3x3()
        ax = {'ahead': (0, 0, 1), 'left': (0, 0, 1), 'right': (0, 0, -1), 'up': (-1, 0, 0), 'down': (1, 0, 0)}[tag]
        loc_ax = bm.inverted() @ mathutils.Vector(ax); pb.rotation_mode = 'QUATERNION'; pb.rotation_quaternion = mathutils.Quaternion(loc_ax, ang)
    for p_, n_ in eyes.items(): arm.pose.bones[n_].rotation_quaternion = (1, 0, 0, 0)            # test the look shapes (the slide), not the bone turn
    for ob in irises:
        for kb in ob.data.shape_keys.key_blocks[1:]: kb.value = 1.0 if kb.name == 'look' + tag.capitalize() else 0.0
    bpy.context.view_layer.update(); sc.render.filepath = '%s_%s.png' % (pre, tag); bpy.ops.render.render(write_still=True)
print('BT tests rendered')
