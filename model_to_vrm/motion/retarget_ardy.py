"""Blender: an ARDY (SOMA-77) clip onto the rigged original:
  blender -b -P retarget_ardy.py -- <rigged.glb> <fit.npz> <clip.npz> <out.glb> <sheet prefix>
Both skeletons have a T-pose: ARDY's rotations are measured from its own neutral T-pose (identity = T-pose), and the rig is
the T-pose twin's skeleton bent into the setting pose by the fit's per-bone world turns Rg. So a bone's world turn in the
clip, relative to the twin's T-pose, is the source bone's world turn (moved into Blender's axes); relative to the rig's own
rest (the setting pose) it is that turn times Rg^-1. Twist comes along with the turn. Bone lengths stay the rig's own.
The hips travel by the clip's root path, scaled by the ratio of hip heights (the twin's over ARDY's, both above the soles)."""
import bpy, sys, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, fitp, clipp, out, pre = a
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]; mesh = [o for o in bpy.data.objects if o.type == 'MESH' and o.find_armature() == arm][0]
F = np.load(fitp); C = np.load(clipp); names = [str(n) for n in F['names']]; Rg = F['Rg']
G, root, fps = C['global_rot_mats'], C['root_positions'], float(C['fps']); T = len(G)
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)          # ARDY (y up, facing +z) -> Blender (z up, facing -y); +x is the left in both
soma = open(r'E:\figures\ardy\soma77_names.txt').read().split('\n'); si = {n: i for i, n in enumerate(soma)}
mp = {'Hips': 'Hips', 'Spine': 'Spine1', 'Spine1': 'Spine2', 'Spine2': 'Chest', 'Neck': 'Neck1', 'Head': 'Head'}
for s in ('Left', 'Right'):
    mp.update({s + 'Shoulder': s + 'Shoulder', s + 'Arm': s + 'Arm', s + 'ForeArm': s + 'ForeArm',
               s + 'UpLeg': s + 'Leg', s + 'Leg': s + 'Shin', s + 'Foot': s + 'Foot', s + 'ToeBase': s + 'ToeBase'})
    # hands and fingers are not carried over: the fit's bends for them are guesses (a hand in a pocket, one around a
    # scrunchie), and undoing a wrong bend throws the hand about; they keep the original's own hand pose, riding the forearm
# world turn per rig bone per frame, relative to the rig's rest; unmapped bones ride with their parent
par = F['parent']; D = np.zeros((T, len(names), 3, 3))
for j, n in enumerate(names):
    k = n.split(':')[-1]
    if k in mp: Q = np.einsum('ij,tjk,lk->til', M, G[:, si[mp[k]]], M); D[:, j] = Q @ Rg[j].T
    else: D[:, j] = D[:, par[j]] if par[j] >= 0 else np.eye(3)
# hips path: ARDY's root offsets from its first frame, scaled by hip height above the soles
s_hip = (F['head0'][0, 2] - F['V0'][:, 2].min()) / (root[0, 1] - 0)  # ARDY's ground is y = 0
dz = np.einsum('ij,tj->ti', M, root - root[0]) * s_hip
print('RT %d frames at %g fps; hip scale %.3f' % (T, fps, s_hip))
bpy.context.view_layer.objects.active = arm; bpy.ops.object.mode_set(mode='POSE')
B = arm.data.bones; P = arm.pose.bones; R = {n: B[n].matrix_local.to_3x3() for n in names}
sc = bpy.context.scene; sc.render.fps = int(round(fps)); sc.frame_start, sc.frame_end = 0, T - 1
Aw = arm.matrix_world.to_3x3()
for pb in P: pb.rotation_mode = 'QUATERNION'
for t in range(T):
    for j, n in enumerate(names):
        Dj = mathutils.Matrix(D[t, j].tolist()); p = par[j]
        Dp = mathutils.Matrix(D[t, p].tolist()) if p >= 0 else mathutils.Matrix.Identity(3)
        basis = R[n].inverted() @ Dp.inverted() @ Dj @ R[n]
        P[n].rotation_quaternion = basis.to_quaternion(); P[n].keyframe_insert('rotation_quaternion', frame=t)
    h = names[0]; P[h].location = R[h].inverted() @ (Aw.inverted() @ mathutils.Vector(dz[t].tolist())); P[h].keyframe_insert('location', frame=t)
bpy.ops.object.mode_set(mode='OBJECT')
# sheet: 8 frames, front and side, shaded grey
sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x, sc.render.resolution_y = 300, 450
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (0.35, 0.38, 0.45); sh = sc.display.shading
sh.light = 'STUDIO'; sh.color_type = 'SINGLE'; sh.single_color = (0.85, 0.85, 0.85)
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'
hgt = F['V0'][:, 2].max() - F['V0'][:, 2].min(); span = np.abs(dz).max() * 2
cam.data.ortho_scale = hgt * 1.3; cam.data.clip_end = 50 * hgt; zc = F['V0'][:, 2].min() + hgt * 0.5
for t in np.linspace(0, T - 1, 8).astype(int):
    sc.frame_set(int(t)); c = mathutils.Vector((float(dz[t, 0]), float(dz[t, 1]), zc))
    for tag, d in (('f', mathutils.Vector((0, -1, 0))), ('s', mathutils.Vector((1, 0, 0)))):
        cam.location = c + d * 10 * hgt; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = '%s_%03d_%s.png' % (pre, t, tag); bpy.ops.render.render(write_still=True)
bpy.ops.object.select_all(action='SELECT'); bpy.ops.export_scene.gltf(filepath=out, export_format='GLB', export_animations=True)
print('RT done ->', out)
