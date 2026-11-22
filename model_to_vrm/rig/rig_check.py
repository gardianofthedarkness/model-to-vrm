"""Blender: check a rigged model: blender -b -P rig_check.py -- <rigged.glb> <prefix>
Renders (front, textured): the rest pose; a weights view (each point coloured by the bone that owns it most); and test poses:
the arms lifted to the sides, one knee bent, the head turned. Bones found by their place in the chain (no names are assumed):
the spine is the root's longest upward chain; arms branch from it at shoulder height; legs hang below the root."""
import bpy, sys, math, colorsys, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, pre = a[0], a[1]
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src)
arm = [o for o in bpy.data.objects if o.type == 'ARMATURE'][0]; mesh = [o for o in bpy.data.objects if o.type == 'MESH' and o.find_armature() == arm][0]
B = arm.data.bones; W = arm.matrix_world
up = lambda b: (W @ b.tail_local - W @ b.head_local).normalized()
root = [b for b in B if b.parent is None][0]
def chain(b):
    out = [b]
    while b.children: b = max(b.children, key=lambda c: (W @ c.tail_local).z if up(c).z > 0 else -1e9); out.append(b) if up(b).z > 0 else None;
    return out
spine = [root]; b = root
while True:
    kids = [c for c in b.children if up(c).z > 0.5]
    if not kids: break
    b = max(kids, key=lambda c: c.length); spine.append(b)
head = spine[-1]
arms = [c for s in spine for c in s.children if c not in spine and abs(up(c).x) > 0.3]
legs = [c for c in root.children if up(c).z < -0.5]
print('RC bones %d; spine %s; arm roots %s; leg roots %s' % (len(B), [s.name for s in spine], [c.name for c in arms], [c.name for c in legs]))
# owners: the bone with the largest weight per point
groups = {g.index: g.name for g in mesh.vertex_groups}; own = []
for v in mesh.data.vertices:
    gw = max(v.groups, key=lambda g: g.weight, default=None); own.append(groups[gw.group] if gw else None)
names = sorted(set(n for n in own if n)); col = {n: colorsys.hsv_to_rgb((i * 0.618) % 1, 0.65, 0.95) for i, n in enumerate(names)}
ca = mesh.data.color_attributes.new('own', 'FLOAT_COLOR', 'POINT')
for i, n in enumerate(own): ca.data[i].color = (*col.get(n, (0.2, 0.2, 0.2)), 1)
print('RC %d points; %d bones own points; unowned %d' % (len(own), len(names), sum(n is None for n in own)))
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x, sc.render.resolution_y = 600, 900
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (0.35, 0.38, 0.45); sh = sc.display.shading
pts = [mesh.matrix_world @ mathutils.Vector(c) for c in mesh.bound_box]; lo = mathutils.Vector([min(p[k] for p in pts) for k in range(3)]); hi = mathutils.Vector([max(p[k] for p in pts) for k in range(3)])
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.ortho_scale = (hi.z - lo.z) * 1.25
c = (lo + hi) / 2; cam.location = c + mathutils.Vector((0, -10 * (hi.z - lo.z), 0)); cam.rotation_euler = mathutils.Vector((0, 1, 0)).to_track_quat('-Z', 'Y').to_euler(); cam.data.clip_end = 30 * (hi.z - lo.z)
def shot(tag, mode):
    if mode == 'tex' and not [i for i in bpy.data.images if i.source in ('FILE', 'PACKED') or i.packed_file]: sh.light = 'STUDIO'; sh.color_type = 'SINGLE'; sh.single_color = (0.85, 0.85, 0.85)   # no texture: shaded grey
    elif mode == 'tex': sh.light = 'FLAT'; sh.color_type = 'TEXTURE'
    else: sh.light = 'STUDIO'; sh.color_type = 'VERTEX'
    sc.render.filepath = '%s_%s.png' % (pre, tag); bpy.ops.render.render(write_still=True)
shot('rest', 'tex'); shot('weights', 'own')
bpy.context.view_layer.objects.active = arm; bpy.ops.object.mode_set(mode='POSE'); P = arm.pose.bones
def rot(b, axis, deg):
    pb = P[b.name]; pb.rotation_mode = 'XYZ'
    # turn about a world axis, expressed in the bone's own frame
    m = (W @ b.matrix_local).to_3x3().inverted(); ax = (m @ mathutils.Vector(axis)).normalized(); pb.rotation_mode = 'QUATERNION'; pb.rotation_quaternion = mathutils.Quaternion(ax, math.radians(deg)) @ pb.rotation_quaternion
for ar in arms:
    s = 1 if (W @ ar.tail_local).x > (W @ ar.head_local).x else -1; nxt = ar.children[0] if ar.children else ar
    rot(nxt, (0, 1, 0), -s * 60)                                                      # the upper arm lifted toward the side
shot('arms_up', 'tex')
_l, _r = cam.location.copy(), cam.rotation_euler.copy(); d3 = mathutils.Vector((0.7, -0.7, 0)); cam.location = c + d3 * 10 * (hi.z - lo.z)
cam.rotation_euler = (-d3).to_track_quat('-Z', 'Y').to_euler(); shot('arms_up_q', 'tex'); cam.location, cam.rotation_euler = _l, _r   # the same from three-quarter view
for pb in P: pb.rotation_quaternion = (1, 0, 0, 0)
if legs: knee = legs[0].children[0] if legs[0].children else legs[0]; rot(legs[0], (1, 0, 0), -50); rot(knee, (1, 0, 0), 80)
shot('knee', 'tex')
for pb in P: pb.rotation_quaternion = (1, 0, 0, 0)
rot(head, (0, 0, 1), 40); rot(spine[-2] if len(spine) > 1 else head, (0, 1, 0), 15); shot('head_turn', 'tex')
# the head turn again from three-quarter view, closer on the upper body (the neck seam shows here)
d3 = mathutils.Vector((0.7, -0.7, 0)); hz = (W @ head.head_local).z; cam.location = mathutils.Vector((c.x, c.y, hz)) + d3 * 10 * (hi.z - lo.z)
cam.rotation_euler = (-d3).to_track_quat('-Z', 'Y').to_euler(); cam.data.ortho_scale = (hi.z - lo.z) * 0.55; shot('head_turn_q', 'tex')
print('RC done')
