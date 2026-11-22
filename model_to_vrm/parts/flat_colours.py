"""Blender: the original model with one flat material per part (from the part sheet) beside Tripo's own texture:
  blender -b -P flat_colours.py -- <parts.glb> <part_sheet.json> <out_prefix> [<out.glb>]
Each part object tripo_part_N gets a Principled material in its sheet colour (no image); renders front, back and a head
close-up both ways (Tripo texture, flat), and optionally saves the flat-coloured model."""
import bpy, sys, json, mathutils
a = sys.argv[sys.argv.index('--') + 1:]; src, sj, pre = a[:3]; outglb = a[3] if len(a) > 3 else None
S = json.load(open(sj))
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); meshes = [o for o in bpy.data.objects if o.type == 'MESH']
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; sc.render.resolution_x, sc.render.resolution_y = 700, 1100
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sh = sc.display.shading; sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 50
def shots(tag):
    for v, d, c, s, res in (('front', (0, -1, 0), (0, 0, 0.49), 1.05, (700, 1100)), ('back', (0, 1, 0), (0, 0, 0.49), 1.05, (700, 1100)), ('head', (0, -1, 0), (0.017, 0, 0.86), 0.2, (700, 700))):
        sc.render.resolution_x, sc.render.resolution_y = res; cam.data.ortho_scale = s; d = mathutils.Vector(d); cam.location = mathutils.Vector(c) + d * 10
        cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); sc.render.filepath = '%s_%s_%s.png' % (pre, tag, v); bpy.ops.render.render(write_still=True)
sh.light = 'FLAT'; sh.color_type = 'TEXTURE'; shots('tripo')
mats = {}
for o in meshes:
    p = o.name.rsplit('_', 1)[1]; h = (S.get(p) or {}).get('base_colour') or '#ff00ff'
    if h not in mats:
        m = bpy.data.materials.new('flat_' + h[1:]); rgb = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        srgb2lin = lambda c: c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
        m.diffuse_color = (*[srgb2lin(c) for c in rgb], 1); m.use_nodes = True; m.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (*[srgb2lin(c) for c in rgb], 1); mats[h] = m
    o.data.materials.clear(); o.data.materials.append(mats[h])
sh.light = 'FLAT'; sh.color_type = 'MATERIAL'; shots('flat')
if outglb: bpy.ops.object.select_all(action='SELECT'); bpy.ops.export_scene.gltf(filepath=outglb, export_format='GLB')
print('FC done: %d parts, %d flat colours' % (len(meshes), len(mats)))
