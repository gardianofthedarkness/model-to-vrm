"""Demo pictures for the model card: the input shell, the generated compartments, and the shell half-transparent over
them, from the front and at three-quarter view. Run inside Blender (5.x):
  blender -b -P render_demo.py -- <compartments.glb with skin_input> <out dir>
Writes <out>/<state>_<view>.png; tile them with any image tool (the model card's picture is tiled with PIL)."""
import bpy, sys, os, math, mathutils
glb, out = sys.argv[sys.argv.index('--') + 1:][:2]; os.makedirs(out, exist_ok=True)
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=glb)
sc = bpy.context.scene; sc.render.engine = 'BLENDER_EEVEE'; sc.render.resolution_x = sc.render.resolution_y = 900
sc.render.film_transparent = True; sc.view_settings.view_transform = 'Standard'   # transparent film: paste onto white afterwards
world = bpy.data.worlds.new('w'); sc.world = world; world.use_nodes = True
world.node_tree.nodes['Background'].inputs[0].default_value = (1, 1, 1, 1); world.node_tree.nodes['Background'].inputs[1].default_value = 0.35
objs = {o.name: o for o in bpy.data.objects if o.type == 'MESH'}
def name_of(o):
    for k in ('skin_input', 'fat_layer', 'other_soft', 'other_bone', 'skull', 'mandible', 'teeth', 'eyes', 'brain', 'chew_left', 'chew_right', 'tongue', 'airway', 'spine'):
        if o.name.startswith(k): return k
    return o.name
part = {name_of(o): o for o in objs.values()}
def set_alpha(o, a):
    m = o.active_material; b = m.node_tree.nodes.get('Principled BSDF'); b.inputs['Alpha'].default_value = a
    m.surface_render_method = 'BLENDED' if a < 1 else 'DITHERED'; m.use_backface_culling = False
part['skin_input'].active_material.node_tree.nodes['Principled BSDF'].inputs['Base Color'].default_value = (0.80, 0.66, 0.60, 1)
for o in part.values():
    bpy.context.view_layer.objects.active = o; bpy.ops.object.shade_smooth()
# light: key + fill suns
for rot, e in (((50, 0, 35), 4.0), ((60, 0, -140), 1.5)):
    l = bpy.data.lights.new('sun', 'SUN'); l.energy = e; lo = bpy.data.objects.new('sun', l); sc.collection.objects.link(lo)
    lo.rotation_euler = [math.radians(x) for x in rot]
cam = bpy.data.cameras.new('cam'); cam.type = 'ORTHO'; cam.ortho_scale = 0.30; co = bpy.data.objects.new('cam', cam); sc.collection.objects.link(co); sc.camera = co
VIEWS = {'front': (0.0,), 'three_quarter': (40.0,), 'side': (90.0,)}
INNER = ['skull', 'brain', 'mandible', 'teeth', 'eyes', 'tongue', 'chew_left', 'chew_right', 'airway', 'spine']
STATES = {'input': {'skin_input': 1.0},
          'output': {**{k: 1.0 for k in INNER}, 'skull': 0.35},
          'both': {**{k: 1.0 for k in INNER}, 'skull': 0.35, 'skin_input': 0.28}}
c = sum((o.matrix_world @ mathutils.Vector(v) for o in [part['skin_input']] for v in o.bound_box), mathutils.Vector()) / 8
for st, show in STATES.items():
    for k, o in part.items():
        o.hide_render = k not in show
        if k in show: set_alpha(o, show[k])
    for vn, (az,) in VIEWS.items():
        a = math.radians(az); d = mathutils.Vector((-math.sin(a), -math.cos(a), 0.0))          # the face points -Y (glTF +Z)
        co.location = c + d * 1.0; co.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler()
        sc.render.filepath = os.path.join(out, '%s_%s.png' % (st, vn)); bpy.ops.render.render(write_still=True)
print('done')
