"""Blender: what is inside the Z-Anatomy atlas file (collections, mesh objects, sizes), for picking bones/muscles/skin:
  blender -b -Y <Startup.blend> -P za_inventory.py -- <out.json>      (-Y: the file's own scripts never run)"""
import bpy, sys, json
out = sys.argv[sys.argv.index('--') + 1]; R = {'collections': {}, 'objects': []}
def walk(c, path):
    R['collections'][path] = [o.name for o in c.objects]
    for ch in c.children: walk(ch, path + '/' + ch.name)
walk(bpy.context.scene.collection, 'root')
for o in bpy.data.objects:
    R['objects'].append({'name': o.name, 'type': o.type, 'verts': len(o.data.vertices) if o.type == 'MESH' else 0,
                         'colls': [c.name for c in o.users_collection], 'loc': [round(x, 3) for x in o.matrix_world.translation], 'dims': [round(x, 3) for x in o.dimensions]})
json.dump(R, open(out, 'w'), indent=0); print('ZI', len(R['objects']), 'objects,', len(R['collections']), 'collections')
