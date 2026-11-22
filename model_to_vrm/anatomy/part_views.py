"""Blender: zoomed renders of one body region alone (other regions' faces removed, so nothing occludes it), from
directions all around it (6 azimuths x 3 elevations + straight up and down), textured and clay, cameras saved in
body_views.py's format (orthographic: matrix_inv, scale, res) so joints3d-style triangulation reads them:
  blender -b -P part_views.py -- <model.glb> <parts_zoom dir> <region>
inputs  <dir>/boxes.json, <dir>/<region>_vids.npy (part_boxes.py; vertex ids in the order of parts/orig_colours.npz,
        i.e. the glb's vertices object by object)
writes  <dir>/<region>/<tex|clay>_e<el>_a<az>.png, cameras.json"""
import bpy, bmesh, sys, os, json, math, mathutils, numpy as np
a = sys.argv[sys.argv.index('--') + 1:]; src, D, reg = a[0], a[1], a[2]; od = os.path.join(D, reg); os.makedirs(od, exist_ok=True)
box = json.load(open(os.path.join(D, 'boxes.json')))[reg]; keep = np.load(os.path.join(D, reg + '_vids.npy'))
for o in list(bpy.data.objects): bpy.data.objects.remove(o)
bpy.ops.import_scene.gltf(filepath=src); obs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
off = 0
for o in obs:                                                     # keep the region's own vertices (ids follow the glb's vertex order)
    n = len(o.data.vertices); kk = keep[(keep >= off) & (keep < off + n)] - off; off += n
    m = np.zeros(n, bool); m[kk] = True; bm = bmesh.new(); bm.from_mesh(o.data); bm.verts.ensure_lookup_table()
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not m[v.index]], context='VERTS'); bm.to_mesh(o.data); bm.free()
c = mathutils.Vector(box['centre']); ext = float(2 * np.linalg.norm(box['half']) * 1.05)   # the box's diagonal: the region fits from any direction
sc = bpy.context.scene; sc.render.engine = 'BLENDER_WORKBENCH'; R = 1024; sc.render.resolution_x = sc.render.resolution_y = R
sc.world = sc.world or bpy.data.worlds.new('w'); sc.world.color = (1, 1, 1); sc.view_settings.view_transform = 'Standard'
cam = bpy.data.objects.new('c', bpy.data.cameras.new('c')); sc.collection.objects.link(cam); sc.camera = cam; cam.data.type = 'ORTHO'; cam.data.clip_end = 100; cam.data.ortho_scale = ext
sh = sc.display.shading; cams = {}
dirs = [(el, az) for el in (-45, 0, 45) for az in range(0, 360, 60)] + [(89.9, 0), (-89.9, 0)]
for mode in ('tex', 'clay'):
    if mode == 'tex': sh.light = 'FLAT'; sh.color_type = 'TEXTURE'; sh.show_cavity = False
    else: sh.light = 'STUDIO'; sh.color_type = 'SINGLE'; sh.single_color = (0.8, 0.8, 0.8); sh.show_cavity = True; sh.cavity_type = 'BOTH'
    for el, az in dirs:
        d = mathutils.Vector((math.sin(math.radians(az)) * math.cos(math.radians(el)), -math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(el))))
        cam.location = c + d * 10; cam.rotation_euler = (-d).to_track_quat('-Z', 'Y').to_euler(); bpy.context.view_layer.update()
        n = '%s_e%+03d_a%03d' % (mode, round(el), az); sc.render.filepath = os.path.join(od, n + '.png'); bpy.ops.render.render(write_still=True)
        cams[n] = {'az': az, 'el': el, 'matrix_inv': [list(r) for r in cam.matrix_world.inverted()], 'scale': ext, 'res': R}
json.dump(cams, open(os.path.join(od, 'cameras.json'), 'w')); print('PV done', reg, len(cams))
