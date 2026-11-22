"""Render the model in its flat colours (front, back, head close-up) beside the AI-textured version:
  python tools/render.py            -> renders/flat_<view>.png and renders/compare.jpg (top: AI texture, bottom: flat)
Each full-detail point takes the colour recorded for the unit that owns its rig point; a unit with no colour is MAGENTA."""
import os, sys, subprocess, numpy as np, cv2, ws
o = ws.owner(); s = ws.sheet(); hexrgb = lambda h: [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
unit_col = {u: (hexrgb(e['colour']) if e.get('colour') else [1, 0, 1]) for u, e in s.items()}
rig_col = np.array([unit_col.get(u, [1, 0, 1]) for u in o.tolist()], np.float32)
full = rig_col[ws.F2R]; np.save(os.path.join(ws.W, 'renders', 'full_colours.npy'), full)
B = r'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe'
r = subprocess.run([B, '-b', '-P', os.path.join(ws.W, 'tools', '_blender_flat.py'), '--', os.path.join(ws.T, '..', 'tripo_parts_orig.glb'),
                    os.path.join(ws.W, 'data.npz'), os.path.join(ws.W, 'renders', 'full_colours.npy'), os.path.join(ws.W, 'renders', 'flat')], capture_output=True, text=True)
if 'BF done' not in r.stdout: print(r.stdout[-1500:], r.stderr[-800:]); sys.exit('render failed')
row = lambda t: np.hstack([cv2.imread(os.path.join(ws.W, 'renders', '%s_%s.png' % (t, v))) for v in ('front', 'back', 'head')])
cv2.imwrite(os.path.join(ws.W, 'renders', 'compare.jpg'), cv2.resize(np.vstack([row('tripo'), row('flat')]), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('rendered renders/compare.jpg (top: AI texture, bottom: flat; magenta = no colour yet); units without colour: %d' % sum(1 for u in set(o.tolist()) if not s.get(u, {}).get('colour')))
