"""Render the tree (about a minute): python tools/render.py  ->  renders/compare.jpg and renders/flesh.jpg
compare: top row the model with its own texture, bottom row every point in its node's colour (the median texture colour
of the node's parts), same cameras (front, back); MAGENTA = parts not yet assigned.
flesh: the parts under the node 'flesh' in their colour, everything else grey, to check what counts as flesh."""
import sys, os, subprocess, numpy as np, cv2; sys.path.insert(0, os.path.dirname(__file__)); from common import *
t = tree(); A = all_parts(); reg = np.load(os.path.join(WS, 'parts.npz'))['region']
lut = np.tile([1.0, 0, 1], (reg.max() + 1, 1)); cache = {}
for p, n in t['assign'].items():
    if n not in cache: cache[n] = entity_colour(t, n, A)
    lut[int(p)] = cache[n]
def under(n, root):
    while n is not None:
        if n == root: return True
        n = t['nodes'][n].get('parent')
    return False
flesh = np.zeros(len(lut), bool)
for p, n in t['assign'].items(): flesh[int(p)] = 'flesh' in t['nodes'] and under(n, 'flesh')
fl = np.where(flesh[:, None], lut, 0.75)
os.makedirs(os.path.join(WS, 'renders'), exist_ok=True); B = r'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe'
for name, cols in (('tree', lut[reg]), ('flesh', fl[reg])):
    np.save(os.path.join(WS, 'renders', name + '.npy'), cols.astype(np.float32))
    subprocess.run([B, '-b', '-P', os.path.join(os.path.dirname(__file__), '_flat.py'), '--', CFG['model'], os.path.join(WS, 'renders', name + '.npy'),
                    os.path.join(CFG['views'], 'cameras.json'), CFG['front'], CFG['back'], os.path.join(WS, 'renders', name)], capture_output=True)
row = lambda files: np.hstack([cv2.resize(cv2.imread(f), (640, 640)) for f in files])
top = row([os.path.join(CFG['views'], CFG['front'] + '.png'), os.path.join(CFG['views'], CFG['back'] + '.png')])
for name in ('tree', 'flesh'):
    cv2.imwrite(os.path.join(WS, 'renders', 'compare.jpg' if name == 'tree' else 'flesh.jpg'),
                np.vstack([top, row([os.path.join(WS, 'renders', '%s_front.png' % name), os.path.join(WS, 'renders', '%s_back.png' % name)])]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print('renders/compare.jpg and renders/flesh.jpg written')
