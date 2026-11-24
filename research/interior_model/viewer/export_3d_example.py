"""Export one test person's labels (truth and each autoencoder reconstruction) as 3D surfaces for the local 3D viewer.
Picks the face test person whose 4 mm reconstruction Dice is the median of the test set (a typical case, not the best).
Surfaces: marching cubes on each structure's voxel mask (no smoothing, so the voxel steps stay visible); the body
outline is extracted at a coarser step as context. Output: <out>/3d/<person>/{mesh.bin, mesh.json}.
args: <viewer folder made by export_compress_viewer.py>"""
import os, sys, json, numpy as np
from skimage.measure import marching_cubes
from scipy.ndimage import find_objects
root = sys.argv[1]; meta = json.load(open(os.path.join(root, 'meta.json'))); part = 'face'; P = meta['parts'][part]
def med(d, v): x = sorted(r[v] for r in d.values() if v in r); return x[len(x) // 2]
people = sorted(P['people'], key=lambda c: med(P['people'][c]['dice'], 't4')); person = people[len(people) // 2]
shape = P['people'][person]['shape']; mm = P['voxel_mm']; names = {int(k): v for k, v in P['classes'].items()}
BASIC = set(range(0, 7)) | {255}
def surf(mask, step):
    if mask.sum() < 20: return None
    m = np.pad(mask, 1).astype(np.float32)
    v, f, _, _ = marching_cubes(m, 0.5, step_size=step, allow_degenerate=False)
    return (v - 1) * mm, f.astype(np.uint32)
blobs, header = [], dict(person=person, part=part, voxel_mm=mm, shape=shape, sources={}, classes={})
off = 0
def add(arr):
    global off; b = arr.tobytes(); blobs.append(b); o = off; off += len(b); pad = (-off) % 4; blobs.append(b'\0' * pad); off += pad; return o
centre = np.array(shape) * mm / 2
for src in ['truth'] + P['variants']:
    L = np.fromfile(os.path.join(root, part, person, src + '.bin'), np.uint8).reshape(shape)
    if src == 'truth': known = L != 255
    else: L = np.where(known, L, 255)
    out = {}
    body = surf((L != 0) & (L != 255), 3)
    if body is not None:
        v, f = body; out['body'] = dict(pos=add(((v - centre) * 4).round().astype(np.int16)), nv=len(v), idx=add(f), nf=len(f))
    for k, bx in enumerate(find_objects(L.astype(np.int32)), 0 + 1):
        if bx is None or k in BASIC: continue
        lo = np.array([max(s.start - 1, 0) for s in bx]); sub = L[tuple(slice(a, s.stop + 1) for a, s in zip(lo, bx))] == k
        n = int(sub.sum()); r = surf(sub, 1 if n < 40000 else 2)
        if r is None: continue
        v, f = r; v = v + lo * mm
        out[str(k)] = dict(pos=add(((v - centre) * 4).round().astype(np.int16)), nv=len(v), idx=add(f), nf=len(f))
        header['classes'][str(k)] = names.get(k, str(k))
    header['sources'][src] = out
    print(src, sum(o['nf'] for o in out.values()), 'triangles', flush=True)
header['dice'] = P['people'][person]['dice']; header['token_mm'] = P['token_mm']
d = os.path.join(root, '3d'); os.makedirs(d, exist_ok=True)
open(os.path.join(d, 'mesh.bin'), 'wb').write(b''.join(blobs)); json.dump(header, open(os.path.join(d, 'mesh.json'), 'w'))
print('person', person, 'bytes', off)
