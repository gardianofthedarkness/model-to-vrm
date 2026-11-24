"""Export milestone 15 compression results (truth vs autoencoder reconstruction) for the local slice viewer.
Writes, per test person, the truth and every variant's reconstruction cropped to the body (raw uint8 .bin), plus
meta.json with class names, colours, crop shapes and per-person per-class Dice. Output stays local and private
(CT-derived labels of real people): serve it with  python -m http.server  from the output folder.
args: <interior_data root> <downloaded m16_full folder> <out folder>"""
import os, sys, json, glob, shutil, numpy as np
from scipy.ndimage import find_objects
root, m16, out = sys.argv[1:4]
UNK = 255
BASIC = {0: 'outside', 1: 'subcutaneous fat', 2: 'other fat', 3: 'other lean', 4: 'gas', 5: 'dense unlabelled', 6: 'other body structure'}
os.makedirs(out, exist_ok=True)
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compress_viewer.html'), os.path.join(out, 'index.html'))

def thickness(b):                                                   # same idea as the trainer: 2 * volume / surface, in voxels
    if b.sum() == 0: return 0.0
    s = sum(int((np.diff(b.astype(np.int8), axis=a) != 0).sum()) for a in range(3))
    return float(2 * b.sum() / max(s, 1))

meta = dict(parts={})
for part, truth_dir, idx_name, mm in (('face', 'face_v1_headframe', 'index.json', 1.0), ('pelvis', 'pelvis_v1', 'index.json', 3.0)):
    idx = json.load(open(os.path.join(root, truth_dir, idx_name)))
    names = {**BASIC, **{v: k for k, v in idx['ids'].items()}} if part == 'face' else dict(enumerate(idx['classes']))
    preds = sorted(glob.glob(os.path.join(m16, '%s_compress' % part, 'pred_*.npz')))
    variants = sorted({os.path.basename(p).split('_')[1] for p in preds}, key=lambda v: int(v[1:]))
    people = sorted({os.path.basename(p)[len('pred_'):-4].split('_', 1)[1] for p in preds})
    summ = json.load(open(glob.glob(os.path.join(m16, '%s_compress' % part, '*.json'))[0]))['variants']
    P = dict(voxel_mm=mm, variants=variants, people={}, classes={int(k): v for k, v in names.items()},
             token_mm={v: summ[v]['token_mm'] for v in variants}, overall={v: dict(median_dice=summ[v]['median_dice'], by_thickness=summ[v]['by_thickness']) for v in variants})
    for c in people:
        T = np.load(os.path.join(root, truth_dir, c + '.npz'))['labels']
        body = (T != 0) & (T != UNK); ii = np.argwhere(body); lo = np.maximum(ii.min(0) - 4, 0); hi = np.minimum(ii.max(0) + 5, T.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi)); d = os.path.join(out, part, c); os.makedirs(d, exist_ok=True)
        np.ascontiguousarray(T[sl]).tofile(os.path.join(d, 'truth.bin')); Tc = T[sl]; ok = Tc != UNK
        dice = {}; boxes = find_objects(Tc.astype(np.int32) % 255)                  # per-class boxes: work inside them only
        Ps = {v: np.load(os.path.join(m16, '%s_compress' % part, 'pred_%s_%s.npz' % (v, c)))['labels'][sl] for v in variants}
        for v, Pv in Ps.items(): np.ascontiguousarray(Pv).tofile(os.path.join(d, v + '.bin'))
        okP = {v: Pv[ok] for v, Pv in Ps.items()}; cntP = {v: np.bincount(x, minlength=256) for v, x in okP.items()}
        for k, bx in enumerate(boxes, 1):
            if bx is None: continue
            b = Tc[bx] == k; n = int(b.sum())
            if n < 10: continue
            cen = (np.argwhere(b).mean(0) + [s.start for s in bx]).round().astype(int).tolist()
            r = dict(voxels=n, thickness_mm=round(thickness(b) * mm, 1), centre=cen)
            for v, Pv in Ps.items(): r[v] = round(float(2 * ((Pv[bx] == k) & b).sum() / (cntP[v][k] + n)), 3)
            dice[k] = r
        P['people'][c] = dict(shape=[int(x) for x in Tc.shape], dice=dice)
        print(part, c, Tc.shape, flush=True)
    meta['parts'][part] = P
json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'))
print('done', out)
