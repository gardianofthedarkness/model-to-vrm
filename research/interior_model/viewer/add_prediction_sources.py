"""Add the milestone 16 transformer's test predictions, and the nearest-person baseline it was scored against, to the
local slice viewer made by export_compress_viewer.py, as two more sources beside the autoencoder reconstructions.
The nearest person is chosen the same way as in transformer_smoke.py: the training person whose body overlaps this
one most (IoU in the part frame); their labels are copied inside this person's body, with unlabelled body filled as
'other lean' (face) or 'dense' (pelvis). Output stays local and private (CT-derived labels of real people).
args: <interior_data root> <downloaded transformer_smoke folder> <viewer folder>"""
import os, sys, json, numpy as np
from scipy.ndimage import find_objects
root, tf, out = sys.argv[1:4]
UNK = 255
meta = json.load(open(os.path.join(out, 'meta.json'))); res = json.load(open(os.path.join(tf, 'transformer_smoke_TF8.json')))
fidx = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pidx = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
hs = sorted(c for c, v in fidx['heads'].items() if v['source'] == 'hanseg')                       # split exactly as the trainer
split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
split.update({c: v['split'] for c, v in fidx['heads'].items() if v['source'] == 'ts'}); split.update({c: v['split'] for c, v in pidx['people'].items()})
SRC = {'face': ('face_v1_headframe', fidx['heads'], 3), 'pelvis': ('pelvis_v1', pidx['people'], 5)}

for part, P in meta['parts'].items():
    folder, members, fill = SRC[part]
    train = sorted(c for c in members if split.get(c) == 'train')
    bodies = {}                                                                                    # packed body masks of training people
    for c in train:
        L = np.load(os.path.join(root, folder, c + '.npz'))['labels']; b = (L > 0) & (L < UNK); bodies[c] = (np.packbits(b, axis=-1), b.shape)
    print(part, len(train), 'training people loaded', flush=True)
    for v in ('tf', 'nn'):
        if v not in P['variants']: P['variants'].insert(0 if v == 'tf' else 1, v)
    P['labels'] = {**{v: '%s mm' % P['token_mm'][v] for v in P['variants'] if v in P['token_mm']}, 'tf': 'Transformer', 'nn': 'Nearest person'}
    P['token_mm'].update(tf=res['token_mm'][part], nn=None)
    for c, rec in P['people'].items():
        T = np.load(os.path.join(root, folder, c + '.npz'))['labels']; body = (T > 0) & (T < UNK)
        ii = np.argwhere(body); lo = np.maximum(ii.min(0) - 4, 0); hi = np.minimum(ii.max(0) + 5, T.shape); sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        Ptf = np.load(os.path.join(tf, 'pred_%s_%s.npz' % (part, c)))['labels']
        def iou(q):
            pk, sh = bodies[q]; bq = np.unpackbits(pk, axis=-1, count=sh[-1]).astype(bool); s = tuple(slice(0, min(x, y)) for x, y in zip(sh, T.shape))
            return (bq[s] & body[s]).sum() / max(1, (bq[s] | body[s]).sum())
        q = max(train, key=iou); Q = np.load(os.path.join(root, folder, q + '.npz'))['labels']; s = tuple(slice(0, min(x, y)) for x, y in zip(Q.shape, T.shape))
        N = np.zeros_like(T); N[s] = np.where(Q[s] == UNK, 0, Q[s]); N[~body] = 0; N[body & (N == 0)] = fill
        Tc = T[sl]; ok = Tc != UNK; boxes = find_objects(Tc.astype(np.int32) % 255)
        for v, Pv in (('tf', Ptf[sl]), ('nn', N[sl])):
            np.ascontiguousarray(Pv).tofile(os.path.join(out, part, c, v + '.bin')); cnt = np.bincount(Pv[ok], minlength=256)
            for k, r in rec['dice'].items():
                bx = boxes[int(k) - 1]; b = Tc[bx] == int(k); n = int(b.sum())
                r[v] = round(float(2 * ((Pv[bx] == int(k)) & b).sum() / (cnt[int(k)] + n)), 3)
        rec['nearest_person'] = 'training person %d' % train.index(q)                              # index only, no id
        print(part, c, 'nearest', train.index(q), 'tf', np.median([r['tf'] for r in rec['dice'].values()]).round(3), 'nn', np.median([r['nn'] for r in rec['dice'].values()]).round(3), flush=True)
    bins = list(next(iter(P['overall'].values()))['by_thickness'])
    def edges(k): a, b = k.replace(' mm', '').split('-'); return float(a), float(b)
    for v in P['variants']:                                                                        # same thickness bins for every source: per-class medians over people
        per = {}
        for rec in P['people'].values():
            for k, r in rec['dice'].items():
                if v in r: per.setdefault(k, ([], []))[0].append(r['thickness_mm']); per[k][1].append(r[v])
        cls = [(np.median(t), np.median(d)) for t, d in per.values()]; bt = {}
        for k in bins:
            a, b = edges(k); x = [d for t, d in cls if a <= t < b]; bt[k] = [round(float(np.median(x)), 3) if x else None, len(x)]
        P['overall'].setdefault(v, {})['by_thickness'] = bt
    for v, who in (('tf', 'model'), ('nn', 'nearest')): P['overall'][v]['median_dice'] = res['test'][part][who + '_median_over_classes']
json.dump(meta, open(os.path.join(out, 'meta.json'), 'w'))
print('done')
