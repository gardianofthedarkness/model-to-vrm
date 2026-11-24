"""Comparison pictures for the local viewer folder: for the typical test person of each part (median transformer Dice),
three slices through the body centre (axial, coronal, sagittal), columns truth / transformer / nearest person /
autoencoder reconstruction, plus a row-wise mistake map for the transformer. Same colours and orientation as the
slice viewer. Output stays local and private (CT-derived): <viewer>/pics/*.png and <viewer>/pics.html.
args: <viewer folder>"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, colorsys
root = sys.argv[1]; meta = json.load(open(os.path.join(root, 'meta.json'))); os.makedirs(os.path.join(root, 'pics'), exist_ok=True)
BASIC = {0: (0, 0, 0), 1: (214, 186, 92), 2: (176, 150, 70), 3: (181, 120, 120), 4: (40, 60, 110), 5: (205, 205, 200), 6: (130, 130, 150)}
PELV = dict(outside=(0, 0, 0), subcut_fat=(214, 186, 92), torso_fat=(176, 150, 70), inter_fat=(230, 210, 140), muscle_other=(181, 90, 90), lean_other=(181, 120, 120), gas=(40, 60, 110), dense=(205, 205, 200), organ=(120, 140, 120))
def lut(part, classes):
    L = np.full((256, 3), 90, np.uint8)
    for k, n in classes.items():
        k = int(k)
        if part == 'face' and k in BASIC: L[k] = BASIC[k]
        elif part == 'pelvis' and n in PELV: L[k] = PELV[n]
        else: L[k] = [round(255 * x) for x in colorsys.hls_to_rgb((k * 0.61803398875) % 1, 0.56, 0.72)]
    L[255] = (40, 40, 40); return L
def med(x): x = sorted(x); return x[len(x) // 2] if x else 0
html = ['<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Transformer pictures</title>',
        '<style>body{background:#0f1211;color:#e8e6e0;font:14px system-ui;margin:0;padding:16px}img{width:100%;max-width:1400px;display:block;margin:8px 0 28px}h2{font-weight:600}</style>',
        '<h1>Transformer vs nearest person vs compression</h1><p>Typical test person per part (median transformer Dice). Last column: red = transformer label differs from truth. Hatched grey = unlabelled in the CT.</p>']
for part, P in meta['parts'].items():
    rec = sorted(P['people'], key=lambda c: med([r['tf'] for r in P['people'][c]['dice'].values() if 'tf' in r])); c = rec[len(rec) // 2]
    shape = P['people'][c]['shape']; L = lut(part, P['classes'])
    ae = 't8' if part == 'face' else 't12'; srcs = ['truth', 'tf', 'nn', ae]
    V = {s: np.fromfile(os.path.join(root, part, c, s + '.bin'), np.uint8).reshape(shape) for s in srcs}
    T = V['truth']; ii = np.argwhere((T > 0) & (T < 255)); cx, cy, cz = ii.mean(0).round().astype(int)
    planes = [('axial', lambda A: np.flipud(A[:, :, cz].T)), ('coronal', lambda A: np.flipud(A[:, cy, :].T)), ('sagittal', lambda A: np.flipud(np.fliplr(A[cx, :, :].T)))]
    names = {'truth': 'Truth', 'tf': 'Transformer %.2f' % med([r['tf'] for r in P['people'][c]['dice'].values()]),
             'nn': 'Nearest person %.2f' % med([r['nn'] for r in P['people'][c]['dice'].values()]), ae: '%s mm compression %.2f' % (P['token_mm'][ae], med([r[ae] for r in P['people'][c]['dice'].values()]))}
    fig, ax = plt.subplots(3, 5, figsize=(20, 12.5), dpi=110, facecolor='#0f1211')
    for r, (pn, cut) in enumerate(planes):
        t = cut(T)
        for j, s in enumerate(srcs):
            a = cut(V[s]); img = L[a]; img[t == 255] = (40, 40, 40); ax[r, j].imshow(img, interpolation='nearest')
            if r == 0: ax[r, j].set_title(names[s], color='#e8e6e0', fontsize=13)
        p = cut(V['tf']); e = np.zeros(t.shape + (3,), np.uint8); e[(t > 0) & (t < 255)] = (52, 56, 54); e[(t != p) & (t < 255) & ((t > 0) | (p > 0))] = (235, 70, 60); e[t == 255] = (40, 40, 40)
        ax[r, 4].imshow(e, interpolation='nearest')
        if r == 0: ax[r, 4].set_title('Transformer mistakes', color='#e8e6e0', fontsize=13)
        ax[r, 0].set_ylabel(pn, color='#e8e6e0', fontsize=13)
        for a in ax[r]: a.set_xticks([]); a.set_yticks([]); [s.set_visible(False) for s in a.spines.values()]
    fig.suptitle('%s, test person %s (%s mm voxels)' % (part.capitalize(), c, P['voxel_mm']), color='#e8e6e0', fontsize=15, x=0.01, ha='left')
    fig.tight_layout(); f = 'pics/%s.png' % part; fig.savefig(os.path.join(root, f), facecolor='#0f1211'); plt.close(fig)
    html.append('<h2>%s · %s</h2><img src="%s">' % (part.capitalize(), c, f)); print(part, c, flush=True)
open(os.path.join(root, 'pics.html'), 'w', encoding='utf8').write('\n'.join(html))
