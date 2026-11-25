"""Data for the frame-check page: the character's head in the head frame, next to what real training heads look like there.
Writes <viewer>/elf/frame.json: her head mesh (mm in the frame, hair/cloth vertices flagged), her keypoints, the average real
head (surface where half of the training heads have body; an average, not a person), and the mean and spread of real landmark
centroids (ear canals, eyes, mandible, teeth). args: <interior_data root> <character shell npz (elf_face.npz)> <mesh parts dir>
<loose.npz> <joints3d.npz> <generated npz or -> <viewer dir>"""
import os, sys, json, numpy as np
from scipy import ndimage as ndi
from skimage import measure
root, shellf, parts, loosef, jf, genf, viewer = sys.argv[1:8]; UNK = 255
fi = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); ids = fi['ids']; lo = np.array(fi['window_lo_mm']); hi = np.array(fi['window_hi_mm'])
E = np.load(shellf); o, A, s = E['origin'], E['axes'], float(E['scale_mm_per_unit'])
V = np.load(os.path.join(parts, 'orig_colours.npz'))['V'].astype(np.float64); F = np.load(os.path.join(parts, 'orig_normals.npz'))['F']; loose = np.load(loosef)['loose']
Vf = (V - o) @ A.T * s; inw = ((Vf > lo - 30) & (Vf < hi + 30)).all(1); Fh = F[inw[F].all(1)]
CELL = 2.5; key = np.floor(Vf / CELL).astype(np.int64); uq, inv = np.unique(key[np.unique(Fh)], axis=0, return_inverse=True)           # vertex clustering to 2.5 mm
used = np.unique(Fh); remap = np.full(len(V), -1); remap[used] = inv.ravel(); nv = len(uq)
P = np.zeros((nv, 3)); cnt = np.zeros(nv); np.add.at(P, remap[used], Vf[used]); np.add.at(cnt, remap[used], 1); P /= cnt[:, None]
hl = np.zeros(nv); np.add.at(hl, remap[used], loose[used]); hair = hl / cnt > 0.5
Fc = remap[Fh]; Fc = Fc[(Fc[:, 0] != Fc[:, 1]) & (Fc[:, 1] != Fc[:, 2]) & (Fc[:, 0] != Fc[:, 2])]; Fc = np.unique(np.sort(Fc, 1), axis=0, return_index=True)[1]; Fc = remap[Fh][(lambda m: m)(np.ones(len(Fh), bool))][:0] if False else None
Fall = remap[Fh]; ok = (Fall[:, 0] != Fall[:, 1]) & (Fall[:, 1] != Fall[:, 2]) & (Fall[:, 0] != Fall[:, 2]); Fall = Fall[ok]; _, ui = np.unique(np.sort(Fall, 1), axis=0, return_index=True); Fall = Fall[ui]
J = (np.load(jf)['X'][:5] - o) @ A.T * s
hs = sorted(c for c, v in fi['heads'].items() if v['source'] == 'hanseg'); split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
split.update({c: v['split'] for c, v in fi['heads'].items() if v['source'] == 'ts'})
LM = ['auditory_canal_left', 'auditory_canal_right', 'eye_left', 'eye_right', 'mandible', 'teeth_upper', 'teeth_lower']; cen = {k: [] for k in LM}; occ = None; n = 0; G = 4
for c in sorted(fi['heads']):
    f = os.path.join(root, 'face_v1_headframe', c + '.npz')
    if split.get(c) != 'train' or not os.path.exists(f): continue
    L = np.load(f)['labels']; b = ((L > 0) & (L < UNK))[:L.shape[0] // G * G, :L.shape[1] // G * G, :L.shape[2] // G * G]
    b = b.reshape(b.shape[0] // G, G, b.shape[1] // G, G, b.shape[2] // G, G).mean((1, 3, 5)); occ = b if occ is None else occ + b; n += 1
    cm = ndi.center_of_mass(np.ones(L.shape, bool), L, [ids[k] for k in LM]); cnts = np.bincount(L.ravel(), minlength=256)
    for k, m in zip(LM, cm):
        if cnts[ids[k]] >= 10: cen[k].append(np.array(m) + lo + 0.5)
occ /= n; v, f, _, _ = measure.marching_cubes(np.pad(occ, 1), 0.5); v = (v - 1 + 0.5) * G + lo
out = dict(scale_mm_per_unit=s, window=[lo.tolist(), hi.tolist()], elf=dict(v=np.round(P, 1).ravel().tolist(), f=Fall.ravel().tolist(), hair=hair.astype(int).tolist()),
           keypoints=dict(zip(['nose', 'left eye', 'right eye', 'left ear', 'right ear'], np.round(J, 1).tolist())),
           average_head=dict(v=np.round(v, 1).ravel().tolist(), f=f.ravel().tolist(), people=n),
           landmarks={k: dict(mean=np.round(np.mean(c_, 0), 1).tolist(), std=np.round(np.std(c_, 0), 1).tolist(), heads=len(c_)) for k, c_ in cen.items()})
if genf != '-':
    Pg = np.load(genf)['draws'][0]; g = {}
    for k in ('eye_left', 'eye_right', 'mandible', 'teeth_upper'):
        ii = np.argwhere(Pg == ids[k])
        if len(ii): g[k] = dict(centre=np.round(ii.mean(0) + lo + 0.5, 1).tolist(), ml=round(len(ii) / 1000, 1), points=np.round(ii[:: max(1, len(ii) // 1500)] + lo + 0.5).astype(int).ravel().tolist())
    out['generated'] = g
os.makedirs(os.path.join(viewer, 'elf'), exist_ok=True); json.dump(out, open(os.path.join(viewer, 'elf', 'frame.json'), 'w'))
print('elf mesh', nv, 'vertices', len(Fall), 'faces; average head', len(v), 'vertices from', n, 'people;', {k: (out['landmarks'][k]['mean'], out['landmarks'][k]['std']) for k in LM}); print('her keypoints', out['keypoints'])
