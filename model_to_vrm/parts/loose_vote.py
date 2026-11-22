"""Jev's boxes of what hangs loose (loose_names.py), voted onto the surface:
  python loose_vote.py <model dir>        -> <model dir>/loose/loose.npz (loose: bool per vertex, share, seen) and loose.jpg
A view votes on the vertices it sees (body/vis.npz): inside Jev's box "loose", outside "body", each with the weight of
Jev's confidence in that box. A vertex is loose when the loose votes outweigh the body votes. Bare skin is never loose
(contacts/material.npz: flesh), whatever box it falls in - a box is a rectangle and takes in the arm or leg beside the
cloth. Hair is loose as a whole (its material). No number of its own."""
import sys, os, json, numpy as np
MD = sys.argv[1]; j = lambda *p: os.path.join(MD, *p); N = json.load(open(j('loose', 'names.json')))['views']; cams = json.load(open(j('body', 'cameras.json')))
MV = np.load(j('parts', 'orig_colours.npz'))['V'].astype(np.float64); VZ = np.load(j('body', 'vis.npz')); vn = [str(x) for x in VZ['names']]; VIS = VZ['vis'].astype(bool)
MT = np.load(j('contacts', 'material.npz'), allow_pickle=True); mat = np.array([str(x) for x in MT['materials']])[MT['material']]
yes = np.zeros(len(MV)); no = np.zeros(len(MV))
for n, a in N.items():
    if a.get('box') is None or n not in vn: continue
    c = cams[n]; Mi = np.array(c['matrix_inv']); cc = MV @ Mi[:3, :3].T + Mi[:3, 3]; u = cc[:, 0] / c['scale'] + 0.5; v = 0.5 - cc[:, 1] / c['scale']; b = a['box']; w = float(a.get('confidence') or 0)
    ins = (u >= b['left']) & (u <= b['right']) & (v >= b['top']) & (v <= b['bottom']); see = VIS[vn.index(n)]; yes[see & ins] += w; no[see & ~ins] += w
    print('LV %s: box %s, confidence %.2f, sees %d vertices, %d in the box' % (n, {k: round(x, 2) for k, x in b.items()}, w, see.sum(), (see & ins).sum()))
share = yes / np.maximum(yes + no, 1e-12); seen = (yes + no) > 0; loose = ((yes > no) & (mat != 'flesh')) | (mat == 'hair')
np.savez(j('loose', 'loose.npz'), loose=loose, share=share.astype(np.float32), seen=seen)
print('LV loose: %d of %d vertices (%s); never seen %d' % (loose.sum(), len(MV), ', '.join('%s %d' % (m, (loose & (mat == m)).sum()) for m in np.unique(mat)), (~seen).sum()))
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(15, 9)); col = np.where(loose, 0, np.where(mat == 'flesh', 1, 2)); cm = np.array(['#d0452c', '#e8c4a8', '#3b6fb5'])
for a_, (i, sg, ti) in zip(ax, [(0, 1, 'front'), (1, 1, 'left side'), (0, -1, 'back')]):
    o = np.argsort(MV[:, 1] if ti == 'back' else (-MV[:, 1] if ti == 'front' else MV[:, 0])); a_.scatter(sg * MV[o, i], MV[o, 2], s=0.15, c=cm[col[o]]); a_.set_aspect('equal'); a_.set_title(ti + ': red = loose, blue = cloth kept as body, tan = skin'); a_.axis('off')
plt.savefig(j('loose', 'loose.jpg'), dpi=80, bbox_inches='tight')
