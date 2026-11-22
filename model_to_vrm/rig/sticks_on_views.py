"""Draw the stick skeleton over the textured body renders (their saved cameras): python sticks_on_views.py <body views dir> <skeleton.npz>
Writes <dir>/sticks_on_<view>.png for the front, sides and back; each joint labelled by its short name."""
import sys, os, json, numpy as np, cv2
D, sp = sys.argv[1:3]; cams = json.load(open(os.path.join(D, 'cameras.json'))); S = np.load(sp); H, par, lab = S['heads'], S['parents'], list(S['labels'])
for n in ('tex_a000', 'tex_a090', 'tex_a180', 'tex_a270'):
    cm = cams[n]; R, Sc = cm['res'], cm['scale']; q = np.c_[H, np.ones(len(H))] @ np.array(cm['matrix_inv']).T
    u = ((q[:, 0] / Sc + 0.5) * R).astype(int); v = ((0.5 - q[:, 1] / Sc) * R).astype(int); im = cv2.imread(os.path.join(D, n + '.png'))
    for i, p in enumerate(par):
        if p >= 0: cv2.line(im, (u[p], v[p]), (u[i], v[i]), (255, 80, 0), 3)
    for i in range(len(H)):
        cv2.circle(im, (u[i], v[i]), 5, (0, 0, 255), -1)
        if not any(k in lab[i] for k in ('pinky', 'ring', 'middle', 'index', 'thumb', 'spine', 'eye', 'jaw')): cv2.putText(im, lab[i].replace('_', ''), (u[i] + 6, v[i]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
    cv2.imwrite(os.path.join(D, 'sticks_on_' + n + '.png'), im)
print('SOV done')
