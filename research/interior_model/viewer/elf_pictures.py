"""Pictures of a generated interior (flow_sample.py output) for a character: three slices per part with a class legend.
args: <interior_data root> <generated npz> <face|pelvis> <out png> [title]"""
import os, sys, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
root, gf, part, outf = sys.argv[1:5]; title = sys.argv[5] if len(sys.argv) > 5 else part
fi = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pi = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
names = {v: k for k, v in fi['ids'].items()} if part == 'face' else dict(enumerate(pi['classes'])); mm = 1.0 if part == 'face' else 3.0
z = np.load(gf); P = z['majority'] if z['draws'].shape[0] > 1 else z['draws'][0]; body = P > 0
rng = np.random.default_rng(7); pal = rng.random((256, 3)) * 0.75 + 0.2; pal[0] = 1.0
vol = np.bincount(P[body], minlength=256); top = [k for k in np.argsort(vol)[::-1] if vol[k] > 0]
bb = np.argwhere(body); a0 = np.maximum(bb.min(0) - 4, 0); P = P[a0[0]:bb[:, 0].max() + 5, a0[1]:bb[:, 1].max() + 5, a0[2]:bb[:, 2].max() + 5]; body = P > 0   # crop to the body
c = np.round(np.argwhere(body).mean(0)).astype(int); ax_names = ('side view (slice across left-right)', 'front view (slice across front-back)', 'top view (slice across height)')
fig, ax = plt.subplots(1, 4, figsize=(22, 7), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.75]))
for k in range(3):
    sl = [slice(None)] * 3; sl[k] = c[k]; im = pal[P[tuple(sl)]]; ax[k].imshow(im.transpose(1, 0, 2), origin='lower'); ax[k].set_title(ax_names[k]); ax[k].set_xticks([]); ax[k].set_yticks([])
ax[3].axis('off'); ax[3].legend(handles=[Patch(color=pal[k], label='%s  %.0f ml' % (names.get(int(k), str(k)), vol[k] * mm ** 3 / 1000)) for k in top[:28]], loc='center', fontsize=9, frameon=False)
fig.suptitle(title); plt.tight_layout(); plt.savefig(outf, dpi=80); print(outf)
