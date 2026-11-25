"""Pictures for plan v8 section 13: the same generated codes for the elf's head decoded with the old decode tiles
(24 tokens, three times the autoencoder's training crop) and the fixed ones (the crop size). No sampling, no training.
args: <interior_data root> <generated npz (codes, idx)> <shell labels npz> <autoencoder .pt> <out png>"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import transformer_smoke as ts
from face_compress import make_ae
from transformer_smoke import DEV, UNK, encode
root, genf, shellf, aef, outf = sys.argv[1:6]
ids = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json')))['ids']; NC = max(ids.values()) + 1
c = torch.load(aef, map_location=DEV); ae = make_ae(NC, c['F'], c['C'], norm=c.get('norm', 'group')).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval(); F = c['F']
g = np.load(genf); codes = g['codes'][0]; idx = g['idx'].astype(np.int64)
L = np.load(shellf)['labels']; G = [n // F for n in L.shape]; L = L[:G[0] * F, :G[1] * F, :G[2] * F]; body = (L > 0) & (L < UNK)
base = encode(ae, np.where(body, 3, L).astype(np.uint8), NC, F); grid = base.copy(); grid[(slice(None),) + tuple(idx.T)] = codes.T

@torch.no_grad()
def decode(tile, ov):
    P = np.zeros(L.shape, np.uint8)
    for a, b, m in ts.tiles(G, tile, ov):
        lg = ts.decode(ae, torch.tensor(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]], device=DEV)).float().argmax(1)[0].cpu().numpy()
        P[(a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = lg[m[0] * F:, m[1] * F:, m[2] * F:]
    P[~body] = 0; return P

old, new = decode(24, 4), decode(*ts.dec_tile(F))
np.savez_compressed(outf.replace('.png', '.npz'), old=old, new=new)
inv = {v: k for k, v in ids.items()}
GROUP = {'bone': ['skull', 'mandible', 'teeth_upper', 'teeth_lower', 'hard_palate'] + ['vertebrae_C%d' % i for i in range(1, 8)],
         'air': ['sinus_maxillary', 'sinus_frontal', 'nasal_cavity_left', 'nasal_cavity_right', 'nasopharynx', 'oropharynx', 'hypopharynx', 'Cavity_Oral', 'trachea'],
         'eye': ['eye_left', 'eye_right', 'eye_lens_left', 'eye_lens_right'], 'brain': ['brain', 'Brainstem'],
         'muscle': ['masseter_left', 'masseter_right', 'temporalis_left', 'temporalis_right', 'lateral_pterygoid_left', 'lateral_pterygoid_right', 'medial_pterygoid_left', 'medial_pterygoid_right', 'tongue', 'digastric_left', 'digastric_right']}
COL = dict(bone=(0.93, 0.89, 0.78), air=(0.25, 0.55, 0.85), eye=(0.95, 0.95, 1.0), brain=(0.80, 0.62, 0.70), muscle=(0.75, 0.25, 0.22), fat=(0.98, 0.84, 0.45), lean=(0.62, 0.42, 0.40), other=(0.55, 0.55, 0.55))
def rgb(P):
    img = np.ones(P.shape + (3,)); img[P > 0] = COL['other']; img[np.isin(P, [1, 2])] = COL['fat']; img[P == 3] = COL['lean']; img[P == 4] = COL['air']; img[P == 5] = COL['bone']
    for gname, names in GROUP.items(): img[np.isin(P, [ids[n] for n in names if n in ids])] = COL[gname]
    return img
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
ii = np.argwhere(body); ctr = ii.mean(0).astype(int)
sinus = np.argwhere(np.isin(new, [ids['sinus_maxillary']])); zs = int(np.median(sinus[:, 2])) if len(sinus) else ctr[2]
views = [('front-to-back slice through the middle', lambda P: P[ctr[0], :, :]), ('top-down slice at the cheekbones', lambda P: P[:, :, zs])]
fig, axs = plt.subplots(2, 2, figsize=(12, 11))
for r, (nm, f) in enumerate(views):
    for col, (P, t) in enumerate(((old, 'old decoding (tiles 3x the training crop)'), (new, 'fixed decoding (tiles = training crop)'))):
        axs[r, col].imshow(rgb(f(P)).transpose(1, 0, 2), origin='lower'); axs[r, col].set_title('%s\n%s' % (t, nm), fontsize=10); axs[r, col].axis('off')
fig.legend(handles=[Patch(color=v, label=k) for k, v in COL.items()], loc='lower center', ncol=8, fontsize=9, frameon=False)
diff = (old != new) & body
fig.suptitle('Elf head, generated interior: same codes, two ways of decoding (%.1f%% of the head changes)' % (100 * diff.sum() / body.sum()), fontsize=12)
plt.tight_layout(rect=(0, 0.04, 1, 0.96)); plt.savefig(outf, dpi=80); print('saved', outf)
for n in ('sinus_maxillary', 'skull', 'teeth_upper', 'Cavity_Oral', 'nasal_cavity_left'):
    print(n, 'old %.0f ml  fixed %.0f ml' % ((old == ids[n]).sum() / 1000, (new == ids[n]).sum() / 1000))
