"""Does the autoencoder's output depend on the tile size used to encode and decode? (GroupNorm normalises over the whole
tile, and training used 64-voxel crops.) Re-encodes the true labels with encode tiles E and decodes with code tiles D
(in tokens), and scores against the labels. args: <autoencoder .pt> <face index.json> <prep npz> <E list voxels> <D list tokens>"""
import os, sys, json, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import transformer_smoke as ts
from face_compress import make_ae
from transformer_smoke import DEV, UNK
aef, idxf, pf, El, Dl = sys.argv[1:6]
ids = json.load(open(idxf))['ids']; NC = max(ids.values()) + 1; inv = {v: k for k, v in ids.items()}
c = torch.load(aef, map_location=DEV); ae = make_ae(NC, c['F'], c['C'], norm=c.get('norm', 'group')).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval(); F = c['F']
z = np.load(pf); L = z['L']; G = [n // F for n in L.shape]; L = L[:G[0] * F, :G[1] * F, :G[2] * F]; known = L != UNK; body = (L > 0) & (L < UNK)

@torch.no_grad()
def encode(L, tile, ov):
    x = np.where(L == UNK, NC, L).astype(np.int64); out = np.zeros((ae.to_code.out_channels, *G), np.float32)
    for a, b, m in ts.tiles(L.shape, tile, ov):
        a = [aa // F * F for aa in a]; blk = x[a[0]:b[0], a[1]:b[1], a[2]:b[2]]; blk = np.pad(blk, [(0, (-d) % F) for d in blk.shape], constant_values=NC)
        h = ae.inp(ae.emb(torch.tensor(blk[None], device=DEV)).permute(0, 4, 1, 2, 3))
        for d in ae.down: h = d(h)
        zz = ae.to_code(h).float()[0].cpu().numpy(); ga = [aa // F for aa in a]; mm = [q // F for q in m]; gb = [min(g, ga_ + zz.shape[i + 1]) for i, (g, ga_) in enumerate(zip(G, ga))]
        out[:, ga[0] + mm[0]:gb[0], ga[1] + mm[1]:gb[1], ga[2] + mm[2]:gb[2]] = zz[:, mm[0]:gb[0] - ga[0], mm[1]:gb[1] - ga[1], mm[2]:gb[2] - ga[2]]
    return out

@torch.no_grad()
def decode(grid, tile, ov):
    P = np.zeros(L.shape, np.uint8)
    for a, b, m in ts.tiles(G, tile, ov):
        lg = ts.decode(ae, torch.tensor(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]], device=DEV)).float().argmax(1)[0].cpu().numpy()
        P[(a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = lg[m[0] * F:, m[1] * F:, m[2] * F:]
    return P

W = ['skull', 'sinus_maxillary', 'teeth_upper', 'lateral_pterygoid_right', '5', '4']
for E in [int(x) for x in El.split(',')]:
    g = encode(L, E, E // 6 // F * F or F)
    for D in [int(x) for x in Dl.split(',')]:
        P = decode(g, D, max(1, D // 6)); P[~body] = 0; d = {}
        for k in sorted(set(np.unique(L[known])) - {0, 255}):
            a = L == k
            if a.sum() >= 50: d[inv.get(int(k), str(k))] = 2 * (a & (P == k) & known).sum() / (a.sum() + ((P == k) & known).sum())
        print('encode tile %3d vox  decode tile %2d tokens  median %.3f  ' % (E, D, np.median(list(d.values()))) + ' '.join('%s %.2f' % (w, d[w]) for w in W if w in d), flush=True)
