"""Exports the Newton flesh and muscle runs for the interactive viewer (one JSON, int16-quantised positions).
  leg     tet nodes every clip frame (elf_leg_flesh.py SAVE_NODES=1); the old run is rebuilt from it exactly: pinned nodes on
          their targets (they sit there to 0.00 mm), free nodes at rest (the old model held no tets, so they never moved).
          The leg skin is clustered to a lighter mesh and embedded in the tets (barycentric), so the page computes it.
  bones   hip bone, femur, tibia, fibula, patella clustered the same way, with the per-frame thigh / shank transforms.
  rectus  surface every 2nd frame of the isometric runs (dynamic and quasi-static) + their curves.
Clustering cell = the tet edge h (bones, skin) - the resolution the flesh solve itself has.
  E:\\splat\\.venv\\Scripts\\python.exe flesh_web_export.py <fits dir> <model dir> <leg_transforms npz> <flesh_nodes dir> <sim dir> <out json>"""
import sys, os, json, base64, numpy as np
from scipy.spatial import cKDTree
FD, MD, TRF, ND, SD, OUT = sys.argv[1:7]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
def quant(P, lo, hi): return np.round((P - lo) / (hi - lo) * 65534 - 32767).astype(np.int16)
def cluster(V, F, cell):
    """vertex clustering: one vertex per occupied grid cell (mean), faces remapped, degenerate faces dropped"""
    key = np.floor(V / cell).astype(np.int64); _, lab = np.unique(key, axis=0, return_inverse=True); lab = lab.ravel()
    W = np.zeros((lab.max() + 1, 3)); np.add.at(W, lab, V); W /= np.bincount(lab)[:, None]
    G = lab[F]; G = G[(G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])]
    return W, G
N = np.load(os.path.join(ND, 'elf_leg_nodes.npz')); Q, X0, T, pin = N['q'].astype(np.float64), N['X0'].astype(np.float64), N['T'], N['pin']
h = float(json.load(open(os.path.join(ND, 'elf_leg_flesh.json')))['summary']['tet_edge_mm']) / 1000
# skin: the leg's own surface piece (as elf_leg_flesh.py), clustered at h/2, embedded in its tet
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB, FB = R['V'].astype(np.float64), R['obj'], R['F_bone']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']
ISL = np.load(os.path.join(MD, 'flesh', 'islands.npz'))['island']
near = ISL[cKDTree(MV).query(V[OB == nid['Femur.l']])[1]]; leg = int(np.bincount(near[near >= 0]).argmax())
sk = np.flatnonzero(ISL == leg); loc = np.full(len(MV), -1); loc[sk] = np.arange(len(sk)); SF = loc[MF[(ISL[MF] == leg).all(1)]]
SV, SFc = cluster(MV[sk], SF, h / 2)
cen = X0[T].mean(1); cand = cKDTree(cen).query(SV, k=8)[1]; et = np.zeros(len(SV), int); eb = np.zeros((len(SV), 4)); best = np.full(len(SV), np.inf)
for k in range(8):
    tet = T[cand[:, k]]; A = np.stack([X0[tet[:, 1]] - X0[tet[:, 0]], X0[tet[:, 2]] - X0[tet[:, 0]], X0[tet[:, 3]] - X0[tet[:, 0]]], 2)
    l = np.linalg.solve(A, (SV - X0[tet[:, 0]])[..., None])[..., 0]; bb = np.c_[1 - l.sum(1), l]; o = -np.minimum(bb, 0).sum(1); m = o < best
    et[m] = cand[m, k]; eb[m] = bb[m]; best[m] = o[m]
# bones
TR = np.load(TRF); grp_b = {'Hip bone.l': 0, 'Femur.l': 1, 'Tibia.l': 2, 'Fibula.l': 2, 'Patella.l': 2}
bones = []
for n, g in grp_b.items():
    fb = FB[OB[FB[:, 0]] == nid[n]]; u, inv = np.unique(fb, return_inverse=True); W, G = cluster(V[u], inv.reshape(-1, 3), h / 2); bones.append((n, g, W, G))
# quantisation box over everything
allP = np.concatenate([Q.reshape(-1, 3), X0, SV] + [b[2] for b in bones]); lo, hi = allP.min(0) - 0.05, allP.max(0) + 0.05
nf = len(Q)   # the old run (pins on target, free nodes at rest) is rebuilt in the page from q_new, x0 and pin
Dm = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2)
leg_out = dict(frames=nf, fps=float(N['fps']), nodes=len(X0), tets=b64(T.astype(np.uint16)), ntets=len(T), Dinv=b64(np.linalg.inv(Dm).astype(np.float32)),
               q_new=b64(quant(Q, lo, hi)), x0=b64(quant(X0, lo, hi)), pin=b64(pin.astype(np.uint8)),
               skin_f=b64(SFc.astype(np.uint32)), skin_n=len(SV), emb_t=b64(et.astype(np.uint16)), emb_b=b64(eb.astype(np.float32)),
               rows_new=json.load(open(os.path.join(ND, 'elf_leg_flesh.json')))['frames'])
bone_out = [dict(name=n, group=g, v=b64(quant(W, lo, hi)), f=b64(G.astype(np.uint32))) for n, g, W, G in bones]
K = len(TR['thigh_R']); tf = np.zeros((K, 2, 12), np.float32)
for i, key in enumerate(('thigh', 'shank')): tf[:, i, :9] = TR[key + '_R'].reshape(K, 9); tf[:, i, 9:] = TR[key + '_t']
# rectus runs
rect = {}
for tag in ('dyn', 'static'):
    D = np.load(os.path.join(SD, 'view_rectus_%s.npz' % tag)); J = json.load(open(os.path.join(SD, 'view_rectus_%s.json' % tag)))
    S = D['surf'].astype(np.float64); S = np.where(np.isfinite(S), S, D['rest'][None]); rect[tag] = dict(frames=[int(x) for x in D['frames']], surf=b64(quant(S, lo, hi)),
        rows=[dict(t=r['t'], a=r['activation'], belly=r['belly_thickness_ratio'], fib=r['fibre_stretch_p5_50_95'][1], vol=r['volume_ratio']) for r in J['frames']])
    rect['F'] = b64(D['F'].astype(np.uint32)); rect['nv'] = int(S.shape[1]); rect['rest'] = b64(quant(D['rest'].astype(np.float64), lo, hi))
json.dump(dict(lo=lo.tolist(), hi=hi.tolist(), leg=leg_out, bones=bone_out, bone_tf=b64(tf), rectus=rect), open(OUT, 'w'), separators=(',', ':'))
print('wrote %s (%.1f MB): skin %d verts / %d faces, bones %s, %d frames' % (OUT, os.path.getsize(OUT) / 1e6, len(SV), len(SFc), [len(b[3]) for b in bones], nf))
