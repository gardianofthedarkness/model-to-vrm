"""Rest shapes for the elf's leg muscles as tubes between their glued ends, rebuilt from the atlas (the fitted shapes of
several muscles were crushed by the warp and repair: rectus femoris 64 -> 5.7 cm^3).
  E:\\splat\\.venv\\Scripts\\python.exe muscle_rest.py <fits dir> <out npz> "<muscle>" ["<muscle>" ...]

  ends      origin / insertion patches as in muscle_tets.py: the muscle's bone-contact vertices on the most proximal and
            the most distal bone segment (pelvis, thigh, shank), from the fitted muscle (same vertex indices in the atlas)
  rest      the atlas muscle (atlas_warp V0) moved by one similarity: the rotation that takes the atlas end-to-end axis
            o0 -> i0 onto the fitted one o1 -> i1 with the least turn, a uniform scale s = |i1 - o1| / |i0 - o0|, and the
            translation putting the atlas origin centroid on the fitted one.  The tube keeps human proportions at the
            elf's size; its ends land on the elf's own attachments.
  check     volume against the fitted shape; share of tube vertices inside each leg bone's solid (to be pushed out by the
            simulation's bone contact)
Writes per muscle: rest vertices, faces, origin / insertion vertex indices and segments."""
import sys, os, re, json, numpy as np
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(HERE, 'model_to_vrm', 'anatomy'))
from solids import bone_solid, sample
FD, OUT = sys.argv[1], sys.argv[2]; ms = sys.argv[3:]
W = np.load(os.path.join(FD, 'atlas_warp.npz')); names = [str(x) for x in W['names']]; nid = {n: i for i, n in enumerate(names)}; V0 = W['V0'].astype(np.float64)
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']; FM, FB = R['F_muscle'], R['F_bone']; contact = R['contact'].astype(bool)
Wt = csr_matrix((R['W_data'], R['W_indices'], R['W_indptr'])); wb = [int(b) for b in R['bones']]
S = '.l'; foot = [i for i, n in enumerate(names) if n.endswith(S) and re.search(r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform|phalan', n)]
SEG = {nid['Hip bone' + S]: 0, nid['Femur' + S]: 1, nid['Tibia' + S]: 2, nid['Fibula' + S]: 2, nid['Patella' + S]: 2, **{b: 2 for b in foot}}
bones = {n: nid[n + S] for n in ('Hip bone', 'Femur', 'Tibia', 'Fibula', 'Patella')}
hstep = 2 * float(np.median(np.linalg.norm(V[FB[:, 0]] - V[FB[:, 1]], axis=1)))
SOL = {k: bone_solid(V, FB[OB[FB[:, 0]] == b], hstep)[:2] for k, b in bones.items()}
def turn(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b); v = np.cross(a, b); c = a @ b
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K / (1 + c)
def volume(P, f): return abs(np.einsum('ij,ij->i', P[f[:, 0]], np.cross(P[f[:, 1]], P[f[:, 2]])).sum()) / 6
out = {}; rep = {}
for m in ms:
    i = nid[m]; f = FM[OB[FM[:, 0]] == i]; u, inv = np.unique(f.ravel(), return_inverse=True); f = inv.reshape(-1, 3)
    c = contact[u]; seg = np.full(len(u), -1)
    if c.any(): seg[np.flatnonzero(c)] = [SEG.get(wb[k], -1) for k in np.asarray(Wt[u[c]].argmax(1)).ravel()]
    present = sorted(set(seg[seg >= 0]))
    if len(present) < 2: print('%s: touches fewer than two segments, skipped' % m); continue
    oi, ii = np.flatnonzero(seg == present[0]), np.flatnonzero(seg == present[-1])
    o0, i0, o1, i1 = V0[u[oi]].mean(0), V0[u[ii]].mean(0), V[u[oi]].mean(0), V[u[ii]].mean(0)
    Rm = turn(i0 - o0, i1 - o1); s = np.linalg.norm(i1 - o1) / np.linalg.norm(i0 - o0)
    P = (V0[u] - o0) @ Rm.T * s + o1
    inb = {k: float((sample(sd, lo, hstep, P) > 0).mean()) for k, (sd, lo) in SOL.items()}
    rep[m] = dict(segments=[int(x) for x in present], scale=round(float(s), 4), atlas_cm3=round(1e6 * volume(V0[u], f), 1), rest_cm3=round(1e6 * volume(P, f), 1),
                  fitted_cm3=round(1e6 * volume(V[u], f), 1), end_distance_mm=round(1000 * float(np.linalg.norm(i1 - o1)), 1),
                  inside_bone={k: round(v, 3) for k, v in inb.items() if v > 0})
    out[m] = dict(P=P, F=f, origin=oi, insertion=ii, seg_o=present[0], seg_i=present[-1])
    print('%-34s scale %.3f | volume atlas %.0f, rest tube %.1f, fitted %.1f cm3 | ends %.0f mm apart | inside bone %s' % (
        m, s, rep[m]['atlas_cm3'], rep[m]['rest_cm3'], rep[m]['fitted_cm3'], rep[m]['end_distance_mm'], rep[m]['inside_bone']), flush=True)
np.savez(OUT, names=np.array(list(out)), **{'%d_%s' % (k, key): v for k, (m, d) in enumerate(out.items()) for key, v in d.items()})
json.dump(rep, open(OUT.replace('.npz', '.json'), 'w'), indent=1); print('wrote', OUT)
