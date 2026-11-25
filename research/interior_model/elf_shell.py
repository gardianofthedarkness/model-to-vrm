"""Put a generated character's head and pelvis into the interior model's part frames, as shell-only label volumes.

Input: the character's surface mesh (V, F, metres, z up), a per-vertex mask of loose parts (hair, hanging cloth), her
detected body keypoints (COCO-wholebody order) and the anatomy atlas already warped into her (for the bones that define
the pelvis frame). Steps:
1. Body solid: generalized winding number (libigl, fast) of the mesh. With a loose mask, faces with a loose vertex are
   dropped first. Tried on the elf 2026-11-24 and rejected: hair and cloth cover her whole back and most of her head,
   so dropping them leaves half a body. Pass '-' to keep every face; hair and cloth then count as body.
2. Frames, as the training data defines them:
   head    origin = midpoint of the ears, x = left ear -> right ear, y towards the midpoint of the eyes (orthogonal to x)
   pelvis  origin = midpoint of the hip joints, x = left -> right hip, z towards the centroid of L4-L5 (orthogonal to x)
3. Size. The mesh has no real-world scale. Per part, the scale (mm per mesh unit) is the one at which her token
   occupancy overlaps a real training person's best (highest best IoU over training people), searched in one dimension.
4. Label volume in the training window and resolution: 0 outside, one generic tissue class inside, 255 = unknown where
   training people have no data either (head: below the median lowest body height of training heads; pelvis: outside
   the height range of her hip bones and sacrum).
Output: <out>/elf_<part>.npz (labels, origin, axes, scale) and elf_shell.json.
args: <interior_data root> <mesh V/F npz dir (parts/)> <loose.npz> <joints3d.npz> <atlas_warp.npz> <out dir>"""
import os, sys, json, numpy as np, igl
from scipy.optimize import minimize_scalar
root, parts, loosef, jf, af, out = sys.argv[1:7]; os.makedirs(out, exist_ok=True); UNK = 255
V = np.load(os.path.join(parts, 'orig_colours.npz'))['V'].astype(np.float64); F = np.load(os.path.join(parts, 'orig_normals.npz'))['F'].astype(np.int64)
keep = ~np.load(loosef)['loose'][F].any(1) if loosef != '-' else np.ones(len(F), bool); Fk = np.ascontiguousarray(F[keep]); print('faces kept', int(keep.sum()), 'of', len(F), flush=True)   # '-' = keep every face
def inside(Q):                                                                                     # generalized winding number > 1/2
    w = np.concatenate([igl.fast_winding_number(V, Fk, np.ascontiguousarray(q)) for q in np.array_split(Q, max(1, len(Q) // 2000000))]); return w > 0.5
J = np.load(jf)['X']; A = np.load(af, allow_pickle=True); an = list(A['names']); AV = A['V'].astype(np.float64); aobj = A['obj']
bone = lambda *names: AV[np.isin(aobj, [an.index(n) for n in names])]
unit = lambda v: v / np.linalg.norm(v)
def frame_head():
    le, re_, eyes = J[3], J[4], (J[1] + J[2]) / 2; o = (le + re_) / 2; x = unit(re_ - le); y = eyes - o; y = unit(y - x * (x @ y)); return o, np.stack([x, y, np.cross(x, y)])
def frame_pelvis():
    lh, rh = J[11], J[12]; o = (lh + rh) / 2; x = unit(rh - lh); z = bone('Vertebra L4', 'Vertebra L5').mean(0) - o; z = unit(z - x * (x @ z)); return o, np.stack([x, np.cross(z, x), z])
fidx = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pidx = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
hs = sorted(c for c, v in fidx['heads'].items() if v['source'] == 'hanseg'); split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
split.update({c: v['split'] for c, v in fidx['heads'].items() if v['source'] == 'ts'}); split.update({c: v['split'] for c, v in pidx['people'].items()})
CFG = dict(face=('face_v1_headframe', fidx['heads'], 1.0, 8, 3, np.array(fidx['window_lo_mm']), np.array(fidx['window_hi_mm']), frame_head),
           pelvis=('pelvis_v1', pidx['people'], 3.0, 4, 5, np.array(pidx['window_lo_mm']), np.array(pidx['window_hi_mm']), frame_pelvis))
res = {}
for part, (folder, members, mm, Ft, fill, lo, hi, fr) in CFG.items():
    o, R = fr(); tok = Ft * mm; G = ((hi - lo) / tok).astype(int); blk = lambda B: B[:G[0] * Ft, :G[1] * Ft, :G[2] * Ft].reshape(G[0], Ft, G[1], Ft, G[2], Ft).any((1, 3, 5))
    occ = []; lowz = []
    for c in sorted(members):
        f = os.path.join(root, folder, c + '.npz')
        if split.get(c) != 'train' or not os.path.exists(f): continue
        L = np.load(f)['labels']; b = (L > 0) & (L < UNK); occ.append(np.packbits(blk(b))); lowz.append(lo[2] + mm * np.argmax(b.any((0, 1))))
    occ = np.unpackbits(np.stack(occ), axis=1)[:, :G.prod()].astype(bool); nocc = occ.sum(1); print(part, 'training people', len(occ), 'tokens', G.tolist(), flush=True)
    hb = (bone('Hip bone.l', 'Hip bone.r', 'Sacrum') - o) @ R.T if part == 'pelvis' else None       # her hip bones and sacrum in the frame (mesh units)
    def zrange(s): return (float(np.median(lowz)), hi[2]) if part == 'face' else (float(hb[:, 2].min() * s), float(hb[:, 2].max() * s))
    def grid(step, s):                                                                             # window sample points (mm) -> inside her body, and known mask
        ax = [lo[i] + step * (np.arange(int(round((hi[i] - lo[i]) / step))) + 0.5) for i in range(3)]; P = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
        z0, z1 = zrange(s); known = (P[:, 2] >= z0) & (P[:, 2] <= z1); B = np.zeros(len(P), bool); B[known] = inside(P[known] / s @ R + o)
        sh = [len(a) for a in ax]; return B.reshape(sh), known.reshape(sh)
    def score(s):
        B, _ = grid(tok / 2, s); t = B[:G[0] * 2, :G[1] * 2, :G[2] * 2].reshape(G[0], 2, G[1], 2, G[2], 2).any((1, 3, 5)).ravel()
        inter = (occ & t).sum(1); return float((inter / (nocc + t.sum() - inter)).max())
    span = V[:, 2].max() - V[:, 2].min(); br = (1200.0 / span, 2200.0 / span)                      # search bracket only: she is between 1.2 and 2.2 m tall
    opt = minimize_scalar(lambda s: -score(s), bounds=br, method='bounded', options=dict(xatol=br[0] * 1e-3)); s = float(opt.x)
    B, known = grid(mm, s); L = np.where(known, np.where(B, fill, 0), UNK).astype(np.uint8)
    np.savez_compressed(os.path.join(out, 'elf_%s.npz' % part), labels=L, origin=o, axes=R, scale_mm_per_unit=s)
    res[part] = dict(scale_mm_per_unit=round(s, 1), implied_height_mm=round(s * span), best_iou_with_a_training_person=round(-float(opt.fun), 3), body_voxels=int(B.sum()), shape=list(L.shape), z_range_mm=[round(x, 1) for x in zrange(s)])
    print(part, res[part], flush=True)
json.dump(res, open(os.path.join(out, 'elf_shell.json'), 'w'), indent=1)
