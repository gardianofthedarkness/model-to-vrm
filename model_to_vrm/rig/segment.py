"""Part separation through the volume, and the contact surfaces where body parts touch:
  python segment.py <model.glb> <skeleton.npz> <out dir>
segments  body pieces of the stick skeleton (head, torso, upper arm, forearm, hand, thigh, shin, foot; left/right); a
          stick (parent joint -> child joint) belongs to the piece that moves it (thigh = hip -> knee ...)
volume    the mesh voxelised and filled (closing then hole filling); voxel size from the target count of voxels inside
          the bounding box, so the grid is the same size for any model
label     each voxel and each surface point goes to the piece with the smallest distance-to-its-sticks divided by that
          piece's own flesh radius; the radius = median stick distance of the surface points that are unambiguous for
          that piece (their second-nearest piece is further than the Otsu split of the nearest/second ratios)
contact   voxels of piece A with a face-neighbour in piece B, for A and B not joined in the skeleton = where two pieces of
          flesh press together (thigh on calf in a crouch, thigh on thigh with crossed legs, arm on torso)
Writes labels.npz (surface vertex labels, names), voxels.npz (grid labels, origin, pitch), contacts.npz and one
contact_<A>__<B>.ply (voxel boxes) per touching pair."""
import sys, os, numpy as np, trimesh
from scipy import ndimage
from scipy.spatial import cKDTree
src, sp, od = sys.argv[1:4]; os.makedirs(od, exist_ok=True)
sc = trimesh.load(src); g = trimesh.util.concatenate(list(sc.geometry.values())) if hasattr(sc, 'geometry') else sc
toB = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float); g.vertices = g.vertices @ toB.T
S = np.load(sp); Hh, par, lab = S['heads'], S['parents'], [str(x) for x in S['labels']]
def piece(name):
    side = name[:2] if name[:2] in ('l_', 'r_') else ''
    if any(k in name for k in ('pinky', 'ring', 'middle', 'index', 'thumb')) or name.endswith('_wrist'): return side + 'hand'
    if 'lowarm' in name or 'wrist_twist' in name: return side + 'forearm'
    if 'uparm' in name: return side + 'upperarm'
    if 'upleg' in name: return side + 'thigh'
    if 'lowleg' in name: return side + 'shin'
    if any(k in name for k in ('foot', 'talocrural', 'subtalar', 'transversetarsal', 'ball')): return side + 'foot'
    if any(k in name for k in ('neck', 'head', 'jaw', 'teeth', 'tongue', 'eye')): return 'head'
    return 'torso'                                                   # root, spine, clavicles
sticks = []                                                          # (piece, a, b): a stick moves with its parent joint's piece
for i, p in enumerate(par):
    if p < 0: continue
    pc = piece(lab[p]) if piece(lab[i]) != 'head' else 'head'        # the neck stick belongs to the head
    if lab[i].endswith('_upleg'): pc = 'torso'                      # hip sticks are pelvis
    sticks.append((pc, Hh[p], Hh[i]))
names = sorted({s[0] for s in sticks}); NP = len(names)
# pieces joined in the skeleton (a stick of one ends where a stick of the other starts) never count as touching
joined = set()
for i, p in enumerate(par):
    if p >= 0 and piece(lab[i]) != piece(lab[p]): joined |= {(piece(lab[i]), piece(lab[p])), (piece(lab[p]), piece(lab[i]))}
joined |= {('head', 'torso'), ('torso', 'head')} | {(s + 'thigh', 'torso') for s in ('l_', 'r_')} | {('torso', s + 'thigh') for s in ('l_', 'r_')}
def seg_dist(P):
    D = np.full((len(P), NP), np.inf, np.float32)
    for pc, a, b in sticks:
        ab = b - a; t = np.clip(((P - a) @ ab) / max(ab @ ab, 1e-12), 0, 1); d = np.linalg.norm(P - (a + t[:, None] * ab), axis=1)
        k = names.index(pc); D[:, k] = np.minimum(D[:, k], d)
    return D
# radius per piece from unambiguous surface points
Vs = g.vertices; Ds = seg_dist(Vs); o = np.sort(Ds, 1); ratio = o[:, 1] / np.maximum(o[:, 0], 1e-9)
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
clear = ratio > otsu(np.log(ratio[np.isfinite(ratio)])) if False else np.log(ratio) > otsu(np.log(ratio[np.isfinite(ratio)]))
first = Ds.argmin(1); rad = np.array([np.median(Ds[clear & (first == k), k]) if (clear & (first == k)).any() else np.median(Ds[:, k]) for k in range(NP)])
print('SEG flesh radius per piece (m):', ', '.join('%s %.3f' % (n, r) for n, r in zip(names, rad)))
vlab = (Ds / rad).argmin(1)
# volume
lo, hi = Vs.min(0), Vs.max(0); pitch = float((np.prod(hi - lo) / 3e6) ** (1 / 3))
vg = g.voxelized(pitch); M = vg.matrix; M = ndimage.binary_closing(M, iterations=2); M = ndimage.binary_fill_holes(M)
idx = np.argwhere(M); P = vg.indices_to_points(idx) if hasattr(vg, 'indices_to_points') else trimesh.transform_points(idx.astype(float), vg.transform)
L = -np.ones(M.shape, np.int16); L[tuple(idx.T)] = (seg_dist(P) / rad).argmin(1)
print('SEG voxels inside %d, pitch %.4f m' % (len(idx), pitch))
# contacts: face-neighbour pairs of non-joined pieces
contacts = {}
for ax in range(3):
    a = [slice(None)] * 3; b = [slice(None)] * 3; a[ax] = slice(0, -1); b[ax] = slice(1, None)
    A, B = L[tuple(a)], L[tuple(b)]; m = (A >= 0) & (B >= 0) & (A != B)
    for ia, ib, pos in zip(A[m], B[m], np.argwhere(m)):
        na, nb = names[ia], names[ib]
        if (na, nb) in joined: continue
        key = tuple(sorted((na, nb))); contacts.setdefault(key, []).append(pos)
np.savez(os.path.join(od, 'labels.npz'), label=vlab, names=np.array(names), radius=rad)
np.savez(os.path.join(od, 'voxels.npz'), L=L, transform=vg.transform, pitch=pitch)
for f in os.listdir(od):
    if f.startswith('contact_'): os.remove(os.path.join(od, f))
summary = []
for (na, nb), pos in sorted(contacts.items(), key=lambda kv: -len(kv[1])):
    pos = np.array(pos); pts = trimesh.transform_points(pos.astype(float), vg.transform)
    area = len(pos) * pitch ** 2; summary.append((na, nb, len(pos), area, pts.mean(0)))
    boxes = trimesh.util.concatenate([trimesh.creation.box(extents=[pitch] * 3, transform=trimesh.transformations.translation_matrix(p)) for p in pts[:: max(1, len(pts) // 4000)]])
    boxes.export(os.path.join(od, 'contact_%s__%s.ply' % (na, nb)))
np.savez(os.path.join(od, 'contacts.npz'), pairs=np.array([(a, b) for a, b, *_ in summary]), area=np.array([s[3] for s in summary]))
for na, nb, n, area, c in summary: print('SEG contact %-12s %-12s %6d voxel faces  %.4f m2  at %s' % (na, nb, n, area, c.round(3)))
print('SEG done: %d pieces, %d touching pairs' % (NP, len(summary)))
