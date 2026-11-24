"""Tetrahedra for closed muscle "tubes" of the elf, with their glued end patches.
  E:\\splat\\.venv-sim\\Scripts\\python.exe muscle_tets.py <fits dir> <out dir> "<muscle name>" ["<muscle name>" ...]

  surface   the fitted muscle mesh (muscle_repair.npz), closed and manifold
  tets      the surface vertices plus an interior grid at spacing h = volume / surface area of the muscle (for a round
            tube of radius r that is r / 2, about four elements across), joined by Delaunay (scipy); tets kept where their
            centre is inside the surface (offset to the nearest surface vertex points against its outward normal); slivers below the
            Otsu split of log volume dropped.  (fTetWild ran out of memory on these thin closed tubes.)
  glue      the muscle's surface vertices in contact with bone, grouped by the bone segment their largest bone share is on
            (pelvis, thigh, shank); origin = contacts on the most proximal segment, insertion = on the most distal one.
            Contacts on a segment in between are left free (the muscle rests on that bone and may slide).  Tet nodes
            within one edge length of a glued surface vertex are glued, each to the bone of its nearest glued vertex.
Writes <out>/<muscle>.npz: X (nodes), T (tets), glue_seg (-1 free, else 0 pelvis / 1 thigh / 2 shank), surface V/F."""
import sys, os, re, numpy as np
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree
FD, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']; FM = R['F_muscle']; contact = R['contact'].astype(bool)
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
Wt = csr_matrix((R['W_data'], R['W_indices'], R['W_indptr'])); wb = [int(b) for b in R['bones']]
S = '.l'; foot = [i for i, n in enumerate(names) if n.endswith(S) and re.search(r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform|phalan', n)]
SEG = {nid['Hip bone' + S]: 0, nid['Femur' + S]: 1, nid['Tibia' + S]: 2, nid['Fibula' + S]: 2, nid['Patella' + S]: 2, **{b: 2 for b in foot}}
for name in sys.argv[3:]:
    i = nid[name]; f = FM[OB[FM[:, 0]] == i]; u, inv = np.unique(f.ravel(), return_inverse=True); Vs, Fs = V[u], inv.reshape(-1, 3)
    vol = abs(np.einsum('ij,ij->i', Vs[Fs[:, 0]], np.cross(Vs[Fs[:, 1]], Vs[Fs[:, 2]])).sum()) / 6
    area = np.linalg.norm(np.cross(Vs[Fs[:, 1]] - Vs[Fs[:, 0]], Vs[Fs[:, 2]] - Vs[Fs[:, 0]]), axis=1).sum() / 2; h = vol / area
    vn = np.zeros_like(Vs); fn = np.cross(Vs[Fs[:, 1]] - Vs[Fs[:, 0]], Vs[Fs[:, 2]] - Vs[Fs[:, 0]])
    for k_ in range(3): np.add.at(vn, Fs[:, k_], fn)
    vn /= np.linalg.norm(vn, axis=1, keepdims=True) + 1e-30
    if np.einsum('ij,ij->', Vs - Vs.mean(0), vn) < 0: vn = -vn                                    # outward
    kdv = cKDTree(Vs)
    def winding(Pq):    # inside test for a closed surface: sign of the offset to the nearest surface vertex along its normal
        d_, j_ = kdv.query(Pq); return ((np.einsum('ij,ij->i', Pq - Vs[j_], vn[j_]) < 0)).astype(float)
    lo, hi = Vs.min(0), Vs.max(0); g = np.stack(np.meshgrid(*[np.arange(l_ + h / 2, u_, h) for l_, u_ in zip(lo, hi)], indexing='ij'), -1).reshape(-1, 3)
    wg = winding(g); g = g[np.abs(wg) > 0.5]; g = g[cKDTree(Vs).query(g)[0] > h / 2]        # interior grid, not crowding the surface
    X = np.r_[Vs, g]; T = Delaunay(X).simplices.astype(np.int64)
    keep = np.abs(winding(X[T].mean(1))) > 0.5; T = T[keep]
    Dm = np.stack([X[T[:, 1]] - X[T[:, 0]], X[T[:, 2]] - X[T[:, 0]], X[T[:, 3]] - X[T[:, 0]]], 2); tv = np.linalg.det(Dm)
    T[tv < 0] = T[tv < 0][:, [0, 2, 1, 3]]; tv = np.abs(tv) / 6; lv = np.log(tv + 1e-30)
    q_ = tv / (np.max(np.linalg.norm(np.stack([X[T[:, a]] - X[T[:, b]] for a in range(4) for b in range(a + 1, 4)], 1), axis=2), axis=1) ** 3)   # volume / longest edge^3
    lq = np.log(q_ + 1e-30); srt = np.sort(lq); m_ = len(srt); cs = np.cumsum(srt); ii = np.arange(1, m_)
    thr = srt[np.argmax(ii * (m_ - ii) * (cs[:-1] / ii - (cs[-1] - cs[:-1]) / (m_ - ii)) ** 2)]; T = T[lq >= thr] if (lq < thr).mean() < 0.2 else T
    used = np.unique(T); remap = np.full(len(X), -1); remap[used] = np.arange(len(used)); X = X[used]; T = remap[T]
    c = contact[u]; seg = np.full(len(u), -1)
    if c.any():
        top = np.asarray(Wt[u[c]].argmax(1)).ravel(); seg[np.flatnonzero(c)] = [SEG.get(wb[k], -1) for k in top]
    present = sorted(set(seg[seg >= 0]))
    glue_s = np.where(np.isin(seg, [present[0], present[-1]]), seg, -1) if len(present) >= 2 else np.full(len(seg), -1)
    gi = np.flatnonzero(glue_s >= 0); d, nn = cKDTree(Vs[gi]).query(X) if len(gi) else (np.full(len(X), np.inf), np.zeros(len(X), int))
    glue = np.where(d <= h, glue_s[gi][nn], -1) if len(gi) else np.full(len(X), -1)
    print('%s: volume %.1f cm^3, area %.1f cm^2, tet edge %.1f mm -> %d nodes, %d tets; segments touched %s; glued nodes origin %d / insertion %d' % (
        name, 1e6 * vol, 1e4 * area, 1000 * h, len(X), len(T), present, (glue == (present[0] if present else -9)).sum(), (glue == (present[-1] if present else -9)).sum()), flush=True)
    np.savez(os.path.join(OUT, re.sub(r'[^A-Za-z0-9]+', '_', name).strip('_') + '.npz'), X=X, T=T, glue_seg=glue, V=Vs, F=Fs, name=name, edge=h)
