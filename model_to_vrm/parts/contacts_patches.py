"""Contact patches with two sides, and a card per patch for the labelling agent:
  python contacts_patches.py <rays.npz> <orig_colours.npz> <orig_normals.npz> <views dir> <skeleton.npz> <out dir> [shares.npz]
  contact  a vertex whose outward ray hits a back-facing surface closer than the Otsu split of all such hit distances
           (log scale); its partner = the hit vertex
  patch    contact vertices and their partners, joined by partner links and by their 8 nearest neighbours among them;
           patches below the Otsu split of the log patch sizes are dropped (texture-scale noise)
  sides    per patch, the main axis of its normals splits it in two: side A = normals along the axis, side B = against
  hints    per side: points, median texture colour, nearest skeleton joint (most common), SAM 3's top concepts (mean share)
Writes patches.npz (patch id and side per vertex, -1 = none), patches.json and cards/patch_<k>.jpg (front, back and the
view where the patch is seen best; side A magenta, side B cyan)."""
import sys, os, json, numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
rp, cp, np_, vd, sp, od = sys.argv[1:7]; shp = sys.argv[7] if len(sys.argv) > 7 else None; os.makedirs(os.path.join(od, 'cards'), exist_ok=True)
Rz = np.load(rp); gap, hit = Rz['gap'], Rz['hit_vertex']; O = np.load(cp); V, COL = O['V'], O['C']; N = np.load(np_)['N']
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[-1] if m else 0
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
fin = np.isfinite(gap); thr = np.exp(otsu(np.log(gap[fin]))); src = np.flatnonzero(fin & (gap < thr))
print('CP hits %d, contact split %.4f m, contact vertices %d' % (fin.sum(), thr, len(src)))
S = np.unique(np.r_[src, hit[src]]); loc = -np.full(len(V), 1, int); loc[S] = np.arange(len(S))
_, nb = cKDTree(V[S]).query(V[S], k=9); r = np.r_[np.repeat(np.arange(len(S)), 8), loc[src]]; c = np.r_[nb[:, 1:].ravel(), loc[hit[src]]]
ncomp, comp = connected_components(coo_matrix((np.ones(len(r)), (r, c)), shape=(len(S), len(S))), directed=False)
sz = np.bincount(comp); keep = np.flatnonzero(sz > np.exp(otsu(np.log(sz)))); keep = keep[np.argsort(-sz[keep])]
print('CP patches %d, kept %d (size split %d points)' % (ncomp, len(keep), int(np.exp(otsu(np.log(sz))))))
SK = np.load(sp); J, JL = SK['heads'], [str(x) for x in SK['labels']]; jt = cKDTree(J)
SH = np.load(shp) if shp else None; con = [str(x) for x in SH['concepts']] if SH is not None else []
import json as _j; cams = _j.load(open(os.path.join(vd, 'cameras.json'))); VZ = np.load(os.path.join(vd, 'vis.npz')); vis = dict(zip([str(n) for n in VZ['names']], VZ['vis']))
pid = -np.ones(len(V), int); side = -np.ones(len(V), int); out = []
def proj(name, idx):
    cm = cams[name]; R = cm['res']; q = np.c_[V[idx], np.ones(len(idx))] @ np.array(cm['matrix_inv']).T
    u = ((q[:, 0] / cm['scale'] + 0.5) * R).astype(int); v = ((0.5 - q[:, 1] / cm['scale']) * R).astype(int); ok = vis[name][idx] & (u >= 0) & (v >= 0) & (u < R) & (v < R); return u[ok], v[ok]
for k, cc in enumerate(keep):
    idx = S[comp == cc]; n = N[idx]; w, e = np.linalg.eigh(n.T @ n); ax = e[:, -1]; sd = (n @ ax < 0).astype(int)
    pid[idx] = k; side[idx] = sd; rec = {'patch': k, 'points': int(len(idx)), 'centre': V[idx].mean(0).round(3).tolist(), 'gap_median_m': round(float(np.median(gap[idx][np.isfinite(gap[idx])])) if np.isfinite(gap[idx]).any() else 0, 4)}
    for s, nm in ((0, 'A'), (1, 'B')):
        m = idx[sd == s]
        if len(m) == 0: rec[nm] = None; continue
        jn = np.bincount(jt.query(V[m])[1], minlength=len(J)); h = {'points': int(len(m)), 'colour': '#%02x%02x%02x' % tuple(int(255 * x) for x in np.median(COL[m], 0)), 'near_joint': JL[int(jn.argmax())]}
        if SH is not None:
            ms = SH['share'][m].mean(0); o = np.argsort(-ms)[:3]; h['sam3'] = ', '.join('%s %.2f' % (con[i], ms[i]) for i in o)
        rec[nm] = h
    views = sorted(vis, key=lambda nm_: -len(proj(nm_, idx)[0])); tiles = []
    for name in [views[0], views[1], views[2]]:
        im = cv2.imread(os.path.join(vd, name + '.png')); dim = (im * 0.55 + 255 * 0.45).astype(np.uint8)
        for s, colr in ((0, (255, 0, 255)), (1, (255, 255, 0))):
            u, v = proj(name, idx[sd == s]); mk = np.zeros(im.shape[:2], np.uint8); mk[v, u] = 255; mk = cv2.dilate(mk, np.ones((5, 5), np.uint8)); dim[mk > 0] = colr
        tiles.append(cv2.resize(dim, (480, 480)))
    card = np.hstack(tiles); cv2.putText(card, 'patch %d: A magenta, B cyan' % k, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(od, 'cards', 'patch_%d.jpg' % k), card, [cv2.IMWRITE_JPEG_QUALITY, 85]); out.append(rec)
np.savez(os.path.join(od, 'patches.npz'), patch=pid, side=side); json.dump(out, open(os.path.join(od, 'patches.json'), 'w'), indent=1)
print('CP done: %d patches with cards' % len(out))
