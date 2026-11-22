"""Split each contact patch into sub-contacts by the SAM 3 concept pair of its rays (one patch can hold a garment's own
folds, thigh on thigh and skirt on thigh at once):
  python contacts_split.py <real dir>            -> contacts/sub.npz, contacts/sub.json
  python contacts_split.py <real dir> cards      -> contacts/sub_cards/sub_<k>.jpg (after show_subcontacts.py rendered)
  sub-contact  the rays of one patch whose (source material, hit material) is one unordered pair; groups below the Otsu split
               of the log group sizes are dropped
  sides        different materials: side = material; the same material: the main axis of the normals, as for patches
  hints        per side material, its top SAM 3 concepts, points, nearest skeleton joint; the view that sees most of the group"""
import sys, os, json, numpy as np, cv2
from scipy.spatial import cKDTree
R = sys.argv[1]; j = lambda *p: os.path.join(R, *p); mode = sys.argv[2] if len(sys.argv) > 2 else 'split'
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[0] if m else 0
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
V = np.load(j('parts', 'orig_colours.npz'))['V']; VZ = np.load(j('body', 'vis.npz')); vn = [str(x) for x in VZ['names']]; cams = json.load(open(j('body', 'cameras.json')))
def proj(name, idx):
    cm = cams[name]; Rr = cm['res']; q = np.c_[V[idx], np.ones(len(idx))] @ np.array(cm['matrix_inv']).T
    u = ((q[:, 0] / cm['scale'] + 0.5) * Rr).astype(int); v = ((0.5 - q[:, 1] / cm['scale']) * Rr).astype(int)
    ok = VZ['vis'][vn.index(name)][idx] & (u >= 0) & (v >= 0) & (u < Rr) & (v < Rr); return u[ok], v[ok]
if mode == 'cards':
    S = np.load(j('contacts', 'sub.npz')); G = json.load(open(j('contacts', 'sub.json'))); od = j('contacts', 'sub_cards'); os.makedirs(od, exist_ok=True)
    for g in G:
        k = g['sub']; m = S['group'] == k; idx, sd = S['vert'][m], S['side'][m]
        im = cv2.imread(j('body', 'views', g['view'] + '.png')) if os.path.exists(j('body', 'views', g['view'] + '.png')) else cv2.imread(j('body', g['view'] + '.png'))
        dim = (im * 0.55 + 255 * 0.45).astype(np.uint8)
        for s, colr in ((0, (255, 0, 255)), (1, (255, 255, 0))):
            u, v = proj(g['view'], idx[sd == s]); mk = np.zeros(im.shape[:2], np.uint8); mk[v, u] = 255; dim[cv2.dilate(mk, np.ones((5, 5), np.uint8)) > 0] = colr
        tiles = [cv2.resize(dim, (480, 480))] + [cv2.resize(cv2.imread(os.path.join(j('contacts', 'sub_renders'), 's%03d_%s.png' % (k, t))), (480, 480)) for t in ('close', 'section')]
        card = np.hstack(tiles); cv2.putText(card, 'contact %d: A magenta, B cyan  (textured | close-up | cut through)' % k, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.imwrite(os.path.join(od, 'sub_%d.jpg' % k), card, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print('SPL cards %d' % len(G)); sys.exit()
C = np.load(j('contacts', 'combined.npz')); lab, mat, names, MN = C['concept'], C['material'], [str(x) for x in C['names']], [str(x) for x in C['materials']]
N = np.load(j('parts', 'orig_normals.npz'))['N']; pid = np.load(j('contacts', 'patches.npz'))['patch']; Rz = np.load(j('contacts', 'rays.npz')); gap, hit = Rz['gap'], Rz['hit_vertex']
SK = np.load(j('body', 'skeleton.npz')); jt = cKDTree(SK['heads']); JL = [str(x) for x in SK['labels']]
fin = np.isfinite(gap); thr = np.exp(otsu(np.log(gap[fin]))); ray = np.flatnonzero(fin & (gap < thr) & (pid >= 0) & (lab >= 0) & (lab[np.maximum(hit, 0)] >= 0))
a, b = mat[ray], mat[hit[ray]]; key = pid[ray] * len(MN) ** 2 + np.minimum(a, b) * len(MN) + np.maximum(a, b)
uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True); split = np.exp(otsu(np.log(cnt))); keep = np.flatnonzero(cnt > split); keep = keep[np.argsort(-cnt[keep])]
print('SPL %d rays in %d (patch, material pair) groups, kept %d above %d rays' % (len(ray), len(uk), len(keep), int(split)))
VV, GG, SS, out = [], [], [], []
for k, g in enumerate(keep):
    r = ray[inv == g]; idx = np.unique(np.r_[r, hit[r]]); ca, cb = sorted((int(mat[r[0]]), int(mat[hit[r[0]]])))
    if ca != cb: sd = (mat[idx] == cb).astype(int)
    else: n = N[idx]; w, e = np.linalg.eigh(n.T @ n); sd = (n @ e[:, -1] < 0).astype(int)
    VV.append(idx); GG.append(np.full(len(idx), k)); SS.append(sd)
    best = max(vn, key=lambda nm: VZ['vis'][vn.index(nm)][idx].sum())
    rec = {'sub': k, 'patch': int(pid[r[0]]), 'rays': int(len(r)), 'pair': '%s-%s' % (MN[ca], MN[cb]),
           'centre': V[idx].mean(0).round(4).tolist(), 'extent': float(np.ptp(V[idx], 0).max()), 'gap_median_m': round(float(np.median(gap[r])), 4), 'view': best, 'az': cams[best]['az']}
    for s, nm in ((0, 'A'), (1, 'B')):
        m = idx[sd == s]; jn = np.bincount(jt.query(V[m])[1], minlength=len(JL)) if len(m) else np.zeros(1, int)
        cn = np.bincount(lab[m], minlength=len(names)) if len(m) else np.zeros(len(names)); co = np.argsort(-cn)[:3]
        rec[nm] = {'concepts': ', '.join('%s %d%%' % (names[i], 100 * cn[i] / max(cn.sum(), 1)) for i in co if cn[i]), 'material': MN[int(np.bincount(mat[m]).argmax())] if len(m) else None, 'points': int(len(m)), 'near_joint': JL[int(jn.argmax())] if len(m) else None}
    out.append(rec)
    print('SPL sub %2d patch %2d %-12s rays %5d  A %s/%s (%s)  B %s/%s (%s)' % (k, rec['patch'], rec['pair'], rec['rays'], rec['A']['material'], rec['A']['concepts'], rec['A']['near_joint'], rec['B']['material'], rec['B']['concepts'], rec['B']['near_joint']))
np.savez(j('contacts', 'sub.npz'), vert=np.concatenate(VV), group=np.concatenate(GG), side=np.concatenate(SS)); json.dump(out, open(j('contacts', 'sub.json'), 'w'), indent=1)
print('SPL done: %d sub-contacts' % len(out))
