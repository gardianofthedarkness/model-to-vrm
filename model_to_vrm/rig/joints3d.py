"""3D body keypoints from whole-body renders: RTMW (rtmlib, 133 COCO-WholeBody points) on every view, then each point
triangulated from the orthographic cameras by score-weighted least squares, views that disagree dropped (a view's
residual above the Otsu split of all residuals of that point is an outlier, e.g. a left/right swap from behind):
  python joints3d.py <views dir> [tex|clay]   ->  <dir>/joints3d.npz (X: 133x3, conf: 133, views used), joints_check.jpg"""
import sys, os, json, numpy as np, cv2
from rtmlib import Wholebody
D = sys.argv[1]; mode = sys.argv[2] if len(sys.argv) > 2 else 'tex'; cams = json.load(open(os.path.join(D, 'cameras.json')))
names = sorted(n for n in cams if n.startswith(mode + '_')); det = Wholebody(mode='balanced', backend='onnxruntime', device='cpu')
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[-1] if m else 0
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
obs = []                                                     # per view: rows of the camera (2x4) per keypoint, pixel, score
for n in names:
    img = cv2.imread(os.path.join(D, n + '.png')); k, s = det(img)
    if len(k) == 0: continue
    j = int(np.argmax(s.mean(1))); cm = cams[n]; M = np.array(cm['matrix_inv']); R, S = cm['res'], cm['scale']
    # pixel (u, v) = ((q.x / S + 0.5) R, (0.5 - q.y / S) R) with q = M @ [X, 1]  ->  q.x = (u / R - 0.5) S, q.y = (0.5 - v / R) S
    obs.append((n, M[:2], np.c_[(k[j][:, 0] / R - 0.5) * S, (0.5 - k[j][:, 1] / R) * S], s[j], k[j], img))
K = 133
# COCO-WholeBody left/right pairs (body, feet, hands); a detector looking from the side or back often swaps them
PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 20), (18, 21), (19, 22)] + [(91 + i, 112 + i) for i in range(21)]
SW = np.arange(K)
for a_, b_ in PAIRS: SW[a_], SW[b_] = b_, a_
def tri(sel, flips):
    """score-weighted least squares per keypoint over the views in sel, each view read swapped if flips[v]"""
    X = np.zeros((K, 3)); conf = np.zeros(K); used = np.zeros((K, len(obs)), bool)
    for p in range(K):
        rows, rhs, w, vs = [], [], [], []
        for vi in sel:
            n, A, q, s_, _, _ = obs[vi]; pp = SW[p] if flips[vi] else p
            for r in range(2): rows.append(A[r, :3]); rhs.append(q[pp, r] - A[r, 3]); w.append(s_[pp])
            vs.append(vi)
        A_, b_, w_ = np.array(rows), np.array(rhs), np.array(w); keep = np.ones(len(vs), bool)
        for _ in range(2):
            m = np.repeat(keep, 2); W = np.sqrt(w_[m])
            if m.sum() < 4: break
            x = np.linalg.lstsq(A_[m] * W[:, None], b_[m] * W, rcond=None)[0]
            res = np.sqrt(((A_ @ x - b_) ** 2).reshape(-1, 2).sum(1)); keep = res <= max(otsu(res), np.median(res))
        X[p] = x; used[p, np.array(vs)[keep]] = True
        conf[p] = float(np.mean([obs[v][3][SW[p] if flips[v] else p] for v in np.array(vs)[keep]])) if keep.any() else 0
    return X, conf, used
def reproj(X, vi, flip):
    n, A, q, s_, _, _ = obs[vi]; idx = SW if flip else np.arange(K); pr = X @ A[:, :3].T + A[:, 3]
    e = np.linalg.norm(pr[:23] - q[idx][:23], axis=1); return float((e * s_[idx][:23]).sum() / s_[idx][:23].sum())
# 1. seed: the views where the face is seen best (nose + both eyes most confident) are taken as correctly labelled
face = np.array([o[3][[0, 1, 2]].mean() for o in obs]); seed = [int(v) for v in np.flatnonzero(face >= otsu(face))]
flips = np.zeros(len(obs), bool); X, conf, used = tri(seed, flips)
# 2. every view: read as-is or swapped, whichever agrees with the seed skeleton; then all views together, twice
for _ in range(2):
    for vi in range(len(obs)): flips[vi] = reproj(X, vi, True) < reproj(X, vi, False)
    X, conf, used = tri(range(len(obs)), flips)
print('J3 seed views', [obs[v][0] for v in seed], 'swapped views', [obs[v][0] for v in np.flatnonzero(flips)])
np.savez(os.path.join(D, 'joints3d.npz'), X=X, conf=conf, used=used, views=np.array([o[0] for o in obs]))
tiles = []
for n, A, q, s, k, img in obs[:8]:
    im = img.copy(); cm = cams[n]; M = np.array(cm['matrix_inv']); qq = np.c_[X, np.ones(K)] @ M.T; R, S = cm['res'], cm['scale']
    for p in range(23):                                       # body + feet: detected (green) vs reprojected 3D (red)
        cv2.circle(im, (int(k[p, 0]), int(k[p, 1])), 6, (0, 200, 0), -1)
        cv2.circle(im, (int((qq[p, 0] / S + 0.5) * R), int((0.5 - qq[p, 1] / S) * R)), 4, (0, 0, 255), -1)
    tiles.append(cv2.resize(im, (384, 384)))
cv2.imwrite(os.path.join(D, 'joints_check.jpg'), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print('J3 done: %d views, body points mean conf %.2f' % (len(obs), conf[:23].mean()))
