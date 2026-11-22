"""Critical points of the intricate ends from their zoomed views (part_views.py), triangulated to 3D:
  E:\\splat\\.venv\\Scripts\\python.exe zoom_joints3d.py <parts_zoom dir> <region> [tex|clay]
  hand_r / hand_l   RTMPose hand (rtmlib, 21 points: wrist, 4 per finger from base to tip)
  head              RTMW whole-body (133 points) read on the head alone: the 68 face points (eye centres = mean of
                    each eye's 6 contour points, written as l_eye / r_eye)
  per view          box = the rendered part's own pixels (the view shows the region alone), one detection
  3D (skin)         face points lie on the skin: in each view the detected pixel is cast onto the front-most region
                    surface under it (orthographic ray, within one median mesh edge), if that surface faces the camera;
                    the views' surface points are combined by their score-weighted geometric median, views farther
                    from it than the Otsu split of the distances dropped. Second pass: the face's forward direction = the mean
                    surface normal at the first pass's points, and each view is weighted by cos^2 of its angle to it (side
                    and back views put far-side points toward the midline). (Triangulating them failed: from behind a
                    face still gets confident points, and filtering views left 2 near-parallel ones with unstable depth.)
  3D (joints)       hand points are joint centres (inside the finger, seen from every side): all views, then views
                    whose residual is above the Otsu split dropped, twice
writes <dir>/<region>/points3d.npz (X, conf, names, views used), <dir>/<region>_points_check.jpg (detected green,
reprojected 3D red, on the best-scored views)"""
import sys, os, json, numpy as np, cv2
from rtmlib import RTMPose
D, reg = sys.argv[1], sys.argv[2]; mode = sys.argv[3] if len(sys.argv) > 3 else 'tex'; od = os.path.join(D, reg)
cams = json.load(open(os.path.join(od, 'cameras.json'))); ck = os.path.expanduser('~/.cache/rtmlib/hub/checkpoints/')
if reg.startswith('hand'):
    det = RTMPose(ck + 'rtmpose-m_simcc-hand5_pt-aic-coco_210e-256x256-74fb594_20230320.onnx', model_input_size=(256, 256), backend='onnxruntime', device='cpu')
    keep = np.arange(21); fing = ['thumb', 'index', 'middle', 'ring', 'little']
    NM = ['wrist'] + ['%s%d' % (f, i) for f in fing for i in range(1, 5)]
    EDGES = [(0, 1 + 4 * f) for f in range(5)] + [(1 + 4 * f + i, 2 + 4 * f + i) for f in range(5) for i in range(3)]
else:
    det = RTMPose(ck + 'rtmw-dw-x-l_simcc-cocktail14_270e-256x192_20231122.onnx', model_input_size=(192, 256), backend='onnxruntime', device='cpu')
    keep = np.r_[23:91]; NM = ['face%d' % i for i in range(68)]       # the 68 face points (body eye/ear points: left/right swapped from the side; elf ears are not human ears)
    EDGES = [(i, i + 1) for i in range(67) if i not in (16, 21, 26, 30, 35, 41, 47, 59)]
K = len(keep)
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[-1] if m else 0
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
obs = []
for n in sorted(c for c in cams if c.startswith(mode + '_')):
    img = cv2.imread(os.path.join(od, n + '.png')); ys, xs = np.where(img.min(2) < 250)
    if not len(xs): continue
    k, s = det(img, bboxes=[[xs.min(), ys.min(), xs.max(), ys.max()]]); k, s = k[0][keep], s[0][keep]
    cm = cams[n]; M = np.array(cm['matrix_inv']); R, S = cm['res'], cm['scale']
    obs.append((n, M[:2], np.c_[(k[:, 0] / R - 0.5) * S, (0.5 - k[:, 1] / R) * S], s, k, img))
from scipy.spatial import cKDTree
MD = os.path.dirname(os.path.abspath(D)); vid = np.load(os.path.join(D, reg + '_vids.npy'))   # the region's model vertices and normals
PV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'][vid].astype(np.float64); PN = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['N'][vid].astype(np.float64)
PF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']; inr = np.zeros(len(np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V']), bool); inr[vid] = True
PFr = PF[inr[PF].all(1)]; EDG = float(np.median(np.linalg.norm(np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'][PFr[:, 0]] - np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'][PFr[:, 1]], axis=1)))
ptree = cKDTree(PV); TOC = np.array([np.array(cams[o[0]]['matrix_inv'])[2, :3] for o in obs])   # per view: unit direction toward the camera
PRJ = []                                                           # per view: the region's vertices in camera coordinates and a 2D tree
for o in obs:
    M3 = np.array(cams[o[0]]['matrix_inv'])[:3]; q = PV @ M3[:, :3].T + M3[:, 3]; PRJ.append((q, cKDTree(q[:, :2])))
def solve(p, kp):
    A_ = np.array([o[1][r, :3] for o in obs for r in range(2)]); b_ = np.array([o[2][p, r] - o[1][r, 3] for o in obs for r in range(2)])
    w_ = np.repeat([o[3][p] for o in obs], 2); m = np.repeat(kp, 2); W = np.sqrt(w_[m]); x = np.linalg.lstsq(A_[m] * W[:, None], b_[m] * W, rcond=None)[0]
    return x, np.sqrt(((A_ @ x - b_) ** 2).reshape(-1, 2).sum(1))
def wmedian(P, w):                                                 # weighted geometric median (Weiszfeld)
    x = (P * w[:, None]).sum(0) / w.sum()
    for _ in range(50):
        d = np.linalg.norm(P - x, axis=1) + 1e-9; x = (P * (w / d)[:, None]).sum(0) / (w / d).sum()
    return x
ON_SURFACE = not reg.startswith('hand')                         # face points lie on the skin; hand points are joint centres inside the fingers
X = np.zeros((K, 3)); conf = np.zeros(K); used = np.zeros((K, len(obs)), bool); rms = np.zeros(K); FWD = None
for PASS in range(2 if ON_SURFACE else 1):                        # skin points, pass 2: the face's forward direction = mean surface normal at pass 1's points;
    used[:] = False                                                # views are then weighted by how frontal they are, so side and back views (which put far-side
    for p in range(K):
        if not ON_SURFACE:                                             # joint centres: triangulate over all views, drop views above the Otsu split of the residuals, twice
            kp = np.ones(len(obs), bool)
            for _ in range(2): x, res = solve(p, kp); kp = res <= otsu(res)
            x, res = solve(p, kp); X[p] = x; used[p] = kp; rms[p] = float(np.sqrt(np.mean(res[kp] ** 2)))
        else:                                                          # skin points: each view's pixel cast onto the front-most region surface under it
            C, w, vv = [], [], []                                      # (within one median edge of the ray), kept when that surface faces the camera
            for v, (q, t2) in enumerate(PRJ):
                ii = t2.query_ball_point(obs[v][2][p], EDG)
                if not ii: continue
                h = ii[int(np.argmax(q[ii, 2]))]
                if TOC[v] @ PN[h] <= 0: continue
                C.append(PV[h]); w.append(obs[v][3][p]); vv.append(v)
            C, w, vv = np.array(C), np.array(w), np.array(vv, int)
            if FWD is not None: w = w * np.maximum(TOC[vv] @ FWD, 0) ** 2; C, w, vv = C[w > 0], w[w > 0], vv[w > 0]   # second pass: a view counts by how frontal it is (cos^2)
            if not len(C): X[p] = np.nan; continue
            x = wmedian(C, w); d = np.linalg.norm(C - x, axis=1); kp = d <= otsu(d) if len(C) > 2 else np.ones(len(C), bool)   # views far from the others' agreement dropped
            x = wmedian(C[kp], w[kp])
            X[p] = x; used[p, vv[kp]] = True; rms[p] = float(np.sqrt(np.mean(np.linalg.norm(C[kp] - x, axis=1) ** 2)))
        conf[p] = float(np.mean([obs[v][3][p] for v in np.flatnonzero(used[p])])) if used[p].any() else 0
    if ON_SURFACE and PASS == 0:                                   # points on whatever they do see, toward the midline: the atlas's eye corners came out 63 mm apart,
        okp_ = np.isfinite(X).all(1); FWD = PN[ptree.query(X[okp_])[1]].mean(0); FWD /= np.linalg.norm(FWD)   # 84 mm in its front view) count little
if not reg.startswith('hand'):                                    # eye centres from the eye contours (face 36-41 right, 42-47 left in the image's subject)
    X = np.r_[X, np.nanmean(X[42:48], 0, keepdims=True), np.nanmean(X[36:42], 0, keepdims=True)]; NM = NM + ['l_eye', 'r_eye']
    conf = np.r_[conf, conf[42:48].mean(), conf[36:42].mean()]; rms = np.r_[rms, rms[42:48].mean(), rms[36:42].mean()]; used = np.r_[used, used[42:43], used[36:37]]
np.savez(os.path.join(od, 'points3d.npz'), X=X, conf=conf, rms=rms, names=np.array(NM), used=used, views=np.array([o[0] for o in obs]))
okp = np.isfinite(X).all(1); dsk = ptree.query(X[okp])[0]                      # a face point with no surface under it in any view (an open eye socket, a hole) stays nan
print('ZJ distance to the skin: median %.2f mm, p90 %.2f mm (face points should lie on it; hand joints inside); points without a surface: %s' % (1000 * np.median(dsk), 1000 * np.percentile(dsk, 90), [NM[i] for i in np.flatnonzero(~okp)]))
print('ZJ %s: %d points from %d views; mean score %.2f; views kept per point median %d; spread (rms) median %.2f mm (worst %s %.2f mm)' % (
    reg, K, len(obs), conf.mean(), np.median(used.sum(1)), 1000 * np.nanmedian(rms), NM[int(np.nanargmax(rms))], 1000 * np.nanmax(rms)))
tiles = []
for vi in np.argsort([-o[3].mean() for o in obs])[:8]:
    n, A, q, s, k, img = obs[vi]; im = img.copy(); cm = cams[n]; R, S = cm['res'], cm['scale']; pr = X @ A[:, :3].T + A[:, 3]
    px = np.nan_to_num(np.c_[(pr[:, 0] / S + 0.5) * R, (0.5 - pr[:, 1] / S) * R], nan=-1e4).astype(int)
    for a, b in EDGES: cv2.line(im, tuple(px[a]), tuple(px[b]), (40, 40, 220), 2)
    for p in range(K): cv2.circle(im, tuple(k[p].astype(int)), 6, (0, 190, 0), -1); cv2.circle(im, tuple(px[p]), 4, (0, 0, 255), -1)
    cv2.putText(im, n, (10, 40), 0, 1.1, (0, 0, 0), 2); tiles.append(cv2.resize(im, (380, 380)))
tiles += [np.full((380, 380, 3), 255, np.uint8)] * (8 - len(tiles))
cv2.imwrite(os.path.join(D, reg + '_points_check.jpg'), np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])]))
