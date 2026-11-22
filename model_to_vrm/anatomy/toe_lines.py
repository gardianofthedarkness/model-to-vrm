"""The toes of one foot as separate parts of the surface, each with a centre line and a tip, from SAM 3's "toe" instances
(parts/part_concept_masks.py) on the foot's close-ups. No toe detector exists; this is the toes' "points and lines".
  E:\\splat\\.venv\\Scripts\\python.exe toe_lines.py <model dir> <region>           region: foot_r foot_l
  instances   the instances SAM 3 is sure of: one Otsu split of the scores pooled over all views.
  on the mesh each instance -> the foot's vertices visible in that view (nearest surface per pixel block, within two mesh
              edges) whose pixel lies in the mask.
  several     an instance that covers two or more separate instances of another view is several toes at once; left out
  same toe    two instances from different views are the same toe when most of the smaller one's vertices are in the
              other; groups of linked instances seen in most of the views are toes. A vertex belongs to the toe most
              of its views say.
  rules       exactly five toes, or the step fails and says so. Numbered first to fifth from the detected big-toe point
              to the detected small-toe point (the atlas numbers the big toe first); the first must be the largest.
  lines       per toe: its main axis (first principal direction of its vertices, pointing away from the heel), its
              centre, and its tip (the far end along the axis).
Writes <model dir>/parts_zoom/<region>/toes.npz (label per foot vertex, centre, axis, tip, length per toe) and
<region>_toes.jpg."""
import sys, os, json, numpy as np, cv2
from skimage.filters import threshold_otsu
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
MD, REG = sys.argv[1:3]; PZ = os.path.join(MD, 'parts_zoom'); D = os.path.join(PZ, REG); sd = REG[-1]
def log(*a): print('TL', *a, flush=True)
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']; vid = np.load(os.path.join(PZ, REG + '_vids.npy')); V = MV[vid]
edge = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1))); cams = json.load(open(os.path.join(D, 'cameras.json')))
Z = np.load(os.path.join(MD, 'body', 'joints3d.npz'))['X']; o = 0 if sd == 'l' else 1; big, small, heel = Z[17 + 3 * o], Z[18 + 3 * o], Z[19 + 3 * o]      # COCO-WholeBody order
def cam_xyz(X, cam): M = np.array(cam['matrix_inv']); return X @ M[:3, :3].T + M[:3, 3]
def pix(c, cam): r = cam['res']; return np.c_[(c[:, 0] / cam['scale'] + 0.5) * r, (0.5 - c[:, 1] / cam['scale']) * r]
inst = []; views = sorted(f[:-4] for f in os.listdir(os.path.join(D, 'masks')) if f.endswith('.npz')); seen = np.zeros(len(V), int)
TH = float(threshold_otsu(np.concatenate([np.load(os.path.join(D, 'masks', n_ + '.npz'))['s_toe'] for n_ in views])))        # one split of the scores pooled over all views (a view from behind has no toe at all)
for name in views:
    m = np.load(os.path.join(D, 'masks', name + '.npz')); S = m['s_toe']; res = int(m['res']); M = np.unpackbits(m['m_toe'], axis=1)[:, :res * res].reshape(-1, res, res).astype(bool)
    keep = np.flatnonzero(S > TH); cam = cams[name]; c = cam_xyz(V, cam); p = pix(c, cam); blk = max(1, int(np.ceil(edge / (cam['scale'] / cam['res']))))
    bx = (p[:, 0] // blk).astype(int); by = (p[:, 1] // blk).astype(int); key = bx * 100000 + by; order = np.argsort(key); ku, st = np.unique(key[order], return_index=True)
    zmax = np.maximum.reduceat(c[order, 2], st); near = np.empty(len(V)); near[order] = np.repeat(zmax, np.diff(np.r_[st, len(V)])); vis = c[:, 2] >= near - 2 * edge   # the camera looks down its -z: larger z is nearer
    pi = np.clip(np.rint(p).astype(int), 0, res - 1); seen += vis
    for k in keep:
        ii = np.flatnonzero(vis & M[k][pi[:, 1], pi[:, 0]])
        if len(ii): inst.append((name, float(S[k]), ii))
    log('%s: %d instances above the split (%.2f): scores %s' % (name, len(keep), TH, np.round(np.sort(S[keep])[::-1], 2).tolist()))
# an instance that covers two or more separate instances of another view is a mask of several toes at once (SAM 3 also
# returns "the toes" as one instance); kept, it would link every toe into one group, so it is dropped
sets = [set(i[2].tolist()) for i in inst]; same = lambda a_, b_: len(sets[a_] & sets[b_]) > 0.5 * min(len(sets[a_]), len(sets[b_]))
multi = set()
for a_ in range(len(inst)):
    for w in {i[0] for i in inst} - {inst[a_][0]}:
        inside = [b_ for b_ in range(len(inst)) if inst[b_][0] == w and len(sets[a_] & sets[b_]) > 0.5 * len(sets[b_]) and len(sets[b_]) < len(sets[a_])]
        if any(not same(b1, b2) for i1, b1 in enumerate(inside) for b2 in inside[i1 + 1:]): multi.add(a_); break
if multi: log('%d instances cover several toes at once; left out' % len(multi))
inst = [x for k_, x in enumerate(inst) if k_ not in multi]
n = len(inst); sets = [set(i[2].tolist()) for i in inst]; rows, cols = [], []
for a in range(n):
    for b in range(a + 1, n):
        if inst[a][0] != inst[b][0] and len(sets[a] & sets[b]) > 0.5 * min(len(sets[a]), len(sets[b])): rows.append(a); cols.append(b)
nc, lab = connected_components(coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)), directed=False)
nv = {c_: len({inst[i][0] for i in np.flatnonzero(lab == c_)}) for c_ in range(nc)}; nw = len({i[0] for i in inst}); toes = [c_ for c_ in range(nc) if nv[c_] > nw / 2]                 # most of the views that show any toe
log('%d instances in %d views -> %d groups; seen in most views: %d (views per group: %s)' % (n, len(views), nc, len(toes), sorted(nv.values(), reverse=True)))
if len(toes) != 5: sys.exit('TL RULE FAILED: %d toes found, not five' % len(toes))
votes = np.zeros((len(V), 5), int)
for t, c_ in enumerate(toes):
    for i in np.flatnonzero(lab == c_): votes[inst[i][2], t] += 1
best = votes.argmax(1); L = np.where(votes.max(1) > votes.sum(1) / 2, best, -1); L[votes.sum(1) == 0] = -1
cen = np.array([V[L == t].mean(0) for t in range(5)]); u = (small - big) / np.linalg.norm(small - big); order = np.argsort((cen - big) @ u); remap = np.empty(5, int); remap[order] = np.arange(5); L = np.where(L >= 0, remap[np.maximum(L, 0)], -1)
cen = np.array([V[L == t].mean(0) for t in range(5)]); size = np.array([(L == t).sum() for t in range(5)]); axis, tip, length = [], [], []
for t in range(5):
    P = V[L == t] - cen[t]; w, U = np.linalg.eigh(P.T @ P); a = U[:, -1]; a = a if a @ (cen[t] - heel) > 0 else -a; s_ = P @ a; k = np.argsort(s_)[-max(3, len(P) // 100):]
    axis.append(a); tip.append(V[L == t][k].mean(0)); length.append(float(s_.max() - s_.min()))
axis, tip = np.array(axis), np.array(tip)
log('toes first to fifth: vertices %s; visible length mm %s; first is the largest: %s' % (size.tolist(), np.round(1000 * np.array(length), 1).tolist(), bool(size.argmax() == 0)))
if size.argmax() != 0: log('RULE WARNING: the first toe is not the largest')
np.savez(os.path.join(D, 'toes.npz'), label=L, vids=vid, centre=cen, axis=axis, tip=tip, length=np.array(length), size=size)
# picture: the view from above, toes coloured, centre lines and tips
name = 'tex_e+90_a000' if 'tex_e+90_a000' in cams else views[0]; cam = cams[name]; im = cv2.imread(os.path.join(D, name + '.png')); COL = [(60, 60, 230), (40, 170, 240), (60, 200, 60), (220, 160, 40), (200, 60, 200)]
p = pix(cam_xyz(V, cam), cam).astype(int)
for t in range(5):
    for x, y in p[L == t][::3]: cv2.circle(im, (int(x), int(y)), 1, COL[t], -1)
for t in range(5):
    a = pix(cam_xyz(np.array([tip[t] - axis[t] * length[t], tip[t]]), cam), cam).astype(int); cv2.line(im, tuple(a[0]), tuple(a[1]), (0, 0, 0), 2, cv2.LINE_AA); cv2.circle(im, tuple(a[1]), 6, (255, 255, 255), -1, cv2.LINE_AA); cv2.circle(im, tuple(a[1]), 6, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(im, str(t + 1), (a[1][0] - 8, a[1][1] + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)
ys, xs = np.nonzero((im < 250).any(2)); im = im[max(0, ys.min() - 20):ys.max() + 50, max(0, xs.min() - 20):xs.max() + 20]; cv2.imwrite(os.path.join(PZ, REG + '_toes.jpg'), im, [cv2.IMWRITE_JPEG_QUALITY, 90]); log('done')
