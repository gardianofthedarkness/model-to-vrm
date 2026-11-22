"""Each eye's gaze from its iris ring on the setting picture (one circle on a ball fixes the direction of focus):
  python gaze_from_rings.py <setting_head.png> <face_hi.png> <ab_free.npz> <twin_full.npz> <out_prefix>
  rings    SAM 3 'eye' on the picture; inside each eye, the dark part (Otsu on brightness) - its largest piece, the
           iris outline ring
  map      picture -> model face: rtmlib's 68 face points on the picture and on the model's front close-up (face_hi,
           orthographic, centre (0.017,-0.041,0.865), 0.13 m wide, 2048 px) give a similarity (RANSAC); then pixels -> (x, z)
  balls    the twin's eye pieces (parts 23, 25) fitted with spheres in T-pose, carried into the setting pose by the fit's
           Head bone turn (ab_free); each ring pixel lands on its sphere's front (the side facing the viewer, -y)
  gaze     a plane through each ring's points on the sphere; its normal (outward) is the gaze. The turn from the eye
           piece's own rest axis (sphere centre -> piece centre) to the gaze, in the head's frame, is applied to the twin."""
import sys, json, numpy as np, cv2, torch
from PIL import Image
pic_p, face_p, fit_p, tw_p, pre = sys.argv[1:6]
def otsu(v):
    v = np.sort(np.asarray(v, float)); n = len(v); c = np.cumsum(v); i = np.arange(1, n); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (n - i)
    return v[np.argmax(i * (n - i) * (m0 - m1) ** 2)]
# --- rings
from transformers import Sam3Processor, Sam3Model
proc = Sam3Processor.from_pretrained('facebook/sam3'); model = Sam3Model.from_pretrained('facebook/sam3', torch_dtype=torch.float16).eval().to('cuda')
im = Image.open(pic_p).convert('RGB'); A = np.asarray(im)
x = proc(images=im, text='eye', return_tensors='pt').to('cuda'); x['pixel_values'] = x['pixel_values'].half()
with torch.no_grad(): o = model(**x)
r = proc.post_process_instance_segmentation(o, threshold=0.5, mask_threshold=0.5, target_sizes=x['original_sizes'].tolist())[0]
eyes = [m.cpu().numpy().astype(bool) for m in r['masks']]; L = cv2.cvtColor(A, cv2.COLOR_RGB2LAB)[..., 0].astype(float)
rings = []
for m in eyes:
    d = m & (L <= otsu(L[m])); n, lab = cv2.connectedComponents(d.astype(np.uint8)); k = 1 + np.argmax([(lab == j).sum() for j in range(1, n)])
    rings.append(np.argwhere(lab == k)[:, ::-1].astype(float))                       # (x, y) pixels
rings.sort(key=lambda p: p[:, 0].mean())                                                # picture left first
print('GZ rings: %s pixels' % [len(p) for p in rings])
del model; torch.cuda.empty_cache()
# --- picture -> model face
from rtmlib import Wholebody
wb = Wholebody(mode='balanced', backend='onnxruntime', device='cpu')
def face68(img):
    big = cv2.copyMakeBorder(img, 300, 900, 300, 300, cv2.BORDER_CONSTANT, value=(255, 255, 255)); kp, sc = wb(big)
    k = int(np.argmax(sc[:, 23:91].mean(1))); return kp[k][23:91] - 300, sc[k][23:91]
# the two eyes anchor the map: their centres in the picture (SAM 3 boxes, *_pupils.json) and on the model's close-up
def eye_centres(js):
    E = json.load(open(js)); c = [[(e['box'][0] + e['box'][2]) / 2, (e['box'][1] + e['box'][3]) / 2] for e in E]; return np.array(sorted(c))
pa = eye_centres(pic_p.replace('_head.png', '_pupils.json')); fb = eye_centres(face_p.replace('_hi.png', '_pupils.json'))
M = cv2.estimateAffinePartial2D(pa.astype(np.float32), fb.astype(np.float32))[0]
print('GZ picture -> model face from the two eye centres: scale %.3f, turn %.1f deg' % (np.hypot(M[0, 0], M[0, 1]), np.degrees(np.arctan2(M[1, 0], M[0, 0]))))
def to_world_xz(p):
    q = p @ M[:, :2].T + M[:, 2]; return np.c_[0.017 + (q[:, 0] - 1024) / 2048 * 0.13, 0.865 - (q[:, 1] - 1024) / 2048 * 0.13]
# --- eye balls in the setting pose
F = np.load(fit_p); T = np.load(tw_p); names = [str(n).split(':')[-1] for n in F['names']]; h = names.index('Head')
Rh, H0h, Ph = F['Rg'][h], F['head0'][h], F['head'][h]; bend = lambda X: (X - H0h) @ Rh.T + Ph
balls = []
for p in (23, 25):
    X = T['V'][T['part'] == p]; s = np.linalg.lstsq(np.c_[2 * X, np.ones(len(X))], (X ** 2).sum(1), rcond=None)[0]; c = s[:3]; rad = np.sqrt(s[3] + c @ c)
    import os
    if os.path.exists('E:/figures/rikka/tripo/transfer/flame/anime_eyeballs.npy'):          # FLAME's eye scaled to the anime eye opening
        AB = np.load('E:/figures/rikka/tripo/transfer/flame/anime_eyeballs.npy'); j = int(np.argmin(np.abs(AB[:, 0] - X[:, 0].mean()))); c, rad = AB[j, :3], AB[j, 3]
    ax = np.array([0.0, -1.0, 0.0]); balls.append({'part': p, 'c_t': c, 'r': rad, 'ax_t': ax, 'c_s': bend(c), 'ax_s': Rh @ ax})
balls.sort(key=lambda b: b['c_s'][0])                                                   # model left (-x) first, as the picture's left
out = {}
for ring, b in zip(rings, balls):
    xz = to_world_xz(ring); c, rad = b['c_s'], b['r']; dx, dz = xz[:, 0] - c[0], xz[:, 1] - c[2]; ins = dx ** 2 + dz ** 2 < rad ** 2
    P = np.c_[xz[ins, 0], c[1] - np.sqrt(rad ** 2 - dx[ins] ** 2 - dz[ins] ** 2), xz[ins, 1]]   # the sphere's front (toward -y)
    cen = P.mean(0); _, _, Vt = np.linalg.svd(P - cen); nrm = Vt[2]; nrm *= np.sign(nrm @ b['ax_s']) or 1
    g_head = Rh.T @ nrm; a_head = b['ax_t']                                                     # both in the head's (T-pose) frame
    yaw = (np.degrees(np.arctan2(g_head[0], -g_head[1]) - np.arctan2(a_head[0], -a_head[1])) + 180) % 360 - 180; pitch = np.degrees(np.arcsin(np.clip(g_head[2], -1, 1)) - np.arcsin(np.clip(a_head[2], -1, 1)))
    out[b['part']] = {'ring_pts_on_ball': int(ins.sum()), 'of': len(ring), 'gaze_world': nrm.tolist(), 'gaze_head': g_head.tolist(), 'rest_axis_head': a_head.tolist(), 'yaw_deg': float(yaw), 'pitch_deg': float(pitch)}
    print('GZ eye part %d: %d of %d ring pixels on the ball; gaze vs rest axis: yaw %+.1f deg, pitch %+.1f deg' % (b['part'], ins.sum(), len(ring), yaw, pitch))
json.dump(out, open(pre + '_gaze.json', 'w'), indent=1)
# --- show: the picture's rings, and the twin's face with eyes at rest and turned to the gaze
vis = A[:, :, ::-1].copy()
for ring in rings: vis[ring[:, 1].astype(int), ring[:, 0].astype(int)] = (255, 0, 255)
sys.path.insert(0, __file__.replace('\\', '/').rsplit('/', 1)[0]); from quick_render import shade
V, Fc, Pt = T['V'], T['F'], T['part']; keepF = Fc[np.isin(Pt[Fc[:, 0]], [0, 9, 23, 25])]; cc = np.array([0.0, -0.03, 0.865])
def turn(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b); v = np.cross(a, b); cs = a @ b
    if np.linalg.norm(v) < 1e-9: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K / (1 + cs)
Vg = V.copy()
for b in balls:
    g = np.array(out[b['part']]['gaze_head']); Rr = turn(b['ax_t'], g); m = Pt == b['part']; Vg[m] = (V[m] - b['c_t']) @ Rr.T + b['c_t']
t1 = shade(V, keepF, 'front', 500, center=cc, scale=4500); t2 = shade(Vg, keepF, 'front', 500, center=cc, scale=4500)
cv2.putText(t1, 'twin eyes at rest', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2); cv2.putText(t2, 'turned to the picture gaze', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
pv = cv2.resize(vis[600:1150, 450:1000], (500, 500)); cv2.putText(pv, 'setting picture: iris rings', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
fh = cv2.imread(face_p)
for ring in rings:
    q = ring @ M[:, :2].T + M[:, 2]; ok = (q[:, 0] >= 0) & (q[:, 1] >= 0) & (q[:, 0] < 2048) & (q[:, 1] < 2048); fh[q[ok, 1].astype(int), q[ok, 0].astype(int)] = (255, 0, 255)
fm = cv2.resize(fh[500:1500, 500:1500], (500, 500)); cv2.putText(fm, 'rings mapped onto the model', (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
cv2.imwrite(pre + '_gaze.jpg', np.hstack([pv, fm, t1, t2])); print('GZ done')
