"""Anime gaze as the iris circle's place inside the eye (user, 2026-11-18): the iris centre moves within the eye's outline and
never past it; the virtual eyeball behind it is not bounded by the skull.
  python iris_place.py <setting_head.png> <out_prefix>
  eye      SAM 3 'eye' regions on the setting picture (each kept); its outline is the boundary
  iris     inside each eye, the dark part (Otsu on brightness), its largest piece; the circle is the ellipse fitted to it
  place    the iris centre in the eye's own frame: u along the eye's long axis (corner to corner), v across it, each
           divided by the eye's half-extent on that axis (so -1..1 spans the eye); a centre outside the outline is pulled
           back to the nearest point of the outline
  gaze     a virtual eyeball of radius = the eye's half-width (unbounded): yaw = asin(u), pitch = asin(v) in degrees
Writes <prefix>_iris.json and <prefix>_iris.jpg."""
import sys, json, numpy as np, cv2, torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
pic, pre = sys.argv[1:3]
def otsu(v):
    v = np.sort(np.asarray(v, float)); n = len(v); c = np.cumsum(v); i = np.arange(1, n); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (n - i)
    return v[np.argmax(i * (n - i) * (m0 - m1) ** 2)]
proc = Sam3Processor.from_pretrained('facebook/sam3'); model = Sam3Model.from_pretrained('facebook/sam3', torch_dtype=torch.float16).eval().to('cuda')
im = Image.open(pic).convert('RGB'); A = np.asarray(im); L = cv2.cvtColor(A, cv2.COLOR_RGB2LAB)[..., 0].astype(float)
x = proc(images=im, text='eye', return_tensors='pt').to('cuda'); x['pixel_values'] = x['pixel_values'].half()
with torch.no_grad(): o = model(**x)
r = proc.post_process_instance_segmentation(o, threshold=0.5, mask_threshold=0.5, target_sizes=x['original_sizes'].tolist())[0]
eyes = sorted([m.cpu().numpy().astype(bool) for m in r['masks']], key=lambda m: np.argwhere(m)[:, 1].mean())
vis = A[:, :, ::-1].copy(); res = []
for k, m in enumerate(eyes):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); outline = max(cs, key=cv2.contourArea)
    d = m & (L <= otsu(L[m])); n, lab = cv2.connectedComponents(d.astype(np.uint8)); big = 1 + np.argmax([(lab == j).sum() for j in range(1, n)])
    ci, _ = cv2.findContours((lab == big).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); (icx, icy), (iw, ih), ia = cv2.fitEllipse(max(ci, key=cv2.contourArea))
    P = np.argwhere(m)[:, ::-1].astype(float); c = P.mean(0); U, S, Vt = np.linalg.svd(P - c, full_matrices=False); ax_u, ax_v = Vt[0] * np.sign(Vt[0][0] or 1), Vt[1] * np.sign(Vt[1][1] or 1)
    pu, pv = (P - c) @ ax_u, (P - c) @ ax_v; hu, hv = (pu.max() - pu.min()) / 2, (pv.max() - pv.min()) / 2; cu, cv_ = (pu.max() + pu.min()) / 2, (pv.max() + pv.min()) / 2
    ic = np.array([icx, icy]); inside = cv2.pointPolygonTest(outline, (float(icx), float(icy)), False) >= 0
    if not inside: q = outline[:, 0, :].astype(float); ic = q[np.argmin(((q - ic) ** 2).sum(1))]          # pulled back onto the outline
    u = ((ic - c) @ ax_u - cu) / hu; v = ((ic - c) @ ax_v - cv_) / hv
    yaw, pitch = np.degrees(np.arcsin(np.clip(u, -1, 1))), np.degrees(np.arcsin(np.clip(-v, -1, 1)))   # picture y grows downward: up is -v
    res.append({'eye': k, 'iris_centre_px': ic.tolist(), 'iris_axes_px': [iw, ih], 'inside_outline': bool(inside), 'u': float(u), 'v': float(v), 'yaw_deg': float(yaw), 'pitch_deg': float(pitch),
                'eye_half_width_px': float(hu), 'eye_half_height_px': float(hv)})
    print('IP eye %d (picture %s): iris centre %s inside the outline: %s; u %+.2f v %+.2f -> yaw %+.1f deg, pitch %+.1f deg' % (k, 'left' if k == 0 else 'right', np.round(ic, 1), inside, u, v, yaw, pitch))
    cv2.drawContours(vis, [outline], -1, (0, 200, 0), 2); cv2.ellipse(vis, ((icx, icy), (iw, ih), ia), (255, 0, 255), 2); cv2.circle(vis, tuple(int(t) for t in ic), 5, (0, 0, 255), -1)
    e0 = c + ax_u * cu + ax_v * cv_; cv2.drawMarker(vis, tuple(int(t) for t in e0), (255, 120, 0), cv2.MARKER_CROSS, 18, 2)
json.dump(res, open(pre + '_iris.json', 'w'), indent=1)
ys, xs = np.nonzero(np.any(eyes, 0)); cv2.imwrite(pre + '_iris.jpg', vis[max(ys.min() - 120, 0):ys.max() + 120, max(xs.min() - 120, 0):xs.max() + 120])
