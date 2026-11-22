"""SAM 3 eye and iris masks, kept: python sam3_iris.py <picture.png> <out_prefix>
  eye   SAM 3 'eye' on the picture (every instance above its 0.5 cut)
  iris  SAM 3 'iris' on each eye enlarged to SAM 3's input size (picture read once); the best candidate is kept even when
        below the cut, and its score is reported; regions mapped back to picture pixels
Writes <prefix>_iris.npz (eye masks, iris masks, iris scores, fitted iris ellipses) and <prefix>_iris.jpg."""
import sys, numpy as np, cv2, torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
pic, pre = sys.argv[1:3]
proc = Sam3Processor.from_pretrained('facebook/sam3'); model = Sam3Model.from_pretrained('facebook/sam3', torch_dtype=torch.float16).eval().to('cuda')
def seg(img, text, keep_best=False):
    x = proc(images=img, text=text, return_tensors='pt').to('cuda'); x['pixel_values'] = x['pixel_values'].half()
    with torch.no_grad(): o = model(**x)
    r = proc.post_process_instance_segmentation(o, threshold=0.0 if keep_best else 0.5, mask_threshold=0.5, target_sizes=x['original_sizes'].tolist())[0]
    M = [m.cpu().numpy().astype(bool) for m in r['masks']]; S = [float(s) for s in r['scores']]
    if keep_best and M: k = int(np.argmax(S)); return [M[k]], [S[k]]
    return M, S
im = Image.open(pic).convert('RGB'); A = np.asarray(im); H, W = A.shape[:2]
eyes, es = seg(im, 'eye'); order = np.argsort([np.argwhere(m)[:, 1].mean() for m in eyes]); eyes = [eyes[i] for i in order]; es = [es[i] for i in order]
print('SI eyes %d %s' % (len(eyes), np.round(es, 2).tolist()), flush=True)
irises, iscores, ells, vis = [], [], [], A[:, :, ::-1].copy()
for k, m in enumerate(eyes):
    ys, xs = np.nonzero(m); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); w, h = x1 - x0, y1 - y0
    X0, X1, Y0, Y1 = max(int(x0 - w / 2), 0), min(int(x1 + w / 2), W - 1), max(int(y0 - h / 2), 0), min(int(y1 + h / 2), H - 1)
    crop = A[Y0:Y1 + 1, X0:X1 + 1]; f = 1008 / max(crop.shape[:2]); big = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    M, S = seg(Image.fromarray(big), 'iris', keep_best=True)
    full = np.zeros((H, W), bool)
    if M: full[Y0:Y1 + 1, X0:X1 + 1] = cv2.resize(M[0].astype(np.uint8), (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    irises.append(full); iscores.append(S[0] if S else 0.0)
    cs, _ = cv2.findContours(full.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    e = cv2.fitEllipse(max(cs, key=cv2.contourArea)) if cs and len(max(cs, key=cv2.contourArea)) >= 5 else ((0, 0), (0, 0), 0); ells.append([*e[0], *e[1], e[2]])
    vis[full] = (vis[full] * 0.4 + np.array([255, 120, 40]) * 0.6).astype(np.uint8)
    ce, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(vis, ce, -1, (0, 200, 0), 2)
    if e[1][0] > 0: cv2.ellipse(vis, e, (255, 0, 255), 2)
    print('SI eye %d: iris score %.2f, area %d px, ellipse centre %s axes %s' % (k, iscores[-1], full.sum(), np.round(e[0], 1), np.round(e[1], 1)), flush=True)
np.savez_compressed(pre + '_iris.npz', eyes=np.array(eyes), irises=np.array(irises), eye_scores=np.array(es), iris_scores=np.array(iscores), ellipses=np.array(ells))
ys, xs = np.nonzero(np.any(eyes, 0)); cv2.imwrite(pre + '_iris.jpg', vis[max(ys.min() - 100, 0):ys.max() + 100, max(xs.min() - 100, 0):xs.max() + 100])
