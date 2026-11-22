"""Eyes, irises and pupils on a face picture: SAM 3 (picture encoded once, every prompt reuses it) plus texture carving.
  python pupil_find.py <face.png> <out_prefix>
  eyes   SAM 3 'eye' on the whole picture, each instance kept
  iris   SAM 3 'iris' on each enlarged eye; where it finds none, the eye's darker part (Otsu on brightness L inside the
         eye) - the dark blob nearest the eye's centre, so the lash line along the rim is left out
  pupil  inside the iris, the darker part again (Otsu on L) - the blob nearest the iris centre: the pupil marks gaze
Writes <prefix>_pupils.png and <prefix>_pupils.json (eye boxes, iris and pupil centroids and areas in picture pixels)."""
import sys, json, time, numpy as np, cv2, torch
from PIL import Image
from transformers import Sam3Processor, Sam3Model
img_p, pre = sys.argv[1], sys.argv[2]; DEV = 'cuda'
proc = Sam3Processor.from_pretrained('facebook/sam3'); model = Sam3Model.from_pretrained('facebook/sam3', torch_dtype=torch.float16).eval().to(DEV)
def seg_many(img, prompts):
    x = proc(images=img, return_tensors='pt').to(DEV); pv = x['pixel_values'].half(); sizes = x['original_sizes'].tolist(); out = {}
    with torch.no_grad():
        ve = model.get_vision_features(pixel_values=pv)                                 # the picture, read once
        for p in prompts:
            t = proc(text=p, return_tensors='pt').to(DEV)
            o = model(vision_embeds=ve, input_ids=t['input_ids'], attention_mask=t['attention_mask'])
            r = proc.post_process_instance_segmentation(o, threshold=0.5, mask_threshold=0.5, target_sizes=sizes)[0]
            out[p] = ([m.cpu().numpy().astype(bool) for m in r['masks']], [float(s) for s in r['scores']])
    return out
def otsu(v):
    v = np.sort(np.asarray(v, float)); n = len(v); c = np.cumsum(v); i = np.arange(1, n); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (n - i)
    return v[np.argmax(i * (n - i) * (m0 - m1) ** 2)]
def dark_core(region, L, inner=False):
    """The darker part of region (Otsu on L), the connected blob nearest the region's centre; with inner=True only blobs
    that do not touch the region's edge (so an outline ring cannot be taken for a pupil)."""
    cut = otsu(L[region]); d = region & (L <= cut); n, lab = cv2.connectedComponents(d.astype(np.uint8))
    if n < 2: return d
    edge = region & ~cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    cy, cx = np.argwhere(region).mean(0); best, bd = 0, 1e18
    for k in range(1, n):
        yx = np.argwhere(lab == k)
        if len(yx) < 4: continue
        if inner and (lab[edge] == k).any(): continue
        dd = ((yx.mean(0) - [cy, cx]) ** 2).sum()
        if dd < bd: best, bd = k, dd
    return lab == best
t0 = time.time(); im = Image.open(img_p).convert('RGB'); A = np.asarray(im); H, W = A.shape[:2]
eyes, scores = seg_many(im, ['eye'])['eye']; print('PF eyes %d %s (%.0f s)' % (len(eyes), [round(s, 2) for s in scores], time.time() - t0), flush=True)
rec, tiles = [], []
for k, m in enumerate(eyes):
    ys, xs = np.nonzero(m); x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max(); w, h = x1 - x0, y1 - y0
    X0, X1, Y0, Y1 = max(int(x0 - w / 2), 0), min(int(x1 + w / 2), W - 1), max(int(y0 - h / 2), 0), min(int(y1 + h / 2), H - 1)
    crop = A[Y0:Y1 + 1, X0:X1 + 1]; f = 1008 / max(crop.shape[:2]); big = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    L = cv2.cvtColor(big, cv2.COLOR_RGB2LAB)[..., 0].astype(float)
    eye_big = cv2.resize(m[Y0:Y1 + 1, X0:X1 + 1].astype(np.uint8), (big.shape[1], big.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
    got = seg_many(Image.fromarray(big), ['iris'])['iris']
    if got[0]: iris = np.any(got[0], 0); src = 'SAM 3 (%.2f)' % max(got[1])
    else:
        d_ = eye_big & (L <= otsu(L[eye_big])); n_, lb_ = cv2.connectedComponents(d_.astype(np.uint8)); big_k = 1 + np.argmax([(lb_ == j).sum() for j in range(1, n_)]) if n_ > 1 else 0
        iris = cv2.morphologyEx((lb_ == big_k).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
        cs_, _ = cv2.findContours(iris.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); iris = cv2.drawContours(np.zeros_like(iris, np.uint8), cs_, -1, 1, -1).astype(bool); src = 'carved from the eye'
    pupil = dark_core(iris, L, inner=True)
    vis = big.copy(); vis[iris] = (vis[iris] * 0.5 + np.array([60, 120, 255]) * 0.5).astype(np.uint8); vis[pupil] = (vis[pupil] * 0.3 + np.array([255, 40, 160]) * 0.7).astype(np.uint8)
    cnt = lambda q: [float(X0 + np.argwhere(q)[:, 1].mean() / f), float(Y0 + np.argwhere(q)[:, 0].mean() / f)]
    pc, ic = cnt(pupil), cnt(iris); r = {'eye': k, 'box': [X0, Y0, X1, Y1], 'iris_from': src, 'iris_c': ic, 'pupil_c': pc, 'iris_area': float(iris.sum() / f / f), 'pupil_area': float(pupil.sum() / f / f)}
    rec.append(r); print('PF eye %d: iris %s, pupil centre %s (area %.0f px, iris %.0f px)' % (k, src, np.round(pc, 1), r['pupil_area'], r['iris_area']), flush=True)
    v = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR); cv2.putText(v, 'eye %d: iris %s' % (k, src), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2); tiles.append(cv2.resize(v, (504, 504)))
cv2.imwrite(pre + '_pupils.png', np.hstack(tiles)); json.dump(rec, open(pre + '_pupils.json', 'w'), indent=1)
print('PF done in %.0f s' % (time.time() - t0))
