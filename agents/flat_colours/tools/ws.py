"""Shared state of the labelling workspace (one folder per character).
  data.npz     rig points of the original model: V, part, colour (texture colour 0..1), full_to_rig (full-detail point -> rig point)
  owner.npy    which unit each rig point belongs to: a part id ('6') or a region carved out of it ('6.2')
  sheet.json   one entry per unit: name, category, material, stiffness, colour, movable, notes, candidates
The 16 textured views of the original (../views) give cards; nothing here invents a colour - candidates are measured."""
import os, json, numpy as np, cv2
W = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); T = os.path.dirname(W)
D = np.load(os.path.join(W, 'data.npz')); V, PART, COL, F2R = D['V'], D['part'], D['colour'], D['full_to_rig']
VIEWS = np.load(os.path.join(T, 'views', 'views.npz'))
def owner():
    p = os.path.join(W, 'owner.npy')
    return np.load(p) if os.path.exists(p) else PART.astype(str).astype('<U12')
def save_owner(o): np.save(os.path.join(W, 'owner.npy'), o)
def sheet():
    p = os.path.join(W, 'sheet.json'); return json.load(open(p)) if os.path.exists(p) else {}
def save_sheet(s): json.dump(s, open(os.path.join(W, 'sheet.json'), 'w'), indent=1)
def lab(c): return cv2.cvtColor((np.clip(c, 0, 1)[None] * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)[0].astype(float)
def candidates(idx):
    """The unit's colours: its texture colours clustered in Lab, the count chosen by BIC (1..5); hex and share."""
    from sklearn.mixture import GaussianMixture
    X = lab(COL[idx]); X = X[np.random.default_rng(0).choice(len(X), min(len(X), 8000), replace=False)]
    if len(X) < 10: return [{'hex': hexc(np.median(COL[idx], 0)), 'share': 1.0}]
    g = min((GaussianMixture(k, random_state=0).fit(X) for k in range(1, min(5, len(X) // 5) + 1)), key=lambda g: g.bic(X)); out = []
    for k in np.argsort(-g.weights_):
        rgb = cv2.cvtColor(np.clip(g.means_[k], 0, 255).astype(np.uint8)[None, None], cv2.COLOR_LAB2RGB)[0, 0]
        out.append({'hex': '#%02x%02x%02x' % tuple(int(v) for v in rgb), 'share': round(float(g.weights_[k]), 3)})
    return out
def hexc(rgb01): return '#%02x%02x%02x' % tuple(int(255 * min(max(v, 0), 1)) for v in rgb01)
def otsu(v):
    v = np.sort(np.asarray(v, float))
    if len(v) < 3: return v[0]
    n = len(v); c = np.cumsum(v); i = np.arange(1, n); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (n - i)
    return v[np.argmax(i * (n - i) * (m0 - m1) ** 2)]
def card(uid, idx):
    """The unit outlined in magenta in the view that sees most of it (left: zoom; right: whole view)."""
    uv, vis = VIEWS['uv'], VIEWS['vis']; seen = vis[:, idx].sum(1); v = int(np.argmax(seen)); im = cv2.imread(os.path.join(T, 'views', 'v%02d.png' % v))
    pts = uv[v, idx][vis[v, idx]].astype(int)
    if len(pts) == 0: return None
    m = np.zeros(im.shape[:2], np.uint8); m[np.clip(pts[:, 1], 0, im.shape[0] - 1), np.clip(pts[:, 0], 0, im.shape[1] - 1)] = 255
    m = cv2.morphologyEx(cv2.dilate(m, np.ones((5, 5), np.uint8)), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); x, y, w, h = cv2.boundingRect(np.vstack(cs)); pad = max(w, h) // 2 + 15
    x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, im.shape[1]), min(y + h + pad, im.shape[0])
    zoom = im[y0:y1, x0:x1].copy(); cv2.drawContours(zoom, [c - [x0, y0] for c in cs], -1, (255, 0, 255), 2); s = 500 / max(zoom.shape[:2]); zoom = cv2.resize(zoom, None, fx=s, fy=s)
    ctx = im.copy(); cv2.drawContours(ctx, cs, -1, (255, 0, 255), 3); cv2.rectangle(ctx, (x0, y0), (x1, y1), (0, 160, 0), 2); ctx = cv2.resize(ctx, (int(ctx.shape[1] * 500 / ctx.shape[0]), 500))
    out = np.full((500, zoom.shape[1] + ctx.shape[1] + 10, 3), 255, np.uint8); out[:zoom.shape[0], :zoom.shape[1]] = zoom; out[:, zoom.shape[1] + 10:] = ctx
    p = os.path.join(W, 'cards', 'card_%s.jpg' % uid.replace('.', '_')); cv2.imwrite(p, out, [cv2.IMWRITE_JPEG_QUALITY, 90]); return p
