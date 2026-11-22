"""Cards: python tools/card.py <part id> [...]  ->  cards/card_<id>.jpg
The part's points painted magenta on textured renders of the model: front, back, and the view where it is seen at the
largest size. Points hidden from a view are not painted in it."""
import sys, os, numpy as np, cv2; sys.path.insert(0, os.path.dirname(__file__)); from common import *
reg = np.load(os.path.join(WS, 'parts.npz'))['region']; V = np.load(CFG['colours'])['V']; VW = views(); byname = {n: (d, cm, vv) for d, n, cm, vv in VW}
def proj(cm, m):
    R = cm['res']; P = np.c_[V[m], np.ones(m.sum())] @ np.array(cm['matrix_inv']).T
    return ((P[:, 0] / cm['scale'] + 0.5) * R).astype(int), ((0.5 - P[:, 1] / cm['scale']) * R).astype(int)
def seen(name, m):
    d, cm, vv = byname[name]; u, v = proj(cm, m); R = cm['res']; return int((vv[m] & (u >= 0) & (v >= 0) & (u < R) & (v < R)).sum())
def paint(name, m):
    d, cm, vv = byname[name]; R = cm['res']; u, v = proj(cm, m); ok = vv[m] & (u >= 0) & (v >= 0) & (u < R) & (v < R)
    im = cv2.imread(os.path.join(d, name + '.png')); mask = np.zeros((R, R), np.uint8); mask[v[ok], u[ok]] = 255; mask = cv2.dilate(mask, np.ones((3, 3), np.uint8))
    out = im.copy(); out[mask > 0] = (out[mask > 0] * 0.3 + np.array([255, 0, 255]) * 0.7).astype(np.uint8)
    cs, _ = cv2.findContours(cv2.dilate(mask, np.ones((15, 15), np.uint8)), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(out, cs, -1, (255, 0, 255), 2); return out
A = all_parts()
for a in sys.argv[1:]:
    i = int(a); m = reg == i
    if not m.any(): print('no part', i); continue
    close = max((n for n in byname if n not in (CFG['front'], CFG['back'])), key=lambda n: seen(n, m) / byname[n][1]['scale'] ** 2)
    img = np.hstack([cv2.resize(paint(n, m), (512, 512)) for n in (CFG['front'], CFG['back'], close)])
    p = A[i]; cv2.putText(img, 'part %d  %s  %s' % (i, p['kind'], p.get('concept', p.get('sam3_hint', ''))), (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imwrite(os.path.join(WS, 'cards', 'card_%d.jpg' % i), img, [cv2.IMWRITE_JPEG_QUALITY, 85]); print('cards/card_%d.jpg  (close-up: %s)' % (i, close))
