"""The philtrum as an area on her face, boxed by Jev on the front close-up of the head:
  python philtrum_area.py <model dir>      -> <model dir>/parts_zoom/philtrum.npz (vids: her surface vertices in the area;
                                              box, confidence) and philtrum_check.jpg
A few requests (coarse to fine, at most four). The area is the surface the front view sees inside Jev's box (nearest surface per pixel, to one mesh edge).
The key comes from the vault, goes only into the request header, and is never printed or written."""
import sys, os, json, base64, time, urllib.request, urllib.error, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common')); from vault import get
from models import JEV_MODEL, JEV_ENDPOINT
MD = sys.argv[1]; j = lambda *p: os.path.join(MD, *p); view = 'tex_e+00_a000'; cams = json.load(open(j('parts_zoom', 'head', 'cameras.json'))); png = j('parts_zoom', 'head', view + '.png')
if int(os.environ.get('LLM_MAX_CALLS', '200')) < 1: raise SystemExit('over the LLM call budget')
KEY = get('jev'); import cv2
Q = {'philtrum': {'type': 'box', 'instructions': 'In the "view" picture, the philtrum: the area of skin between the base of the nose and the upper lip, as wide as the groove and its two ridges. Not the nose, not the lips.'}}
def ask(img_bytes, what):
    body = {'model': JEV_MODEL, 'state': {'state': what, 'view': 'view_image'}, 'assets': {'view_image': {'type': 'image', 'format': 'png', 'data_base64': base64.b64encode(img_bytes).decode()}}, 'questions': Q}
    req = urllib.request.Request(JEV_ENDPOINT, json.dumps(body).encode(), {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + KEY, 'TypeSafe-Danger-Zone-Images': 'true'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=90) as res: return json.loads(res.read())['answers']['philtrum']
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429: raise SystemExit('Jev refused: %d %s' % (e.code, e.read()[:300].decode('utf-8', 'replace')))
        except (urllib.error.URLError, TimeoutError): pass
        time.sleep(1 + 2 * attempt)
    raise SystemExit('Jev failed 3 times')
# coarse to fine: Jev's box on the whole head is rough (it took in the nose), so it is asked again on a close-up of its own
# box with as much again around it, and once more, while the box gets smaller and still holds the face detector's own
# point between the nose base and the upper lip (left alone, Jev kept zooming in to nothing); the budget is LLM_MAX_CALLS
PT = np.load(j('parts_zoom', 'head', 'points3d.npz'))['X'].astype(np.float64); c_ = cams[view]; Mi_ = np.array(c_['matrix_inv']); mp_ = ((PT[33] + PT[51]) / 2) @ Mi_[:3, :3].T + Mi_[:3, 3]; mpu, mpv = mp_[0] / c_['scale'] + 0.5, 0.5 - mp_[1] / c_['scale']   # the face detector's own point between nose base and upper lip
full = cv2.imread(png); R_ = full.shape[0]; win = (0.0, 0.0, 1.0, 1.0); a = None; calls = 0; hist_ = []
while calls < min(4, int(os.environ.get('LLM_MAX_CALLS', '200'))):
    x0, y0, x1, y1 = win; crop = full[int(y0 * R_):int(y1 * R_), int(x0 * R_):int(x1 * R_)]; sc_ = R_ / max(crop.shape[:2]); crop = cv2.resize(crop, (max(1, int(round(crop.shape[1] * sc_))), max(1, int(round(crop.shape[0] * sc_)))), interpolation=cv2.INTER_CUBIC)   # longer side = the view's own size (the decision API refuses larger images)
    r = ask(cv2.imencode('.png', crop)[1].tobytes(), 'A front render of the head of a 3D figure.' if calls == 0 else 'A close-up of the middle of the face of a 3D figure, front view.'); calls += 1
    if not r or not r.get('box'): break
    bb = r['box']; g = dict(left=x0 + bb['left'] * (x1 - x0), right=x0 + bb['right'] * (x1 - x0), top=y0 + bb['top'] * (y1 - y0), bottom=y0 + bb['bottom'] * (y1 - y0)); area = (g['right'] - g['left']) * (g['bottom'] - g['top']); hist_.append((g, float(r.get('confidence') or 0), area))
    print('PA ask %d: box %s, confidence %.2f' % (calls, {k: round(x, 3) for k, x in g.items()}, hist_[-1][1]), flush=True)
    if not (g['left'] <= mpu <= g['right'] and g['top'] <= mpv <= g['bottom']) and len(hist_) > 1: hist_.pop(); continue   # the box has left the detector's own philtrum point: ask again on the same close-up (answers vary), within the call budget
    if len(hist_) > 1 and area >= hist_[-2][2]: hist_.pop(); break                                   # no smaller: the one before stands
    a = dict(box=g, confidence=hist_[-1][1]); w_, h_ = g['right'] - g['left'], g['bottom'] - g['top']; win = (max(0, g['left'] - w_ / 2), max(0, g['top'] - h_ / 2), min(1, g['right'] + w_ / 2), min(1, g['bottom'] + h_ / 2))
if not a: raise SystemExit('Jev gave no box')
if len(hist_) < 2: print('PA WARNING: no close-up box held the face detector philtrum point; the whole-head box is used and may take in the nose', flush=True)
b = a['box']; MV = np.load(j('parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(j('parts', 'orig_normals.npz'))['F']; vid = np.load(j('parts_zoom', 'head_vids.npy')); est = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1)))
c = cams[view]; Mi = np.array(c['matrix_inv']); cc = MV[vid] @ Mi[:3, :3].T + Mi[:3, 3]; u = cc[:, 0] / c['scale'] + 0.5; v = 0.5 - cc[:, 1] / c['scale']; res = int(c['res'])
px = np.clip((u * res).astype(int), 0, res - 1) + res * np.clip((v * res).astype(int), 0, res - 1); near = np.full(res * res, -np.inf); np.maximum.at(near, px, cc[:, 2])       # the camera looks down its -z: the nearest surface has the largest z
ins = (u >= b['left']) & (u <= b['right']) & (v >= b['top']) & (v <= b['bottom']) & (cc[:, 2] >= near[px] - est)
np.savez(j('parts_zoom', 'philtrum.npz'), vids=vid[ins], box=np.array([b['left'], b['top'], b['right'], b['bottom']]), confidence=float(a.get('confidence') or 0))
print('PA box %s, confidence %.2f: %d surface vertices, %.1f x %.1f mm' % ({k: round(x, 3) for k, x in b.items()}, float(a.get('confidence') or 0), ins.sum(), *((1000 * np.ptp(MV[vid[ins]][:, [0, 2]], 0)) if ins.any() else (0, 0))))
im = cv2.imread(png); cv2.rectangle(im, (int(b['left'] * res), int(b['top'] * res)), (int(b['right'] * res), int(b['bottom'] * res)), (0, 0, 255), 2); cv2.imwrite(j('parts_zoom', 'philtrum_check.jpg'), im[res // 4:, res // 5:4 * res // 5])
