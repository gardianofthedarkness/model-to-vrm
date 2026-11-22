"""The iris as a movable part on the twin's eyes: python iris_move.py <setting_iris.npz> <twin_full.npz> <out_prefix>
  measure  on the setting picture (SAM 3 masks): each eye's outline and its iris ellipse; the iris centre in the eye's own
           frame (u along the eye's long axis, v across, each in units of the eye's half-extent) and the iris size in the
           same units
  carry    on the twin: each eye piece (parts 23, 25) seen from the front gives its outline (hull) and frame the same way;
           the iris is that ellipse at (u, v), same relative size - the eye points inside it are the iris
  move     gaze sets a new (u, v); the centre is kept inside the outline (pulled back onto it when past); the iris points
           are reselected; a virtual eyeball of radius = the eye's half-width (not bounded by the skull) gives the angle
Writes <prefix>_iris_move.jpg (gaze poses) and <prefix>_iris_rig.npz (per eye: outline, frame, iris size, setting (u, v))."""
import sys, numpy as np, cv2
S_, T_, pre = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
def frame(P):
    c = P.mean(0); _, _, Vt = np.linalg.svd(P - c, full_matrices=False); au = Vt[0] * np.sign(Vt[0][0] or 1); av = np.array([-au[1], au[0]])
    pu, pv = (P - c) @ au, (P - c) @ av; o = c + au * (pu.max() + pu.min()) / 2 + av * (pv.max() + pv.min()) / 2
    return o, au, av, (pu.max() - pu.min()) / 2, (pv.max() - pv.min()) / 2
meas = []
for e, ir in zip(S_['eyes'], S_['irises']):
    o, au, av, hu, hv = frame(np.argwhere(e)[:, ::-1].astype(float)); Q = np.argwhere(ir)[:, ::-1].astype(float); qc = Q.mean(0)
    su, sv = np.ptp((Q - qc) @ au) / 2 / hu, np.ptp((Q - qc) @ av) / 2 / hv
    meas.append({'u': float((qc - o) @ au / hu), 'v': float((qc - o) @ av / hv), 'su': float(su), 'sv': float(sv)})
    print('IM picture eye: iris centre u %+.2f v %+.2f (of the eye half-extents); iris half-size %.2f x %.2f of the eye' % (meas[-1]['u'], meas[-1]['v'], su, sv))
V, P = T_['V'], T_['part']; c0 = np.array([0.0, -0.05, 0.879]); SC = 5500; SZ = 900
to_px = lambda X: np.c_[(X[:, 0] - c0[0]) * SC + SZ / 2, SZ / 2 - (X[:, 2] - c0[2]) * SC]
eyes = []
for p, m in zip((23, 25), meas):                                                       # part 23 at -x is the picture's left eye
    X = V[P == p]; q = to_px(X); o, au, av, hu, hv = frame(q); hull = cv2.convexHull(q.astype(np.float32))
    eyes.append({'part': p, 'X': X, 'q': q, 'o': o, 'au': au, 'av': av, 'hu': hu, 'hv': hv, 'hull': hull, 'm': m})
def iris_points(E, u, v):
    c = E['o'] + E['au'] * u * E['hu'] + E['av'] * v * E['hv']
    if cv2.pointPolygonTest(E['hull'], (float(c[0]), float(c[1])), False) < 0: h = E['hull'][:, 0, :]; c = h[np.argmin(((h - c) ** 2).sum(1))]   # kept inside the outline
    d = E['q'] - c; a, b = E['m']['su'] * E['hu'], E['m']['sv'] * E['hv']
    return ((d @ E['au']) / a) ** 2 + ((d @ E['av']) / b) ** 2 <= 1, c
def tile(gaze, label):
    im = np.full((SZ, SZ, 3), 255, np.uint8); F, Pf = T_['F'], T_['part']
    sys.path.insert(0, __file__.replace('\\', '/').rsplit('/', 1)[0])
    for E, (u, v) in zip(eyes, gaze):
        sel, c = iris_points(E, u, v); col = np.where(sel[:, None], np.array([200, 110, 40]), np.array([235, 235, 235])).astype(np.uint8)
        order = np.argsort(-E['X'][:, 1]); pts = E['q'][order].astype(int); ok = (pts >= 1).all(1) & (pts < SZ - 1).all(1)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1): im[pts[ok, 1] + dy, pts[ok, 0] + dx] = col[order][ok]
        cv2.polylines(im, [E['hull'].astype(np.int32)], True, (0, 160, 0), 2); cv2.circle(im, tuple(int(t) for t in c), 4, (0, 0, 255), -1)
    cv2.putText(im, label, (160, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2); return cv2.resize(im[250:650, 150:750], (600, 400))
deg = lambda u: np.degrees(np.arcsin(np.clip(u, -1, 1)))
poses = [('setting gaze', [(e['m']['u'], e['m']['v']) for e in eyes]), ('ahead', [(0, 0)] * 2), ('look left', [(-0.6, 0)] * 2),
         ('look right', [(0.6, 0)] * 2), ('look up', [(0, -0.5)] * 2), ('look down', [(0, 0.5)] * 2)]
tiles = [tile(g, '%s (%+.0f / %+.0f deg)' % (n, deg(g[0][0]), -deg(g[0][1]))) for n, g in poses]
cv2.imwrite(pre + '_iris_move.jpg', np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:])]))
np.savez(pre + '_iris_rig.npz', **{'eye%d_%s' % (i, k): np.asarray(v) for i, E in enumerate(eyes) for k, v in (('o', E['o']), ('au', E['au']), ('av', E['av']), ('hu', E['hu']), ('hv', E['hv']))},
         setting_uv=np.array([[e['m']['u'], e['m']['v']] for e in eyes]), iris_size=np.array([[e['m']['su'], e['m']['sv']] for e in eyes]))
print('IM done')
