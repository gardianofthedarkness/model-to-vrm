"""Eyes and mouth on the twin's own face: python face_rig.py <twin_full.npz> <body_fit.npz> <face68_px.npy> <out_prefix>
  eyes   each eye piece (parts 23, 25) is a lens-shaped ball; it turns about the centre of the sphere fitted to it.
         Gaze range: the eye's normal range of motion (about 45 degrees each way, clinical norm); the demo uses 25.
  jaw    jaw weights on the face (part 9) from the detected face landmarks (rtmlib, 68 points): 1 below the mouth slit and
         inside the jawline, 0 above the slit and outside the jawline, harmonic over the face surface in between.
         The jaw turns about SOMA's Jaw joint (placed by the stage-2 fit), across the head; full opening 25 degrees
         (a normal adult opening of ~45 mm at the incisors).
  lips   the mouth corners from the landmarks (3-D: the front-most face point under each); spread or round moves the
         face around each corner with a falloff over half the mouth's width. Spread up to 20% of the mouth width
         (a smile), rounding up to 35% inward and half that forward (speech kinematics, rounded vowels).
  shapes aa (open), ih, ee (spread), oh, ou (rounded), as jaw fraction + lip amounts."""
import sys, numpy as np, scipy.sparse as sp, scipy.sparse.linalg as sl, cv2
d, B, L68, pre = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]
V, F, P = d['V'], d['F'], d['part']; names = open('E:/figures/ardy/soma77_names.txt').read().split('\n'); J = B['J']
# the face render's camera (quick_render, front): x = (px - 300)/4500, z = 0.865 - (py - 300)/4500
to_xz = lambda px: np.c_[(px[:, 0] - 300) / 4500.0, 0.865 - (px[:, 1] - 300) / 4500.0]
lm = to_xz(L68); face = np.flatnonzero(P == 9)
def front_point(x, z):
    m = face[(np.abs(V[face, 0] - x) < 0.0015) & (np.abs(V[face, 2] - z) < 0.0015)]; return V[m[np.argmin(V[m, 1])]]
cornerR, cornerL = front_point(*lm[48]), front_point(*lm[54]); mc = (cornerR + cornerL) / 2; mw = np.linalg.norm(cornerL - cornerR)
print('FR mouth width %.4f, corners %s %s' % (mw, cornerR.round(4), cornerL.round(4)))
# jaw weights
pts2 = (np.c_[V[face, 0] * 4500 + 300, 300 - (V[face, 2] - 0.865) * 4500]).astype(np.float32)
jaw_poly = L68[0:17].astype(np.float32); z_up = lm[62, 1]; z_lo = lm[66, 1]
inside = np.array([cv2.pointPolygonTest(np.r_[jaw_poly, jaw_poly[:1]], (float(x), float(y)), False) >= 0 for x, y in pts2])
inside |= (V[face, 2] > lm[[0, 16], 1].min())                                        # above the jawline's ends counts as inside the face
w = np.full(len(face), np.nan); w[(V[face, 2] < z_lo) & inside & (V[face, 2] > lm[8, 1] - 0.003)] = 1; w[(V[face, 2] > z_up)] = 0; w[~inside] = 0
idx = {v: i for i, v in enumerate(face)}; fF = F[np.isin(F, face).all(1)]; fF = np.vectorize(idx.get)(fF)
E = np.vstack([fF[:, [0, 1]], fF[:, [1, 2]], fF[:, [2, 0]]]); n = len(face)
A = sp.coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(n, n)); A = ((A + A.T) > 0).astype(float).tocsr(); Lp = sp.diags(np.asarray(A.sum(1)).ravel()) - A
fx = np.flatnonzero(~np.isnan(w)); fr = np.flatnonzero(np.isnan(w))
if len(fr): w[fr] = sl.splu(Lp[fr][:, fr].tocsc() + sp.eye(len(fr)) * 1e-12).solve(-(Lp[fr][:, fx] @ w[fx]))
w = np.clip(w, 0, 1); print('FR jaw weights: %d face points, %d fully on the jaw' % (n, (w > 0.99).sum()))
jaw = J[names.index('Jaw')]
# eye centres
eyes = {}
for p in (23, 25):
    X = V[P == p]; A_ = np.c_[2 * X, np.ones(len(X))]; s = np.linalg.lstsq(A_, (X ** 2).sum(1), rcond=None)[0]; eyes[p] = s[:3]
def rot(axis, deg):
    a = np.asarray(axis, float); a /= np.linalg.norm(a); t = np.radians(deg); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
SHAPES = {'neutral': (0, 0, 0), 'aa': (1.0, 0, 0), 'ih': (0.3, 0.5, 0), 'ee': (0.2, 1.0, 0), 'oh': (0.6, 0, 0.6), 'ou': (0.25, 0, 1.0)}
def pose(shape='neutral', yaw=0.0, pitch=0.0):
    Vp = V.copy(); jf, spread, rnd = SHAPES[shape]
    # jaw: open = the chin goes down and back; pick the turn direction that lowers the chin landmark
    R = rot((1, 0, 0), 25 * jf); chin = front_point(*lm[8])
    if (R @ (chin - jaw) + jaw)[2] > chin[2]: R = rot((1, 0, 0), -25 * jf)
    Q = V[face] - jaw; moved = (Q @ R.T) + jaw; Vp[face] = V[face] + w[:, None] * (moved - V[face])
    # lips
    for cn, sx in ((cornerR, -1), (cornerL, 1)):
        fall = np.exp(-0.5 * (np.linalg.norm(V[face] - cn, axis=1) / (mw / 2)) ** 2)
        dv = np.array([sx * 0.20 * spread * mw - sx * 0.35 * rnd * mw, -0.175 * rnd * mw, 0.0]); Vp[face] += fall[:, None] * dv
    for p in (23, 25):                                                              # eyes: yaw about up (z), pitch about across (x)
        m = P == p; c = eyes[p]; Rg = rot((0, 0, 1), yaw) @ rot((1, 0, 0), pitch); Vp[m] = (V[m] - c) @ Rg.T + c
    return Vp
if __name__ == '__main__':
    sys.path.insert(0, __file__.rsplit('\\', 1)[0].rsplit('/', 1)[0]); from quick_render import shade
    keepF = F[np.isin(P[F[:, 0]], [0, 9, 23, 25])]; c = np.array([0.0, -0.03, 0.86])
    tile = lambda Vp, label: cv2.putText(shade(Vp, keepF, 'front', 330, center=c, scale=3600), label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    gaze = [tile(pose(yaw=y, pitch=p_), lab) for lab, y, p_ in (('look ahead', 0, 0), ('look left', 25, 0), ('look right', -25, 0), ('look up', 0, 20), ('look down', 0, -20))]
    mouth = [tile(pose(shape=s), s) for s in ('neutral', 'aa', 'ih', 'ee', 'oh', 'ou')]
    cv2.imwrite(pre + '_gaze.jpg', np.hstack(gaze)); cv2.imwrite(pre + '_mouth.jpg', np.vstack([np.hstack(mouth[:3]), np.hstack(mouth[3:])]))
    np.savez(pre + '_rig.npz', jaw_w=w, face=face, jaw=jaw, eyes=np.array([eyes[23], eyes[25]]), corners=np.array([cornerR, cornerL]), mw=mw)
    print('FR done')
