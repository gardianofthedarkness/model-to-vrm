"""Fit Meta's MHR body model (Momentum Human Rig, Apache-2.0) inside a target mesh, the same way fit_anny.py fits Anny:
  python fit_mhr.py <model.glb> <joints3d.npz> <mhr assets dir> <out.npz> [lod]
Unknowns: identity (45: body 20, head 20, hands 5) and the 204 model parameters (anatomical joint angles plus scales),
expression held at zero. MHR is Y-up in centimetres; converted to Blender's Z-up metres, front -Y.
Keypoints: COCO-WholeBody foot points 17-22 (toes, heels) on MHR foot vertices, and body points matched to MHR joints by anatomy (shoulder = upper-arm joint, elbow = lower-arm
joint, wrist, hip = upper-leg joint, knee = lower-leg joint, ankle = talocrural joint); hand points 91-132 on the finger
joints (knuckle, middle, end joint, tip) as their own term 'hand' - without them the hands' size drifts freely (Rikka's
finger chains came out 2-3x too long); nose and eyes on the head (they hold the neck, head and collarbones).
Loss terms and stages as in fit_anny.py (kp; then kp + prior + inside + cover; each normalised at its stage start, the
prior on kp's scale)."""
import sys, numpy as np, torch, trimesh
from pathlib import Path
from scipy.spatial import cKDTree
from mhr.mhr import MHR
src, jp, ad, out = sys.argv[1:5]; lod = int(sys.argv[5]) if len(sys.argv) > 5 else 3
sc = trimesh.load(src); g = trimesh.util.concatenate(list(sc.geometry.values())) if hasattr(sc, 'geometry') else sc
toB = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
pts, fi = trimesh.sample.sample_surface(g, 40000, seed=0); T = pts @ toB.T; TN = g.face_normals[fi] @ toB.T; tree = cKDTree(T)
J = np.load(jp); m = MHR.from_files(folder=Path(ad), device=torch.device('cpu'), lod=lod); names = list(m.character.skeleton.joint_names)
MAP = {5: 'l_uparm', 6: 'r_uparm', 7: 'l_lowarm', 8: 'r_lowarm', 9: 'l_wrist', 10: 'r_wrist', 11: 'l_upleg', 12: 'r_upleg',
       13: 'l_lowleg', 14: 'r_lowleg', 15: 'l_talocrural', 16: 'r_talocrural'}
ki = list(MAP); ji = [names.index(MAP[k]) for k in ki]
# hands: COCO-WholeBody 91-111 (left) and 112-132 (right); per finger knuckle, middle, end joint, tip -> MHR's joints
FING = {1: 'thumb1', 2: 'thumb2', 3: 'thumb3', 4: 'thumb_null', 5: 'index1', 6: 'index2', 7: 'index3', 8: 'index_null',
        9: 'middle1', 10: 'middle2', 11: 'middle3', 12: 'middle_null', 13: 'ring1', 14: 'ring2', 15: 'ring3', 16: 'ring_null',
        17: 'pinky1', 18: 'pinky2', 19: 'pinky3', 20: 'pinky_null'}
HAND = {base + k: side + v for side, base in (('l_', 91), ('r_', 112)) for k, v in FING.items() if side + v in names}
hk = list(HAND); hj = [names.index(HAND[k]) for k in hk]
# foot landmarks on MHR's own mesh (rest pose, Y up, front +Z): per side, the foot's vertices are those whose nearest
# joint is in that foot's chain; big toe = front-most of the inner half, small toe = front-most of the outer half,
# heel = back-most of the lower half (halves split at the foot's own median), matched to COCO-WholeBody 17-22
with torch.no_grad():
    v0, s0 = m(torch.zeros(1, 45), torch.zeros(1, 204), torch.zeros(1, 72)); v0 = v0[0].numpy(); j0 = s0[0, :, :3].numpy()
nearj = cKDTree(j0).query(v0)[1]; FOOTK = {}
for side, (big, small, heel) in (('l_', (17, 18, 19)), ('r_', (20, 21, 22))):
    chain = [names.index(side + n) for n in ('foot', 'talocrural', 'subtalar', 'transversetarsal', 'ball')]
    fv = np.flatnonzero(np.isin(nearj, chain)); ax = np.abs(v0[fv, 0]); inner, outer = fv[ax <= np.median(ax)], fv[ax > np.median(ax)]
    low = fv[v0[fv, 1] <= np.median(v0[fv, 1])]
    FOOTK[big] = int(inner[np.argmax(v0[inner, 2])]); FOOTK[small] = int(outer[np.argmax(v0[outer, 2])]); FOOTK[heel] = int(low[np.argmin(v0[low, 2])])
# face: without it nothing holds the neck, head and collarbones (Rikka's T-pose twin came out with the neck folded 7 cm
# forward and down). Nose (COCO 0) = front-most vertex of the head (vertices whose nearest joint is in the head chain);
# eyes (COCO 1, 2) = the eye joint's end point when it lies in front of the eye centre (the eye's surface), else the centre
headc = [names.index(n) for n in names if n.startswith('c_head') or 'eye' in n or n.startswith(('c_jaw', 'c_teeth', 'c_tongue'))]
hv = np.flatnonzero(np.isin(nearj, headc)); FOOTK[0] = int(hv[np.argmax(v0[hv, 2])])
for k, side in ((1, 'l_'), (2, 'r_')):
    a, b = names.index(side + 'eye'), names.index(side + 'eye_null'); MAP[k] = names[b] if j0[b, 2] > j0[a, 2] else names[a]
ki = list(MAP); ji = [names.index(MAP[k]) for k in ki]
fk = list(FOOTK); fvi = [FOOTK[k] for k in fk]
X = torch.tensor(J['X'][ki], dtype=torch.float32); C = torch.tensor(J['conf'][ki], dtype=torch.float32)
XF = torch.tensor(J['X'][fk], dtype=torch.float32); CF = torch.tensor(J['conf'][fk], dtype=torch.float32)
XH = torch.tensor(J['X'][hk], dtype=torch.float32); CH = torch.tensor(np.nan_to_num(J['conf'][hk]), dtype=torch.float32)
toBt = torch.tensor(toB, dtype=torch.float32) / 100.0
idc = torch.zeros(1, 45, requires_grad=True); mp = torch.zeros(1, 204, requires_grad=True); fe = torch.zeros(1, 72); tr = torch.zeros(3, requires_grad=True)
def run():
    v, sk = m(idc, mp, fe); V = v[0] @ toBt.T + tr; Jp = sk[0, :, :3] @ toBt.T + tr; return V, Jp, sk
with torch.no_grad():
    V, Jp, _ = run(); t = [ki.index(k) for k in (5, 6, 11, 12)]               # shoulders and hips
    tr += X[t].mean(0) - Jp[ji][t].mean(0)
def gm(d, s): return (d ** 2 / (d ** 2 + s ** 2)).mean()
def terms(V, Jp):
    L = {'kp': ((C[:, None] * (Jp[ji] - X) ** 2).sum() + (CF[:, None] * (V[fvi] - XF) ** 2).sum()) / (C.sum() + CF.sum())}
    L['hand'] = (CH[:, None] * (Jp[hj] - XH) ** 2).sum() / CH.sum().clamp(min=1e-9)
    Vn = V.detach().numpy(); d, i = tree.query(Vn); Ti = torch.tensor(T[i], dtype=torch.float32); Ni = torch.tensor(TN[i], dtype=torch.float32)
    L['inside'] = torch.relu(((V - Ti) * Ni).sum(1)).pow(2).mean()
    d2, j = cKDTree(Vn).query(T); L['cover'] = gm((torch.tensor(T, dtype=torch.float32) - V[j]).norm(dim=1), float(np.median(d2)))
    L['prior'] = idc.pow(2).mean() + mp.pow(2).mean(); return L
stages = [(['kp'], [mp, tr], 150), (['kp', 'hand'], [mp, tr, idc], 400), (['kp', 'hand', 'inside', 'cover'], [mp, tr, idc], 300)]
for names_, params, n in stages:
    opt = torch.optim.Adam(params, lr=0.02); norm = None
    for it in range(n):
        opt.zero_grad(); V, Jp, _ = run(); L = terms(V, Jp)
        if norm is None: norm = {a: max(float(L[a]), 1e-12) for a in names_}; norm['prior'] = norm['kp']
        loss = sum(L[a] / norm[a] for a in names_); loss.backward(); opt.step()
        if it % 50 == 0 or it == n - 1: print('FM stage %s it %3d  ' % ('+'.join(names_), it) + '  '.join('%s %.4g' % (a, float(L[a])) for a in L), flush=True)
with torch.no_grad():
    V, Jp, sk = run()
par = np.array(m.character.skeleton.joint_parents)
F = np.array(m.character.mesh.faces)
err = (Jp[ji] - X).norm(dim=1).numpy(); print('FM per-keypoint error (m):', ' '.join('%s:%.3f' % (MAP[k], e) for k, e in zip(ki, err)))
eh = (Jp[hj] - XH).norm(dim=1).numpy(); print('FM hand keypoint error (m): median %.3f, max %.3f over %d points' % (np.median(eh), eh.max(), len(eh)))
ef = (V[fvi] - XF).norm(dim=1).numpy(); print('FM foot and nose landmark error (m):', ' '.join('%d:%.3f' % (k, e) for k, e in zip(fk, ef)))
np.savez(out, V=V.numpy(), F=F, heads=Jp.numpy(), parents=par, labels=np.array(names), kp=np.r_[Jp[ji].numpy(), V[fvi].numpy()], X=np.r_[X.numpy(), XF.numpy()], identity=idc.detach().numpy(), model_params=mp.detach().numpy(), trans=tr.detach().numpy())
print('FM done; keypoint rms %.4f' % float(np.sqrt((err ** 2).mean())))
