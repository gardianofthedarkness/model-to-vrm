"""SMPL body fitted straight to the model (for OSSO and other SMPL-based anatomy), no stick skeleton needed:
  E:\\splat\\.venv-body\\Scripts\\python.exe fit_smpl.py <model dir> <SMPL model dir> [female|male]
Data: the model's surface (parts/orig_colours.npz + orig_normals.npz) and the multi-view keypoints (body/joints3d.npz,
COCO-WholeBody). Keypoints on SMPL:
  joints    shoulder, elbow, wrist, hip, knee, ankle -> SMPL's joints of the same names
  vertices  nose, eyes, ears, big toes, small toes, heels -> SMPL's standard keypoint vertices (as in SPIN / VIBE:
            nose 332, eyes 2800 / 6260, ears 583 / 4071, toes and heels 3216 3226 3387 / 6617 6624 6787)
Unknowns: 10 shape numbers, 72 pose numbers, position, one overall size (the model's units are not metres).
Terms (as fit_mhr.py): keypoints (confidence-weighted), inside (SMPL stays under the surface: clothes are only an outer
bound), cover (the surface is reached where it is bare: robust Geman-McClure at the median distance), prior (shape and
pose stay small). Stages: keypoints with size and position; + pose and shape; + inside and cover; each term normalised
at its stage start.
Writes anatomy/smpl_fit.npz (same keys as mhr_to_smpl.py: V, F in the world, betas, body_pose, global_orient, transl,
scale) and prints the keypoint errors and the distance to the surface."""
import sys, os, numpy as np, torch, smplx
from scipy.spatial import cKDTree
R, sd = sys.argv[1:3]; G = sys.argv[3] if len(sys.argv) > 3 else 'female'
j = lambda *p: os.path.join(R, *p); torch.manual_seed(0)
MV = np.load(j('parts', 'orig_colours.npz'))['V'].astype(np.float64); MN = np.load(j('parts', 'orig_normals.npz'))['N'].astype(np.float64); mtree = cKDTree(MV)
JD = np.load(j('body', 'joints3d.npz')); KX, KC = JD['X'].astype(np.float64), np.nan_to_num(JD['conf'].astype(np.float64))
sm = smplx.SMPL(model_path=os.path.join(sd, 'SMPL_%s.pkl' % G.upper()), gender=G)
JM = {5: 16, 6: 17, 7: 18, 8: 19, 9: 20, 10: 21, 11: 1, 12: 2, 13: 4, 14: 5, 15: 7, 16: 8}       # COCO -> SMPL joint
VM = {0: 332, 1: 2800, 2: 6260, 3: 583, 4: 4071, 17: 3216, 18: 3226, 19: 3387, 20: 6617, 21: 6624, 22: 6787}   # COCO -> SMPL vertex
kj, jj = list(JM), list(JM.values()); kv, vv = list(VM), list(VM.values())
ok = lambda ks: [k for k in ks if np.isfinite(KX[k]).all()]
kj = [k for k in kj if np.isfinite(KX[k]).all()]; jj = [JM[k] for k in kj]; kv = [k for k in kv if np.isfinite(KX[k]).all()]; vv = [VM[k] for k in kv]
toB = torch.tensor([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=torch.float64)   # SMPL Y up, front +Z -> world Z up, front -Y
T = lambda a: torch.tensor(a, dtype=torch.float64)
XJ, CJ, XV, CV = T(KX[kj]), T(KC[kj]), T(KX[kv]), T(KC[kv])
betas = torch.zeros(1, 10, dtype=torch.float64, requires_grad=True); pose = torch.zeros(1, 69, dtype=torch.float64, requires_grad=True)
orient = torch.zeros(1, 3, dtype=torch.float64, requires_grad=True); trans = torch.zeros(3, dtype=torch.float64, requires_grad=True); ls = torch.zeros(1, dtype=torch.float64, requires_grad=True)
sm = sm.double()
def run():
    o = sm(betas=betas, body_pose=pose, global_orient=orient)
    V = torch.exp(ls) * o.vertices[0] @ toB.T + trans; Jt = torch.exp(ls) * o.joints[0, :24] @ toB.T + trans; return V, Jt
with torch.no_grad():                                                # start: size and position from the body keypoints
    V, Jt = run(); A, B = Jt[jj].numpy(), KX[kj]; s = np.sqrt(((B - B.mean(0)) ** 2).sum() / ((A - A.mean(0)) ** 2).sum())
    ls += np.log(s); V, Jt = run(); trans += T(B.mean(0) - Jt[jj].numpy().mean(0))
MVt, MNt = T(MV), T(MN)
def gm(d, s): return (d ** 2 / (d ** 2 + s ** 2)).mean()
def terms(V, Jt):
    L = {'kp': ((CJ[:, None] * (Jt[jj] - XJ) ** 2).sum() + (CV[:, None] * (V[vv] - XV) ** 2).sum()) / (CJ.sum() + CV.sum())}
    Vn = V.detach().numpy(); _, i = mtree.query(Vn); L['inside'] = torch.relu(((V - MVt[i]) * MNt[i]).sum(1)).pow(2).mean()
    sub = MV[::max(1, len(MV) // 40000)]; d2, k = cKDTree(Vn).query(sub); L['cover'] = gm((T(sub) - V[k]).norm(dim=1), float(np.median(d2)))
    L['prior'] = betas.pow(2).mean() + pose.pow(2).mean(); return L
stages = [(['kp'], [orient, trans, ls], 150), (['kp'], [orient, trans, ls, pose, betas], 600), (['kp', 'inside', 'cover'], [orient, trans, ls, pose, betas], 400)]
for names_, params, n in stages:
    opt = torch.optim.Adam(params, lr=0.03); norm = None
    for it in range(n):
        opt.zero_grad(); V, Jt = run(); L = terms(V, Jt)
        if norm is None: norm = {a: max(float(L[a]), 1e-12) for a in names_}; norm['prior'] = norm['kp']
        loss = sum(L[a] / norm[a] for a in names_) + (L['prior'] / norm['prior'] if len(params) > 3 else 0); loss.backward(); opt.step()
        if it % 50 == 0 or it == n - 1: print('SM stage %s it %3d  ' % ('+'.join(names_), it) + '  '.join('%s %.4g' % (a, float(L[a])) for a in L), flush=True)
with torch.no_grad(): V, Jt = run()
V, Jt = V.numpy(), Jt.numpy(); ej = np.linalg.norm(Jt[jj] - KX[kj], axis=1); ev = np.linalg.norm(V[vv] - KX[kv], axis=1)
d = mtree.query(V)[0]; _, i = mtree.query(V); out = ((V - MV[i]) * MN[i]).sum(1) > 0
print('SM joint keypoint error:', ' '.join('%d:%.3f' % (k, e) for k, e in zip(kj, ej)))
print('SM vertex keypoint error:', ' '.join('%d:%.3f' % (k, e) for k, e in zip(kv, ev)))
print('SM size %.3f; distance to the surface median %.4f, p90 %.4f; vertices outside %.1f%%' % (float(torch.exp(ls)), np.median(d), np.percentile(d, 90), 100 * out.mean()))
os.makedirs(j('anatomy'), exist_ok=True)
np.savez(j('anatomy', 'smpl_fit.npz'), V=V, F=np.asarray(sm.faces, np.int64), J=Jt, betas=betas.detach().numpy(), body_pose=pose.detach().numpy(), global_orient=orient.detach().numpy(),
         transl=trans.detach().numpy(), scale=float(torch.exp(ls)), toB=toB.numpy(), gender=G, source='fit_smpl.py')
print('SM done')
