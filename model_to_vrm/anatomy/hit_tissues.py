"""HIT (Keller et al., CVPR 2024: fat / lean / bone inside a SMPL body, learned from MRI of 157 men and 241 women) on the
model's fitted SMPL body (mhr_to_smpl.py) -> posed tissue surfaces in the model's world:
  E:\\splat\\.venv\\Scripts\\python.exe hit_tissues.py <model dir> [female|male]
  needs     HIT code at E:\\code\\external\\HIT (its hit_config.py points at the checkpoints and SMPL files under
            E:\\figures\\models\\hit), the stand-in psbody package in anatomy/hit_shims (viewer only)
  route     tissues are predicted in the body's canonical space (its shape, in HIT's X pose) and surfaces extracted
            there. Posing: each surface point is anchored to its nearest canonical SMPL vertex in that vertex's local
            frame (normal, tangent) and rebuilt on the posed SMPL body (<tissue>_V). HIT's own learned posing
            (pose_unposed_tissue_mesh) is kept as <tissue>_Vhit: on test model R (shape values up to 9, HIT trained on +-2)
            it put 65-75% of the forearm tissue outside the SMPL forearm. HIT's demo extracts
            with LEAP's MISE; here scikit-image marching cubes on a regular grid instead (LEAP does not build on Windows)
  grid      over the canonical SMPL body's box; voxel = half the SMPL mesh's median edge (the body is known that finely)
  surface   per tissue c: the zero level of (score_c - best other score) - where c stops being the most likely class
  classes   lean (muscle + organs + inner fat), fat (subcutaneous), bone (long bones only: femur, radius-ulna, tibia, fibula)
  frames    world = p_smpl @ toB.T + trans
  writes    anatomy/hit_tissues.npz: per tissue V/F posed in the world, the canonical label grid; anatomy/hit_cuts.png"""
import sys, os, time, numpy as np, torch, cv2
from skimage.measure import marching_cubes
HIT = r'E:\code\external\HIT'; sys.path[:0] = [HIT, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hit_shims')]
os.environ.setdefault('WANDB_MODE', 'disabled')
R = sys.argv[1]; S = np.load(os.path.join(R, 'anatomy', 'smpl_fit.npz')); gender = sys.argv[2] if len(sys.argv) > 2 else str(S['gender'])
toB, tr = S['toB'], S['trans']
cwd = os.getcwd(); os.chdir(HIT)                                   # HIT finds its assets relative to its folder
from hit.utils.model import HitLoader
hl = HitLoader.from_expname('hit_' + gender, ckpt_choice='best'); hl.load(); hl.hit_model.apply_compression = False; os.chdir(cwd)
dev = next(hl.hit_model.parameters()).device
T = lambda k: torch.tensor(S[k], dtype=torch.float32, device=dev)
B_RANGE = 2.0   # HIT's valid shape range: its demo (demos/infer_smpl.py) says betas "should be between -2 and 2"; outside it the
                # learned beta field extrapolates (on test model R, betas up to 9.2 put the canonical forearm bone 100% outside the arm)
betas_used = np.clip(S['betas'], -B_RANGE, B_RANGE); print('HT betas %s -> used %s' % (S['betas'].round(2), betas_used.round(2)), flush=True)
_T = T; T = lambda k: torch.tensor(betas_used, dtype=torch.float32, device=dev) if k == 'betas' else _T(k)
with torch.no_grad():
    so = hl.smpl(betas=T('betas'), body_pose=T('body_pose'), global_orient=T('global_orient'), transl=T('transl'))   # transl (HIT's demo passes trans=, which is ignored)
    sx = hl.smpl.forward(betas=T('betas'), body_pose=hl.smpl.x_cano().to(dev), global_orient=None, transl=None)   # canonical
    Vc = sx.vertices[0].cpu().numpy()
F = S['F']; e = np.linalg.norm(Vc[F[:, [1, 2, 0]]] - Vc[F], axis=2); vox = float(np.median(e)) / 2
lo, hi = Vc.min(0) - 2 * vox, Vc.max(0) + 2 * vox; n = np.ceil((hi - lo) / vox).astype(int)
print('HT canonical body %s m, voxel %.4f m (half the median edge), grid %s = %.1f M points' % (np.ptp(Vc, 0).round(3), vox, n, n.prod() / 1e6), flush=True)
axes = [lo[i] + np.arange(n[i]) * vox for i in range(3)]; L = np.zeros(tuple(n) + (4,), np.float16)
chunk = 100000; t0 = time.time()
for i in range(n[0]):                                             # one x-slab at a time
    G = np.stack(np.meshgrid([axes[0][i]], axes[1], axes[2], indexing='ij'), -1).reshape(-1, 3); sc = np.zeros((len(G), 4), np.float32)
    for j in range(0, len(G), chunk):
        with torch.no_grad():
            out = hl.hit_model.query(torch.tensor(G[j:j + chunk], dtype=torch.float32, device=dev)[None], sx, unposed=True, eval_mode=True)
        sc[j:j + chunk] = out['pred_occ'][0].float().cpu().numpy()
    L[i] = sc.reshape(n[1], n[2], 4)
    if i % max(1, n[0] // 8) == 0: print('HT slab %d/%d (%.0f s)' % (i, n[0], time.time() - t0), flush=True)
lab = L.argmax(-1).astype(np.uint8); names = ['none', 'lean', 'fat', 'bone']
cnt = np.bincount(lab.ravel(), minlength=4) * vox ** 3
print('HT canonical volumes (L): lean %.2f, fat %.2f, bone %.2f; fat %.1f%% of the body' % (cnt[1] * 1e3, cnt[2] * 1e3, cnt[3] * 1e3, 100 * cnt[2] / cnt[1:].sum()), flush=True)
import trimesh
from scipy.spatial import cKDTree
Vp = so.vertices[0].cpu().numpy()
def vframes(Vv):                                                  # per SMPL vertex: normal, a tangent toward a fixed neighbour, their cross
    m_ = trimesh.Trimesh(Vv, F, process=False); nn = m_.vertex_normals; nb = np.array([sorted(x)[0] for x in m_.vertex_neighbors])
    tt = Vv[nb] - Vv; tt -= nn * (tt * nn).sum(1, keepdims=True); tt /= np.linalg.norm(tt, axis=1, keepdims=True) + 1e-12
    return np.stack([nn, tt, np.cross(nn, tt)], 1)              # (n, 3 axes, 3)
Fc_, Fp_ = vframes(Vc), vframes(Vp); kc = cKDTree(Vc)
def anchor_pose(v):                                               # canonical point -> nearest SMPL vertex + local coords -> posed
    i = kc.query(v)[1]; loc = np.einsum('nij,nj->ni', Fc_[i], v - Vc[i]); return Vp[i] + np.einsum('nij,ni->nj', Fp_[i], loc)
res = {}
for c in (1, 2, 3):
    Lf = L.astype(np.float32); s = Lf[..., c] - np.delete(Lf, c, axis=-1).max(-1)   # > 0 where tissue c is the most likely
    v, f, _, _ = marching_cubes(s, 0.0, spacing=(vox, vox, vox)); v = v + lo
    mesh_s = trimesh.Trimesh(v, f[:, ::-1], process=False)
    with torch.no_grad():
        mp = hl.hit_model.pose_unposed_tissue_mesh(mesh_s, so)    # HIT's learned deformation into the pose
    Wh = np.asarray(mp.vertices) @ toB.T + tr; W = anchor_pose(v) @ toB.T + tr; res[names[c]] = (W, np.asarray(mp.faces), v, Wh)
    print('HT %s surface: %d vertices, posed' % (names[c], len(W)), flush=True)
np.savez_compressed(os.path.join(R, 'anatomy', 'hit_tissues.npz'), **{'%s_V' % k: v[0] for k, v in res.items()}, **{'%s_F' % k: v[1] for k, v in res.items()},
                    **{'%s_Vc' % k: v[2] for k, v in res.items()}, **{'%s_Vhit' % k: v[3] for k, v in res.items()}, labels_c=lab, origin_c=lo, voxel=vox, names=np.array(names), gender=gender,
                    smpl_V=so.vertices[0].cpu().numpy() @ toB.T + tr, smpl_F=F, smpl_Vc=Vc, betas_used=betas_used)
# canonical cuts: front (x-y) through the body's middle depth, side (y-z) through the middle, and levels up the body
col = np.array([[255, 255, 255], [70, 70, 200], [40, 180, 230], [225, 235, 235]], np.uint8)   # BGR: lean red, fat yellow, bone ivory
k = max(1, int(500 / n[1])); im = lambda a: cv2.resize(col[a], None, fx=k, fy=k, interpolation=cv2.INTER_NEAREST)
front = im(np.flipud(lab[:, :, n[2] // 2].T)); side = im(np.flipud(lab[n[0] // 2, :, :]))
lv = [im(lab[:, j, :].T) for j in np.linspace(0, n[1] - 1, 10)[1:-1].astype(int)]
h = max(front.shape[0], side.shape[0]); pad = lambda a: np.vstack([a, np.full((h - a.shape[0], a.shape[1], 3), 255, np.uint8)])
cv2.imwrite(os.path.join(R, 'anatomy', 'hit_cuts.png'), np.hstack([pad(front), pad(side)]))
cv2.imwrite(os.path.join(R, 'anatomy', 'hit_levels.png'), np.vstack([np.hstack(lv[:4]), np.hstack(lv[4:8])]))
print('HT done')
