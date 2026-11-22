"""Fit the Anny body model (NAVER, Apache-2.0) inside a target mesh (Tripo output, clothed, any proportions):
  python fit_anny.py <model.glb> <joints3d.npz> <out.npz>
Unknowns: Anny phenotype (gender, age, muscle, weight, height, proportions; sigmoid to [0,1]); per bone a local
rotation (rotation vector) and a log scale (bone lengths, so a big anime head or thin limbs are reachable); root
translation. Coordinates: Blender's (Z up, front -Y), which Anny shares.
Loss terms (each normalised by its own value at the start of its stage, so no term weights are tuned by hand):
  kp      triangulated COCO-WholeBody keypoints 0-22 (joints3d.py) vs Anny's own COCO regressor, weighted by confidence
  inside  Anny vertices outside the target surface (signed distance > 0 by the nearest target point's normal)
  cover   target points near the body pulled onto it: robust (Geman-McClure) target-to-Anny distance whose scale is the
          median of those distances, so clothes and hair far from the body barely pull
  prior   rotation vectors and log scales towards zero (rest pose, rest proportions)
Stages: 1 root only (kp) - 2 all bones (kp) - 3 kp + inside + cover. The prior is reported only: weighting it by kp
the start), the others by their own start value. Writes vertices, faces, bone heads/poses and params."""
import sys, numpy as np, torch, roma, trimesh, anny
from anny.keypoints import KeypointsRegressor
from scipy.spatial import cKDTree
src, jp, out = sys.argv[1:4]; torch.manual_seed(0)
sc = trimesh.load(src); g = trimesh.util.concatenate(list(sc.geometry.values())) if hasattr(sc, 'geometry') else sc
toB = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)                      # glTF Y-up -> Blender Z-up
pts, fi = trimesh.sample.sample_surface_even(g, 40000, seed=0) if hasattr(trimesh.sample, 'sample_surface_even') else trimesh.sample.sample_surface(g, 40000)
T = pts @ toB.T; TN = g.face_normals[fi] @ toB.T; tree = cKDTree(T)
J = np.load(jp); X = torch.tensor(J['X'][:23], dtype=torch.float32); C = torch.tensor(J['conf'][:23], dtype=torch.float32)
m = anny.Anny().to(dtype=torch.float32); kreg = KeypointsRegressor.coco(m); B = m.bone_count; F = torch.as_tensor(m.faces)
ph = torch.zeros(len(m.phenotype_labels), requires_grad=True); rv = torch.zeros(B, 3, requires_grad=True); ls = torch.zeros(B, requires_grad=True)
tr = torch.zeros(3, requires_grad=True)
# bones that carry a keypoint chain move; fingers, toes, twist halves and eyes stay at rest (no data drives them,
# and freeing them let the mesh fly apart on 2026-11-19)
MAIN = ('root', 'pelvis', 'spine', 'neck', 'head', 'clavicle', 'shoulder01', 'upperarm01', 'lowerarm01', 'wrist', 'upperleg01', 'lowerleg01', 'foot')
mask = torch.tensor([any(l.split('.')[0].startswith(k) for k in MAIN) for l in m.bone_labels], dtype=torch.float32)
print('FA free bones:', int(mask.sum()), 'of', B)
def run():
    Rm = roma.rotvec_to_rotmat(rv) * torch.exp(ls)[:, None, None]; P = torch.eye(4).repeat(B, 1, 1).clone()
    P = torch.cat([torch.cat([Rm, torch.zeros(B, 3, 1)], 2), torch.tensor([[[0, 0, 0, 1.]]]).repeat(B, 1, 1)], 1)
    P = P.clone(); P[0, :3, 3] = tr
    o = m(pose_parameters=P[None], phenotype_kwargs={k: torch.sigmoid(ph[i])[None] for i, k in enumerate(m.phenotype_labels)}, pose_parameterization='local-bone')
    return o, kreg(o)[0]
# start: root placed and scaled by the torso keypoints (shoulders, hips): scale = ratio of their spans, translation = centres
with torch.no_grad():
    o, k = run(); t = [5, 6, 11, 12]; s0 = (X[t] - X[t].mean(0)).norm(dim=1).mean() / (k[t] - k[t].mean(0)).norm(dim=1).mean()
    ls[0] = torch.log(s0); o, k = run(); tr += X[t].mean(0) - k[t].mean(0)
def gm(d, s): return (d ** 2 / (d ** 2 + s ** 2)).mean()
def terms(o, k):
    V = o['vertices'][0]; L = {'kp': (C[:, None] * (k - X) ** 2).sum() / C.sum()}
    Vn = V.detach().numpy(); d, i = tree.query(Vn); Ti = torch.tensor(T[i], dtype=torch.float32); Ni = torch.tensor(TN[i], dtype=torch.float32)
    L['inside'] = torch.relu(((V - Ti) * Ni).sum(1)).pow(2).mean()
    d2, j = cKDTree(Vn).query(T); near = torch.tensor(d2, dtype=torch.float32); s = float(np.median(d2))
    L['cover'] = gm((torch.tensor(T, dtype=torch.float32) - V[j]).norm(dim=1), s)
    L['prior'] = rv.pow(2).sum(1).mean() + ls[1:].pow(2).mean(); return L
stages = [(['kp'], [rv, tr, ls], 'root', 150), (['kp'], [rv, tr, ls, ph], 'all', 400), (['kp', 'inside', 'cover'], [rv, tr, ls, ph], 'all', 300)]
for names, params, which, n in stages:
    opt = torch.optim.Adam(params, lr=0.02); norm = None
    for it in range(n):
        opt.zero_grad(); o, k = run(); L = terms(o, k)
        if norm is None: norm = {a: max(float(L[a]), 1e-12) for a in names}; norm['prior'] = norm['kp'] if 'prior' in names else None   # prior shares kp's scale
        loss = sum(L[a] / norm[a] for a in names); loss.backward()
        if which == 'root': rv.grad[1:] = 0; ls.grad[1:] = 0
        rv.grad *= mask[:, None]; ls.grad *= mask
        opt.step()
        if it % 50 == 0 or it == n - 1: print('FA stage %s it %3d  ' % ('+'.join(names), it) + '  '.join('%s %.4g' % (a, float(L[a])) for a in L), flush=True)
with torch.no_grad():
    o, k = run(); k = k.detach()
np.savez(out, V=o['vertices'][0].numpy(), F=F.numpy(), heads=o['bone_poses'][0, :, :3, 3].numpy(), poses=o['bone_poses'][0].numpy(), labels=np.array(m.bone_labels),
         parents=np.array(m.bone_parents), kp=k.numpy(), X=X.numpy(), phenotype=torch.sigmoid(ph).detach().numpy(), phenotype_labels=np.array(m.phenotype_labels),
         rotvec=rv.detach().numpy(), logscale=ls.detach().numpy(), trans=tr.detach().numpy())
err = (k - X).norm(dim=1).numpy(); print('FA per-keypoint error (m):', ' '.join('%d:%.3f' % (i, e) for i, e in enumerate(err)))
print('FA done; phenotype', dict(zip(m.phenotype_labels, torch.sigmoid(ph).detach().numpy().round(2))), 'keypoint rms %.4f' % float(((k - X) ** 2).sum(1).mean().sqrt()))
