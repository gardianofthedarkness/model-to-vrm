"""Synthetic check of the T0b job's pieces: (1) the affine ICP recovers a known affine of a bone shape from a partial,
noisy copy; (2) the smooth-field fit with height-block cross-validation recovers a smooth field from noisy samples on
the atlas's own pelvis skin mesh, with held-out error near the noise. Made-up numbers only."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t0b_pelvis_job as J
A = dict(np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'atlas_pelvis.npz')))
rng = np.random.default_rng(0)
# (1) ICP
Va = A['bone_femur_left_V'].astype(float); Mtrue = np.c_[np.diag([1.05, 0.95, 1.1]) @ np.array([[0.995, -0.1, 0], [0.1, 0.995, 0], [0, 0, 1]]), [0.01, -0.02, 0.03]]
P = (np.c_[Va, np.ones(len(Va))] @ Mtrue.T); P = P[P[:, 2] > np.percentile(P[:, 2], 40)] + rng.normal(0, 0.0007, (0, 3)).sum() + rng.normal(0, 0.0007, P[P[:, 2] > np.percentile(P[:, 2], 40)].shape)   # the upper 60% only (a cut femur), 0.7 mm noise
from scipy.spatial.transform import Rotation as Rot                                   # start as the job does: the truth seen through a whole-pelvis fit, off by a few degrees and centimetres
Rd = Rot.from_rotvec([0.05, -0.04, 0.06]).as_matrix(); c = Va.mean(0); A0 = Rd @ Mtrue[:, :3]; M0 = np.c_[A0, Mtrue[:, :3] @ c + Mtrue[:, 3] - A0 @ c + [0.02, -0.015, 0.025]]
M, d = J.affine_icp(P, Va, M0)
X = np.c_[Va, np.ones(len(Va))]; err = np.linalg.norm(X @ M.T - X @ Mtrue.T, axis=1)
print('ICP: matches median %.2f mm; placed-atlas error against the truth: median %.2f mm, p90 %.2f mm (on the part that was seen: %.2f mm)' % (1000 * np.median(d), 1000 * np.median(err), 1000 * np.percentile(err, 90), 1000 * np.median(err[(X @ Mtrue.T)[:, 2] > np.percentile(P[:, 2], 0)])))
# (2) smooth field
SV = A['skin_V'].astype(float); L, degm = J.laplacian(len(SV), A['skin_F'])
truth = 0.012 + 0.006 * np.sin(SV[:, 2] * 40) + 0.004 * np.cos(SV[:, 0] * 30)          # a smooth field, metres
idx = rng.choice(len(SV), int(0.8 * len(SV)), replace=False); y = truth[idx] + rng.normal(0, 0.0008, len(idx))
r = dict(s_idx=idx, s_y=y, t_idx=idx[:10], t_y=y[:10], z=SV[:, 2])
f = J.fit_and_score(r, L, degm)['skin']
print('field: held-out median %.2f mm (noise median %.2f mm), lambda/base %.3g, error against the truth everywhere: median %.2f mm' % (f['heldout_median_mm'], 1000 * 0.6745 * 0.0008, f['lambda_rel'], 1000 * np.median(np.abs(f['field'] - truth))))
