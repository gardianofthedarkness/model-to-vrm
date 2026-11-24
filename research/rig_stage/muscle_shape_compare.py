"""Picture: a muscle's atlas shape (scaled uniformly to the fitted length) against its fitted, repaired shape on the elf.
  E:\splat\.venv\Scripts\python.exe muscle_shape_compare.py <fits dir> <out png> "<muscle>" ["<muscle>" ...]"""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
FD, OUT = sys.argv[1:3]; ms = sys.argv[3:]
W = np.load(os.path.join(FD, 'atlas_warp.npz')); names = [str(x) for x in W['names']]; R = np.load(os.path.join(FD, 'muscle_repair.npz')); OB, FM = R['obj'], R['F_muscle']
light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
fig = plt.figure(figsize=(4.2 * len(ms), 8))
for j, m in enumerate(ms):
    f = FM[OB[FM[:, 0]] == names.index(m)]; u, inv = np.unique(f.ravel(), return_inverse=True); f = inv.reshape(-1, 3)
    for r, (lab, P) in enumerate((('atlas, scaled to the fitted length', W['V0'][u].astype(float)), ('fitted and repaired on the elf', R['V'][u].astype(float)))):
        P = P - P.mean(0)
        if r == 0:
            L1 = np.ptp(R['V'][u] @ np.linalg.svd(R['V'][u] - R['V'][u].mean(0))[2][0]); L0 = np.ptp(P @ np.linalg.svd(P)[2][0]); P = P * L1 / L0
        vol = abs(np.einsum('ij,ij->i', P[f[:, 0]], np.cross(P[f[:, 1]], P[f[:, 2]])).sum()) / 6
        ax = fig.add_subplot(2, len(ms), r * len(ms) + j + 1, projection='3d'); n = np.cross(P[f[:, 1]] - P[f[:, 0]], P[f[:, 2]] - P[f[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        k = 0.4 + 0.6 * np.abs(n @ light); ax.add_collection3d(Poly3DCollection(P[f], facecolors=np.c_[0.72 * k, 0.28 * k, 0.25 * k], linewidths=0))
        h = np.abs(P).max(); ax.set_xlim(-h, h); ax.set_ylim(-h, h); ax.set_zlim(-h, h); ax.set_box_aspect((1, 1, 1)); ax.view_init(10, 200); ax.axis('off')
        ax.set_title('%s\n%s: %.1f cm3' % (m.replace(' muscle', ''), lab, 1e6 * vol), fontsize=8)
fig.tight_layout(); fig.savefig(OUT, dpi=90); print('wrote', OUT)
