"""Pictures of a muscle tube simulation (muscle_sim.py): the muscle surface at saved frames, coloured by how much each
surface triangle is stretched (red) or squashed (blue) against rest, on the moving leg bones, from her left side.
  E:/splat/.venv/Scripts/python.exe muscle_sim_picture.py <fits dir> <sim npz> <leg_transforms.npz> <out png> [frames ...]"""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
FD, SIMF, TRF, OUT = sys.argv[1:5]; want = [int(x) for x in sys.argv[5:]]
D = np.load(SIMF); S, F, P0, fr = D['surf'], D['F'], D['rest'], list(D['frames'])
pick = [fr.index(f) for f in want if f in fr] or list(range(0, len(fr), max(1, len(fr) // 4)))[:4]
TR = np.load(TRF); fps = float(TR['fps'])
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB, FB = R['V'], R['obj'], R['F_bone']
nm = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]
grp = {nm.index('Hip bone.l'): 0, nm.index('Femur.l'): 1, nm.index('Tibia.l'): 2, nm.index('Fibula.l'): 2, nm.index('Patella.l'): 2}
light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
def shade(P, Fc, col):
    n = np.cross(P[Fc[:, 1]] - P[Fc[:, 0]], P[Fc[:, 2]] - P[Fc[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return np.clip(col * (0.4 + 0.6 * np.abs(n @ light))[:, None], 0, 1)
a0 = np.linalg.norm(np.cross(P0[F[:, 1]] - P0[F[:, 0]], P0[F[:, 2]] - P0[F[:, 0]]), axis=1)
fig = plt.figure(figsize=(4.6 * len(pick), 6.5)); allP = np.concatenate([S[k] for k in pick])
c = (allP.min(0) + allP.max(0)) / 2; h = np.ptp(allP, 0).max() / 2 * 1.15
for j, k in enumerate(pick):
    f = fr[k]; fb_ = int(D['hold']) if 'hold' in D.files and int(D['hold']) >= 0 else f; ax = fig.add_subplot(1, len(pick), j + 1, projection='3d', computed_zorder=False)
    for b, g in grp.items():
        fb = FB[OB[FB[:, 0]] == b][::2]; P = V.astype(float)
        if g: Rm, t = (TR['thigh_R'][fb_], TR['thigh_t'][fb_]) if g == 1 else (TR['shank_R'][fb_], TR['shank_t'][fb_]); P = P @ Rm.T + t      # bones at the held frame in isometric runs
        ax.add_collection3d(Poly3DCollection(P[fb], facecolors=shade(P, fb, np.array([0.85, 0.82, 0.74])), linewidths=0, zorder=1))
    X = S[k]; a1 = np.linalg.norm(np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]), axis=1); r = np.log2(np.clip(a1 / np.maximum(a0, 1e-12), 1 / 4, 4)) / 2
    col = np.c_[0.75 + 0.25 * np.clip(r, 0, 1), 0.3 * (1 - np.abs(r)), 0.3 + 0.6 * np.clip(-r, 0, 1)]
    ax.add_collection3d(Poly3DCollection(X[F], facecolors=shade(X, F, col), linewidths=0, zorder=2))
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h); ax.set_box_aspect((1, 1, 1)); ax.view_init(0, 180); ax.axis('off')
    ax.set_title(os.environ.get('LABEL_%d' % f, 't = %.1f s (frame %d)' % (f / fps, f)), fontsize=9)
fig.suptitle(os.environ.get('TITLE', 'Rectus femoris as a glued tube (passive Blemker muscle)') + ', ARDY "sit", from her left: red = surface stretched, blue = squashed (full = 4x)', fontsize=10)
fig.tight_layout(); fig.savefig(OUT, dpi=90); print('wrote', OUT)
