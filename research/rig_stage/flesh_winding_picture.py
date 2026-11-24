"""Before/after picture of the tet-winding fix in elf_leg_flesh.py: the leg skin at saved frames of the sit clip for the
old run (all tets silently dropped by Newton) and the fixed run, coloured by how much each skin triangle is stretched (red)
or squashed (blue) against rest, plus inverted tets and flipped skin triangles over the clip.
  E:\\splat\\.venv\\Scripts\\python.exe flesh_winding_picture.py <old run dir> <fixed run dir> <out png> [saved index ...]"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
OLD, NEW, OUT = sys.argv[1:4]; pick = [int(x) for x in sys.argv[4:]] or [0, 6, 10]
runs = [('old (0 tets in the model)', OLD), ('fixed winding (18 753 tets)', NEW)]
light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
fig = plt.figure(figsize=(4.2 * len(pick), 11)); gs = fig.add_gridspec(3, len(pick), height_ratios=[1, 1, 0.7])
for r, (lab, d) in enumerate(runs):
    D = np.load(os.path.join(d, 'elf_leg_flesh.npz')); S, F, X0 = D['skin'], D['faces'], D['rest']
    a0 = np.linalg.norm(np.cross(X0[F[:, 1]] - X0[F[:, 0]], X0[F[:, 2]] - X0[F[:, 0]]), axis=1)
    n0 = np.cross(X0[F[:, 1]] - X0[F[:, 0]], X0[F[:, 2]] - X0[F[:, 0]])
    for j, k in enumerate(pick):
        X = S[k].astype(float); ax = fig.add_subplot(gs[r, j], projection='3d')
        n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]); a1 = np.linalg.norm(n, axis=1)
        q = np.log2(np.clip(a1 / np.maximum(a0, 1e-12), 1 / 4, 4)) / 2; flip = (n * n0).sum(1) < 0
        col = np.c_[0.75 + 0.25 * np.clip(q, 0, 1), 0.72 * (1 - np.abs(q)), 0.55 + 0.45 * np.clip(-q, 0, 1)]; col[flip] = [0.1, 0.1, 0.1]
        nn = n / (a1[:, None] + 1e-12); ax.add_collection3d(Poly3DCollection(X[F], facecolors=np.clip(col * (0.45 + 0.55 * np.abs(nn @ light))[:, None], 0, 1), linewidths=0))
        c = (X.min(0) + X.max(0)) / 2; h = np.ptp(X, 0).max() / 2
        ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h); ax.set_box_aspect((1, 1, 1)); ax.view_init(0, 180); ax.axis('off')
        ax.set_title('%s\nclip frame %d (t = %.1f s), %.1f%% skin triangles flipped' % (lab, 15 * k, 15 * k / 30, 100 * flip.mean()), fontsize=9)
ax = fig.add_subplot(gs[2, :])
for lab, d in runs:
    J = json.load(open(os.path.join(d, 'elf_leg_flesh.json'))); fr = J['frames']; t = [r['frame'] / 30 for r in fr]; nt = J['summary']['tets']
    ax.plot(t, [100 * r['inverted_tets'] / nt for r in fr], label='inverted tets %%, %s' % lab)
    ax.plot(t, [100 * r['skin_flipped'] for r in fr], '--', label='flipped skin triangles %%, %s' % lab)
ax.set_xlabel('time in the sit clip (s)'); ax.set_ylabel('%'); ax.legend(fontsize=8); ax.grid(alpha=0.3)
fig.suptitle('Elf leg flesh on the ARDY sit clip, before and after fixing the tet winding (colour: red = skin stretched, blue = squashed, black = flipped)', fontsize=11)
fig.tight_layout(); fig.savefig(OUT, dpi=80); print('wrote', OUT)
