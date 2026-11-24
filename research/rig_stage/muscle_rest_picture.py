"""Picture: the rebuilt muscle tubes (muscle_rest.py) on the elf's fitted left-leg bones at rest, from her left and from the front.
  E:/splat/.venv/Scripts/python.exe muscle_rest_picture.py <fits dir> <muscle_rest.npz> <out png>"""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
FD, RF, OUT = sys.argv[1:4]
D = np.load(RF); names = [str(x) for x in D['names']]
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB, FB = R['V'], R['obj'], R['F_bone']
nm = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]
bones = [nm.index(b + '.l') for b in ('Hip bone', 'Femur', 'Tibia', 'Fibula', 'Patella')]
light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
def shade(P, F, c):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return np.clip(np.outer(0.4 + 0.6 * np.abs(n @ light), c), 0, 1)
cols = plt.get_cmap('tab10')
fig = plt.figure(figsize=(11, 9))
for v, (title, el, az) in enumerate((('from her left side', 0, 180), ('from the front', 0, -90))):
    ax = fig.add_subplot(1, 2, v + 1, projection='3d', computed_zorder=False)
    fb = FB[np.isin(OB[FB[:, 0]], bones)][::2]; ax.add_collection3d(Poly3DCollection(V[fb], facecolors=shade(V, fb, np.array([0.85, 0.82, 0.74])), linewidths=0, zorder=1))
    allP = [V[np.unique(fb)]]
    for k, m in enumerate(names):
        if 'gastrocnemius' in m: continue
        P, F = D['%d_P' % k], D['%d_F' % k]; allP.append(P)
        ax.add_collection3d(Poly3DCollection(P[F], facecolors=np.c_[shade(P, F, np.array(cols(k)[:3])), np.full(len(F), 0.8)], linewidths=0, zorder=2))
    A = np.concatenate(allP); c = (A.min(0) + A.max(0)) / 2; h = np.ptp(A, 0).max() / 2
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h); ax.set_box_aspect((1, 1, 1)); ax.view_init(el, az); ax.axis('off'); ax.set_title(title, fontsize=9)
from matplotlib.patches import Patch
fig.legend(handles=[Patch(color=cols(k)[:3], label=m.replace(' muscle', '').replace('.l', '')) for k, m in enumerate(names) if 'gastrocnemius' not in m], loc='lower center', ncol=4, fontsize=8)
fig.suptitle('Elf left leg: muscles rebuilt as tubes between their fitted attachments (atlas shape, uniform scale), on the fitted bones', fontsize=10)
fig.tight_layout(); fig.savefig(OUT, dpi=90); print('wrote', OUT)
