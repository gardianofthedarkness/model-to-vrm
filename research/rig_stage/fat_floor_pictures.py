"""Pictures of the fat layer set-up in skin_fat.py (no solve): (1) fat depth on the welded leg skin at rest; (2) the fat's
floor (inner shell, moved with the deep tissue) at one clip frame, with the muscle-rig shares and with harmonic shares,
coloured by edge stretch (blue = flipped triangles).
  E:\\splat\\.venv\\Scripts\\python.exe fat_floor_pictures.py <fits dir> <model dir> <leg_transforms.npz> <out prefix> [frame]"""
import sys, os, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
FD, MD, TRF, PRE = sys.argv[1:5]; FRAME = int(sys.argv[5]) if len(sys.argv) > 5 else 105
HERE = os.path.dirname(os.path.abspath(__file__)); SRC = open(os.path.join(HERE, 'skin_fat.py'), encoding='utf-8').read()
IA, IB, IC = SRC.index('# --- global matrix over all 2n nodes'), SRC.index('def dq_blend'), SRC.index('def polar')
def setup(shares):
    os.environ['SHARES'] = shares; g = {}; sys.argv = ['x', FD, MD, TRF, os.path.dirname(PRE)]
    exec(SRC[:IA], g); exec(SRC[IB:IC], g); return g
light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
def draw(ax, P, F, col, title, el, az, lim):
    n = np.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    k = 0.45 + 0.55 * np.abs(n @ light); ax.add_collection3d(Poly3DCollection(P[F], facecolors=np.clip(col * k[:, None], 0, 1), linewidths=0))
    c, half = lim; ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half); ax.set_zlim(c[2] - half, c[2] + half)
    ax.view_init(elev=el, azim=az); ax.set_box_aspect((1, 1, 1)); ax.axis('off'); ax.set_title(title, fontsize=9)
g = setup('rig'); P0, F, h, Q0 = g['P0'], g['F'], g['h'], g['Q0']
lo, hi = P0.min(0), P0.max(0); lim = ((lo + hi) / 2, (hi - lo).max() / 2)
cm = plt.get_cmap('viridis'); hv = np.clip(h[F].mean(1) / np.percentile(h, 95), 0, 1); col = cm(hv)[:, :3]
fig = plt.figure(figsize=(10, 8))
for i, (nm, (el, az)) in enumerate((('from her left side', (0, 180)), ('from the front', (0, -90)))):
    draw(fig.add_subplot(1, 2, i + 1, projection='3d'), P0, F, col, 'fat depth at rest, %s' % nm, el, az, lim)
fig.suptitle('Elf left leg: room between skin and muscle/bone, filled with fat (median %.1f mm; yellow = %.0f mm or more)' % (1000 * np.median(h), 1000 * np.percentile(h, 95)), fontsize=10)
fig.tight_layout(); fig.savefig(PRE + '_fat_depth.png', dpi=90); plt.close(fig); print('wrote', PRE + '_fat_depth.png')
fig = plt.figure(figsize=(10, 8))
for i, sh in enumerate(('rig', 'harmonic')):
    g = setup(sh); Q0, F, Wa, anc, d_in, V = g['Q0'], g['F'], g['Wa'], g['anc'], g['d_in'], g['V']
    Qa, Ra = g['dq_blend'](V[anc], Wa, g['transforms'](FRAME)); Qf = Qa + np.einsum('nij,nj->ni', Ra, d_in)
    E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; e0 = np.linalg.norm(Q0[E[:, 0]] - Q0[E[:, 1]], axis=1); r = np.ones(len(E)); ok = e0 > 1e-9
    r[ok] = np.linalg.norm(Qf[E[ok, 0]] - Qf[E[ok, 1]], axis=1) / e0[ok]; r = np.maximum(r, 1 / np.maximum(r, 1e-9)).reshape(3, -1).max(0)
    s = np.clip(np.log2(r) / 2, 0, 1); colr = np.c_[np.ones(len(F)), 1 - 0.85 * s, 1 - 0.9 * s] * np.array([0.93, 0.82, 0.72])
    flip = (np.cross(Q0[F[:, 1]] - Q0[F[:, 0]], Q0[F[:, 2]] - Q0[F[:, 0]]) * np.cross(Qf[F[:, 1]] - Qf[F[:, 0]], Qf[F[:, 2]] - Qf[F[:, 0]])).sum(1) < 0; colr[flip] = [0.15, 0.35, 0.95]
    lo, hi = Qf.min(0), Qf.max(0); lim2 = ((lo + hi) / 2, (hi - lo).max() / 2)
    draw(fig.add_subplot(1, 2, i + 1, projection='3d'), Qf, F, colr, '%s shares: flipped %.1f%%' % ('muscle-rig' if sh == 'rig' else 'smooth (harmonic)', 100 * flip.mean()), 0, 180, lim2)
fig.suptitle('Fat floor (rides the muscle/bone) at t = %.1f s of "sit", seen from her left: red = stretched or squashed (full = 4x), blue = flipped' % (FRAME / 30), fontsize=10)
fig.tight_layout(); fig.savefig(PRE + '_floor_f%d.png' % FRAME, dpi=90); print('wrote', PRE + '_floor_f%d.png' % FRAME)
