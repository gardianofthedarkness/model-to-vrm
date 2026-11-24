"""Still pictures of the elf's left-leg flesh run (elf_leg_flesh.py): skin coloured by stretch, and the fitted bones inside,
at chosen saved frames, from her left side and from the front.
  E:\\splat\\.venv\\Scripts\\python.exe leg_pictures.py <fits dir> <rig_stage dir> <out png> [saved frame indices]"""
import sys, os, re, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
FD, RS, OUT = sys.argv[1:4]; pick = [int(x) for x in sys.argv[4:]] or [0, 3, 7]
D = np.load(os.path.join(RS, os.environ.get('SKIN', 'elf_leg_flesh.npz'))); S, F, X0 = D['skin'], D['faces'], D['rest']
FR = [int(x) for x in os.environ['FRAMES'].split(',')] if os.environ.get('FRAMES') else None    # clip frame of each saved skin (default every 15th)
TR = np.load(os.path.join(RS, 'sit_leg_transforms.npz')); step = 15; fps = float(TR['fps'])
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB, FB = R['V'], R['obj'], R['F_bone']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
foot = [i for i, n in enumerate(names) if n.endswith('.l') and re.search(r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform|phalan', n)]
groups = {0: [nid['Hip bone.l']], 1: [nid['Femur.l']], 2: [nid['Tibia.l'], nid['Fibula.l'], nid['Patella.l']] + foot}
E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; e0 = np.linalg.norm(X0[E[:, 0]] - X0[E[:, 1]], axis=1); ok = e0 > 0
n0 = np.cross(X0[F[:, 1]] - X0[F[:, 0]], X0[F[:, 2]] - X0[F[:, 0]])
light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
def shade(P, Fc, base):
    n = np.cross(P[Fc[:, 1]] - P[Fc[:, 0]], P[Fc[:, 2]] - P[Fc[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    k = 0.45 + 0.55 * np.abs(n @ light); return np.clip(base * k[:, None], 0, 1)
def bones_at(k):
    out = []
    for g, ids in groups.items():
        fb = FB[np.isin(OB[FB[:, 0]], ids)][::3]; P = V.astype(np.float64)
        if g:
            Rm, t = (TR['thigh_R'], TR['thigh_t']) if g == 1 else (TR['shank_R'], TR['shank_t']); f = FR[k] if FR else min(k * step, len(Rm) - 1); P = P @ Rm[f].T + t[f]
        out.append((P, fb))
    return out
views = [('from her left side', (0, 180)), ('from the front', (0, -90))]
fig = plt.figure(figsize=(5.2 * len(pick), 10.5))
allP = np.concatenate([S[k] for k in pick]); lo, hi = allP.min(0), allP.max(0); c = (lo + hi) / 2; half = (hi - lo).max() / 2
for row, (vname, (el, az)) in enumerate(views):
    for col, k in enumerate(pick):
        ax = fig.add_subplot(2, len(pick), row * len(pick) + col + 1, projection='3d', computed_zorder=False)
        X = S[k]; r = np.ones(len(E)); r[ok] = np.linalg.norm(X[E[ok, 0]] - X[E[ok, 1]], axis=1) / e0[ok]; r = np.maximum(r, 1 / np.maximum(r, 1e-6))
        sv = np.ones(len(X)); np.maximum.at(sv, E[:, 0], r); np.maximum.at(sv, E[:, 1], r); s = np.clip(np.log2(sv[F].max(1)) / 2, 0, 1)
        flip = (n0 * np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])).sum(1) < 0
        base = np.c_[np.ones(len(F)), 1 - 0.85 * s, 1 - 0.9 * s] * np.array([0.93, 0.82, 0.72]); base[flip] = [0.15, 0.35, 0.95]
        for P, fb in bones_at(k): ax.add_collection3d(Poly3DCollection(P[fb], facecolors=shade(P, fb, np.array([0.55, 0.52, 0.45])), linewidths=0, zorder=1))
        ax.add_collection3d(Poly3DCollection(X[F], facecolors=np.c_[shade(X, F, base), np.full(len(F), 0.35)], linewidths=0, zorder=2))
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half); ax.set_zlim(c[2] - half, c[2] + half)
        ax.view_init(elev=el, azim=az); ax.set_box_aspect((1, 1, 1)); ax.axis('off')
        ax.set_title('t = %.1f s, %s\nflipped skin %.1f%%' % ((FR[k] if FR else k * step) / fps, vname, 100 * flip.mean()), fontsize=9)
fig.suptitle(os.environ.get('TITLE', 'Elf left leg, Newton flesh') + ', ARDY "sit": skin tinted red where stretched or squashed (full red = 4x), blue = flipped triangles; bones inside', fontsize=10)
fig.tight_layout(); fig.savefig(OUT, dpi=90); print('wrote', OUT)
