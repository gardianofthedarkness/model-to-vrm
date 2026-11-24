"""Dynamic vs quasi-static (STATIC=1) runs of the active-strain muscle: the straight block with one end free
(muscle_block_test.py) and the rectus tube held isometric at clip frame 0 (newton_muscle_active.py HOLD=0), plus the
rectus surface at full activation in both runs.
  E:\\splat\\.venv\\Scripts\\python.exe muscle_static_picture.py <block dyn json> <block static json> <rectus dyn npz> <rectus static npz> <out png>"""
import sys, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
BD, BS, RD, RS, OUT = sys.argv[1:6]
fig = plt.figure(figsize=(17, 10)); gs = fig.add_gridspec(2, 4)
for j, (key, lab) in enumerate((('length_ratio', 'block length / rest'), ('bend_mm', 'block bend (mm)'))):
    ax = fig.add_subplot(gs[0, j])
    for path, ls, nm in ((BD, '-', 'dynamic'), (BS, '--', 'quasi-static')):
        for r in json.load(open(path))['runs']:
            if r['case'] != 'free': continue
            h = r['history']; ax.plot([x['t'] for x in h], [x[key] for x in h], ls, label='%s, a = %.2f' % (nm, r['activation']))
            if key == 'length_ratio': ax.axhline(r['last_second']['expected_length_ratio_if_free'], color='0.8', lw=0.6)
    ax.set_xlabel('time (s)'); ax.set_title('Straight block, one end free: ' + lab, fontsize=10); ax.legend(fontsize=7); ax.grid(alpha=0.3)
light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
for j, (path, nm) in enumerate(((RD, 'dynamic'), (RS, 'quasi-static'))):
    D = np.load(path); J = json.load(open(path.replace('.npz', '.json'))); rows = J['frames']; S, F, fr = D['surf'], D['F'], list(D['frames'])
    k = int(np.argmin([abs(rows[f]['t'] - 1.4) for f in fr]))        # late in the active phase (excitation on 0.5-1.5 s)
    X = S[k]; ax = fig.add_subplot(gs[0, 2 + j], projection='3d'); P = X[np.isfinite(X[:, 0])]; c = (P.min(0) + P.max(0)) / 2; h = np.ptp(P, 0).max() / 2
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ax.add_collection3d(Poly3DCollection(X[F], facecolors=np.clip(np.array([0.8, 0.35, 0.3]) * (0.45 + 0.55 * np.abs(n @ light))[:, None], 0, 1), linewidths=0))
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h); ax.set_box_aspect((1, 1, 1)); ax.view_init(0, 180); ax.axis('off')
    ax.set_title('Rectus, isometric, %s\nt = %.2f s, activation %.2f' % (nm, rows[fr[k]]['t'], rows[fr[k]]['activation']), fontsize=10)
for j, (key, lab) in enumerate((('belly_thickness_ratio', 'rectus belly thickness / rest'), ('fibre_stretch_p5_50_95', 'rectus median fibre stretch'), ('volume_ratio', 'rectus volume / rest'))):
    ax = fig.add_subplot(gs[1, j])
    for path, ls, nm in ((RD, '-', 'dynamic'), (RS, '--', 'quasi-static')):
        rows = json.load(open(path.replace('.npz', '.json')))['frames']; v = [r[key][1] if isinstance(r[key], list) else r[key] for r in rows]
        ax.plot([r['t'] for r in rows], v, ls, label=nm)
    ax.axvspan(0.5, 1.5, color='0.92', zorder=0); ax.set_xlabel('time (s); grey = excitation on'); ax.set_title(lab, fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = fig.add_subplot(gs[1, 3])
for path, ls, nm in ((BD, '-', 'dynamic'), (BS, '--', 'quasi-static')):
    for r in json.load(open(path))['runs']:
        h = r['history']; ax.plot([x['t'] for x in h], [x['fibre_stress_Pa_p5_50_95'][1] for x in h], ls, label='%s %s a=%.2f' % (nm, r['case'], r['activation']))
ax.set_xlabel('time (s)'); ax.set_ylabel('Pa'); ax.set_title('Block median fibre stress (+ = tension)', fontsize=10); ax.legend(fontsize=6); ax.grid(alpha=0.3)
fig.suptitle('Active-strain muscle: is the folding/S-bend inertia (dynamic) or the equilibrium (quasi-static)?', fontsize=12)
fig.tight_layout(); fig.savefig(OUT, dpi=80); print('wrote', OUT)
