"""Close-up of an activation run of newton_muscle_active.py: the muscle surface at chosen frames (cropped to the muscle,
coloured by local thickness change), its thickness along the muscle (origin -> insertion) per frame, and activation,
belly thickness and median fibre stretch over time.
  E:\\splat\\.venv\\Scripts\\python.exe muscle_activation_profile.py <mesh npz> <run npz> <out png> frame [frame ...]"""
import sys, os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
M = np.load(sys.argv[1]); D = np.load(sys.argv[2]); OUT = sys.argv[3]; want = [int(x) for x in sys.argv[4:]]
J = json.load(open(sys.argv[2].replace('.npz', '.json'))); rows = J['frames']; fps = 30.0
S, F, P0, fr = D['surf'], D['F'], D['rest'], list(D['frames']); pick = [fr.index(f) for f in want if f in fr]
ph = M['phi'][M['srf'][M['srf'] >= 0]] if False else None
srf = M['srf']; phi_s = np.full(len(srf), np.nan); ok = srf >= 0; phi_s[ok] = M['phi'][srf[ok]]
edges = np.linspace(0, 1, 21); mids = (edges[:-1] + edges[1:]) / 2
def profile(X):
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        m = (phi_s >= a) & (phi_s < b) & np.isfinite(X[:, 0])
        out.append(np.linalg.norm(X[m] - X[m].mean(0), axis=1).mean() if m.sum() > 3 else np.nan)
    return np.array(out)
p0 = profile(P0)
fig = plt.figure(figsize=(17, 9)); gs = fig.add_gridspec(2, len(pick), height_ratios=[1.3, 1])
allP = np.concatenate([S[k] for k in pick]); allP = allP[np.isfinite(allP[:, 0])]; c = (allP.min(0) + allP.max(0)) / 2; h = np.ptp(allP, 0).max() / 2 * 1.02
light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
for j, k in enumerate(pick):
    X = S[k]; ax = fig.add_subplot(gs[0, j], projection='3d'); ratio = profile(X) / p0
    r = np.interp(np.nanmean(phi_s[F], 1), mids, np.nan_to_num(ratio, nan=1.0)); v = np.clip((r - 1) / 0.3, -1, 1)
    col = np.c_[0.75 + 0.25 * np.clip(v, 0, 1), 0.35 * (1 - np.abs(v)), 0.35 + 0.6 * np.clip(-v, 0, 1)]
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    ax.add_collection3d(Poly3DCollection(X[F], facecolors=np.clip(col * (0.45 + 0.55 * np.abs(n @ light))[:, None], 0, 1), linewidths=0))
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h); ax.set_zlim(c[2] - h, c[2] + h); ax.set_box_aspect((1, 1, 1)); ax.view_init(0, 180); ax.axis('off')
    rr = rows[fr[k]]; ax.set_title('t = %.2f s, activation %.2f' % (rr['t'], rr['activation']), fontsize=10)
ax = fig.add_subplot(gs[1, :len(pick) // 2])
for k in pick: ax.plot(mids, profile(S[k]) / p0, marker='.', label='t = %.2f s (a = %.2f)' % (rows[fr[k]]['t'], rows[fr[k]]['activation']))
ax.axhline(1, color='0.6', lw=0.8); ax.set_xlabel('along the muscle: origin (0) -> insertion (1)'); ax.set_ylabel('thickness / rest'); ax.legend(fontsize=8); ax.set_title('Thickness along the muscle', fontsize=10)
ax = fig.add_subplot(gs[1, len(pick) // 2:]); t = [r['t'] for r in rows]
ax.plot(t, [r['activation'] for r in rows], 'k', label='activation a'); ax.plot(t, [r['belly_thickness_ratio'] for r in rows], label='belly thickness / rest')
ax.plot(t, [r['fibre_stretch_p5_50_95'][1] for r in rows], label='median fibre stretch'); ax.plot(t, [r['volume_ratio'] for r in rows], ':', label='volume / rest')
ax.set_xlabel('time (s)'); ax.legend(fontsize=8); ax.set_title('Over time', fontsize=10)
fig.suptitle(os.environ.get('TITLE', 'Muscle activation run') + '  (surface colour: red = thicker than rest, blue = thinner)', fontsize=12)
fig.tight_layout(); fig.savefig(OUT, dpi=80); print('wrote', OUT)
