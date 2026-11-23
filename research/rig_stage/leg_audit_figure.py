"""Figure for docs/rig-stage-strategies.md: what the existing one-leg rig audit (model_to_vrm/rig/leg_bend_test.py, elf,
hip 60 deg / knee 90 deg) shows about binding the visible skin to the fitted anatomy by nearest vertex.
  E:\\splat\\.venv\\Scripts\\python.exe leg_audit_figure.py <fits dir> <out png>
Reads only leg_bend_<side>.json/.npz (numbers and the elf's own fitted mesh points; the elf is a fictional figure).
Left: skin points of the bent leg seen from the side, coloured by each point's largest edge stretch (rest = 1).
Right: per-frame skin edge p99 stretch, flipped skin triangles, and closed-muscle volume ratio range."""
import sys, os, json, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
FD, OUT = sys.argv[1], sys.argv[2]
J = json.load(open(os.path.join(FD, 'leg_bend_l.json'))); D = np.load(os.path.join(FD, 'leg_bend_l.npz'))
S, F = D['skin'], D['skin_faces']
e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]
L0 = np.linalg.norm(S[0][e[:, 0]] - S[0][e[:, 1]], axis=1); L1 = np.linalg.norm(S[-1][e[:, 0]] - S[-1][e[:, 1]], axis=1)
ok = L0 > 0; st = np.ones(len(S[0])); r = np.zeros(len(e)); r[ok] = L1[ok] / L0[ok]
np.maximum.at(st, e[:, 0], r); np.maximum.at(st, e[:, 1], r)
fig = plt.figure(figsize=(11, 5.2))
ax = fig.add_subplot(1, 2, 1)
P = S[-1]; c = np.log2(np.clip(st, 1, None))
o = np.argsort(c); sc = ax.scatter(P[o, 1], P[o, 2], c=c[o], s=0.4, cmap='magma', vmin=0, vmax=np.percentile(c, 99.5))
ax.scatter(S[0][::7, 1], S[0][::7, 2], s=0.2, c='0.75', zorder=0)
ax.set_aspect('equal'); ax.set_xlabel('front-back (m)'); ax.set_ylabel('up (m)')
ax.set_title('Elf left leg: grey = rest, colour = bent\n(colour: log2 of the largest edge stretch at each skin point)', fontsize=9)
plt.colorbar(sc, ax=ax, fraction=0.04, label='log2 stretch')
ax2 = fig.add_subplot(1, 2, 2); fr = J['frames']; u = [f['hip_deg'] for f in fr]
ax2.plot(u, [f['skin_edge']['p99'] for f in fr], 'o-', label='skin edge stretch, p99')
ax2.plot(u, [100 * f['skin_triangles_flipped'] for f in fr], 's-', label='skin triangles flipped (%)')
ax2.plot(u, [f['closed_muscle_volume_ratio']['max'] for f in fr], '^-', label='muscle volume ratio, max')
ax2.plot(u, [f['closed_muscle_volume_ratio']['min'] for f in fr], 'v-', label='muscle volume ratio, min')
ax2.set_yscale('log'); ax2.set_xlabel('hip flexion (deg); knee rises with it to 90'); ax2.grid(alpha=.3); ax2.legend(fontsize=8)
ax2.set_title('Nearest-vertex skin binding + dual-quaternion muscles\n(skin binding median %.1f mm, max %.1f mm)'
              % (J['skin_binding_mm']['median'], J['skin_binding_mm']['max']), fontsize=9)
fig.tight_layout(); fig.savefig(OUT, dpi=110); print('wrote', OUT)
