"""Figure for docs/motion-to-anatomy-mapping.md: skin quality of the Newton flesh run (elf_leg_flesh.py, sit clip) against the
nearest-vertex binding of the one-leg audit.  Numbers only (no picture of the body).
  E:\\splat\\.venv\\Scripts\\python.exe elf_leg_flesh_figure.py <fits dir> <rig_stage dir> <out png>
Skin above and below the ankle are counted apart (ankle height = the tibia points facing the talus, nearest 2 %)."""
import sys, os, json, numpy as np, matplotlib
from scipy.spatial import cKDTree
matplotlib.use('Agg'); import matplotlib.pyplot as plt
FD, RS, OUT = sys.argv[1:4]
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB = R['V'], R['obj']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
tib, tal = V[OB == nid['Tibia.l']], V[OB == nid['Talus.l']]; d = cKDTree(tal).query(tib)[0]; ank_z = tib[d < np.percentile(d, 2)].mean(0)[2]
D = np.load(os.path.join(RS, 'elf_leg_flesh.npz')); S, F, X0 = D['skin'], D['faces'], D['rest']
A = json.load(open(os.path.join(RS, 'ardy_to_elf_leg_calibrated.json')))
n0 = np.cross(X0[F[:, 1]] - X0[F[:, 0]], X0[F[:, 2]] - X0[F[:, 0]]); up = X0[F].mean(1)[:, 2] > ank_z
E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; e0 = np.linalg.norm(X0[E[:, 0]] - X0[E[:, 1]], axis=1); eu = np.r_[up, up, up] & (e0 > 0)
fa, fb, st = [], [], []
for X in S:
    fl = (n0 * np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])).sum(1) < 0; fa.append(100 * fl[up].mean()); fb.append(100 * fl[~up].mean())
    st.append(np.percentile(np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)[eu] / e0[eu], 99))
T = json.load(open(os.path.join(RS, 'elf_leg_flesh.json'))); fps = 30.0; t = np.arange(len(S)) * 15 / fps
L = json.load(open(os.path.join(FD, 'leg_bend_l.json')))['frames'][-1]
fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
ax[0].plot(t, fa, 'o-', label='flesh sim: flipped skin above ankle (%)'); ax[0].plot(t, fb, 's-', label='flesh sim: flipped skin below ankle (%)')
ax[0].axhline(100 * L['skin_triangles_flipped'], c='k', ls='--', label='nearest-vertex audit at hip 60 / knee 90 (%)'); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
ax[0].set_xlabel('time in the sit clip (s)'); ax[0].set_title('Skin triangles flipped', fontsize=10)
ax[1].plot(t, st, 'o-', label='flesh sim, above ankle'); ax[1].axhline(L['skin_edge']['p99'], c='k', ls='--', label='nearest-vertex audit (hip 60 / knee 90)')
ax[1].set_yscale('log'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3); ax[1].set_xlabel('time in the sit clip (s)'); ax[1].set_title('Skin edge stretch, 99th percentile', fontsize=10)
s = T['summary']; fig.suptitle('Elf left leg, Newton VBD flesh (%d tets, %.1f mm edge, %d s on the GTX 1650), sit clip: hip to 85 deg, knee to 102 deg' % (s['tets'], s['tet_edge_mm'], s['seconds']), fontsize=10)
fig.tight_layout(); fig.savefig(OUT, dpi=105); print('wrote', OUT)
