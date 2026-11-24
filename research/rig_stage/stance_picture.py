"""Still picture of the loaded-vs-hanging thigh (stance_runs.sh output): each muscle's final surface in both cases, coloured
by how far the loaded surface sits from the hanging one (mm, per vertex), from the front and the side; plus a table of the
MyoSuite activation and the end-of-run belly thickness ratio per case.
  E:\\splat\\.venv\\Scripts\\python.exe stance_picture.py <sim dir> <myo_single_leg.json> <out png>"""
import sys, os, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SD, MJ, OUT = sys.argv[1:4]; J = json.load(open(MJ))
rows, pts = [], []
for m in J['cases']['loaded']['activation']:
    if m.startswith('Sartorius'): continue
    tag = m.replace(' ', '_').replace('.', '_'); fin = {}
    for case in ('loaded', 'hanging'):
        a = J['cases'][case]['activation'][m]; f = os.path.join(SD, 'stance', '%s_a%.4f' % (tag, a))
        S = np.load(f + '.npz')['surf'][-1].astype(np.float64); fin[case] = (a, S, json.load(open(f + '.json'))['frames'][-1]['belly_thickness_ratio'])
    d = np.linalg.norm(fin['loaded'][1] - fin['hanging'][1], axis=1) * 1e3
    rows.append((m.replace(' muscle', '').replace('.l', ''), fin['hanging'][0], fin['loaded'][0], fin['hanging'][2], fin['loaded'][2], np.percentile(d, 95)))
    pts.append((fin['loaded'][1], fin['hanging'][1], d))
fig = plt.figure(figsize=(15, 8.5)); vmax = max(r[5] for r in rows)
for k, (ax_i, ax_j, title) in enumerate(((0, 2, 'front (x-z)'), (1, 2, 'side (y-z)'))):
    for c, (case, col) in enumerate((('hanging', 1), ('loaded', 0))):
        ax = fig.add_subplot(2, 3, 1 + k * 3 + c)
        for P in pts:
            Q = P[col]; o = np.argsort(Q[:, 1 - ax_i if ax_i < 2 else 0]); sc = ax.scatter(Q[:, ax_i], Q[:, ax_j], c=P[2], s=0.4, cmap='magma', vmin=0, vmax=vmax)
        ax.set_aspect('equal'); ax.set_title('%s, %s' % (case, title), fontsize=10); ax.set_xticks([]); ax.set_yticks([])
fig.colorbar(sc, ax=fig.axes, shrink=0.5, location='left', label='loaded vs hanging surface distance (mm)')
ax = fig.add_subplot(1, 3, 3); ax.axis('off')
tab = ax.table(cellText=[[r[0], '%.3f' % r[1], '%.3f' % r[2], '%.2f' % r[3], '%.2f' % r[4], '%.1f' % r[5]] for r in rows],
               colLabels=['muscle', 'a hang', 'a load', 'belly hang', 'belly load', 'p95 mm'], loc='center', cellLoc='center')
tab.auto_set_font_size(False); tab.set_fontsize(8.5); tab.scale(1, 1.6); tab.auto_set_column_width(range(6))
ax.set_title('Same pose (ARDY sit clip frame %d), left leg\nhanging vs carrying %.2f bw; end of 3 s quasi-static hold' % (J['clip_frame'], J['cases']['loaded']['left_foot_bw']), fontsize=10)
plt.savefig(OUT, dpi=110, bbox_inches='tight'); print('wrote', OUT)
for r in rows: print('SP %-32s a %.4f/%.4f belly %.3f/%.3f p95 %.1f mm' % r)
