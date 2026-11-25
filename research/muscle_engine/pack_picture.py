"""Picture of an anatomy pack: cross-sections across the limb (tets cut by each plane, coloured by region; bone grey, fat
yellow, each muscle its own colour) and one front view of the muscle and bone regions' outlines, with a table of region
volumes against the source.
  E:\\splat\\.venv\\Scripts\\python.exe pack_picture.py <pack.npz> <out.png> [n slices]"""
import sys, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
PK, OUT = sys.argv[1:3]; NS = int(sys.argv[3]) if len(sys.argv) > 3 else 6
P = np.load(PK, allow_pickle=True); X, T, REG = P['X'], P['T'], P['region']; NM = [str(s) for s in P['region_names']]; KD = [str(s) for s in P['region_kind']]
src = json.loads(str(P['source_volume_cm3'])) if 'source_volume_cm3' in P.files else {}
rng = np.random.default_rng(0); cmap = plt.get_cmap('tab20')
col = np.array([(0.55, 0.55, 0.55, 1) if k == 'bone' else (1.0, 0.85, 0.35, 1) if k == 'fat' else cmap(i % 20) for i, k in enumerate(KD)])
def cut(z):
    """polygons where the plane z cuts each tet"""
    Z = X[T][:, :, 2] - z; s = np.sign(Z); hit = (s.min(1) < 0) & (s.max(1) > 0); polys, regs = [], []
    for t, r in zip(T[hit], REG[hit]):
        p = X[t]; d = p[:, 2] - z; pts = []
        for a in range(4):
            for b in range(a + 1, 4):
                if d[a] * d[b] < 0: w = d[a] / (d[a] - d[b]); pts.append(p[a] + w * (p[b] - p[a]))
        pts = np.array(pts)[:, :2]; c = pts.mean(0); o = np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])); polys.append(pts[o]); regs.append(r)
    return polys, np.array(regs, int)
z0, z1 = X[:, 2].min(), X[:, 2].max(); zs = np.linspace(z0, z1, NS + 2)[1:-1][::-1]
fig = plt.figure(figsize=(4 * NS / 2 + 6, 9)); gs = fig.add_gridspec(2, NS // 2 + 2)
for k, z in enumerate(zs):
    ax = fig.add_subplot(gs[k // (NS // 2), k % (NS // 2)]); pl, rg = cut(z)
    ax.add_collection(PolyCollection(pl, facecolors=col[rg], edgecolors='none')); ax.autoscale(); ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])
    ax.set_title('z = %.0f mm' % (1e3 * z), fontsize=9)
ax = fig.add_subplot(gs[:, NS // 2]); m = np.isin(REG, [i for i, k in enumerate(KD) if k != 'fat']); c = X[T[m]].mean(1)
o = np.argsort(c[:, 1]); ax.scatter(c[o, 0], c[o, 2], s=0.3, c=col[REG[m]][o]); ax.set_aspect('equal'); ax.set_title('front view, muscle and bone tets', fontsize=9); ax.set_xticks([]); ax.set_yticks([])
D = np.stack([X[T[:, i]] - X[T[:, 0]] for i in (1, 2, 3)], 2); v = np.abs(np.linalg.det(D)) / 6
rows = [(n, '%.1f' % (v[REG == i].sum() * 1e6), '%s' % src.get(n, '-')) for i, n in enumerate(NM)]
ax = fig.add_subplot(gs[:, NS // 2 + 1]); ax.axis('off')
tb = ax.table(cellText=[[r[0].replace(' muscle', '').replace('.l', '')[:26], r[1], r[2]] for r in rows], colLabels=['region', 'pack cm3', 'source cm3'], loc='center',
              cellColours=[[col[i], 'w', 'w'] for i in range(len(rows))])
tb.auto_set_font_size(False); tb.set_fontsize(6); tb.scale(1, 0.9)
fig.suptitle('%s: %d nodes, %d tets, %d regions' % (PK.replace('\\', '/').split('/')[-1], len(X), len(T), len(NM)), fontsize=11)
plt.savefig(OUT, dpi=110, bbox_inches='tight'); print('wrote', OUT)
