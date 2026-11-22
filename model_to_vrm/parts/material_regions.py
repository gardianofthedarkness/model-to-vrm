"""Material per colour region instead of per vertex (per-vertex SAM 3 votes flip on flat-coloured anime models: sleeves
voted skin, a leg voted cloth):
  python material_regions.py <model dir>        -> sam_body/region_material.npz (material, region, cluster per vertex)
  colours   the texture colour per vertex in Lab; a Gaussian mixture with the number of clusters picked by BIC (2..16,
            on a 50k-vertex sample), every vertex assigned
  regions   vertices of one cluster joined along the mesh (UV-seam twins welded) -> connected regions
  vote      per region, SAM 3's shares summed into materials (concept -> material from sam_body/materials.json) over the
            region's vertices a camera saw; the region takes the largest. Regions no camera saw get -1 (the material
            step fills them along the mesh)."""
import sys, os, json, numpy as np, cv2
from sklearn.mixture import GaussianMixture
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
R = sys.argv[1]; j = lambda *p: os.path.join(R, *p)
O = np.load(j('parts', 'orig_colours.npz')); V, C = O['V'], O['C'].astype(np.float32); F = np.load(j('parts', 'orig_normals.npz'))['F']
Sh = np.load(j('sam_body', 'shares.npz')); share, con = Sh['share'], [str(x) for x in Sh['concepts']]
MAT = json.load(open(j('sam_body', 'materials.json'))); MN = ['flesh', 'cloth', 'hair', 'prop']
M = np.zeros((len(con), 4)); M[np.arange(len(con)), [MN.index(MAT[c]) for c in con]] = 1; ms = share @ M; seen = share.sum(1) > 0
lab = cv2.cvtColor(C.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3)
rng = np.random.default_rng(0); sub = lab[rng.choice(len(lab), min(50000, len(lab)), replace=False)]
fits = [GaussianMixture(k, covariance_type='full', random_state=0).fit(sub) for k in range(2, 17)]; bic = [f.bic(sub) for f in fits]
g = fits[int(np.argmin(bic))]; cl = g.predict(lab); K = g.n_components
_, rep, inv = np.unique(V, axis=0, return_index=True, return_inverse=True); inv = inv.ravel(); d = np.flatnonzero(rep[inv] != np.arange(len(V)))
e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]], np.c_[d, rep[inv[d]]]]; e = e[cl[e[:, 0]] == cl[e[:, 1]]]
n, reg = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V))), directed=False)
score = np.zeros((n, 4)); np.add.at(score, reg[seen], ms[seen]); has = score.sum(1) > 0
rmat = np.where(has, score.argmax(1), -1); mat = rmat[reg]
sz = np.bincount(reg); big = np.argsort(-sz)[:12]
print('MR colour clusters %d (BIC over 2..16); regions %d, %d with SAM 3 votes; vertices without a region vote %d' % (K, n, has.sum(), (mat < 0).sum()))
for r in big: print('MR region %5d  %7d vertices  colour %s  -> %s (%s)' % (r, sz[r], (np.median(C[reg == r], 0) * 255).astype(int), MN[rmat[r]] if rmat[r] >= 0 else 'unseen', ', '.join('%s %.0f' % (MN[i], score[r, i]) for i in range(4))))
print('MR materials:', {MN[i]: int((mat == i).sum()) for i in range(4)})
np.savez(j('sam_body', 'region_material.npz'), material=mat, region=reg, cluster=cl, K=K)
