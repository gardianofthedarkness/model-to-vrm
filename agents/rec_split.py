"""One recursion step of the part split: a part looked at alone, then split where its relief boundaries and SAM 3 agree:
  python rec_split.py <model dir> <part dir>
  part dir: face_ids.npy (the part's faces in the model), body/ (views of the part alone, SAM 3 masks, vis.npz),
            parts/orig_colours.npz (the part's own vertices) - made by part_glb.py, body_views.py, hf_sam3_jobs.py,
            part_colours.py, views_vis.py
  1  vote    SAM 3 on the part alone (body_vote.py) -> a concept per part vertex, carried to the model's vertices by position
  2  regions the model's relief boundaries (sam_body/relief.npz, geo_colour_parts.py) inside the part: regions between
             them, boundary vertices joined to the region most of their neighbours are in, faces by majority
  3  filter  neighbouring regions with the same dominant SAM 3 concept are joined: a boundary stays only where both agree
  4  crumbs  groups below the Otsu split of the log group sizes go to the neighbouring group closest in true colour
  writes <part dir>/split.npz: sub (sub-part per part face), names (dominant SAM 3 concept per sub-part)"""
import sys, os, subprocess, numpy as np, cv2
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
R, D = sys.argv[1:3]; j = lambda *p: os.path.join(R, *p); d = lambda *p: os.path.join(D, *p)
PY = r'E:\splat\.venv\Scripts\python.exe'; VOTE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'model_to_vrm', 'parts', 'body_vote.py')
if not os.path.exists(d('sam', 'top_concept.npz')):
    subprocess.run([PY, VOTE, d('body'), d('parts', 'orig_colours.npz'), '-', d('sam')], check=True, capture_output=True)
T = np.load(d('sam', 'top_concept.npz'), allow_pickle=True); plab, pnames = np.asarray(T['label']).astype(int), [str(x) for x in T['names']]
O = np.load(j('parts', 'orig_colours.npz')); V, C = O['V'], O['C'].astype(np.float32); F = np.load(j('parts', 'orig_normals.npz'))['F']
PV = np.load(d('parts', 'orig_colours.npz'))['V']; fi = np.load(d('face_ids.npy'))
vin = np.unique(F[fi]); dd, k = cKDTree(PV).query(V[vin]); con = np.full(len(V), -1); con[vin] = np.where(dd < 1e-6, plab[k], -1)
print('RS part: %d faces, %d vertices; SAM 3 on the part alone: %d vertices labelled, concepts %s' % (len(fi), len(vin), (con[vin] >= 0).sum(),
      {pnames[c]: int((con[vin] == c).sum()) for c in np.unique(con[vin]) if c >= 0}), flush=True)
RL = np.load(j('sam_body', 'relief.npz')); UW = RL['welded']; bnd_u = RL['boundary']
_, rep, inv = np.unique(V, axis=0, return_index=True, return_inverse=True); Wd = rep[inv.ravel()]; uinv = np.searchsorted(UW, Wd)
FU = uinv[F[fi]]; nU = len(UW); inpart = np.zeros(nU, bool); inpart[FU.ravel()] = True
EU = np.unique(np.sort(np.r_[FU[:, [0, 1]], FU[:, [1, 2]], FU[:, [2, 0]]], 1), axis=0)
ok = ~bnd_u[EU[:, 0]] & ~bnd_u[EU[:, 1]]
nc, vc = connected_components(csr_matrix((np.ones(ok.sum()), (EU[ok, 0], EU[ok, 1])), shape=(nU, nU)), directed=False); vc = vc.copy(); vc[bnd_u | ~inpart] = -1
for _ in range(200):
    a_ = np.r_[EU[:, 0], EU[:, 1]]; b_ = np.r_[EU[:, 1], EU[:, 0]]; m = (vc[a_] < 0) & (vc[b_] >= 0) & inpart[a_]
    if not m.any(): break
    up, cnt = np.unique(a_[m].astype(np.int64) * (nc + 1) + vc[b_[m]], return_counts=True); va, lb = up // (nc + 1), up % (nc + 1)
    o = np.lexsort((-cnt, va)); f = o[np.r_[True, va[o][1:] != va[o][:-1]]]; vc[va[f]] = lb[f]
fl = vc[FU]; f1 = np.where(fl[:, 0] == fl[:, 1], fl[:, 0], np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0]))
_, r = np.unique(np.maximum(f1, 0), return_inverse=True); r = r.ravel(); nr = r.max() + 1
# face adjacency inside the part (welded edges)
e = np.r_[FU[:, [0, 1]], FU[:, [1, 2]], FU[:, [2, 0]]]; fid = np.tile(np.arange(len(fi)), 3); e.sort(1)
key = e[:, 0].astype(np.int64) * nU + e[:, 1]; o = np.argsort(key); ks = key[o]; same = ks[1:] == ks[:-1]; FA, FB = fid[o][:-1][same], fid[o][1:][same]
fcon = con[F[fi]]; fcon = np.where(fcon[:, 0] >= 0, fcon[:, 0], np.where(fcon[:, 1] >= 0, fcon[:, 1], fcon[:, 2]))
cm = np.zeros((nr, len(pnames) + 1)); np.add.at(cm, (r, fcon + 1), 1); cm[:, 0] = 0; dcon = cm.argmax(1) - 1
m = r[FA] != r[FB]; pa, pb = np.minimum(r[FA][m], r[FB][m]), np.maximum(r[FA][m], r[FB][m]); up = np.unique(pa.astype(np.int64) * nr + pb); pa, pb = up // nr, up % nr
keep = (dcon[pa] == dcon[pb]) & (dcon[pa] >= 0)
_, g1 = connected_components(csr_matrix((np.ones(keep.sum()), (pa[keep], pb[keep])), shape=(nr, nr)), directed=False); g1 = g1[r]
LAB = cv2.cvtColor(C.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64); fab = LAB[F[fi]][:, :, 1:].mean(1)
def otsu(v):
    v = np.sort(v); n = len(v); c = np.cumsum(v); i = np.arange(1, n); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (n - i); return v[np.argmax(i * (n - i) * (m0 - m1) ** 2)]
_, g1 = np.unique(g1, return_inverse=True); g1 = g1.ravel(); ng = g1.max() + 1
for _ in range(50):
    cg = np.bincount(g1, minlength=ng).astype(float); live = cg > 0
    if live.sum() < 3: break
    th = np.exp(otsu(np.log(cg[live]))); mg = np.c_[np.bincount(g1, fab[:, 0], ng), np.bincount(g1, fab[:, 1], ng)] / np.maximum(cg, 1)[:, None]
    m = g1[FA] != g1[FB]; a_, b_ = np.r_[g1[FA][m], g1[FB][m]], np.r_[g1[FB][m], g1[FA][m]]; kk = cg[a_] < th; a_, b_ = a_[kk], b_[kk]
    if not len(a_): break
    d_ = np.linalg.norm(mg[a_] - mg[b_], axis=1); o = np.lexsort((d_, a_)); f = o[np.r_[True, a_[o][1:] != a_[o][:-1]]]
    rm = np.arange(ng); src, dst = a_[f], b_[f]; okk = (cg[dst] > cg[src]) | ((cg[dst] == cg[src]) & (dst > src)); rm[src[okk]] = dst[okk]
    if not okk.any(): break
    for _ in range(64):
        nx = rm[rm]
        if (nx == rm).all(): break
        rm = nx
    g1 = rm[g1]
u, sub = np.unique(g1, return_inverse=True); sub = sub.ravel()
names = []
for s in range(len(u)):
    c_ = np.bincount(fcon[(sub == s) & (fcon >= 0)], minlength=len(pnames)); names.append(pnames[c_.argmax()] if c_.sum() else '?')
np.savez(d('split.npz'), sub=sub, names=np.array(names))
print('RS %d relief regions in the part, %d borders, %d kept (SAM 3 changes) -> %d sub-parts: %s' % (nr, len(pa), (~keep).sum(), len(u),
      ', '.join('%s %d' % (n_, c_) for n_, c_ in zip(names, np.bincount(sub)))), flush=True)
