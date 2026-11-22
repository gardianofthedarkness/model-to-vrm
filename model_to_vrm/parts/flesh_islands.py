"""Material per vertex with the LLM's fixes, then the flesh cut into separate pieces at the real flesh-flesh contacts:
  python flesh_islands.py <real dir>      -> contacts/material.npz, flesh/islands.npz, flesh/islands.json
  weld      the import splits the mesh at UV seams; vertices at the same position are joined (zero-length edges)
  material  seeds = vertices a camera saw (SAM 3 concept -> material) plus the sides of sub-contacts the LLM corrected
            (contacts/sub_fix.json, these win); every other vertex takes the material of the nearest seed along the mesh
  flesh     faces whose three vertices are flesh; edges between side A and side B of a real flesh-flesh contact
            (sub_labels kind two_things) are removed, so two body parts pressed or fused together come apart
  islands   connected pieces of what is left; pieces below the Otsu split of the log piece sizes are fragments (-2)
  names     per island the skeleton joints nearest to its vertices (top three with shares)"""
import sys, os, json, numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree
R = sys.argv[1]; j = lambda *p: os.path.join(R, *p); os.makedirs(j('flesh'), exist_ok=True)
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[0] if m else 0
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
C = np.load(j('contacts', 'combined.npz')); MN = [str(x) for x in C['materials']]; names = [str(x) for x in C['names']]
T = np.load(j('sam_body', 'top_concept.npz')); MAT = json.load(open(j('sam_body', 'materials.json'))); mid = np.array([MN.index(MAT[n]) for n in names])
F = np.load(j('parts', 'orig_normals.npz'))['F']; V = np.load(j('parts', 'orig_colours.npz'))['V']
S = np.load(j('contacts', 'sub.npz')); G = {g['sub']: g for g in json.load(open(j('contacts', 'sub.json')))}
L = json.load(open(j('contacts', 'sub_labels.json'))); FX = json.load(open(j('contacts', 'sub_fix.json')))
seed = np.where(T['label'] >= 0, mid[np.maximum(T['label'], 0)], -1); nfix = 0
SEGM = np.load(j('sam_body', 'segment_material.npz'))['material'] if os.path.exists(j('sam_body', 'segment_material.npz')) else None
if os.path.exists(j('sam_body', 'segment_material.npz')):         # the LLM's per-segment judgement wins over SAM 3's vertex vote
    sm = np.load(j('sam_body', 'segment_material.npz'))['material']; seed = np.where(sm >= 0, sm, seed)
    print('FI segment labels (LLM per part) on %d vertices' % (sm >= 0).sum())
for k, d in FX.items():
    for s, m in d.items():
        v = S['vert'][(S['group'] == int(k)) & (S['side'] == 'AB'.index(s))]
        if os.path.exists(j('sam_body', 'segment_material.npz')): v = v[sm[v] < 0]   # a contact fix never overrides a segment label
        seed[v] = MN.index(m); nfix += len(v)
_, rep, inv = np.unique(V, axis=0, return_index=True, return_inverse=True); inv = inv.ravel(); dup = np.flatnonzero(rep[inv] != np.arange(len(V)))
seam = np.c_[dup, rep[inv[dup]]]                                  # the import splits the mesh at UV seams: weld twins
e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]], seam]; w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1) + 1e-9
print('FI welded %d seam twins' % len(seam))
Gm = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V), len(V))).tocsr()
_, _, src = dijkstra(Gm, directed=False, indices=np.flatnonzero(seed >= 0), min_only=True, return_predecessors=True)
mat = np.where(src >= 0, seed[np.maximum(src, 0)], -1); old = C['material']
def clean(mat):                                                   # specks: a same-material region below the Otsu split of all
    split = None                                                  # region sizes takes its neighbours' majority material
    for it in range(50):
        same = mat[e[:, 0]] == mat[e[:, 1]]; se = e[same]
        n, comp = connected_components(coo_matrix((np.ones(len(se)), (se[:, 0], se[:, 1])), shape=(len(V), len(V))), directed=False)
        sz = np.bincount(comp)
        if split is None: split = np.exp(otsu(np.log(sz)))
        small = sz[comp] < split; bd = e[~same]; bd = np.r_[bd, bd[:, ::-1]]; bd = bd[small[bd[:, 0]] & (mat[bd[:, 1]] >= 0)]
        if not len(bd): break
        cnt = np.bincount(comp[bd[:, 0]] * 4 + mat[bd[:, 1]], minlength=n * 4).reshape(n, 4); new = cnt.argmax(1)
        ch = small & (cnt.sum(1)[comp] > 0) & (new[comp] != mat); print('FI clean round %d: split %d vertices, %d small regions, %d vertices changed' % (it, int(split), int((sz < split).sum()), ch.sum()))
        if not ch.any() or (it and ch.sum() == last): break       # done, or a few regions only swap back and forth
        last = ch.sum()
        mat = np.where(ch, new[comp], mat)
    return mat
mat = clean(mat)
def holes(mat):                                                   # a non-flesh region whose whole border is flesh is skin
    fl = MN.index('flesh'); same = mat[e[:, 0]] == mat[e[:, 1]]; se = e[same]   # the segmenter missed (a hole in an island)
    n, comp = connected_components(coo_matrix((np.ones(len(se)), (se[:, 0], se[:, 1])), shape=(len(V), len(V))), directed=False)
    bd = e[~same]; bd = np.r_[bd, bd[:, ::-1]]; tot = np.bincount(comp[bd[:, 0]], minlength=n); tf = np.bincount(comp[bd[:, 0]], weights=(mat[bd[:, 1]] == fl), minlength=n)
    enc = (tot > 0) & (tf == tot); ch = enc[comp] & (mat != fl)
    if SEGM is not None: ch &= SEGM < 0                           # a part the LLM labelled (a shoe on a leg) is not a hole
    print('FI holes: %d enclosed regions, %d vertices -> flesh' % ((enc & (np.bincount(comp, weights=mat != fl) > 0)).sum(), ch.sum()))
    return np.where(ch, fl, mat)
def mode(mat):                                                    # majority of each vertex and its 1-ring, until nothing changes
    E2 = np.r_[e, e[:, ::-1], np.c_[np.arange(len(V)), np.arange(len(V))]]; last = -1
    for it in range(100):
        cnt = np.bincount(E2[:, 0] * 4 + np.maximum(mat[E2[:, 1]], 0), weights=(mat[E2[:, 1]] >= 0), minlength=len(V) * 4).reshape(-1, 4)
        best = cnt.argmax(1); keep = cnt[np.arange(len(V)), np.maximum(mat, 0)] >= cnt.max(1)       # ties keep the current label
        ch = (~keep) & (mat >= 0); print('FI majority round %d: %d vertices changed' % (it, ch.sum()))
        if not ch.any() or ch.sum() == last: break
        last = ch.sum(); mat = np.where(ch, best, mat)
    return mat
mat = holes(mode(holes(mat)))
print('FI fixed seeds %d vertices; material changed on %d vertices vs before' % (nfix, (mat != old).sum()), {MN[i]: int((mat == i).sum()) for i in range(4)})
np.savez(j('contacts', 'material.npz'), material=mat, materials=np.array(MN))
fl = mat == MN.index('flesh'); ff = F[fl[F].all(1)]; fe = np.r_[ff[:, [0, 1]], ff[:, [1, 2]], ff[:, [2, 0]], seam[fl[seam].all(1)]]
cut = np.zeros(len(fe), bool); ncut = 0
for k, g in G.items():
    if L.get(str(k), {}).get('kind') != 'two_things': continue
    m = S['group'] == k; v, sd = S['vert'][m], S['side'][m]
    if not (fl[v[sd == 0]].mean() > 0.5 and fl[v[sd == 1]].mean() > 0.5): continue   # both sides (mostly) flesh after the fixes
    sa = np.zeros(len(V), bool); sb = np.zeros(len(V), bool); sa[v[sd == 0]] = True; sb[v[sd == 1]] = True
    c = (sa[fe[:, 0]] & sb[fe[:, 1]]) | (sb[fe[:, 0]] & sa[fe[:, 1]]); cut |= c; ncut += 1
    print('FI cut contact %d (%s | %s): %d edges' % (k, L[str(k)].get('A', '')[:40], L[str(k)].get('B', '')[:40], c.sum()))
ke = fe[~cut]; n, comp = connected_components(coo_matrix((np.ones(len(ke)), (ke[:, 0], ke[:, 1])), shape=(len(V), len(V))), directed=False)
isl = -np.ones(len(V), int); cf = comp[fl]; u, inv, sz = np.unique(cf, return_inverse=True, return_counts=True)
split = np.exp(otsu(np.log(sz))); big = np.flatnonzero(sz > split); big = big[np.argsort(-sz[big])]; rank = -2 * np.ones(len(u), int); rank[big] = np.arange(len(big))
isl[fl] = rank[inv]; print('FI flesh %d vertices in %d pieces after %d cuts; %d islands above %d vertices, %d vertices in fragments' % (fl.sum(), len(u), ncut, len(big), int(split), (isl == -2).sum()))
SK = np.load(j('body', 'skeleton.npz')); JL = [str(x) for x in SK['labels']]; jt = cKDTree(SK['heads']); nj = jt.query(V[fl])[1]
out = []
for i in range(len(big)):
    m = isl[fl] == i; b = np.bincount(nj[m], minlength=len(JL)); o = np.argsort(-b)[:3]
    rec = {'island': i, 'vertices': int(m.sum()), 'joints': ', '.join('%s %d%%' % (JL[t], 100 * b[t] / b.sum()) for t in o if b[t])}
    out.append(rec); print('FI island %2d  %7d  %s' % (i, rec['vertices'], rec['joints']))
np.savez(j('flesh', 'islands.npz'), island=isl); json.dump(out, open(j('flesh', 'islands.json'), 'w'), indent=1)
C0 = np.load(j('parts', 'orig_colours.npz'))['C']; Lc = np.where(C0 <= 0.04045, C0 / 12.92, ((C0 + 0.055) / 1.055) ** 2.4)     # for show_sticks.py
np.savez(j('flesh', 'islands_lab.npz'), label=np.where(isl >= 0, isl, -1))
np.savez(j('flesh', 'texture_lab.npz'), label=np.where(isl != -1, np.arange(len(isl)), -1), palette=Lc)
print('FI done: %d islands' % len(big))
