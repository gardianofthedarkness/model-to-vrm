"""Contact sides from SAM 3's per-vertex material, pair type and real-or-not from the LLM:
  python contacts_combine.py <real dir>      (reads sam_body/, parts/orig_normals.npz, body/vis.npz, body/cameras.json,
                                              contacts/; writes contacts/combined.npz and contacts/combined.json)
  concept    SAM 3's top concept per vertex; vertices no camera saw (inside creases) take the concept of the nearest seen
             vertex ALONG THE MESH (Dijkstra over edges), so a label never jumps the contact gap
  material   concept -> flesh/cloth/hair/prop as the LLM named it (sam_body/materials.json, agents/concept_material.py)
  pair       per contact ray: material of the vertex and of the vertex its ray hits; per patch the pair counts, the main
             pair, and the LLM's pair and real_contact for comparison
  view       per patch the 8-ring view that sees most of it (for close-ups)"""
import sys, os, json, numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
R = sys.argv[1]; j = lambda *p: os.path.join(R, *p)
T = np.load(j('sam_body', 'top_concept.npz')); lab, names = T['label'].copy(), [str(x) for x in T['names']]
MAT = json.load(open(j('sam_body', 'materials.json'))); MN = ['flesh', 'cloth', 'hair', 'prop']
F = np.load(j('parts', 'orig_normals.npz'))['F']; V = np.load(j('parts', 'orig_colours.npz'))['V']
Z = np.load(j('contacts', 'patches.npz')); pid = Z['patch']; Rz = np.load(j('contacts', 'rays.npz')); gap, hit = Rz['gap'], Rz['hit_vertex']
L = json.load(open(j('contacts', 'labels.json'))) if os.path.exists(j('contacts', 'labels.json')) else {}   # optional per-patch LLM labels
seen = lab >= 0; print('CC seen %d of %d vertices' % (seen.sum(), len(lab)))
e = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; w = np.linalg.norm(V[e[:, 0]] - V[e[:, 1]], axis=1) + 1e-9
G = coo_matrix((np.r_[w, w], (np.r_[e[:, 0], e[:, 1]], np.r_[e[:, 1], e[:, 0]])), shape=(len(V), len(V))).tocsr()
d, _, src = dijkstra(G, directed=False, indices=np.flatnonzero(seen), min_only=True, return_predecessors=True)
ok = src >= 0; lab[ok] = lab[src[ok]]; print('CC filled %d, still unknown %d (pieces with no seen vertex)' % ((ok & ~seen).sum(), (lab < 0).sum()))
mid = np.array([MN.index(MAT[n]) for n in names]); mat = np.where(lab >= 0, mid[np.maximum(lab, 0)], -1)
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i)
    return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
fin = np.isfinite(gap); thr = np.exp(otsu(np.log(gap[fin]))); ray = np.flatnonzero(fin & (gap < thr) & (pid >= 0))
VZ = np.load(j('body', 'vis.npz')); vn = [str(x) for x in VZ['names']]; cams = json.load(open(j('body', 'cameras.json')))
out = []; pm = lambda a, b: '-'.join(sorted((MN[a] if a >= 0 else '?', MN[b] if b >= 0 else '?')))
for k in range(pid.max() + 1):
    r = ray[pid[ray] == k]; idx = np.flatnonzero(pid == k)
    pairs = [pm(a, b) for a, b in zip(mat[r], mat[hit[r]])]; u, c = np.unique(pairs, return_counts=True); o = np.argsort(-c)
    cp = ['-'.join(sorted((names[a] if a >= 0 else '?', names[b] if b >= 0 else '?'))) for a, b in zip(lab[r], lab[hit[r]])]; cu, cc = np.unique(cp, return_counts=True); co = np.argsort(-cc)[:3]
    an = L.get(str(k), {}); llm = '-'.join(sorted((an['A']['material'], an['B']['material']))) if isinstance(an.get('A'), dict) and isinstance(an.get('B'), dict) else None
    best = max(vn, key=lambda n: VZ['vis'][vn.index(n)][idx].sum())
    rec = {'patch': k, 'rays': int(len(r)), 'pair': str(u[o[0]]) if len(u) else None, 'pair_share': round(float(c[o[0]] / c.sum()), 2) if len(u) else 0,
           'pairs': {str(u[i]): int(c[i]) for i in o}, 'concept_pairs': {str(cu[i]): int(cc[i]) for i in co},
           'llm_pair': llm, 'real_contact': an.get('real_contact'), 'llm_why': an.get('why'),
           'agree': (llm == str(u[o[0]])) if llm and len(u) else None, 'centre': V[idx].mean(0).round(4).tolist(),
           'extent': float(np.ptp(V[idx], 0).max()), 'view': best, 'az': cams[best]['az']}
    out.append(rec); print('CC patch %2d  %-12s %.2f  llm %-12s real %-5s  %s' % (k, rec['pair'], rec['pair_share'], llm, rec['real_contact'], list(rec['concept_pairs'])[:2]))
np.savez(j('contacts', 'combined.npz'), concept=lab, material=mat, seen=seen, names=np.array(names), materials=np.array(MN))
json.dump(out, open(j('contacts', 'combined.json'), 'w'), indent=1)
a = [r['agree'] for r in out if r['agree'] is not None]; print('CC done: %d patches, main pair agrees with the LLM on %d of %d' % (len(out), sum(a), len(a)))
