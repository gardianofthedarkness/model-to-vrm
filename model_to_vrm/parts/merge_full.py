"""Combined parts under the rule: colour regions for the big parts, SAM 3 for every detail part.
  python merge_full.py <carve.npz> <full_votes.npz> <orig_colours.npz> <part_sheet.json> <out dir>
  claim   per full point and concept: share = summed mask score / times seen; concepts tried from the fewest claimed points
          (most specific) to the most; a point goes to the first whose share clears that concept's Otsu split
  pieces  connected pieces of each concept (8 nearest points); pieces whose log size falls below the Otsu split of that
          concept's piece sizes are noise and go back to their colour region
  parent  each detail piece's parent = the detail concept or colour region most of its points would otherwise belong to
Writes parts.npz (region per point), part_list.json (the combined list) and part_list.md."""
import sys, os, json, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
K, Dv, OC = [np.load(a) for a in sys.argv[1:4]]; sheet = json.load(open(sys.argv[4])); od = sys.argv[5]; os.makedirs(od, exist_ok=True)
cr = K['colour_region']; V, COL, tp = OC['V'], OC['C'], OC['part']
votes, seen, C, area = Dv['votes'], Dv['seen'].astype(np.float32), list(Dv['concepts']), Dv['area']
share = np.where(seen > 0, votes / np.maximum(seen, 1), 0)
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[0] - 1e-9 if m else 1
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
claimed = {}                                        # concept -> mask of points over its own split (before precedence)
for ci in range(len(C)):
    s = share[:, ci]; t = s[s > 0]
    if len(t) >= 3: claimed[ci] = s > otsu(t)
area = np.array([claimed[ci].sum() if ci in claimed else np.inf for ci in range(len(C))], float)   # size = points claimed
order = [ci for ci in np.argsort(area) if np.isfinite(area[ci])]                                  # most specific first
lab = -np.ones(len(V), int)
for ci in order:
    if ci in claimed: lab[(lab < 0) & claimed[ci]] = ci
region = cr.copy(); nxt = cr.max() + 1; details = []
for ci in order:
    m = np.flatnonzero(lab == ci)
    if len(m) < 2: continue
    _, nb = cKDTree(V[m]).query(V[m], k=min(9, len(m))); r = np.repeat(np.arange(len(m)), nb.shape[1] - 1)
    n, comp = connected_components(coo_matrix((np.ones(len(r)), (r, nb[:, 1:].ravel())), shape=(len(m), len(m))), directed=False)
    sz = np.bincount(comp); keep = sz > np.exp(otsu(np.log(sz))) if n >= 3 else np.ones(n, bool)
    for u in np.flatnonzero(keep):
        pts = m[comp == u]; region[pts] = nxt
        # parent: another (bigger) concept also over its split here, else the colour region underneath
        par = [cj for cj in order if cj != ci and area[cj] > area[ci] and cj in claimed and claimed[cj][pts].mean() > 0.5]
        details.append({'id': int(nxt), 'concept': C[ci], 'points': int(len(pts)), 'centre': V[pts].mean(0).round(3).tolist(),
                        'colour': np.median(COL[pts], 0).round(3).tolist(), 'mean_share': round(float(share[pts, ci].mean()), 3),
                        'inside_concept': C[min(par, key=lambda cj: area[cj])] if par else None,
                        'on_colour_region': int(np.bincount(cr[pts]).argmax()), 'tripo_part': int(np.bincount(tp[pts]).argmax())})
        nxt += 1
    print('MF %-14s %6d points, %d pieces kept of %d' % (C[ci], len(m), keep.sum(), n))
big = []
for k in range(cr.max() + 1):
    pts = np.flatnonzero(region == k)
    if len(pts) == 0: continue
    t = np.bincount(tp[pts]).argmax()
    big.append({'id': int(k), 'points': int(len(pts)), 'centre': V[pts].mean(0).round(3).tolist(), 'colour': np.median(COL[pts], 0).round(3).tolist(),
                'tripo_part': int(t), 'tripo_part_name': sheet.get(str(t), {}).get('name', '?')})
big.sort(key=lambda b: -b['points'])
np.savez(os.path.join(od, 'parts.npz'), region=region, is_detail=region > cr.max())
json.dump({'colour_regions': big, 'sam3_details': details}, open(os.path.join(od, 'part_list.json'), 'w'), indent=1)
hexc = lambda c: '#%02x%02x%02x' % tuple(int(255 * min(max(x, 0), 1)) for x in c)
with open(os.path.join(od, 'part_list.md'), 'w') as f:
    f.write('## Big parts (colour differencing)\n\n| id | points | colour | on Tripo part |\n|---|---|---|---|\n')
    for b in big: f.write('| %d | %d | %s | %s |\n' % (b['id'], b['points'], hexc(b['colour']), b['tripo_part_name']))
    f.write('\n## Detail parts (SAM 3)\n\n| id | concept | points | colour | inside | on Tripo part |\n|---|---|---|---|---|---|\n')
    for d in details: f.write('| %d | %s | %d | %s | %s | %s |\n' % (d['id'], d['concept'], d['points'], hexc(d['colour']), d['inside_concept'] or '-', sheet.get(str(d['tripo_part']), {}).get('name', '?')))
print('MF done: %d colour regions, %d detail pieces' % (len(big), len(details)))
