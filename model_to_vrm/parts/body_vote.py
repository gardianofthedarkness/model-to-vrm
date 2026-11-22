"""SAM 3 whole-body masks voted onto every vertex (visibility from views_vis.py), then the combined part list:
  python body_vote.py <views dir> <orig_colours.npz> <carve.npz | -> <out dir> [geo organs dir]
  top     top_concept.npz: per vertex the concept with the highest share (-1 where no camera saw it), names, palette
          (carve '-' stops here: no colour-region part list)
  found   a concept counts only if its best mask score over all views is above the Otsu split of all concepts' best
  keep    a mask counts when its score is above the Otsu split of its concept's pooled scores (a view's top mask also
          when above the pool's median)
  share   per vertex and concept: summed counted mask scores / views that asked it and see the vertex
  parts   big parts = the colour regions of carve.npz; each gets its SAM 3 hint = the concept with the highest mean share
          over its vertices (and the runner-up), so the agent sees what SAM 3 thinks each region is
  details the geometry-only face organs (geo_vote.py organs.npz: eye, eyelid, brow, nose, lips, ear) as detail parts
Writes <out>/part_list.json, parts.npz (region per vertex), shares.npz."""
import sys, os, json, numpy as np
D, ocp, kp, od = sys.argv[1:5]; geo = sys.argv[5] if len(sys.argv) > 5 else None; os.makedirs(od, exist_ok=True)
cams = json.load(open(os.path.join(D, 'cameras.json'))); VZ = np.load(os.path.join(D, 'vis.npz')); vis = dict(zip([str(n) for n in VZ['names']], VZ['vis']))
import colorsys
O = np.load(ocp); V, COL = O['V'], O['C']; Ph = np.c_[V, np.ones(len(V))]
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[0] - 1e-9 if m else 1
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
md = os.path.join(D, 'masks'); F = {f[:-4]: np.load(os.path.join(md, f)) for f in sorted(os.listdir(md)) if f.endswith('.npz') and f[:-4] in vis}
ALL = sorted({k[2:] for z in F.values() for k in z.files if k.startswith('s_')})
pool = {c: np.concatenate([z['s_' + c] for z in F.values() if 's_' + c in z.files]) for c in ALL}
best = {c: pool[c].max() for c in ALL}; found = otsu(list(best.values())); ALL = [c for c in ALL if best[c] > found]
print('BV concepts kept:', ALL, ' dropped:', [c for c in best if c not in ALL])
split = {c: otsu(pool[c]) for c in ALL}; med = {c: np.median(pool[c]) for c in ALL}
votes = np.zeros((len(V), len(ALL)), np.float32); seen = np.zeros((len(V), len(ALL)), np.float32)
for name, z in F.items():
    cm = cams[name]; R = cm['res']; q = Ph @ np.array(cm['matrix_inv']).T; u = (q[:, 0] / cm['scale'] + 0.5) * R; v = (0.5 - q[:, 1] / cm['scale']) * R
    idx = np.flatnonzero(vis[name] & (u >= 0) & (v >= 0) & (u < R) & (v < R)); pix = v[idx].astype(int) * R + u[idx].astype(int)
    for ci, c in enumerate(ALL):
        if 's_' + c not in z.files: continue
        seen[idx, ci] += 1; S = z['s_' + c]
        if len(S) == 0: continue
        keep = S > split[c]; keep[np.argmax(S)] |= S.max() > med[c]
        if not keep.any(): continue
        M = np.unpackbits(z['m_' + c][keep], axis=1, count=R * R).astype(bool); votes[idx, ci] += (M * S[keep][:, None]).max(0)[pix]
share = np.where(seen > 0, votes / np.maximum(seen, 1), 0); np.savez(os.path.join(od, 'shares.npz'), share=share, concepts=np.array(ALL))
np.savez(os.path.join(od, 'top_concept.npz'), label=np.where(share.max(1) > 0, share.argmax(1), -1), names=np.array(ALL),
         palette=np.array([colorsys.hsv_to_rgb(h, 0.65, 0.95) for h in np.arange(len(ALL)) / max(len(ALL), 1)]))
if kp == '-': print('BV done: shares and top concepts for %d concepts' % len(ALL)); sys.exit()
cr = np.load(kp)['colour_region']
region = cr.copy(); big = []
for k in range(cr.max() + 1):
    m = cr == k
    if not m.any(): continue
    s = share[m].mean(0); o = np.argsort(-s)
    big.append({'id': int(k), 'points': int(m.sum()), 'centre': V[m].mean(0).round(3).tolist(), 'colour': np.median(COL[m], 0).round(3).tolist(),
                'sam3_hint': '%s %.2f, %s %.2f' % (ALL[o[0]], s[o[0]], ALL[o[1]], s[o[1]]), 'tripo_part_name': 'sam3: ' + ALL[o[0]]})
big.sort(key=lambda b: -b['points']); details = []; nxt = cr.max() + 1
if geo:
    G = np.load(os.path.join(geo, 'organs.npz')); gl, gn, gi = G['label'], [str(x) for x in G['names']], G['idx']
    for k, nm in enumerate(gn):
        if nm in ('face', 'hair'): continue
        pts = gi[gl == k]
        if len(pts) == 0: continue
        region[pts] = nxt
        details.append({'id': int(nxt), 'concept': nm, 'points': int(len(pts)), 'centre': V[pts].mean(0).round(3).tolist(), 'colour': np.median(COL[pts], 0).round(3).tolist(),
                        'inside_concept': 'face', 'on_colour_region': int(np.bincount(cr[pts]).argmax()), 'tripo_part': 0, 'mean_share': 1.0, 'source': 'geometry'})
        nxt += 1
np.savez(os.path.join(od, 'parts.npz'), region=region)
json.dump({'colour_regions': big, 'sam3_details': details}, open(os.path.join(od, 'part_list.json'), 'w'), indent=1)
print('BV done: %d colour regions, %d geometry details' % (len(big), len(details)))
