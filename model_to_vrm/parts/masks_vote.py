"""Votes on the full-detail points from SAM 3's saved raw masks:
  python masks_vote.py <sam3views dir> <orig_colours.npz> <views.npz> <pi_ws data.npz> <out.npz>
  keep    per concept, the scores of all its masks over all close-ups are pooled; a mask counts when its score is above
          the Otsu split of that pool (junk masks score far below real ones), and the top mask of a render always counts
          when it is above the pool's median (so a concept seen weakly in one view is not lost)
  vote    each counted mask adds its score to the points the render sees inside it; seen = renders that asked the concept
          and see the point
  area    median model-space area of a concept's counted masks (orders concepts from specific to general)
Also writes kept_<render>.jpg overlays (counted masks, label = concept and score)."""
import sys, os, json, numpy as np, cv2, colorsys
D, oc_p, vis_p, w_p, out = sys.argv[1:6]; md = os.path.join(D, 'masks'); cams = json.load(open(os.path.join(D, 'cameras.json')))
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v)
    if m < 3: return v[0] - 1e-9 if m else 1
    c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
F = {f[:-4]: np.load(os.path.join(md, f)) for f in sorted(os.listdir(md)) if f.endswith('.npz')}
ALL = sorted({k[2:] for z in F.values() for k in z.files if k.startswith('s_')})
pool = {c: np.concatenate([z['s_' + c] for z in F.values() if 's_' + c in z.files]) for c in ALL}
split = {c: otsu(pool[c]) for c in ALL}; med = {c: np.median(pool[c]) if len(pool[c]) else 1 for c in ALL}
for c in ALL: print('MV %-14s %4d masks, scores %.2f..%.2f, Otsu split %.2f' % (c, len(pool[c]), pool[c].min() if len(pool[c]) else 0, pool[c].max() if len(pool[c]) else 0, split[c]))
P = np.load(oc_p)['V']; f2r = np.load(w_p)['full_to_rig']; VIS = np.load(vis_p)['vis']; Ph = np.c_[P, np.ones(len(P))]
votes = np.zeros((len(P), len(ALL)), np.float32); seen = np.zeros((len(P), len(ALL)), np.float32); area = {c: [] for c in ALL}
for name, z in F.items():
    cm = cams[name]; R = cm['res']; q = Ph @ np.array(cm['matrix_inv']).T; u = (q[:, 0] / cm['scale'] + 0.5) * R; v = (0.5 - q[:, 1] / cm['scale']) * R
    idx = np.flatnonzero(VIS[cm['view']][f2r] & (u >= 0) & (v >= 0) & (u < R) & (v < R)); pix = v[idx].astype(int) * R + u[idx].astype(int)
    base = cv2.imread(os.path.join(D, name + '.png')); panels = []
    for c in ALL:
        if 's_' + c not in z.files: continue
        ci = ALL.index(c); seen[idx, ci] += 1; S = z['s_' + c]
        if len(S) == 0: continue
        keep = S > split[c]; keep[np.argmax(S)] |= S.max() > med[c]
        if not keep.any(): continue
        M = np.unpackbits(z['m_' + c][keep], axis=1, count=R * R).astype(bool); best = (M * S[keep][:, None]).max(0)
        votes[idx, ci] += best[pix]; area[c] += [float(m.sum()) / (R * R) * cm['scale'] ** 2 for m in M]
        mk = best.reshape(R, R) > 0; pn = base.copy(); pn[mk] = (pn[mk] * 0.3 + np.array([255, 0, 255]) * 0.7).astype(np.uint8)
        pn = cv2.resize(pn, (384, 384)); cv2.putText(pn, '%s %.2f (%d)' % (c, S[keep].max(), keep.sum()), (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2); panels.append(pn)
    while len(panels) % 4: panels.append(np.full((384, 384, 3), 255, np.uint8))
    cv2.imwrite(os.path.join(D, 'kept_' + name + '.jpg'), np.vstack([np.hstack(panels[i:i + 4]) for i in range(0, len(panels), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
np.savez(out, votes=votes, seen=seen, concepts=np.array(ALL), area=np.array([np.median(area[c]) if area[c] else np.inf for c in ALL]))
print('MV area order: ' + ', '.join('%s %.5f' % (ALL[i], np.median(area[ALL[i]])) for i in np.argsort([np.median(area[c]) if area[c] else np.inf for c in ALL]) if area[ALL[i]]))
print('MV done')
