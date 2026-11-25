"""Mini test (plan v8, fascia): can fascia be read off the boundaries the interior model already predicts?

Fascia lies mostly where two predicted regions meet. For each pelvis, true labels vs a model's prediction, three surfaces
(voxel faces between 6-neighbours, in mm):
  deep      under-skin fat | muscle                       (deep fascia; fascia lata over the thigh)
  septa     muscle | a different muscle                   (intermuscular septa, epimysium between named muscles)
  bone      muscle | bone                                 (where muscle, tendon and periosteum meet bone)
Scores: area of each surface (true, predicted); surface distance both ways (median, 90th percentile); share of the true
surface within one and two voxels of the predicted one. Baseline: another test person's true labels in the same frame
(what a shape-blind guess gives). Only voxels known in the truth count.
args: <pelvis labels dir> <predictions dir> <out json> [pictures dir]"""
import os, sys, json, glob, numpy as np
from scipy.spatial import cKDTree
UNK = 255
FAT_SUB = {1}; MUSCLE = {4, 13, 14, 15, 16, 17}; BONE = {8, 9, 10, 11, 12, 18}
KINDS = ('deep', 'septa', 'bone')

def faces(L, known, mm):
    """centre points of the faces between neighbouring voxels, per surface kind"""
    out = {k: [] for k in KINDS}
    for ax in range(3):
        a = np.moveaxis(L, ax, 0); b = a[1:]; a = a[:-1]; kn = np.moveaxis(known, ax, 0); kn = kn[1:] & kn[:-1]
        ma, mb = np.isin(a, list(MUSCLE)), np.isin(b, list(MUSCLE)); fa, fb = np.isin(a, list(FAT_SUB)), np.isin(b, list(FAT_SUB)); ba, bb = np.isin(a, list(BONE)), np.isin(b, list(BONE))
        sel = dict(deep=(ma & fb) | (fa & mb), septa=ma & mb & (a != b), bone=(ma & bb) | (ba & mb))
        for k, m in sel.items():
            i = np.argwhere(m & kn).astype(np.float32); i[:, 0] += 1.0                              # face between index n and n+1 (voxel centres at n+0.5)
            p = np.zeros_like(i); order = [ax] + [d for d in range(3) if d != ax]   # moved axis back to its place
            for j, d in enumerate(order): p[:, d] = i[:, j]
            out[k].append(p * mm)
    return {k: np.concatenate(v) if v else np.zeros((0, 3)) for k, v in out.items()}

def compare(T, P, mm):
    r = {}
    for k in KINDS:
        t, p = T[k], P[k]
        if len(t) < 10 or len(p) < 10: r[k] = None; continue
        dt = cKDTree(p).query(t, workers=-1)[0]; dp = cKDTree(t).query(p, workers=-1)[0]
        r[k] = dict(area_true_cm2=round(len(t) * mm * mm / 100, 1), area_pred_cm2=round(len(p) * mm * mm / 100, 1),
                    true_to_pred_median_mm=round(float(np.median(dt)), 2), true_to_pred_p90_mm=round(float(np.percentile(dt, 90)), 2),
                    pred_to_true_median_mm=round(float(np.median(dp)), 2), pred_to_true_p90_mm=round(float(np.percentile(dp, 90)), 2),
                    within_1vox=round(float((dt <= mm * 1.01).mean()), 3), within_2vox=round(float((dt <= 2 * mm * 1.01).mean()), 3))
    return r

if __name__ == '__main__':
    lab, pred, outj = sys.argv[1:4]; pics = sys.argv[4] if len(sys.argv) > 4 else None
    mm = json.load(open(os.path.join(lab, 'index.json')))['voxel_mm']
    fs = sorted(glob.glob(os.path.join(pred, 'pred_pelvis_*.npz'))); ids = [os.path.basename(f)[12:-4] for f in fs]
    res = {}; truths = {}
    for c in ids:
        L = np.load(os.path.join(lab, c + '.npz'))['labels']; P = np.load(os.path.join(pred, 'pred_pelvis_%s.npz' % c))['labels']
        s = tuple(min(a, b) for a, b in zip(L.shape, P.shape)); L, P = L[:s[0], :s[1], :s[2]], P[:s[0], :s[1], :s[2]]
        truths[c] = L
        known = L != UNK; T = faces(L, known, mm); Pf = faces(np.where(known, P, UNK), known, mm)
        res[c] = dict(model=compare(T, Pf, mm))
    for i, c in enumerate(ids):                                                       # baseline: the next person's truth as the guess
        o = ids[(i + 1) % len(ids)]; L = truths[c]; B = truths[o]; s = tuple(min(a, b) for a, b in zip(L.shape, B.shape))
        L, B = L[:s[0], :s[1], :s[2]], B[:s[0], :s[1], :s[2]]; known = (L != UNK) & (B != UNK)
        res[c]['other_person'] = compare(faces(L, known, mm), faces(B, known, mm), mm)
        print(c, {k: (v['model'][k] or {}).get('within_2vox') for k, v in [(c, res[c])] for k in KINDS}, flush=True)
    summ = {}
    for src in ('model', 'other_person'):
        for k in KINDS:
            rows = [res[c][src][k] for c in ids if res[c][src][k]]
            summ['%s/%s' % (src, k)] = {m: round(float(np.median([r[m] for r in rows])), 3) for m in rows[0]} | dict(people=len(rows))
    json.dump(dict(summary=summ, people=res, note='F1 pelvis predictions (decoding fix), 3 mm labels'), open(outj, 'w'), indent=1)
    for k, v in summ.items(): print(k, v)
    if pics:
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        c = ids[0]; L = truths[c]; P = np.load(os.path.join(pred, 'pred_pelvis_%s.npz' % c))['labels'][:L.shape[0], :L.shape[1], :L.shape[2]]
        known = L != UNK; T = faces(L, known, mm); Pf = faces(np.where(known, P, UNK), known, mm)
        fig, axs = plt.subplots(1, 3, figsize=(18, 6)); zc = int(np.median(np.argwhere(np.isin(L, list(BONE)))[:, 2]))
        for a, k in zip(axs, KINDS):
            for pts, col, lab_ in ((T[k], '#2a6fdb', 'true'), (Pf[k], '#e8590c', 'predicted')):
                m = np.abs(pts[:, 2] / mm - zc - 0.5) < 0.6; a.scatter(pts[m, 0], pts[m, 1], s=3, c=col, label=lab_)
            a.set_aspect('equal'); a.set_title('%s surface, one axial slice' % k); a.legend(loc='upper right'); a.axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(pics, 'fascia_surfaces_%s.png' % c), dpi=80); print('picture saved')
