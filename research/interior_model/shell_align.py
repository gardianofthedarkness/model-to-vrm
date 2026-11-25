"""Plan v8, test P2: place a part from its outer surface alone, the same way for CT people and for generated characters.

The reference is the population's average shell: the mean signed distance to the skin over the training people, in the
part frame (anatomy-based for CT). A new surface (CT skin, or a character's bare skin) is placed by the similarity
transform (scale s, rotation Q, shift b) that puts its skin points on the zero level of that mean field:
    min over (s, Q, b) of  sum_i rho( phibar(s Q x_i + b) ),
rho a Cauchy loss whose scale is 1.4826 x the median absolute residual at the start (from the data, not a set number).
Skin points: body voxels next to the outside (not next to unknown scan space), for a character its bare-skin vertices.
For a CT person the fitted scale is reported but not applied (CT is already in mm); a character has no unit, so its
fitted scale is its size.
This file: build the mean field from training people, then test on held-out people: start from their true frame moved
by a known size, shift and turn, fit, and report how far the fitted frame ends from the true one (mm at the landmarks
that define the frame). That residual is what training must tolerate (test P3).
args: <interior_data root> <face|pelvis> <out json> [held-out people=20]"""
import os, sys, json, time, numpy as np
from scipy import ndimage as ndi
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
UNK = 255

def skin_points(L, mm, lo, step=1):
    """mm positions (part frame) of body voxels that touch the outside; body next to unknown scan space is not skin."""
    body = (L > 0) & (L < UNK); out = ndi.binary_dilation(L == 0) & body
    ii = np.argwhere(out)[::step]; return (ii + 0.5) * mm + lo

def signed_distance(L, mm):
    body = (L > 0) & (L < UNK); d = (ndi.distance_transform_edt(body) - ndi.distance_transform_edt(~body)) * mm; d[L == UNK] = np.nan; return d.astype(np.float32)

class MeanShell:
    def __init__(s, sdfs, mm, lo, G):                                                      # sdfs: an iterable; summed one at a time (memory)
        tot = cnt = None; npeople = 0
        for d in sdfs:
            k = np.isfinite(d); tot = np.where(k, d, 0).astype(np.float64) if tot is None else tot + np.where(k, d, 0); cnt = k.astype(np.int32) if cnt is None else cnt + k; npeople += 1
        s.n = cnt; s.phi = np.where(cnt > 0, tot / np.maximum(cnt, 1), np.nan).astype(np.float32)
        s.phi[cnt < max(2, npeople // 2)] = np.nan; s.mm, s.lo, s.G = mm * G, np.asarray(lo, float), G   # kept where at least half of the people have data
        s.fill()
    def fill(s):                                                                               # where fewer than half the people have data, take the nearest defined value:
        bad = ~np.isfinite(s.phi)                                                              # a point leaving the defined region must not score as a perfect fit, and the
        if bad.any():                                                                          # filled region must hold no false skin: |nearest value| + distance to the defined region
            d, j = ndi.distance_transform_edt(bad, return_distances=True, return_indices=True); s.phi = (np.abs(s.phi[tuple(j)]) + d * s.mm).astype(np.float32)
    @classmethod
    def load(cls, f):
        z = np.load(f); m = cls.__new__(cls); m.phi, m.n, m.lo, m.mm = z['phi'], z['n'], z['lo'], float(z['mm']); m.fill(); return m
    def __call__(s, p):
        q = (p - s.lo) / s.mm - 0.5; return ndi.map_coordinates(s.phi, q.T, order=1, mode='nearest')

def fit(points, mean, x0=None, scale=True):
    """similarity (s, Q, b) with mean(s Q x + b) ~ 0 on the skin points. x0 = [log s, rotvec(3), b(3)]."""
    x0 = np.zeros(7) if x0 is None else np.asarray(x0, float)
    def apply(x, p): return np.exp(x[0]) * Rotation.from_rotvec(x[1:4]).apply(p) + x[4:7]
    def res(x):
        r = mean(apply(x if scale else np.r_[0.0, x[1:]], points)); return np.where(np.isfinite(r), r, 0.0)
    r0 = mean(apply(x0, points)); r0 = r0[np.isfinite(r0)]; f = 1.4826 * np.median(np.abs(r0 - np.median(r0))) if len(r0) else 1.0
    sol = least_squares(res, x0, loss='cauchy', f_scale=max(f, 1e-3), x_scale=np.r_[0.05, 0.05, 0.05, 0.05, 5, 5, 5])
    return sol.x, apply

def perturb(points, s, rotvec, b):                                                         # move a shell by a known similarity
    return s * Rotation.from_rotvec(rotvec).apply(points) + b

if __name__ == '__main__':
    root, part, outf = sys.argv[1:4]; nheld = int(sys.argv[4]) if len(sys.argv) > 4 else 20; t0 = time.time()
    fi = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pi = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
    if part == 'face':
        folder, members, mm, lo, G = 'face_v1_headframe', fi['heads'], 1.0, np.array(fi['window_lo_mm']), 2   # mean field on a 2 mm grid
        hs = sorted(c for c, v in members.items() if v['source'] == 'hanseg'); split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
        split.update({c: v['split'] for c, v in members.items() if v['source'] == 'ts'})
        LM = {k: fi['ids'][k] for k in ('auditory_canal_left', 'auditory_canal_right', 'eye_left', 'eye_right')}
    else:
        folder, members, mm, lo, G = 'pelvis_v1', pi['people'], 3.0, np.array(pi['window_lo_mm']), 1
        split = {c: v['split'] for c, v in members.items()}; cl = pi['classes']; LM = {k: cl.index(k) for k in ('femur_l', 'femur_r', 'sacrum', 'lumbar')}
    load = lambda c: np.load(os.path.join(root, folder, c + '.npz'))['labels']
    train = sorted(c for c in members if split.get(c) == 'train'); held = sorted(c for c in members if split.get(c) in ('val', 'test'))[:nheld]
    mf = outf.replace('.json', '_mean.npz')
    if os.path.exists(mf): mean = MeanShell.load(mf); print('mean shell loaded', flush=True)
    else: mean = MeanShell((signed_distance(load(c), mm)[::G, ::G, ::G] for c in train), mm, lo, G); print(part, 'mean shell from', len(train), 'people in %.0f s' % (time.time() - t0), flush=True)
    if not os.path.exists(mf): np.savez_compressed(mf, phi=mean.phi, n=mean.n, lo=lo, mm=mm * G)
    rng = np.random.default_rng(0); CASES = [('none', 1.0, 0, 0)] + [('size x%.2f' % k, k, 0, 0) for k in (0.8, 1.25)] + [('shift %d mm' % d, 1.0, d, 0) for d in (10, 20)] + \
            [('turn %d deg' % a, 1.0, 0, a) for a in (5, 10)] + [('all three', 1.15, 15, 8)]
    rows = {n: [] for n, *_ in CASES}; self_fit = []
    for c in held:
        L = load(c); P = skin_points(L, mm, lo, step=4 if part == 'face' else 1)
        cen = {k: np.argwhere(L == v).mean(0) * mm + lo + mm / 2 for k, v in LM.items() if (L == v).sum() >= 10}
        x, ap = fit(P, mean); self_fit.append(dict(scale=float(np.exp(x[0])), shift_mm=float(np.linalg.norm(x[4:])), turn_deg=float(np.degrees(np.linalg.norm(x[1:4])))))
        for n, k, d, a in CASES:
            v = rng.normal(size=3); v /= np.linalg.norm(v); w = rng.normal(size=3); w /= np.linalg.norm(w)
            Pm = perturb(P, k, np.radians(a) * w, d * v)                                         # the shell as it would arrive, misplaced
            xm, ap = fit(Pm, mean); back = lambda q: ap(xm, perturb(q, k, np.radians(a) * w, d * v))   # landmark -> misplaced -> fitted back
            ref = lambda q: ap(x, q)                                                             # where the unperturbed fit puts it
            err = [float(np.linalg.norm(back(p[None])[0] - ref(p[None])[0])) for p in cen.values()]
            rows[n].append(dict(landmark_error_mm=float(np.mean(err)), size_error=float(np.exp(xm[0]) * k / np.exp(x[0]) - 1)))
        print(c, {n: round(r[-1]['landmark_error_mm'], 1) for n, r in rows.items()}, flush=True)
    summ = {n: dict(median_mm=round(float(np.median([r['landmark_error_mm'] for r in rr])), 2), p90_mm=round(float(np.percentile([r['landmark_error_mm'] for r in rr], 90)), 2),
                    median_size_error=round(float(np.median([abs(r['size_error']) for r in rr])), 4)) for n, rr in rows.items()}
    sf = {k: dict(median=round(float(np.median([r[k] for r in self_fit])), 3), p90=round(float(np.percentile([r[k] for r in self_fit], 90)), 3)) for k in self_fit[0]}
    json.dump(dict(part=part, held_out=len(held), recovery=summ, fit_of_unmoved_shell_to_mean=sf, seconds=round(time.time() - t0)), open(outf, 'w'), indent=1)
    print('SUMMARY', json.dumps(summ), json.dumps(sf))
    if os.environ.get('PICS'):                                                                # pictures: recovery error per case (aggregate) and the mean shell (an average over people)
        import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw=dict(width_ratios=[1.5, 1]))
        names_ = list(rows); x_ = np.arange(len(names_))
        for j, r in enumerate(rows[n] for n in names_): ax[0].scatter(np.full(len(r), j) + np.random.default_rng(j).uniform(-.15, .15, len(r)), [q['landmark_error_mm'] for q in r], s=18, color='#4a6fd1')
        ax[0].plot(x_, [summ[n]['median_mm'] for n in names_], 'k_', ms=26, mew=2); ax[0].set_xticks(x_); ax[0].set_xticklabels([n.replace(' ', '\n', 1) for n in names_], fontsize=9)
        ax[0].set_ylabel('landmark error after the fit (mm)'); ax[0].set_title('%s: shell moved by a known amount, then fitted back (%d held-out people; bar = median)' % (part, len(held)), fontsize=10); ax[0].spines[['top', 'right']].set_visible(False)
        ph = mean.phi; k = ph.shape[0] // 2; ax[1].imshow(np.clip(ph[k].T, -40, 40), origin='lower', cmap='RdBu'); ax[1].contour(ph[k].T, [0], colors='k', linewidths=1.2)
        ax[1].set_title('mean shell, midline slice (signed distance, black = average skin)', fontsize=10); ax[1].axis('off')
        plt.tight_layout(); plt.savefig(os.path.join(os.environ['PICS'], 'align_%s.png' % part), dpi=95); print('picture saved')
