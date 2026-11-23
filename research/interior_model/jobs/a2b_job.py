# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "matplotlib"]
# ///
"""Milestones 12 (Tissue) and 14 (Baseline): the Classifier test (surface -> tissue labels, pelvis, 3 mm) with
  LABELSRC = v1      fat and lean from brightness (as milestone 10), or
  LABELSRC = tissue  licensed TotalSegmentator tissue labels: subcutaneous fat, torso fat, intermuscular fat, muscle;
  WINDOW   = labels  the pelvis window's height range from the hip-bone labels (as milestone 10), or
  WINDOW   = surface the height range predicted from the body outline alone, for validation and test people: a 1D
                     network over per-slice width, depth and area (mm, mm, cm2) trained on the training people;
and one more baseline in every run: a surface-fitted population atlas (per training person, each slice of the window
is stretched to a common box by the outline's width and depth, and the window's height to a common length; class
frequencies are averaged over training people; for a test person the most frequent class is stretched back into
their own outline). Scores as in milestone 10, plus subcutaneous-fat (fascia) depth error with tissue labels and the
window's error in mm with WINDOW=surface. args: <repo> [limit per split] [max epochs]"""
import os, sys, csv, json, time, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
SRC = os.environ.get('LABELSRC', 'v1'); WIN = os.environ.get('WINDOW', 'labels')
NAMED = [('hip', ['hip_left']), ('hip_r', ['hip_right']), ('sacrum', ['sacrum']), ('femur_l', ['femur_left']), ('femur_r', ['femur_right']),
         ('gluteus_max', ['gluteus_maximus_left', 'gluteus_maximus_right']), ('gluteus_med', ['gluteus_medius_left', 'gluteus_medius_right']),
         ('gluteus_min', ['gluteus_minimus_left', 'gluteus_minimus_right']), ('iliopsoas', ['iliopsoas_left', 'iliopsoas_right']),
         ('autochthon', ['autochthon_left', 'autochthon_right']), ('lumbar', ['vertebrae_L4', 'vertebrae_L5'])]
BASE = ['outside', 'fat', 'lean', 'gas', 'dense'] if SRC == 'v1' else ['outside', 'subcut_fat', 'torso_fat', 'inter_fat', 'muscle_other', 'lean_other', 'gas', 'dense']
CLASSES = BASE + [n for n, _ in NAMED] + ['organ']; K = len(CLASSES); NB = len(BASE); ORGAN = K - 1; ix = {n: i for i, n in enumerate(CLASSES)}
BONES = [ix[n] for n in ('hip', 'hip_r', 'sacrum', 'femur_l', 'femur_r', 'lumbar')]; FATS = [ix[n] for n in CLASSES if 'fat' in n]

def remap(L, ids, tissue=None):
    """labels_v1 ids (and the licensed tissue map) -> this run's classes."""
    named = np.zeros(256, np.uint8)
    for k, (_, names) in enumerate(NAMED):
        for n in names:
            if n in ids: named[ids[n]] = NB + k
    isnamed = named[L] > 0; out = np.full(L.shape, ORGAN, np.uint8)
    if SRC == 'v1':
        lut = np.full(256, ORGAN, np.uint8); lut[[0, 1, 2, 3, 4, 5]] = [0, 1, 1, 2, 3, 4]; out = lut[L]
    else:
        basic = L < 10
        out[basic & np.isin(L, [1, 2])] = ix['torso_fat']; out[basic & (L == 3)] = ix['lean_other']; out[basic & (L == 4)] = ix['gas']; out[basic & (L == 5)] = ix['dense']; out[L == 0] = 0
        out[basic & (tissue == 1)] = ix['subcut_fat']; out[basic & (tissue == 2)] = ix['torso_fat']; out[basic & (tissue == 3)] = ix['muscle_other']; out[basic & (tissue == 4)] = ix['inter_fat']; out[L == 0] = 0
    out[isnamed] = named[L][isnamed]; return out

def down2(L):
    X, Y, Z = (n // 2 * 2 for n in L.shape); L = L[:X, :Y, :Z]
    b = L.reshape(X // 2, 2, Y // 2, 2, Z // 2, 2).transpose(0, 2, 4, 1, 3, 5).reshape(X // 2, Y // 2, Z // 2, 8)
    cnt = np.zeros(b.shape[:3] + (K,), np.uint8)
    for k in range(K): cnt[..., k] = (b == k).sum(-1)
    return (K - 1 - np.argmax(cnt[..., ::-1], -1)).astype(np.uint8)

def place(A, G, fill, shift):
    out = np.full(G, fill, A.dtype); src, dst = [], []
    for a, (n, g) in enumerate(zip(A.shape, G)):
        o = (g - n) // 2 + (shift[a] if a < 2 else 0); s0, d0 = max(0, -o), max(0, o); ln = max(0, min(n - s0, g - d0))
        src.append(slice(s0, s0 + ln)); dst.append(slice(d0, d0 + ln))
    out[tuple(dst)] = A[tuple(src)]; return out, (src, dst)

def condition(M, sp=3.0):
    from scipy.ndimage import distance_transform_edt
    d = (distance_transform_edt(M, sampling=sp) - distance_transform_edt(~M, sampling=sp)) / 10.0
    g = [np.linspace(-1, 1, n) for n in M.shape]; X, Y, Z = np.meshgrid(*g, indexing='ij')
    return np.stack([d, M.astype(np.float32), X, Y, Z]).astype(np.float32)

def make_net(cin=5, cout=K, w=32):
    import torch, torch.nn as nn
    def blk(a, b): return nn.Sequential(nn.Conv3d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(), nn.Conv3d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())
    class U(nn.Module):
        def __init__(s):
            super().__init__(); s.e1 = blk(cin, w); s.e2 = blk(w, 2 * w); s.e3 = blk(2 * w, 4 * w); s.e4 = blk(4 * w, 8 * w); s.m = blk(8 * w, 8 * w)
            s.d4 = blk(16 * w, 4 * w); s.d3 = blk(8 * w, 2 * w); s.d2 = blk(4 * w, w); s.d1 = blk(2 * w, w); s.o = nn.Conv3d(w, cout, 1); s.p = nn.MaxPool3d(2)
            s.u = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        def forward(s, x):
            a = s.e1(x); b = s.e2(s.p(a)); c = s.e3(s.p(b)); d = s.e4(s.p(c)); m = s.m(s.p(d))
            y = s.d4(torch.cat([s.u(m), d], 1)); y = s.d3(torch.cat([s.u(y), c], 1)); y = s.d2(torch.cat([s.u(y), b], 1)); y = s.d1(torch.cat([s.u(y), a], 1)); return s.o(y)
    return U()

def scores(P, T, M, lm, sp=3.0):
    from scipy.ndimage import distance_transform_edt, binary_erosion
    r = {}
    for k in range(1, K):
        a, b = P == k, T == k; r['dice_' + CLASSES[k]] = float(2 * (a & b).sum() / (a.sum() + b.sum())) if b.sum() > 20 else None
    bp, bt = np.isin(P, BONES), np.isin(T, BONES)
    if bp.any() and bt.any():
        ep, et = bp & ~binary_erosion(bp), bt & ~binary_erosion(bt)
        r['bone_surface_mm'] = float((distance_transform_edt(~et, sampling=sp)[ep].mean() + distance_transform_edt(~ep, sampling=sp)[et].mean()) / 2)
    for side, k in (('left', ix['femur_l']), ('right', ix['femur_r'])):
        key = 'femoral_head_%s' % side
        if key in lm and (P == k).any():
            p = np.argwhere(P == k); z = p[:, 2]; top = p[z >= z.max() - 0.1 * max(z.max() - z.min(), 1)].mean(0); r['femhead_%s_mm' % side] = float(np.linalg.norm((top - np.array(lm[key])) * sp))
    skin = M & ~binary_erosion(M); deep = binary_erosion(M, iterations=2)
    lp, lt = M & ~np.isin(P, FATS + [0]) & deep, M & ~np.isin(T, FATS + [0]) & deep
    if lp.any() and lt.any(): r['fat_thickness_err_mm'] = float(np.abs(distance_transform_edt(~lp, sampling=sp)[skin] - distance_transform_edt(~lt, sampling=sp)[skin]).mean())
    if SRC == 'tissue':                                                                       # depth of the subcutaneous layer = where the fascia lies
        sp_, st_ = M & (P != ix['subcut_fat']) & deep, M & (T != ix['subcut_fat']) & deep
        if sp_.any() and st_.any(): r['fascia_depth_err_mm'] = float(np.abs(distance_transform_edt(~sp_, sampling=sp)[skin] - distance_transform_edt(~st_, sampling=sp)[skin]).mean())
    return r

# ---------------- surface-fitted population atlas
def to_canon(L3, M3, C):
    """stretch each slice's outline box to C[0] x C[1], and the height to C[2] (nearest)."""
    out = np.zeros(C, np.uint8); zz = np.minimum((np.arange(C[2]) + 0.5) / C[2] * L3.shape[2], L3.shape[2] - 1).astype(int)
    for k, z in enumerate(zz):
        m = M3[:, :, z]
        if not m.any(): continue
        xs = np.flatnonzero(m.any(1)); ys = np.flatnonzero(m.any(0)); xi = np.minimum(xs[0] + ((np.arange(C[0]) + 0.5) / C[0] * (xs[-1] - xs[0] + 1)).astype(int), xs[-1]); yi = np.minimum(ys[0] + ((np.arange(C[1]) + 0.5) / C[1] * (ys[-1] - ys[0] + 1)).astype(int), ys[-1])
        out[:, :, k] = L3[np.ix_(xi, yi, [z])][:, :, 0]
    return out
def from_canon(A, M3):
    out = np.zeros(M3.shape, np.uint8); C = A.shape
    for z in range(M3.shape[2]):
        m = M3[:, :, z]
        if not m.any(): continue
        k = min(int((z + 0.5) / M3.shape[2] * C[2]), C[2] - 1); xs = np.flatnonzero(m.any(1)); ys = np.flatnonzero(m.any(0))
        xi = np.minimum(((np.arange(xs[0], xs[-1] + 1) - xs[0] + 0.5) / (xs[-1] - xs[0] + 1) * C[0]).astype(int), C[0] - 1); yi = np.minimum(((np.arange(ys[0], ys[-1] + 1) - ys[0] + 0.5) / (ys[-1] - ys[0] + 1) * C[1]).astype(int), C[1] - 1)
        out[xs[0]:xs[-1] + 1, ys[0]:ys[-1] + 1, z] = A[np.ix_(xi, yi, [k])][:, :, 0]
    out[~M3] = 0; return out

# ---------------- window from the outline alone
def profile(M, sp=1.5, f=2):
    """per 3 mm slice: outline width, depth (mm) and area (cm2)."""
    Z = M.shape[2] // f * f; m = M[:, :, :Z].reshape(M.shape[0], M.shape[1], Z // f, f).any(-1)
    w = m.any(1).sum(0) * sp; d = m.any(0).sum(0) * sp; a = m.sum((0, 1)) * sp * sp / 100.0
    return np.stack([w, d, a]).astype(np.float32)
def train_window(P, Y, Pv, Yv, dev, log=print):
    """P: list of (3, Z) profiles; Y: list of (Z,) 0/1 inside the hip range. A 1D conv net; best validation loss."""
    import torch, torch.nn as nn, copy
    mu = np.concatenate(P, 1).mean(1, keepdims=True); sd = np.concatenate(P, 1).std(1, keepdims=True)
    net = nn.Sequential(nn.Conv1d(3, 32, 9, padding=4), nn.SiLU(), nn.Conv1d(32, 32, 9, padding=8, dilation=2), nn.SiLU(), nn.Conv1d(32, 32, 9, padding=16, dilation=4), nn.SiLU(), nn.Conv1d(32, 32, 9, padding=32, dilation=8), nn.SiLU(), nn.Conv1d(32, 1, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), 1e-3); T = lambda p: torch.tensor((p - mu) / sd, device=dev)[None]; best = (np.inf, None, 0)
    for ep in range(int(os.environ.get('WIN_EPOCHS', '200'))):
        net.train()
        for i in np.random.default_rng(ep).permutation(len(P)):
            loss = nn.functional.binary_cross_entropy_with_logits(net(T(P[i]))[0, 0], torch.tensor(Y[i], device=dev)); opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl = float(np.mean([nn.functional.binary_cross_entropy_with_logits(net(T(p))[0, 0], torch.tensor(y, device=dev)).item() for p, y in zip(Pv, Yv)]))
        if vl < best[0]: best = (vl, copy.deepcopy(net.state_dict()), ep)
        if ep - best[2] > 20: break
    net.load_state_dict(best[1]); net.eval(); log('window net: best val %.4f at epoch %d' % (best[0], best[2]))
    def predict(p):
        with torch.no_grad(): q = torch.sigmoid(net(T(p))[0, 0]).cpu().numpy() > 0.5
        if not q.any(): return None
        d = np.diff(np.r_[0, q.astype(int), 0]); a = np.flatnonzero(d == 1); b = np.flatnonzero(d == -1); j = int(np.argmax(b - a)); return int(a[j]), int(b[j])   # longest run, in 3 mm slices
    return predict

def load_case(case, codes_dir, labels_dir, tissue_dir, ids):
    meta = json.load(open(os.path.join(codes_dir, case + '.json'))); cz = np.load(os.path.join(codes_dir, case + '.npz')); shp = meta['crop_shape']
    M = np.unpackbits(cz['mask'])[:int(np.prod(shp))].reshape(shp).astype(bool); Lr = np.load(os.path.join(labels_dir, case + '.npz'))['labels']
    Tm = np.load(os.path.join(tissue_dir, case + '.npz'))['tissue'] if SRC == 'tissue' else None
    lmk = meta['landmarks']; z0 = min(lmk[n]['lo'][2] for n in ('hip_left', 'hip_right', 'sacrum')); z1 = max(lmk[n]['hi'][2] for n in ('hip_left', 'hip_right', 'sacrum')) + 1
    return dict(L=remap(Lr, ids, Tm), M=M, lmk=lmk, true=(z0, z1))
def window(d, z0, z1):
    L, M = d['L'][:, :, z0:z1], d['M'][:, :, z0:z1]; L3 = down2(L); M3 = down2(M.astype(np.uint8)) > 0; L3[~M3] = 0
    lm = {k: [v['centroid'][0] / 2, v['centroid'][1] / 2, (v['centroid'][2] - z0) / 2] for k, v in d['lmk'].items() if k.startswith('femoral_head')}; return L3, M3, lm

def run(cases, codes_dir, labels_dir, tissue_dir, ids, dev, max_epochs, log=print, figdir=None):
    import torch, copy
    raw = {c: load_case(c, codes_dir, labels_dir, tissue_dir, ids) for v in cases.values() for c in v}; log('loaded', len(raw))
    win = {c: raw[c]['true'] for c in raw}; res = dict(labelsrc=SRC, window=WIN, classes=CLASSES)
    if WIN == 'surface':
        prof = {c: profile(raw[c]['M']) for c in raw}
        def ylab(c): y = np.zeros(prof[c].shape[1], np.float32); y[raw[c]['true'][0] // 2:raw[c]['true'][1] // 2] = 1; return y
        pred = train_window([prof[c] for c in cases['train']], [ylab(c) for c in cases['train']], [prof[c] for c in cases['val']], [ylab(c) for c in cases['val']], dev, log); errs = []
        for s in ('val', 'test'):
            for c in cases[s]:
                w = pred(prof[c])
                if w is None: w = (raw[c]['true'][0] // 2, raw[c]['true'][1] // 2); log('no window predicted for', c)
                z0, z1 = w[0] * 2, min(w[1] * 2, raw[c]['M'].shape[2]); win[c] = (z0, z1)
                if s == 'test': errs.append((abs(z0 - raw[c]['true'][0]) * 1.5, abs(z1 - raw[c]['true'][1]) * 1.5))
        e = np.array(errs); res['window_error_mm'] = dict(lower_median=float(np.median(e[:, 0])), upper_median=float(np.median(e[:, 1])), lower_p90=float(np.percentile(e[:, 0], 90)), upper_p90=float(np.percentile(e[:, 1], 90))); log('window error', res['window_error_mm'])
    data = {c: window(raw[c], *win[c]) for c in raw}; del raw
    shapes = np.array([data[c][0].shape for c in cases['train']]); G = tuple(int(np.ceil(np.percentile(shapes[:, a], 95) / 16) * 16) for a in range(3)); log('grid (3 mm)', G)
    Cn = tuple(int(np.median(shapes[:, a])) for a in range(3))                                # canonical box of the atlas: the median training window
    def boxed(c):
        L3, M3, lm = data[c]; cen = np.argwhere(M3).mean(0); sh = [int(round(L3.shape[a] / 2 - cen[a])) for a in (0, 1)]
        Lb, (src, dst) = place(L3, G, 0, sh); Mb, _ = place(M3, G, False, sh); return Lb, Mb, {k: [v[a] - src[a].start + dst[a].start for a in range(3)] for k, v in lm.items()}
    B = {c: boxed(c) for c in data}
    freq = np.zeros(Cn + (K,), np.uint16)
    for c in cases['train']:
        A = to_canon(data[c][0], data[c][1], Cn); freq[np.arange(Cn[0])[:, None, None], np.arange(Cn[1])[None, :, None], np.arange(Cn[2])[None, None, :], A] += 1
    freq[..., 0] = 0; atlas = freq.argmax(-1).astype(np.uint8); log('atlas canonical box', Cn)
    X = {s: np.stack([condition(B[c][1]).astype(np.float16) for c in cases[s]]) for s in cases}; Y = {s: np.stack([B[c][0] for c in cases[s]]).astype(np.uint8) for s in cases}
    share = np.bincount(Y['train'].ravel(), minlength=K) / Y['train'].size; wts = torch.tensor(np.where(share > 0, share, 1.0) ** -0.5, dtype=torch.float32); wts = wts / wts.mean()
    net = make_net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); lossf = torch.nn.CrossEntropyLoss(weight=wts.to(dev)); best = (np.inf, None, 0); hist = []; bs = 2
    Xtr, Ytr = torch.tensor(X['train']), torch.tensor(Y['train']); Xva, Yva = torch.tensor(X['val']), torch.tensor(Y['val'])
    for ep in range(max_epochs):
        net.train(); perm = torch.randperm(len(Xtr)); tl = []
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'): loss = lossf(net(Xtr[j].to(dev).float()).float(), Ytr[j].to(dev).long())
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        net.eval()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            vl = float(np.mean([lossf(net(Xva[i:i + bs].to(dev).float()).float(), Yva[i:i + bs].to(dev).long()).item() for i in range(0, len(Xva), bs)]))
        hist.append((float(np.mean(tl)), vl))
        if vl < best[0]: best = (vl, copy.deepcopy(net.state_dict()), ep)
        if ep % 5 == 0: log('epoch %d train %.4f val %.4f (best %.4f at %d)' % (ep, hist[-1][0], vl, best[0], best[2]))
        if ep - best[2] > max(15, ep // 5): break
    net.load_state_dict(best[1]); net.eval(); trM = np.stack([B[c][1] for c in cases['train']])
    res.update(grid=list(G), people={s: len(v) for s, v in cases.items()}, best_epoch=best[2], epochs=len(hist), curve=[[round(a, 4), round(b, 4)] for a, b in hist])
    for sp_ in ('val', 'test'):
        rows = dict(model=[], nearest_body=[], surface_atlas=[])
        for k, c in enumerate(cases[sp_]):
            Lb, Mb, lmb = B[c]
            with torch.no_grad(): P = net(torch.tensor(X[sp_][k:k + 1]).to(dev).float()).float().argmax(1)[0].cpu().numpy().astype(np.uint8)
            fill = ix['lean'] if SRC == 'v1' else ix['lean_other']; P[~Mb] = 0; P[Mb & (P == 0)] = fill
            iou = (trM & Mb).sum((1, 2, 3)) / np.maximum((trM | Mb).sum((1, 2, 3)), 1); Nn = Y['train'][int(iou.argmax())].copy(); Nn[~Mb] = 0; Nn[Mb & (Nn == 0)] = fill
            A3 = from_canon(atlas, data[c][1]); cen = np.argwhere(data[c][1]).mean(0); sh = [int(round(data[c][1].shape[a] / 2 - cen[a])) for a in (0, 1)]; A, _ = place(A3, G, 0, sh); A[~Mb] = 0; A[Mb & (A == 0)] = fill
            for name, V in (('model', P), ('nearest_body', Nn), ('surface_atlas', A)): rows[name].append(dict(case=c, **scores(V, Lb, Mb, lmb)))
            if figdir and sp_ == 'test' and k < 4: draw(Lb, P, Nn, A, os.path.join(figdir, 'a2b_%s_%s_%s.png' % (SRC, WIN, c)), c)
        agg = {}
        for name, rr in rows.items():
            keys = sorted({kk for r in rr for kk in r if kk != 'case'}); agg[name] = {kk: round(float(np.median([r[kk] for r in rr if r.get(kk) is not None])), 3) for kk in keys if any(r.get(kk) is not None for r in rr)}
        wins = {}
        for kk in ('bone_surface_mm', 'dice_hip', 'dice_femur_l', 'fat_thickness_err_mm', 'fascia_depth_err_mm'):
            for other in ('nearest_body', 'surface_atlas'):
                a = np.array([r.get(kk) if r.get(kk) is not None else np.nan for r in rows['model']], float); b = np.array([r.get(kk) if r.get(kk) is not None else np.nan for r in rows[other]], float); ok = np.isfinite(a) & np.isfinite(b)
                if ok.any(): wins['%s_vs_%s' % (kk, other)] = '%d of %d' % (int(((a < b) if 'mm' in kk else (a > b))[ok].sum()), int(ok.sum()))
        res[sp_] = dict(median=agg, model_better=wins, per_person=rows); log(sp_, json.dumps(agg)); log(sp_, 'model better:', json.dumps(wins))
    return res

def draw(T, P, Nn, A, path, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    cm = plt.get_cmap('tab20', K); zc = T.shape[2] // 2; yc = int(np.argwhere(T > 0)[:, 1].mean()); f, a = plt.subplots(2, 4, figsize=(18, 8))
    for j, (V, t) in enumerate(((T, 'labels (truth)'), (P, 'model, surface only'), (Nn, 'nearest training body'), (A, 'surface-fitted atlas'))):
        a[0, j].imshow(V[:, :, zc].T, cmap=cm, vmin=0, vmax=K - 1, interpolation='nearest'); a[0, j].set_title(t); a[1, j].imshow(V[:, yc, :].T[::-1], cmap=cm, vmin=0, vmax=K - 1, interpolation='nearest')
        for r in (0, 1): a[r, j].set_xticks([]); a[r, j].set_yticks([])
    f.suptitle('%s (%s labels, %s window): axial (top), coronal (bottom)' % (title, SRC, WIN)); f.savefig(path, dpi=100, bbox_inches='tight'); plt.close(f)

if __name__ == '__main__':
    import torch
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0; max_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 200
    ext = list(csv.DictReader(open(hf_hub_download(repo, 'census/label_extents.csv', repo_type='dataset'))))
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    by = {}
    for e in ext: by.setdefault(e['case'], {})[e['label']] = e
    whole = lambda e: bool(e) and e['present'] == '1' and not any(e[k] == '1' for k in ('cut_inf', 'cut_sup', 'cut_side', 'cut_ap'))
    sel = sorted(c for c, d in by.items() if whole(d.get('hip_left')) and whole(d.get('hip_right')) and whole(d.get('sacrum')) and not (meta[c].get('age') and float(meta[c]['age']) < 18))
    cases = {s: [c for c in sel if meta[c]['split'] == s] for s in ('train', 'val', 'test')}
    if limit: cases = {s: v[:limit if s == 'train' else max(2, limit // 4)] for s, v in cases.items()}
    allc = [c for v in cases.values() for c in v]; log('people', {s: len(v) for s, v in cases.items()}, 'labels', SRC, 'window', WIN)
    pats = ['codes_v1/%s.%s' % (c, e) for c in allc for e in ('json', 'npz')] + ['labels_v1/%s.npz' % c for c in allc] + ['labels_v1/label_ids.json'] + (['tissue_v1/%s.npz' % c for c in allc] if SRC == 'tissue' else [])
    snap = snapshot_download(repo, repo_type='dataset', allow_patterns=pats, max_workers=16); ids = json.load(open(os.path.join(snap, 'labels_v1', 'label_ids.json')))['ids']; log('data ready')
    os.makedirs('figs', exist_ok=True)
    res = run(cases, os.path.join(snap, 'codes_v1'), os.path.join(snap, 'labels_v1'), os.path.join(snap, 'tissue_v1'), ids, 'cuda', max_epochs, log, 'figs'); res['gpu'] = torch.cuda.get_device_name(0)
    tag = '%s_%s_%s' % (SRC, WIN, 'smoke' if limit else 'full'); json.dump(res, open('figs/a2b_%s.json' % tag, 'w'), indent=1)
    api.upload_folder(folder_path='figs', path_in_repo='a2b/%s' % tag, repo_id=repo, repo_type='dataset', commit_message='A2b %s' % tag)
    res.pop('curve'); [res[s].pop('per_person') for s in ('val', 'test')]; log('SUMMARY ' + json.dumps(res))
