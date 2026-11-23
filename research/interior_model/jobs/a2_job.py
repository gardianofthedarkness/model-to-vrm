# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "matplotlib"]
# ///
"""Design v7, test A2: does a classifier learn the pelvis's tissue labels from the outside surface alone?

People: adults with a whole pelvis, official split. Window: the hip bones' and sacrum's height range (from the stored
landmarks), whole crop in x and y, centred on the body outline (surface only). Grid 3 mm (labels_v1 at 1.5 mm, 2x2x2
majority). Target classes (labels v1, grouped): 0 outside, 1 fat (subcutaneous and other: v1 cannot split them),
2 lean, 3 gas, 4 dense unlabelled, then named: hip L, hip R, sacrum, femur L, femur R, gluteus maximus/medius/minimus
(left and right merged per muscle), iliopsoas, autochthon, lumbar vertebrae (L4, L5 merged), and 'organ' (every other
labelled structure). Input: signed distance to the surface (cm) and inside share at 3 mm, plus grid coordinates.
Loss: cross-entropy with class weights (training voxel share)^-1/2, fixed from data. Checkpoint by validation loss.
Scores on test people (outside fixed to the true outline in every method): Dice per class; bone surface distance (mm);
femoral-head centre distance (top tenth of each predicted femur against the stored landmark); fat thickness at the
skin (distance to the first lean or muscle voxel). Baselines: nearest training body (outline IoU) and the training
mean (per-voxel class frequency, argmax). args: <repo> [limit per split] [max epochs]"""
import os, sys, csv, json, time, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
NAMED = [('hip', ['hip_left']), ('hip_r', ['hip_right']), ('sacrum', ['sacrum']), ('femur_l', ['femur_left']), ('femur_r', ['femur_right']),
         ('gluteus_max', ['gluteus_maximus_left', 'gluteus_maximus_right']), ('gluteus_med', ['gluteus_medius_left', 'gluteus_medius_right']),
         ('gluteus_min', ['gluteus_minimus_left', 'gluteus_minimus_right']), ('iliopsoas', ['iliopsoas_left', 'iliopsoas_right']),
         ('autochthon', ['autochthon_left', 'autochthon_right']), ('lumbar', ['vertebrae_L4', 'vertebrae_L5'])]
CLASSES = ['outside', 'fat', 'lean', 'gas', 'dense'] + [n for n, _ in NAMED] + ['organ']
K = len(CLASSES); ORGAN = K - 1

def remap(L, ids):
    """labels_v1 ids -> A2 classes."""
    lut = np.full(256, ORGAN, np.uint8); lut[0] = 0; lut[1] = 1; lut[2] = 1; lut[3] = 2; lut[4] = 3; lut[5] = 4
    for k, (_, names) in enumerate(NAMED):
        for n in names:
            if n in ids: lut[ids[n]] = 5 + k
    return lut[L]

def down2(L):
    """3 mm majority of 2x2x2 blocks (ties to the higher id: structures over brightness classes)."""
    X, Y, Z = (n // 2 * 2 for n in L.shape); L = L[:X, :Y, :Z]
    b = L.reshape(X // 2, 2, Y // 2, 2, Z // 2, 2).transpose(0, 2, 4, 1, 3, 5).reshape(X // 2, Y // 2, Z // 2, 8)
    cnt = np.zeros(b.shape[:3] + (K,), np.uint8)
    for k in range(K): cnt[..., k] = (b == k).sum(-1)
    return (K - 1 - np.argmax(cnt[..., ::-1], -1)).astype(np.uint8)

def place(A, G, fill, shift):
    """put A into a box G with A's xy outline centroid at the box centre (shift given), z centred."""
    out = np.full(G, fill, A.dtype); src, dst = [], []
    for a, (n, g) in enumerate(zip(A.shape, G)):
        o = (g - n) // 2 + (shift[a] if a < 2 else 0)
        s0, d0 = max(0, -o), max(0, o); ln = min(n - s0, g - d0)
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
        a, b = P == k, T == k; s_ = a.sum() + b.sum(); r['dice_' + CLASSES[k]] = float(2 * (a & b).sum() / s_) if b.sum() > 20 else None
    bone_p = np.isin(P, [5, 6, 7, 8, 9, 15]); bone_t = np.isin(T, [5, 6, 7, 8, 9, 15])
    if bone_p.any() and bone_t.any():
        ep, et = bone_p & ~binary_erosion(bone_p), bone_t & ~binary_erosion(bone_t)
        r['bone_surface_mm'] = float((distance_transform_edt(~et, sampling=sp)[ep].mean() + distance_transform_edt(~ep, sampling=sp)[et].mean()) / 2)
    else: r['bone_surface_mm'] = None
    for side, k in (('left', 8), ('right', 9)):
        key = 'femoral_head_%s' % side
        if key in lm and (P == k).any():
            p = np.argwhere(P == k); z = p[:, 2]; top = p[z >= z.max() - 0.1 * max(z.max() - z.min(), 1)].mean(0)
            r['femhead_%s_mm' % side] = float(np.linalg.norm((top - np.array(lm[key])) * sp))
    skin = M & ~binary_erosion(M); deep = binary_erosion(M, iterations=2)
    lp, lt = np.isin(P, [2] + list(range(5, K))) & deep, np.isin(T, [2] + list(range(5, K))) & deep
    if lp.any() and lt.any(): r['fat_thickness_err_mm'] = float(np.abs(distance_transform_edt(~lp, sampling=sp)[skin] - distance_transform_edt(~lt, sampling=sp)[skin]).mean())
    return r

def prepare(case, codes_dir, labels_dir, ids):
    """one person's window at 3 mm: labels, body mask, landmarks in window voxels."""
    meta = json.load(open(os.path.join(codes_dir, case + '.json'))); cz = np.load(os.path.join(codes_dir, case + '.npz'))
    shp = meta['crop_shape']; M = np.unpackbits(cz['mask'])[:int(np.prod(shp))].reshape(shp).astype(bool)
    L = remap(np.load(os.path.join(labels_dir, case + '.npz'))['labels'], ids)
    lmk = meta['landmarks']; z0 = min(lmk[n]['lo'][2] for n in ('hip_left', 'hip_right', 'sacrum')); z1 = max(lmk[n]['hi'][2] for n in ('hip_left', 'hip_right', 'sacrum')) + 1
    L, M = L[:, :, z0:z1], M[:, :, z0:z1]; L3 = down2(L); M3 = down2(M.astype(np.uint8)) > 0; L3[~M3] = 0
    lm = {k: [v['centroid'][0] / 2, v['centroid'][1] / 2, (v['centroid'][2] - z0) / 2] for k, v in lmk.items() if k.startswith('femoral_head')}
    return L3, M3, lm

def run(cases, codes_dir, labels_dir, ids, dev, max_epochs, log=print, figdir=None):
    import torch, copy
    from multiprocessing import Pool
    data = {}
    for sp_ in ('train', 'val', 'test'):
        for c in cases[sp_]: data[c] = prepare(c, codes_dir, labels_dir, ids)
    shapes = np.array([data[c][0].shape for c in cases['train']]); G = tuple(int(np.ceil(np.percentile(shapes[:, a], 95) / 16) * 16) for a in range(3)); log('grid (3 mm)', G)
    def boxed(c):
        L3, M3, lm = data[c]; cen = np.argwhere(M3).mean(0); sh = [int(round(L3.shape[a] / 2 - cen[a])) for a in (0, 1)]
        Lb, (src, dst) = place(L3, G, 0, sh); Mb, _ = place(M3, G, False, sh)
        lmb = {k: [v[a] - src[a].start + dst[a].start for a in range(3)] for k, v in lm.items()}; return Lb, Mb, lmb
    B = {c: boxed(c) for c in data}
    X = {s: np.stack([condition(B[c][1]).astype(np.float16) for c in cases[s]]) for s in cases}; Y = {s: np.stack([B[c][0] for c in cases[s]]).astype(np.uint8) for s in cases}
    share = np.bincount(Y['train'].ravel(), minlength=K) / Y['train'].size; wts = torch.tensor(np.where(share > 0, share, 1.0) ** -0.5, dtype=torch.float32); wts = wts / wts.mean()
    net = make_net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); lossf = torch.nn.CrossEntropyLoss(weight=wts.to(dev))
    best = (np.inf, None, 0); hist = []; bs = 2
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
        if ep % 5 == 0 or ep == max_epochs - 1: log('epoch %d train %.4f val %.4f (best %.4f at %d)' % (ep, hist[-1][0], vl, best[0], best[2]))
        if ep - best[2] > max(15, ep // 5): break
    net.load_state_dict(best[1]); net.eval()
    freq = np.zeros(G + (K,), np.float32)
    for y in Y['train']: freq[np.arange(G[0])[:, None, None], np.arange(G[1])[None, :, None], np.arange(G[2])[None, None, :], y] += 1
    mean_lab = freq.argmax(-1).astype(np.uint8); trM = np.stack([B[c][1] for c in cases['train']])
    res = dict(classes=CLASSES, grid=list(G), people={s: len(v) for s, v in cases.items()}, best_epoch=best[2], epochs=len(hist), curve=[[round(a, 4), round(b, 4)] for a, b in hist])
    for sp_ in ('val', 'test'):
        rows = dict(model=[], nearest_body=[], training_mean=[])
        for k, c in enumerate(cases[sp_]):
            Lb, Mb, lmb = B[c]
            with torch.no_grad(): P = net(torch.tensor(X[sp_][k:k + 1]).to(dev).float()).float().argmax(1)[0].cpu().numpy().astype(np.uint8)
            P[~Mb] = 0; P[Mb & (P == 0)] = 2
            iou = (trM & Mb).sum((1, 2, 3)) / np.maximum((trM | Mb).sum((1, 2, 3)), 1); Nn = Y['train'][int(iou.argmax())].astype(np.uint8).copy(); Nn[~Mb] = 0; Nn[Mb & (Nn == 0)] = 2
            A = mean_lab.copy(); A[~Mb] = 0; A[Mb & (A == 0)] = 2
            for name, V in (('model', P), ('nearest_body', Nn), ('training_mean', A)): rows[name].append(dict(case=c, **scores(V, Lb, Mb, lmb)))
            if figdir and sp_ == 'test' and k < 4: draw(Lb, P, Nn, A, os.path.join(figdir, 'a2_%s.png' % c), c)
        agg = {}
        for name, rr in rows.items():
            keys = sorted({kk for r in rr for kk in r if kk != 'case'}); agg[name] = {kk: round(float(np.median([r[kk] for r in rr if r.get(kk) is not None])), 3) for kk in keys if any(r.get(kk) is not None for r in rr)}
        res[sp_] = dict(median=agg, per_person=rows); log(sp_, json.dumps(agg))
    return res

def draw(T, P, Nn, A, path, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    cols = ['#000000', '#e8c170', '#b5524a', '#3b3b3b', '#dddddd', '#f2f2f2', '#e6e6e6', '#cfcfcf', '#ffffff', '#fafafa', '#7a3e8c', '#4f6db8', '#3f9e8f', '#c46d2a', '#8c8c2a', '#bbbbbb', '#5aa0d6']
    cm = ListedColormap(cols[:K]); zc = T.shape[2] // 2; yc = int(np.argwhere(T > 0)[:, 1].mean())
    f, a = plt.subplots(2, 4, figsize=(18, 8))
    for j, (V, t) in enumerate(((T, 'labels (truth)'), (P, 'model, surface only'), (Nn, 'nearest training body'), (A, 'training mean'))):
        a[0, j].imshow(V[:, :, zc].T, cmap=cm, vmin=0, vmax=K - 1, interpolation='nearest'); a[0, j].set_title(t); a[1, j].imshow(V[:, yc, :].T[::-1], cmap=cm, vmin=0, vmax=K - 1, interpolation='nearest')
        for r in (0, 1): a[r, j].set_xticks([]); a[r, j].set_yticks([])
    f.suptitle('%s: axial (top), coronal (bottom); bones white, muscles coloured, fat yellow, lean red' % title); f.savefig(path, dpi=100, bbox_inches='tight'); plt.close(f)

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
    allc = [c for v in cases.values() for c in v]; log('people', {s: len(v) for s, v in cases.items()})
    snap = snapshot_download(repo, repo_type='dataset', allow_patterns=['codes_v1/%s.%s' % (c, e) for c in allc for e in ('json', 'npz')] + ['labels_v1/%s.npz' % c for c in allc] + ['labels_v1/label_ids.json'], max_workers=16)
    ids = json.load(open(os.path.join(snap, 'labels_v1', 'label_ids.json')))['ids']; log('data ready')
    os.makedirs('figsA2', exist_ok=True)
    res = run(cases, os.path.join(snap, 'codes_v1'), os.path.join(snap, 'labels_v1'), ids, 'cuda', max_epochs, log, 'figsA2'); res['gpu'] = torch.cuda.get_device_name(0)
    tag = 'smoke' if limit else 'full'; json.dump(res, open('figsA2/a2_%s.json' % tag, 'w'), indent=1)
    api.upload_folder(folder_path='figsA2', path_in_repo='a2/%s' % tag, repo_id=repo, repo_type='dataset', commit_message='A2 %s' % tag)
    res.pop('curve'); [res[s].pop('per_person') for s in ('val', 'test')]; log('SUMMARY ' + json.dumps(res))
