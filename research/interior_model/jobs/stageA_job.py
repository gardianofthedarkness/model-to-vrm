# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "torch", "monai==1.4.0", "matplotlib"]
# ///
"""Plan v6.4, stage A: does a network learn the inside of the pelvis from its outside?

People: adults with a whole pelvis, in the dataset's own split (train / val / test). Region: the body piece holding
the hip bones, slice by slice, over the hip bones' height. (Caveat: the window's height range is set from the hip
labels; everything inside the window is predicted from the surface alone. All baselines share the window.)
Grid: every crop is centred on its body mask in a common box whose size is the 95th percentile of the training crops
(rounded up to 32 voxels, so the code grid halves three times). Code: frozen MAISI autoencoder, posterior mean (4 channels, a quarter size per axis).
Input to the network: the signed distance to the body surface (mm) and the inside share, pooled to the code grid, plus
the grid coordinates. Output: the code. Trained with L1 on standardised codes; stopped on the validation people.
Scored on the test people, on the decoded volume against the CT (both median-filtered 3x3x3 before thresholds):
  HU error inside the body; bone Dice and bone surface distance (mm); skin-to-bone depth error at skin voxels (mm);
  fat-thickness error at skin voxels (mm: distance from skin to the nearest lean or bone voxel deeper than 2 voxels);
  fat share error (fat voxels / body voxels).
Compared with: (1) the mean CT of the training people in the grid, cut to the test body; (2) the CT of the training
person whose body mask overlaps the test body most (IoU), cut to the test body; (3) the autoencoder's own round trip of
the true CT (the ceiling).
args: <repo> [limit per split, 0 = all] [max epochs]"""
import os, sys, io, csv, json, gzip, time, zipfile, numpy as np
FAT, LEAN = (-190, -30), (-29, 150)
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def otsu(v):
    v = np.asarray(v, float); h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; tie = np.flatnonzero(s >= np.nanmax(s) * (1 - 1e-9)); return float(c[tie].mean())

# ------------------------------------------------------------------ data
_Z = None
def _init(zp):
    global _Z; _Z = zipfile.ZipFile(zp)
def crop_case(args):
    """one person's pelvis crop -> npy files (HU int16 clipped to +-1000, body mask). Returns case, shape."""
    import nibabel as nib
    from scipy.ndimage import label as cc, binary_fill_holes
    pre, case, out = args
    try:
        rd = lambda n: nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(pre + n)))
        CT = np.asanyarray(rd('ct.nii.gz').dataobj).astype(np.float32); HIP = np.zeros(CT.shape, bool)
        for b in ('hip_left', 'hip_right', 'sacrum'): HIP |= np.asanyarray(rd('segmentations/%s.nii.gz' % b).dataobj) > 0
        bt0 = otsu(np.clip(CT[:, :, CT.shape[2] // 2], -1024, 200)); body = CT > bt0; M = np.zeros(CT.shape, bool)
        for z in np.flatnonzero(HIP.any((0, 1))):
            lab, _ = cc(body[:, :, z]); ids = np.unique(lab[HIP[:, :, z]]); ids = ids[ids > 0]
            if len(ids): M[:, :, z] = binary_fill_holes(np.isin(lab, ids))
        ii = np.argwhere(M); lo = ii.min(0); hi = ii.max(0) + 1; sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        T = np.clip(CT[sl], -1000, 1000); TH = M[sl]; T[~TH] = -1000
        np.save(os.path.join(out, case + '_ct.npy'), T.astype(np.int16)); np.save(os.path.join(out, case + '_m.npy'), TH)
        return case, T.shape
    except Exception as e: return case, 'failed: %s %s' % (type(e).__name__, str(e)[:120])

def to_grid(A, G, fill):
    """centre array A in a box of shape G (crop or pad)."""
    out = np.full(G, fill, A.dtype); src = []; dst = []
    for n, g in zip(A.shape, G):
        if n >= g: o = (n - g) // 2; src.append(slice(o, o + g)); dst.append(slice(0, g))
        else: o = (g - n) // 2; src.append(slice(0, n)); dst.append(slice(o, o + n))
    out[tuple(dst)] = A[tuple(src)]; return out

def centre_on_mask(T, M):
    """shift so the body mask's centroid sits at the array centre in x and y (surface-only information)."""
    c = np.argwhere(M).mean(0); s = [int(round(T.shape[a] / 2 - c[a])) for a in (0, 1)]
    def sh(A, fill):
        B = np.full(A.shape, fill, A.dtype)
        sx = (slice(max(s[0], 0), A.shape[0] + min(s[0], 0)), slice(max(s[1], 0), A.shape[1] + min(s[1], 0)))
        tx = (slice(max(-s[0], 0), A.shape[0] + min(-s[0], 0)), slice(max(-s[1], 0), A.shape[1] + min(-s[1], 0)))
        B[sx] = A[tx]; return B
    return sh(T, -1000), sh(M, False)

def condition(M, sp=1.5, f=4):
    """surface condition on the code grid: signed distance (cm, positive inside), inside share, coordinates."""
    from scipy.ndimage import distance_transform_edt
    d = (distance_transform_edt(M, sampling=sp) - distance_transform_edt(~M, sampling=sp)) / 10.0
    pool = lambda A: A.reshape(A.shape[0] // f, f, A.shape[1] // f, f, A.shape[2] // f, f).mean((1, 3, 5))
    g = [np.linspace(-1, 1, n // f) for n in M.shape]; X, Y, Zz = np.meshgrid(*g, indexing='ij')
    return np.stack([pool(d), pool(M.astype(np.float32)), X, Y, Zz]).astype(np.float32)

# ------------------------------------------------------------------ scores
def masks(V, M):
    from scipy.ndimage import median_filter
    S = median_filter(V, 3); return S, (S > BONE_T) & M, (S >= FAT[0]) & (S <= FAT[1]) & M, (S >= LEAN[0]) & M
BONE_T = 300.0
def bulk(L):
    """the main body of lean tissue: opened by one voxel, largest connected piece (specks and septa in the fat dropped)."""
    from scipy.ndimage import binary_opening, label
    L = binary_opening(L); lab, n = label(L)
    return lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n else L
def score_file(a):
    path, bone_t = a; global BONE_T; BONE_T = bone_t; d = np.load(path); M = d['M']
    return {n: score(d[n].astype(np.float32), d['T'].astype(np.float32), M) for n in ('model', 'mean_ct', 'nearest_body', 'autoencoder_ceiling')}
def score(P, T, M, sp=1.5):
    """P predicted HU, T true HU, M body mask (the given surface)."""
    from scipy.ndimage import distance_transform_edt, binary_erosion
    Ps, pb, pf, pl = masks(P, M); Ts, tb, tf, tl = masks(T, M)
    skin = M & ~binary_erosion(M); deep = binary_erosion(M, iterations=2)
    r = dict(hu_mae=float(np.abs(P - T)[M].mean()), fat_share_err=float(abs(pf.sum() - tf.sum()) / M.sum()))
    s = pb.sum() + tb.sum(); r['bone_dice'] = float(2 * (pb & tb).sum() / s) if s else None
    if pb.any() and tb.any():
        dp = distance_transform_edt(~pb, sampling=sp); dt = distance_transform_edt(~tb, sampling=sp)
        ep = pb & ~binary_erosion(pb); et = tb & ~binary_erosion(tb)
        r['bone_surface_mm'] = float((dt[ep].mean() + dp[et].mean()) / 2); r['skin_to_bone_err_mm'] = float(np.abs(dp[skin] - dt[skin]).mean())
    else: r['bone_surface_mm'] = None; r['skin_to_bone_err_mm'] = None
    pl, tl = bulk(pl & deep), bulk(tl & deep)
    if pl.any() and tl.any():
        r['fat_thickness_err_mm'] = float(np.abs(distance_transform_edt(~pl, sampling=sp)[skin] - distance_transform_edt(~tl, sampling=sp)[skin]).mean())
    else: r['fat_thickness_err_mm'] = None
    return r
KEYS = ('hu_mae', 'bone_dice', 'bone_surface_mm', 'skin_to_bone_err_mm', 'fat_thickness_err_mm', 'fat_share_err')
def agg(rows):
    out = {}
    for k in KEYS:
        v = [r[k] for r in rows if r.get(k) is not None]; out[k] = dict(median=round(float(np.median(v)), 3), mean=round(float(np.mean(v)), 3), n=len(v)) if v else None
    return out

# ------------------------------------------------------------------ network
def make_net(cin=5, cout=4, w=48):
    import torch.nn as nn
    def blk(a, b): return nn.Sequential(nn.Conv3d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU(), nn.Conv3d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())
    class U(nn.Module):
        def __init__(s):
            super().__init__(); s.e1 = blk(cin, w); s.e2 = blk(w, 2 * w); s.e3 = blk(2 * w, 4 * w); s.m = blk(4 * w, 4 * w)
            s.d3 = blk(8 * w, 2 * w); s.d2 = blk(4 * w, w); s.d1 = blk(2 * w, w); s.o = nn.Conv3d(w, cout, 1); s.p = nn.AvgPool3d(2); s.u = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        def forward(s, x):
            import torch
            a = s.e1(x); b = s.e2(s.p(a)); c = s.e3(s.p(b)); m = s.m(s.p(c))
            y = s.d3(torch.cat([s.u(m), c], 1)); y = s.d2(torch.cat([s.u(y), b], 1)); y = s.d1(torch.cat([s.u(y), a], 1)); return s.o(y)
    return U()

def train(Xtr, Ytr, Xva, Yva, dev, max_epochs, log=print):
    """X (n,5,..) conditions, Y (n,4,..) standardised codes. Stops when the validation loss has not improved for a
    fifth of the epochs run so far (at least 20); returns the best validation state."""
    import torch, copy
    net = make_net().to(dev); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4); best = (np.inf, None, 0); hist = []
    Xtr_t, Ytr_t = torch.tensor(Xtr), torch.tensor(Ytr); Xva_t, Yva_t = torch.tensor(Xva).to(dev), torch.tensor(Yva).to(dev); bs = 4
    for ep in range(max_epochs):
        net.train(); perm = torch.randperm(len(Xtr_t)); tl = []
        for i in range(0, len(perm), bs):
            j = perm[i:i + bs]; x, y = Xtr_t[j].to(dev), Ytr_t[j].to(dev)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'): loss = (net(x).float() - y).abs().mean()
            opt.zero_grad(); loss.backward(); opt.step(); tl.append(loss.item())
        net.eval()
        with torch.no_grad(): vl = float(np.mean([(net(Xva_t[i:i + bs]).float() - Yva_t[i:i + bs]).abs().mean().item() for i in range(0, len(Xva_t), bs)]))
        hist.append((float(np.mean(tl)), vl))
        if vl < best[0]: best = (vl, copy.deepcopy(net.state_dict()), ep)
        if ep % 10 == 0 or ep == max_epochs - 1: log('epoch %d  train %.4f  val %.4f  (best %.4f at %d)' % (ep, hist[-1][0], vl, best[0], best[2]))
        if ep - best[2] > max(20, ep // 5): break
    net.load_state_dict(best[1]); return net, hist, best[2]

# ------------------------------------------------------------------ pipeline (autoencoder passed in, so it can be stubbed)
def run(cases, work, enc, dec, dev, max_epochs, log=print, figdir=None):
    """cases: {split: [case,...]} with crops already in `work`. enc(HU volume)->code (4,..); dec(code)->HU volume."""
    global BONE_T
    shapes = np.array([np.load(os.path.join(work, c + '_m.npy'), mmap_mode='r').shape for c in cases['train']])
    G = tuple(int(np.ceil(np.percentile(shapes[:, a], 95) / 32) * 32) for a in range(3)); log('grid', G, 'from', len(shapes), 'training crops')
    def load(c):
        T = np.load(os.path.join(work, c + '_ct.npy')).astype(np.float32); M = np.load(os.path.join(work, c + '_m.npy')); T, M = centre_on_mask(T, M)
        return to_grid(T, G, -1000.0), to_grid(M, G, False)
    X, Y, MSK = {}, {}, {}; mean_ct = np.zeros(G, np.float64); bone_vals = []
    tr_ct = np.lib.format.open_memmap(os.path.join(work, 'train_ct.npy'), mode='w+', dtype=np.int16, shape=(len(cases['train']),) + G)
    for sp_ in ('train', 'val', 'test'):
        X[sp_], Y[sp_], MSK[sp_] = [], [], []
        for k, c in enumerate(cases[sp_]):
            T, M = load(c); X[sp_].append(condition(M)); Y[sp_].append(enc(T)); MSK[sp_].append(np.packbits(M))
            if sp_ == 'train':
                mean_ct += T; tr_ct[k] = T.astype(np.int16)
                if k < 40: bone_vals.append(T[M & (T > LEAN[1])][::50])
        X[sp_] = np.stack(X[sp_]); Y[sp_] = np.stack(Y[sp_]).astype(np.float32); log('encoded', sp_, len(cases[sp_]))
    mean_ct /= len(cases['train']); BONE_T = otsu(np.concatenate(bone_vals)); log('bone threshold', BONE_T)
    unpack = lambda b: np.unpackbits(b)[:int(np.prod(G))].reshape(G).astype(bool)
    mu = Y['train'].mean((0, 2, 3, 4), keepdims=True); sd = Y['train'].std((0, 2, 3, 4), keepdims=True)
    net, hist, best_ep = train(X['train'], (Y['train'] - mu) / sd, X['val'], (Y['val'] - mu) / sd, dev, max_epochs, log)
    import torch
    trM = np.stack([unpack(m)[::4, ::4, ::4] for m in MSK['train']])                      # coarse masks for the nearest-body search
    res = dict(grid=list(G), people={k: len(v) for k, v in cases.items()}, best_epoch=best_ep, epochs_run=len(hist), train_loss=hist[best_ep][0], val_loss=hist[best_ep][1],
               curve=[[round(a, 4), round(b, 4)] for a, b in hist[::max(1, len(hist) // 60)]], bone_threshold=BONE_T)
    for sp_ in ('val', 'test'):
        rows = dict(model=[], mean_ct=[], nearest_body=[], autoencoder_ceiling=[])
        for k, c in enumerate(cases[sp_]):
            T, M = load(c)
            with torch.no_grad(): pc = net(torch.tensor(X[sp_][k:k + 1]).to(dev)).float().cpu().numpy()[0] * sd[0] + mu[0]
            P = dec(pc); P[~M] = -1000; C = dec(Y[sp_][k]); C[~M] = -1000
            A = mean_ct.astype(np.float32).copy(); A[~M] = -1000
            m4 = M[::4, ::4, ::4]; iou = (trM & m4).sum((1, 2, 3)) / np.maximum((trM | m4).sum((1, 2, 3)), 1); nn_ = int(iou.argmax())
            Nn = tr_ct[nn_].astype(np.float32); Nn[~M] = -1000
            i16 = lambda V: np.clip(V, -1000, 1000).astype(np.int16)
            np.savez(os.path.join(work, 'pred_%s.npz' % c), T=i16(T), M=M, model=i16(P), mean_ct=i16(A), nearest_body=i16(Nn), autoencoder_ceiling=i16(C))
            if figdir and sp_ == 'test' and k < 4: draw(T, P, Nn, A, M, os.path.join(figdir, 'test_%s.png' % c), c)
        from multiprocessing import Pool                                                   # scoring is CPU work: all people at once
        jobs = [(os.path.join(work, 'pred_%s.npz' % c), BONE_T) for c in cases[sp_]]
        if os.name == 'nt': sc = [score_file(j) for j in jobs]                             # local self-test on Windows: one at a time
        else:
            with Pool(min(12, os.cpu_count() or 4)) as pool: sc = pool.map(score_file, jobs)
        for c, r in zip(cases[sp_], sc):
            for name in rows: rows[name].append(dict(case=c, **r[name]))
        res[sp_] = {n: agg(v) for n, v in rows.items()}
        m, a, n_ = (np.array([r[kk] if r[kk] is not None else np.nan for r in rows[x]]) for x in ('model', 'mean_ct', 'nearest_body') for kk in ('skin_to_bone_err_mm',))
        res[sp_]['model_beats_mean_ct_people'] = int(np.nansum(m < a)); res[sp_]['model_beats_nearest_body_people'] = int(np.nansum(m < n_)); res[sp_]['per_person'] = rows
        log(sp_, json.dumps({n: {kk: (vv['median'] if vv else None) for kk, vv in res[sp_][n].items()} for n in rows}))
    return res

def draw(T, P, Nn, A, M, path, title):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    zc = T.shape[2] // 2; yc = int(np.argwhere(M).mean(0)[1]); W = dict(cmap='gray', vmin=-250, vmax=1000, interpolation='nearest')
    f, a = plt.subplots(2, 4, figsize=(18, 8))
    for j, (V, t) in enumerate(((T, 'CT (truth)'), (P, 'model, from the surface only'), (Nn, 'nearest training body'), (A, 'mean of training people'))):
        a[0, j].imshow(V[:, :, zc].T, **W); a[0, j].set_title(t); a[1, j].imshow(V[:, yc, :].T[::-1], **W)
        for r in (0, 1): a[r, j].set_xticks([]); a[r, j].set_yticks([])
    f.suptitle('%s: axial (top) and coronal (bottom)' % title); f.savefig(path, dpi=110, bbox_inches='tight'); plt.close(f)

def maisi(dev):
    import torch
    from huggingface_hub import hf_hub_download
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
    net = AutoencoderKlMaisi(spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4, num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2], norm_num_groups=32, norm_eps=1e-6,
                             attention_levels=[False, False, False], with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False, use_checkpointing=False, use_convtranspose=False,
                             norm_float16=True, num_splits=4, dim_split=1)
    net.load_state_dict(torch.load(hf_hub_download('nvidia/NV-Generate-CT', 'models/autoencoder_v1.pt'), map_location='cpu', weights_only=False), strict=True); net = net.to(dev).eval()
    def enc(T):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            return net.encode(torch.tensor((T + 1000) / 2000.0, dtype=torch.float32, device=dev)[None, None])[0].float()[0].cpu().numpy()
    def dec(c):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            return net.decode(torch.tensor(c, dtype=torch.float32, device=dev)[None]).float()[0, 0].cpu().numpy() * 2000.0 - 1000.0
    return enc, dec

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0; max_epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 400
    ext = list(csv.DictReader(open(hf_hub_download(repo, 'census/label_extents.csv', repo_type='dataset'))))
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    by = {}
    for e in ext: by.setdefault(e['case'], {})[e['label']] = e
    whole = lambda e: bool(e) and e['present'] == '1' and not any(e[k] == '1' for k in ('cut_inf', 'cut_sup', 'cut_side', 'cut_ap'))
    sel = sorted(c for c, d in by.items() if whole(d.get('hip_left')) and whole(d.get('hip_right')) and whole(d.get('sacrum')) and not (meta.get(c, {}).get('age') and float(meta[c]['age']) < 18))
    cases = {s: [c for c in sel if meta[c]['split'] == s] for s in ('train', 'val', 'test')}
    if limit: cases = {s: v[:limit if s == 'train' else max(2, limit // 4)] for s, v in cases.items()}
    log('people', {s: len(v) for s, v in cases.items()})
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); names = [i.filename for i in zipfile.ZipFile(zp).infolist() if i.filename.endswith('/ct.nii.gz')]
    pre = {c: [n for n in names if n.endswith('%s/ct.nii.gz' % c)][0][:-len('ct.nii.gz')] for v in cases.values() for c in v}
    os.makedirs('work', exist_ok=True); os.makedirs('figsA', exist_ok=True)
    with Pool(min(8, os.cpu_count() or 4), _init, (zp,)) as p: done = dict(p.imap_unordered(crop_case, [(pre[c], c, 'work') for v in cases.values() for c in v]))
    bad = {c: s for c, s in done.items() if isinstance(s, str)}; log('cropped', len(done) - len(bad), 'failed', bad)
    cases = {s: [c for c in v if c not in bad] for s, v in cases.items()}
    import torch
    enc, dec = maisi('cuda'); res = run(cases, 'work', enc, dec, 'cuda', max_epochs, log, 'figsA'); res['gpu'] = torch.cuda.get_device_name(0); res['limit'] = limit
    tag = 'smoke' if limit else 'full'
    json.dump(res, open('figsA/stageA_%s.json' % tag, 'w'), indent=1)
    api.upload_folder(folder_path='figsA', path_in_repo='stageA/%s' % tag, repo_id=repo, repo_type='dataset', commit_message='stage A %s' % tag)
    res.pop('curve'); [res[s].pop('per_person') for s in ('val', 'test')]; log('SUMMARY ' + json.dumps(res))
