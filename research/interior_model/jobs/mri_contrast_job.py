# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK", "torch", "monai==1.5.1", "pytorch-metric-learning", "matplotlib"]
# ///
"""Contrastive MRI-to-compartment labeller (user 2026-11-27, after N4 + histogram standardisation did not fix IXI).
Same task, data and network as mri_game_job.py (HaN-Seg T1 -> 14 game compartments + ear canals, SegResNet), trained so that
its features do not depend on the scanner's contrast:
- two contrast copies of every crop (MONAI RandBiasField, RandAdjustContrast, RandHistogramShift, RandGaussianSmooth,
  RandGaussianNoise, library defaults); geometry identical, so voxel i of one copy is voxel i of the other;
- CT anchor: a CT network on the same person's registered CT (same grid); the same voxel in the CT is a positive of the MRI
  voxel (CT numbers are physically calibrated, the same on every scanner);
- labels: voxels of the same compartment are positives too (supervised contrastive, Khosla et al. 2020);
- unlabelled IXI T1 crops (ixi_prep_v1) in every batch: two contrast copies, same-voxel positives, and the two copies must
  give the same prediction (consistency). This is where the network meets IXI's own scanners.
Losses from pytorch-metric-learning (NTXentLoss for same-voxel, SupConLoss for same-compartment; library temperatures);
the weights between the losses are learned (uncertainty weighting, Kendall et al. 2018), not set by hand.
Only the MRI network is kept for use; the CT network and projection heads are training aids.
Gate, against the step-1 models (mri_game/segresnet_fold*.pt, _all.pt) on the same folds:
 (a) Dice per compartment on held-out HaN-Seg heads, plain input;
 (b) the same under a contrast stress (one fixed random bias field + gamma + histogram shift per head);
 (c) IXI, no labels: agreement (Dice) between the predictions on two stressed copies of the same head;
 (d) PRIVATE pictures of the same 8 IXI heads as step 1: MRI | step-1 model | contrastive model (largest head piece kept).
env: STEPS (default 4000), FOLDS (default "0,1"), LIMIT (smoke: first N people), N_IXI (IXI heads loaded, default all),
     V (voxels per crop in the contrastive losses, default 256), COMMIT, OUT_TAG (default _contrast). args: <hanseg repo>"""
import os, sys, json, time, zipfile, urllib.request, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
IGN = 255; S = 96

def make_aug(prob=None):
    from monai.transforms import Compose, RandBiasField, RandAdjustContrast, RandHistogramShift, RandGaussianSmooth, RandGaussianNoise
    if prob is None:                                    # training copies
        return Compose([RandBiasField(prob=0.5), RandAdjustContrast(prob=0.5), RandHistogramShift(prob=0.5), RandGaussianSmooth(prob=0.3), RandGaussianNoise(prob=0.3)])
    return Compose([RandBiasField(prob=1.0), RandAdjustContrast(prob=1.0), RandHistogramShift(prob=1.0)])     # the stress test

def feats(net, x):
    """SegResNet forward split at the last decoder features (before conv_final)"""
    h, down = net.encode(x); down.reverse()
    for i, (up, upl) in enumerate(zip(net.up_samples, net.up_layers)): h = up(h) + down[i + 1]; h = upl(h)
    return h, net.conv_final(h)

def segloss(lg, y, ce, NCL):
    import torch
    valid = (y != IGN); p = torch.softmax(lg.float(), 1); oh = torch.nn.functional.one_hot(y.clamp(max=NCL - 1), NCL).permute(0, 4, 1, 2, 3).float() * valid[:, None]
    inter = (p * oh).sum((0, 2, 3, 4)); den = (p * valid[:, None]).sum((0, 2, 3, 4)) + oh.sum((0, 2, 3, 4))
    return ce(lg.float(), y) + (1 - (2 * inter + 1) / (den + 1)).mean()

def train_contrast(cases, W, steps, NCL, ixi, V=256, dev='cuda'):
    import torch, torch.nn as nn
    from monai.networks.nets import SegResNet
    from pytorch_metric_learning.losses import NTXentLoss, SupConLoss
    rng = np.random.default_rng(0); aug = make_aug(); aug.set_random_state(seed=1)
    D = {c: (lambda v: (v['M'], v['T'], v['C']))(np.load('%s/%s.npz' % (W, c))) for c in cases}
    cand = {c: np.argwhere(D[c][1][::6, ::6, ::6] < IGN) * 6 for c in cases}
    share = np.bincount(np.concatenate([D[c][1][::4, ::4, ::4].ravel() for c in cases]), minlength=256)[:NCL].astype(float); share /= share.sum()
    rare = {c: np.argwhere((D[c][1][::2, ::2, ::2] >= 3) & (D[c][1][::2, ::2, ::2] < IGN) & np.isin(D[c][1][::2, ::2, ::2], np.flatnonzero(share < np.median(share[share > 0])))) * 2 for c in cases}
    wts = torch.tensor(np.where(share > 0, share, share[share > 0].min()) ** -0.5, dtype=torch.float32, device=dev); wts = wts / wts.mean()
    icand = {s: np.argwhere(b[::6, ::6, ::6]) * 6 for s, (M, b) in ixi.items()}; isubs = sorted(ixi)
    net = SegResNet(spatial_dims=3, in_channels=1, out_channels=NCL, init_filters=16).to(dev)
    ctn = SegResNet(spatial_dims=3, in_channels=1, out_channels=NCL, init_filters=16).to(dev)
    head = lambda: nn.Sequential(nn.Conv3d(16, 64, 1), nn.ReLU(), nn.Conv3d(64, 32, 1)).to(dev)
    pm, pc = head(), head(); logs = nn.Parameter(torch.zeros(6, device=dev))         # learned loss weights: seg MRI, seg CT, voxel MRI-MRI-CT, compartment, IXI voxel, IXI consistency
    params = list(net.parameters()) + list(ctn.parameters()) + list(pm.parameters()) + list(pc.parameters())
    opt = torch.optim.AdamW([dict(params=params, weight_decay=1e-5), dict(params=[logs], weight_decay=0)], 2e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); scaler = torch.amp.GradScaler(); ce = nn.CrossEntropyLoss(weight=wts, ignore_index=IGN)
    ntx, sup = NTXentLoss(), SupConLoss()
    def crop(A, o, fill): out = np.full((S, S, S), fill, A.dtype); a = A[o[0]:o[0] + S, o[1]:o[1] + S, o[2]:o[2] + S]; out[:a.shape[0], :a.shape[1], :a.shape[2]] = a; return out
    def org(q, shape): return [int(np.clip(a - S // 2, 0, max(0, d - S))) for a, d in zip(q, shape)]
    def two(m): return [np.asarray(aug(m[None].astype(np.float32)))[0] for _ in range(2)]
    def batch(n=2):
        Xa, Xb, Xc, Y = [], [], [], []
        for _ in range(n):
            c = cases[rng.integers(len(cases))]; M, T, C = D[c]; q = rare[c][rng.integers(len(rare[c]))] if (len(rare[c]) and rng.random() < 0.3) else cand[c][rng.integers(len(cand[c]))]
            o = org(q, T.shape); m = crop(M, o, 0).astype(np.float32); a, b = two(m); Xa.append(a); Xb.append(b); Xc.append(crop(C, o, 0).astype(np.float32)); Y.append(crop(T, o, IGN))
        s = isubs[rng.integers(len(isubs))]; Mi, Bi = ixi[s]; o = org(icand[s][rng.integers(len(icand[s]))], Mi.shape); ia, ib = two(crop(Mi, o, 0).astype(np.float32))
        t = lambda L: torch.tensor(np.stack(L).astype(np.float32)[:, None], device=dev)
        return t(Xa), t(Xb), t(Xc), torch.tensor(np.stack(Y).astype(np.int64), device=dev), t([ia]), t([ib]), torch.tensor(crop(Bi, o, False), device=dev)
    def pick(y):
        """V voxels per crop, balanced over the compartments present"""
        out = []
        for i in range(y.shape[0]):
            yi = y[i].reshape(-1); ks = [k for k in torch.unique(yi).tolist() if k != IGN]; per = max(1, V // max(1, len(ks)))
            for k in ks:
                w = torch.nonzero(yi == k)[:, 0]; out.append(torch.stack([torch.full_like(w[:per], i), w[torch.randperm(len(w), device=dev)[:per]]], 1))
        return torch.cat(out)
    def emb(proj, f, ix): z = proj(f.float()); z = z.reshape(z.shape[0], z.shape[1], -1); return nn.functional.normalize(z[ix[:, 0], :, ix[:, 1]], dim=1)
    for it in range(steps):
        net.train(); ctn.train(); xa, xb, xc, y, ua, ub, ubody = batch()
        with torch.autocast('cuda', dtype=torch.float16):
            fa, la = feats(net, xa); fb, lb = feats(net, xb); fc, lc = feats(ctn, xc); ga, qa = feats(net, ua); gb, qb = feats(net, ub)
        L = [None] * 6
        L[0] = 0.5 * (segloss(la, y, ce, NCL) + segloss(lb, y, ce, NCL)); L[1] = segloss(lc, y, ce, NCL)
        ix = pick(y); za, zb, zc = emb(pm, fa, ix), emb(pm, fb, ix), emb(pc, fc, ix); vid = torch.arange(len(ix), device=dev); cls = y.reshape(y.shape[0], -1)[ix[:, 0], ix[:, 1]]
        L[2] = ntx(torch.cat([za, zb, zc]), torch.cat([vid, vid, vid])); L[3] = sup(torch.cat([za, zb, zc]), torch.cat([cls, cls, cls]))
        iv = torch.nonzero(ubody.reshape(-1))[:, 0]; iv = iv[torch.randperm(len(iv), device=dev)[:V]]; ii = torch.stack([torch.zeros_like(iv), iv], 1)
        L[4] = ntx(torch.cat([emb(pm, ga, ii), emb(pm, gb, ii)]), torch.cat([torch.arange(len(ii), device=dev)] * 2))
        L[5] = ((torch.softmax(qa.float(), 1) - torch.softmax(qb.float(), 1)) ** 2).sum(1).mean() + ((torch.softmax(la.float(), 1) - torch.softmax(lb.float(), 1)) ** 2).sum(1).mean()
        loss = sum(torch.exp(-logs[k]) * L[k] + logs[k] for k in range(6))
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % max(1, steps // 8) == 0 or it == steps - 1:
            log('step', it, 'losses', ' '.join('%.3f' % l.item() for l in L), 'weights', ' '.join('%.2f' % w for w in torch.exp(-logs).tolist()))
    return net

def largest_head(P):
    from scipy.ndimage import label
    lab, n = label(P > 0)
    if n: P = np.where(lab == np.bincount(lab.ravel())[1:].argmax() + 1, P, 0).astype(np.uint8)
    return P

if __name__ == '__main__':
    try: import monai, SimpleITK, pytorch_metric_learning
    except ImportError: import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'monai==1.5.1', 'SimpleITK', 'pytorch-metric-learning'], check=True)
    import torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from multiprocessing import Pool
    from monai.networks.nets import SegResNet
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; user = api.whoami()['name']; C_ = os.environ.get('COMMIT', 'main')
    STEPS = int(os.environ.get('STEPS', '4000')); FOLDS = [int(f) for f in os.environ.get('FOLDS', '0,1').split(',') if f != '']
    LIMIT = int(os.environ.get('LIMIT', '0')); V = int(os.environ.get('V', '256'))
    for f in ('jobs/mri_game_job.py', 'game.py'):
        urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C_, f), os.path.basename(f))
    sys.path.insert(0, '.'); import game, mri_game_job as mg
    NAMES = game.GNAMES + ['canal_left', 'canal_right']; NCL = len(NAMES); W = mg.W; os.makedirs(W, exist_ok=True)
    log('gpu', torch.cuda.get_device_name(0))
    # HaN-Seg: registration and targets exactly as step 1 (now also the CT on the same grid)
    zp = hf_hub_download(repo, 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.')
    snapshot_download(repo, repo_type='dataset', allow_patterns=['face_v1/*'], local_dir=W)
    ids = json.load(open('%s/face_v1/face_ids.json' % W))['ids']; canal = (ids['auditory_canal_left'], ids['auditory_canal_right'])
    cases = sorted({n.split('/')[-2] for n in zipfile.ZipFile(zp).namelist() if n.endswith('_IMG_MR_T1.nrrd')}); cases = [c for c in cases if os.path.exists('%s/face_v1/%s.npz' % (W, c))]
    if LIMIT: cases = cases[:LIMIT]
    import SimpleITK as sitk; sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    with Pool(max(1, min(os.cpu_count() or 2, 4)), mg._init, (zp, game.LUT, canal, len(game.GNAMES)), maxtasksperchild=5) as p: s1 = dict(p.imap_unordered(mg.stage1, cases))
    good = [c for c in cases if not s1[c][0]]; log('registered', len(good), 'failed', {c: e[0] for c, e in s1.items() if e[0]})
    # IXI, unlabelled
    irepo = user + '/ixi-t1-t2'; files = api.list_repo_files(irepo, repo_type='dataset')
    isubs = sorted(f.split('/')[-1][:-4] for f in files if f.startswith('ixi_prep_v1/') and f.endswith('.npz'))
    if os.environ.get('N_IXI'): isubs = [isubs[i] for i in np.linspace(0, len(isubs) - 1, int(os.environ['N_IXI'])).astype(int)]
    pics = [isubs[i] for i in np.linspace(0, len(isubs) - 1, 8).astype(int)] if not os.environ.get('N_IXI') else isubs[:min(8, len(isubs))]
    d = snapshot_download(irepo, repo_type='dataset', allow_patterns=['ixi_prep_v1/%s.npz' % s for s in isubs], local_dir='ixi', max_workers=16); ixi = {}
    for s in isubs:
        z = np.load(os.path.join(d, 'ixi_prep_v1', s + '.npz')); sh = tuple(z['shape']); ixi[s] = (z['M'], np.unpackbits(z['body'])[:np.prod(sh)].reshape(sh).astype(bool))
    log(len(ixi), 'IXI heads loaded')
    out = 'mri_game' + os.environ.get('OUT_TAG', '_contrast') + ('_smoke' if LIMIT else ''); os.makedirs(out, exist_ok=True)
    def oldnet(name):
        n = SegResNet(spatial_dims=3, in_channels=1, out_channels=NCL, init_filters=16).cuda(); n.load_state_dict(torch.load(hf_hub_download(repo, 'mri_game/' + name, repo_type='dataset', local_dir='old'), map_location='cuda')); return n
    stress = make_aug(prob=1.0)
    def stressed(M, seed): stress.set_random_state(seed=seed); return np.asarray(stress(M[None].astype(np.float32)))[0]
    def upload(msg):
        for a in range(5):
            try: api.upload_folder(folder_path=out, path_in_repo=out, repo_id=repo, repo_type='dataset', commit_message=msg); return
            except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
    rows = []
    for f in FOLDS:
        test = [c for i, c in enumerate(good) if i % 5 == f]; tr = [c for c in good if c not in test]
        if not test or not tr: continue
        net = train_contrast(tr, W, STEPS, NCL, ixi, V); torch.save(net.state_dict(), '%s/segresnet_fold%d.pt' % (out, f)); old = oldnet('segresnet_fold%d.pt' % f)
        for c in test:
            M = np.load('%s/%s.npz' % (W, c))['M'].astype(np.float32); Ms = stressed(M, 1000 + good.index(c))
            r = dict(case=c, fold=f)
            for k, n_ in (('new', net), ('old', old)):
                r[k] = {kind: {nm: v['dice'] for nm, v in mg.score(c, mg.predict(n_, X), NAMES).items() if isinstance(v, dict)} for kind, X in (('plain', M), ('stress', Ms))}
            rows.append(r); log('fold', f, c, 'brain plain old/new', r['old']['plain'].get('brain'), r['new']['plain'].get('brain'), 'stress old/new', r['old']['stress'].get('brain'), r['new']['stress'].get('brain'))
        del net, old; torch.cuda.empty_cache(); upload('contrastive labeller fold %d' % f)
    summ = {}
    for k in ('old', 'new'):
        for kind in ('plain', 'stress'):
            summ['%s_%s' % (k, kind)] = {nm: round(float(np.nanmedian([r[k][kind][nm] for r in rows if nm in r[k][kind]])), 3) for nm in NAMES[1:] if any(nm in r[k][kind] for r in rows)}
    log('SUMMARY heldout dice ' + json.dumps(summ))
    # all heads model, then IXI checks
    net = train_contrast(good, W, STEPS, NCL, ixi, V); torch.save(net.state_dict(), '%s/segresnet_all.pt' % out); old = oldnet('segresnet_all.pt')
    agree = {'old': [], 'new': []}; chk = [isubs[i] for i in np.linspace(0, len(isubs) - 1, min(20, len(isubs))).astype(int)]
    for s in chk:
        M = ixi[s][0].astype(np.float32)
        for k, n_ in (('old', old), ('new', net)):
            A, B = mg.predict(n_, stressed(M, 1)), mg.predict(n_, stressed(M, 2))
            agree[k].append({nm: round(float(2 * ((A == i) & (B == i)).sum() / max(1, (A == i).sum() + (B == i).sum())), 3) for i, nm in enumerate(NAMES) if nm in ('brain', 'skull', 'eyes', 'other_soft', 'mandible', 'chew_left', 'airway')})
    summ['ixi_two_copy_agreement'] = {k: {nm: round(float(np.median([a[nm] for a in v])), 3) for nm in v[0]} for k, v in agree.items()}
    log('SUMMARY IXI agreement ' + json.dumps(summ['ixi_two_copy_agreement']))
    COL = {'outside': (0, 0, 0), 'fat_layer': (0.98, 0.84, 0.45), 'skull': (0.93, 0.89, 0.78), 'mandible': (0.85, 0.75, 0.55), 'teeth': (1.0, 1.0, 0.92), 'eyes': (0.6, 0.9, 1.0),
           'brain': (0.92, 0.62, 0.70), 'chew_left': (0.80, 0.22, 0.20), 'chew_right': (0.62, 0.15, 0.30), 'tongue': (0.95, 0.40, 0.45), 'airway': (0.30, 0.60, 0.90),
           'spine': (0.70, 0.70, 0.62), 'other_bone': (0.60, 0.55, 0.45), 'other_soft': (0.72, 0.45, 0.40), 'canal_left': (0, 1, 0), 'canal_right': (0, 1, 0)}
    pal = np.array([COL[n] for n in NAMES] + [(0.25, 0.25, 0.25)] * (256 - NCL)); os.makedirs('pics', exist_ok=True)
    for s in pics:
        M = ixi[s][0].astype(np.float32); Po, Pn = largest_head(mg.predict(old, M)), largest_head(mg.predict(net, M))
        e = Pn == NAMES.index('eyes'); ce = np.argwhere(e).mean(0).astype(int) if e.any() else np.array(M.shape) // 2
        fig, axs = plt.subplots(3, 3, figsize=(12.6, 12))
        for r, (sl, nm) in enumerate([((ce[0], slice(None), slice(None)), 'axial'), ((slice(None), ce[1], slice(None)), 'coronal'), ((slice(None), slice(None), ce[2]), 'sagittal')]):
            for c, (A, kind, t) in enumerate([(M, 'mri', 'IXI MRI'), (Po, 'lab', 'step-1 labeller'), (Pn, 'lab', 'contrastive labeller')]):
                ax = axs[r, c]; a = A[sl]
                ax.imshow(a, cmap='gray', vmin=0, vmax=1, origin='lower' if r else 'upper') if kind == 'mri' else ax.imshow(pal[a], origin='lower' if r else 'upper')
                ax.set_title('%s, %s' % (t, nm), fontsize=9); ax.axis('off')
        fig.suptitle(s, fontsize=11); plt.tight_layout(); plt.savefig('pics/%s.png' % s, dpi=60); plt.close(fig)
    api.upload_folder(folder_path='pics', path_in_repo='ixi_contrast_check/pics', repo_id=irepo, repo_type='dataset', commit_message='contrastive labeller IXI pictures (private)')
    json.dump(dict(summary=summ, per_person=rows, ixi_agreement=agree, folds=FOLDS, steps=STEPS, V=V, classes=NAMES, ixi_heads=len(ixi)), open('%s/mri_contrast.json' % out, 'w'), indent=1)
    upload('contrastive labeller (all heads) and gate'); log('SUMMARY done')
