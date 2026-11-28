# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK", "torch", "monai==1.5.1"]
# ///
"""Step 1 of the IXI plan (user agreed 2026-11-27): can MARROW's 14 game compartments be read from T1 MRI alone?
HaN-Seg, 42 adults with CT and T1 MRI of the same person (CC BY-ND 4.0, private use). Numbers and weights only, no images.

1. CPU, per person: the MRI rigidly registered to the CT (mutual information, two starting alignments, best kept, as
   mri_tissue_job.py), both on the CT's grid at 1.5 mm. Targets: the person's face_v1 labels (TotalSegmentator head tasks
   on the CT) mapped to the compartments (game.py), plus the two ear canals as extra classes (they place the head frame
   on MRI-only heads such as IXI). The face_v1 affine's world convention (RAS or LPS) is chosen per person by which one
   puts the labelled body on the CT's body (data, not an assumption). Voxels outside the MRI's view or unlabelled = unknown.
2. GPU: MONAI SegResNet from MRI to these classes, 5-fold cross-validation (fold = index mod 5), so every head is scored by
   a model that never saw it; then one model on all heads (for IXI).
3. Gate, per compartment: Dice and median surface distance on known voxels; ear-canal and eye centroid errors in mm
   (they decide the head frame). Compartments that MRI cannot read stay unknown on IXI heads.
env: STEPS (per fold, default 4000), LIMIT (smoke: first N people), OUT_TAG, COMMIT (for game.py), NOPOOL (smoke: no worker processes). args: <repo for outputs (the HaN-Seg repo)> [limit]"""
import os, sys, json, time, zipfile, urllib.request, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
W = 'work'; IGN = 255; SP = 1.5
_Z = None; LUT = None; CANAL = None; NG = None
def _init(zp, lut, canal, ng):
    global _Z, LUT, CANAL, NG; _Z = zipfile.ZipFile(zp); LUT = lut; CANAL = canal; NG = ng
def find(names, c, suf): return [n for n in names if n.endswith('%s/%s_%s' % (c, c, suf))][0]

def stage1(case):
    import SimpleITK as sitk
    from scipy.ndimage import median_filter, label, binary_fill_holes, map_coordinates
    try:
        names = _Z.namelist()
        def read(n, suffix):
            p = '%s/%s_%s.nrrd' % (W, case, suffix); open(p, 'wb').write(_Z.read(n)); im = sitk.ReadImage(p); os.remove(p); return im
        ct = sitk.Cast(read(find(names, case, 'IMG_CT.nrrd'), 'ct'), sitk.sitkFloat32); mr = sitk.Cast(read(find(names, case, 'IMG_MR_T1.nrrd'), 'mr'), sitk.sitkFloat32)
        size = [int(round(s * z / SP)) for s, z in zip(ct.GetSize(), ct.GetSpacing())]
        ref = sitk.Image(size, sitk.sitkFloat32); ref.SetSpacing([SP] * 3); ref.SetOrigin(ct.GetOrigin()); ref.SetDirection(ct.GetDirection())
        ctr = sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0); best = None
        for mode in (sitk.CenteredTransformInitializerFilter.GEOMETRY, sitk.CenteredTransformInitializerFilter.MOMENTS):
            init = sitk.CenteredTransformInitializer(ctr, mr, sitk.Euler3DTransform(), mode)
            R = sitk.ImageRegistrationMethod(); R.SetMetricAsMattesMutualInformation(50); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.1, 1)
            R.SetInterpolator(sitk.sitkLinear); R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 400); R.SetOptimizerScalesFromPhysicalShift()
            R.SetShrinkFactorsPerLevel([8, 4, 2, 1]); R.SetSmoothingSigmasPerLevel([3, 2, 1, 0]); R.SetInitialTransform(init, inPlace=False)
            t_ = R.Execute(ctr, mr); m_ = R.GetMetricValue()
            if best is None or m_ < best[0]: best = (m_, t_)
        M = sitk.GetArrayFromImage(sitk.Resample(mr, ref, best[1], sitk.sitkLinear, 0.0))          # (z, y, x)
        fov = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(mr * 0 + 1, sitk.sitkFloat32), ref, best[1], sitk.sitkNearestNeighbor, 0.0)) > 0.5
        C = median_filter(sitk.GetArrayFromImage(ctr), 3); lab, n = label(C > -500); body = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        for z in range(body.shape[0]): body[z] = binary_fill_holes(body[z])
        # physical (LPS) points of the reference grid, in array order (z, y, x)
        sz = ref.GetSize(); D = np.array(ref.GetDirection()).reshape(3, 3); O = np.array(ref.GetOrigin())
        kk, jj, ii = np.meshgrid(*[np.arange(n) for n in (sz[2], sz[1], sz[0])], indexing='ij')
        P = (np.stack([ii, jj, kk], -1).reshape(-1, 3) * SP) @ D.T + O
        fz = np.load('%s/face_v1/%s.npz' % (W, case)); L, aff = fz['labels'], fz['affine']; Ainv = np.linalg.inv(aff[:3, :3]); best_t = None
        for conv, flip in (('lps', np.array([1, 1, 1])), ('ras', np.array([-1, -1, 1]))):          # face_v1 world -> LPS
            idx = ((P * flip) - aff[:3, 3]) @ Ainv.T
            Lr = map_coordinates(L, idx.T, order=0, mode='constant', cval=IGN).reshape(body.shape).astype(np.uint8)
            lb = (Lr > 0) & (Lr < IGN); kn = Lr < IGN; ov = float((lb & body & kn).sum() / max(1, ((lb | body) & kn).sum()))
            if best_t is None or ov > best_t[0]: best_t = (ov, conv, Lr)
        ov, conv, Lr = best_t
        T = LUT[Lr].copy(); T[Lr == CANAL[0]] = NG; T[Lr == CANAL[1]] = NG + 1; T[~fov] = IGN
        lo, hi = np.percentile(M[fov & body], [0.5, 99.5]) if (fov & body).any() else (0, 1); Mn = np.clip((M - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float16); Mn[~fov] = 0
        cv = sitk.GetArrayFromImage(ctr); clo, chi = np.percentile(cv[body], [0.5, 99.5]); Cn = np.clip((cv - clo) / max(chi - clo, 1e-6), 0, 1).astype(np.float16)   # CT on the same grid (contrastive anchor)
        np.savez_compressed('%s/%s.npz' % (W, case), M=Mn, T=T, C=Cn, metric=best[0], body_overlap=ov, conv=conv)
        return case, (None, ov, conv)
    except Exception as e: return case, ('%s %s' % (type(e).__name__, str(e)[:200]), None, None)

def surface_mm(a, b, mm=SP):
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    sa = np.argwhere(a & ~binary_erosion(a)); sb = np.argwhere(b & ~binary_erosion(b))
    if len(sa) < 3 or len(sb) < 3: return float('nan')
    sa = sa[:: max(1, len(sa) // 30000)]; sb = sb[:: max(1, len(sb) // 30000)]
    return float(np.median(np.concatenate([cKDTree(sb).query(sa, workers=-1)[0], cKDTree(sa).query(sb, workers=-1)[0]])) * mm)

def train(cases, steps, NCL, dev='cuda'):
    import torch
    from monai.networks.nets import SegResNet
    rng = np.random.default_rng(0); V = {c: (lambda v: (v['M'], v['T']))(np.load('%s/%s.npz' % (W, c))) for c in cases}
    cand = {c: np.argwhere(V[c][1][::6, ::6, ::6] < IGN) * 6 for c in cases}; S = 96
    share = np.bincount(np.concatenate([V[c][1][::4, ::4, ::4].ravel() for c in cases]), minlength=256)[:NCL].astype(float); share /= share.sum()
    rare = {c: np.argwhere((V[c][1][::2, ::2, ::2] >= 3) & (V[c][1][::2, ::2, ::2] < IGN) & np.isin(V[c][1][::2, ::2, ::2], np.flatnonzero(share < np.median(share[share > 0])))) * 2 for c in cases}   # crops centred on the rarer classes
    wts = torch.tensor(np.where(share > 0, share, share[share > 0].min()) ** -0.5, dtype=torch.float32, device=dev); wts = wts / wts.mean()
    net = SegResNet(spatial_dims=3, in_channels=1, out_channels=NCL, init_filters=16).to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); scaler = torch.amp.GradScaler(); ce = torch.nn.CrossEntropyLoss(weight=wts, ignore_index=IGN)
    def batch(n=2):
        X = np.zeros((n, 1, S, S, S), np.float32); Y = np.full((n, S, S, S), IGN, np.int64)
        for i in range(n):
            c = cases[rng.integers(len(cases))]; M, T = V[c]; q = rare[c][rng.integers(len(rare[c]))] if (len(rare[c]) and rng.random() < 0.3) else cand[c][rng.integers(len(cand[c]))]
            o = [int(np.clip(a - S // 2, 0, max(0, d - S))) for a, d in zip(q, T.shape)]
            m = M[o[0]:o[0] + S, o[1]:o[1] + S, o[2]:o[2] + S].astype(np.float32); t = T[o[0]:o[0] + S, o[1]:o[1] + S, o[2]:o[2] + S]
            m = m * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05)                        # brightness and contrast vary between scanners (IXI)
            X[i, 0, :m.shape[0], :m.shape[1], :m.shape[2]] = m; Y[i, :t.shape[0], :t.shape[1], :t.shape[2]] = t
        return torch.tensor(X, device=dev), torch.tensor(Y, device=dev)
    for it in range(steps):
        net.train(); x, y = batch()
        with torch.autocast('cuda', dtype=torch.float16):
            lg = net(x); valid = (y != IGN); p = torch.softmax(lg.float(), 1); oh = torch.nn.functional.one_hot(y.clamp(max=NCL - 1), NCL).permute(0, 4, 1, 2, 3).float() * valid[:, None]
            inter = (p * oh * valid[:, None]).sum((0, 2, 3, 4)); den = (p * valid[:, None]).sum((0, 2, 3, 4)) + oh.sum((0, 2, 3, 4))
            loss = ce(lg.float(), y) + (1 - (2 * inter + 1) / (den + 1)).mean()
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
        if it % max(1, steps // 8) == 0: log('step', it, 'loss %.3f' % loss.item())
    return net

def predict(net, M, dev='cuda'):
    import torch
    from monai.inferers import sliding_window_inference
    net.eval()
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        lg = sliding_window_inference(torch.tensor(M.astype(np.float32)[None, None], device=dev), (96, 96, 96), 2, net, overlap=0.5)
    return lg.float().argmax(1)[0].cpu().numpy().astype(np.uint8)

def score(case, P, names):
    v = np.load('%s/%s.npz' % (W, case)); T = v['T']; K = T != IGN; r = dict(case=case)
    for k in range(1, len(names)):
        a, b = (P == k) & K, (T == k) & K
        if b.sum() < 30: continue
        r[names[k]] = dict(dice=round(float(2 * (a & b).sum() / (a.sum() + b.sum())), 3), surface_mm=round(surface_mm(a, b), 2), volume_ratio=round(float(a.sum() / b.sum()), 3))
        if names[k] in ('canal_left', 'canal_right', 'eyes'):
            r[names[k]]['centroid_mm'] = round(float(np.linalg.norm(np.argwhere(a).mean(0) - np.argwhere(b).mean(0)) * SP), 2) if a.any() else float('nan')
    return r

if __name__ == '__main__':
    try: import monai, SimpleITK                                                       # the Colab runner installs only the common packages
    except ImportError: import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'monai==1.5.1', 'SimpleITK'], check=True)
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    import torch
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else int(os.environ.get('LIMIT', '0')); os.makedirs(W, exist_ok=True); STEPS = int(os.environ.get('STEPS', '4000'))
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/game.py' % os.environ.get('COMMIT', 'main'), 'game.py')
    sys.path.insert(0, '.'); import game
    NAMES_G = game.GNAMES; NAMES = NAMES_G + ['canal_left', 'canal_right']; NCL = len(NAMES)
    zp = hf_hub_download('gardianofthedarkness/hanseg-ct-mr', 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.')
    snapshot_download('gardianofthedarkness/hanseg-ct-mr', repo_type='dataset', allow_patterns=['face_v1/*'], local_dir=W)
    ids = json.load(open('%s/face_v1/face_ids.json' % W))['ids']; canal = (ids['auditory_canal_left'], ids['auditory_canal_right'])
    cases = sorted({n.split('/')[-2] for n in zipfile.ZipFile(zp).namelist() if n.endswith('_IMG_MR_T1.nrrd')})
    cases = [c for c in cases if os.path.exists('%s/face_v1/%s.npz' % (W, c))]
    if limit: cases = cases[:limit]
    try: q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception:
        try: ram = int([l for l in open('/proc/meminfo') if l.startswith('MemAvailable')][0].split()[1]) / 1e6
        except Exception: ram = 16
    nw = max(1, min(int(cpus), int(ram // 6))); log(len(cases), 'people,', nw, 'workers')
    import SimpleITK as sitk; sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    if os.environ.get('NOPOOL'):                                                          # smoke on Windows: no worker processes
        _init(zp, game.LUT, canal, len(NAMES_G)); s1 = dict(stage1(c) for c in cases)
    else:
        with Pool(nw, _init, (zp, game.LUT, canal, len(NAMES_G)), maxtasksperchild=5) as p: s1 = dict(p.imap_unordered(stage1, cases))
    bad = {c: e[0] for c, e in s1.items() if e[0]}; good = [c for c in cases if c not in bad]
    log('registered', len(good), 'failed', bad); log('label-to-CT body overlap and convention', {c: (round(s1[c][1], 3), s1[c][2]) for c in good})
    out = 'mri_game' + os.environ.get('OUT_TAG', '') + ('_smoke' if limit else ''); os.makedirs(out, exist_ok=True); rows = []
    for f in range(5):
        test = [c for i, c in enumerate(good) if i % 5 == f]; tr = [c for c in good if c not in test]
        if not test or not tr: continue
        net = train(tr, STEPS, NCL); torch.save(net.state_dict(), '%s/segresnet_fold%d.pt' % (out, f))
        for c in test:
            r = score(c, predict(net, np.load('%s/%s.npz' % (W, c))['M']), NAMES); r['fold'] = f; rows.append(r)
            log('fold', f, c, {k: v['dice'] for k, v in r.items() if isinstance(v, dict)})
        del net; torch.cuda.empty_cache()
    summ = {}
    for k in NAMES[1:]:
        rr = [r[k] for r in rows if k in r]
        if rr: summ[k] = {m: round(float(np.nanmedian([x[m] for x in rr])), 3) for m in rr[0]} | dict(dice_p10=round(float(np.percentile([x['dice'] for x in rr], 10)), 3), people=len(rr))
    json.dump(dict(summary=summ, per_person=rows, failed=bad, steps_per_fold=STEPS, classes=NAMES), open('%s/mri_game.json' % out, 'w'), indent=1)
    log('SUMMARY ' + json.dumps(summ))
    if not limit:                                                                            # one model on all heads, for IXI
        net = train(good, STEPS, NCL); torch.save(net.state_dict(), '%s/segresnet_all.pt' % out)
    for a in range(5):
        try: api.upload_folder(folder_path=out, path_in_repo=out, repo_id=repo, repo_type='dataset', commit_message='MRI to game compartments (IXI step 1)'); break
        except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
    log('SUMMARY done')
