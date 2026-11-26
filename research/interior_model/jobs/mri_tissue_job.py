# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK", "torch", "monai==1.5.1"]
# ///
"""Test 1 (plan v8 section 14): can tissue types be read from MRI alone, as CT gives them? HaN-Seg, 42 adults with CT and
T1 MRI of the same person (CC BY-ND 4.0, private use). Numbers and weights only, no images.

1. CPU, per person: the MRI rigidly registered to the CT (mutual information, two starting alignments, best kept, as in
   H3b), both on the CT's grid at 1.5 mm. Tissue from the CT: outside, fat under the skin (fat connected to the outside),
   other fat (-190 to -30 HU), lean (-30 to 150), air inside the body (below -500), bone (above 150), eye (expert outlines).
   Voxels outside the MRI's field of view are left out. Scoring region: the body outside the cranial cavity (face, scalp, neck).
2. GPU: a 3D segmentation network (MONAI SegResNet) from MRI to these tissues, 5-fold cross-validation (fold = index mod 5),
   so every head is scored by a model that never saw it; then one model on all heads for later use on other MRIs (IXI).
3. Scores per tissue in the scoring region: Dice and median surface distance; eyes against the expert outlines; bone inside
   the jaw zone (the expert mandible outline, widened).
env: STEPS (per fold, default 4000), SMALL_SHARE (crops centred on eyes and thin fat, default 0.3), OUT_TAG. args: <repo for outputs (the HaN-Seg repo)> [limit]"""
import os, sys, json, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
W = 'work'; NCL = 7; NAMES = ['outside', 'fat_under_skin', 'fat_other', 'lean', 'air', 'bone', 'eye']; IGN = 255
_Z = None
def _init(zp):
    global _Z; _Z = zipfile.ZipFile(zp)
def find(names, c, suf): return [n for n in names if n.endswith('%s/%s_%s' % (c, c, suf))][0]
def otsu(v, bins=256):
    v = np.asarray(v, float); h, e = np.histogram(v, bins); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; tie = np.flatnonzero(s >= np.nanmax(s) * (1 - 1e-9)); return float(c[tie].mean())

def stage1(case):
    import SimpleITK as sitk
    from scipy.ndimage import median_filter, label, binary_fill_holes, binary_dilation, binary_erosion
    try:
        names = _Z.namelist()
        def read(n, suffix):
            p = '%s/%s_%s.nrrd' % (W, case, suffix); open(p, 'wb').write(_Z.read(n)); im = sitk.ReadImage(p); os.remove(p); return im
        ct = sitk.Cast(read(find(names, case, 'IMG_CT.nrrd'), 'ct'), sitk.sitkFloat32); mr = sitk.Cast(read(find(names, case, 'IMG_MR_T1.nrrd'), 'mr'), sitk.sitkFloat32)
        sp = 1.5; size = [int(round(s * z / sp)) for s, z in zip(ct.GetSize(), ct.GetSpacing())]
        ref = sitk.Image(size, sitk.sitkFloat32); ref.SetSpacing([sp] * 3); ref.SetOrigin(ct.GetOrigin()); ref.SetDirection(ct.GetDirection())
        ctr = sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0); best = None
        for mode in (sitk.CenteredTransformInitializerFilter.GEOMETRY, sitk.CenteredTransformInitializerFilter.MOMENTS):
            init = sitk.CenteredTransformInitializer(ctr, mr, sitk.Euler3DTransform(), mode)
            R = sitk.ImageRegistrationMethod(); R.SetMetricAsMattesMutualInformation(50); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.1, 1)
            R.SetInterpolator(sitk.sitkLinear); R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 400); R.SetOptimizerScalesFromPhysicalShift()
            R.SetShrinkFactorsPerLevel([8, 4, 2, 1]); R.SetSmoothingSigmasPerLevel([3, 2, 1, 0]); R.SetInitialTransform(init, inPlace=False)
            t_ = R.Execute(ctr, mr); m_ = R.GetMetricValue()
            if best is None or m_ < best[0]: best = (m_, t_)
        M = sitk.GetArrayFromImage(sitk.Resample(mr, ref, best[1], sitk.sitkLinear, 0.0))
        fov = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(mr * 0 + 1, sitk.sitkFloat32), ref, best[1], sitk.sitkNearestNeighbor, 0.0)) > 0.5
        C = median_filter(sitk.GetArrayFromImage(ctr), 3); lab, n = label(C > -500); body = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        for z in range(body.shape[0]): body[z] = binary_fill_holes(body[z])
        bone_t = otsu(C[body & (C > 150)]); bonem = binary_dilation(C > bone_t, iterations=1); cav = np.zeros_like(bonem)
        for z in range(bonem.shape[0]): cav[z] = binary_fill_holes(bonem[z]) & ~bonem[z]
        cl, cn = label(cav); cav = cl == (np.bincount(cl.ravel())[1:].argmax() + 1) if cn else cav
        region = body & ~binary_dilation(cav, iterations=2) & binary_erosion(fov, iterations=2)
        oars = {}
        for key, suf in (('eye', ('OAR_Eye_AL', 'OAR_Eye_AR', 'OAR_Eye_PL', 'OAR_Eye_PR')), ('mandible', ('OAR_Bone_Mandible',))):
            m = np.zeros(C.shape, bool)
            for s_ in suf:
                try: im = read(find(names, case, s_ + '.seg.nrrd'), 'oar'); m |= sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(im, sitk.sitkUInt8), ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
                except Exception: pass
            oars[key] = m
        T = np.zeros(C.shape, np.uint8); fat = body & (C >= -190) & (C <= -30); fl, _ = label(fat | ~body); oid = np.bincount(fl[~body].ravel()).argmax() if (~body).any() else -1
        T[body & (C > -30) & (C <= 150)] = 3; T[body & (C < -500)] = 4; T[body & (C > 150)] = 5; T[fat] = 2; T[fat & (fl == oid)] = 1; T[oars['eye'] & body] = 6
        T[~fov] = IGN
        lo, hi = np.percentile(M[fov & body], [0.5, 99.5]) if (fov & body).any() else (0, 1); Mn = np.clip((M - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float16); Mn[~fov] = 0
        jaw = binary_dilation(oars['mandible'], iterations=6)
        np.savez_compressed('%s/%s.npz' % (W, case), M=Mn, T=T, region=region, eye=oars['eye'], jaw=jaw, metric=best[0])
        return case, None
    except Exception as e: return case, '%s %s' % (type(e).__name__, str(e)[:200])

def surface_mm(a, b, mm=1.5):
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    sa = np.argwhere(a & ~binary_erosion(a)); sb = np.argwhere(b & ~binary_erosion(b))
    if len(sa) < 3 or len(sb) < 3: return float('nan')
    sa = sa[:: max(1, len(sa) // 30000)]; sb = sb[:: max(1, len(sb) // 30000)]
    return float(np.median(np.concatenate([cKDTree(sb).query(sa, workers=-1)[0], cKDTree(sa).query(sb, workers=-1)[0]])) * mm)

def train(cases, steps, dev='cuda'):
    import torch
    from monai.networks.nets import SegResNet
    rng = np.random.default_rng(0); vols = {c: np.load('%s/%s.npz' % (W, c)) for c in cases}; V = {c: (v['M'], v['T']) for c, v in vols.items()}
    cand = {c: np.argwhere(V[c][1][::6, ::6, ::6] < IGN) * 6 for c in cases}; S = 96
    small = {c: np.argwhere(np.isin(V[c][1][::2, ::2, ::2], [2, 6])) * 2 for c in cases}           # eyes and thin fat: a share of crops is centred on them
    share = np.bincount(np.concatenate([V[c][1][::4, ::4, ::4].ravel() for c in cases]), minlength=256)[:NCL].astype(float); share /= share.sum()
    wts = torch.tensor(np.where(share > 0, share, share[share > 0].min()) ** -0.5, dtype=torch.float32, device=dev); wts = wts / wts.mean()   # rarer tissue, larger weight (as the label autoencoder)
    SMALL = float(os.environ.get('SMALL_SHARE', '0.3'))
    net = SegResNet(spatial_dims=3, in_channels=1, out_channels=NCL, init_filters=16).to(dev); opt = torch.optim.AdamW(net.parameters(), 2e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); scaler = torch.amp.GradScaler(); ce = torch.nn.CrossEntropyLoss(weight=wts, ignore_index=IGN)
    def batch(n=2):
        X = np.zeros((n, 1, S, S, S), np.float32); Y = np.full((n, S, S, S), IGN, np.int64)
        for i in range(n):
            c = cases[rng.integers(len(cases))]; M, T = V[c]; q = small[c][rng.integers(len(small[c]))] if (len(small[c]) and rng.random() < SMALL) else cand[c][rng.integers(len(cand[c]))]; o = [int(np.clip(a - S // 2, 0, max(0, d - S))) for a, d in zip(q, T.shape)]
            m = M[o[0]:o[0] + S, o[1]:o[1] + S, o[2]:o[2] + S].astype(np.float32); t = T[o[0]:o[0] + S, o[1]:o[1] + S, o[2]:o[2] + S]
            m = m * rng.uniform(0.85, 1.15) + rng.uniform(-0.05, 0.05)                        # brightness and contrast vary between scanners (IXI later)
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

def score(case, P):
    v = np.load('%s/%s.npz' % (W, case)); T, R = v['T'], v['region'] & (v['T'] != IGN); r = dict(case=case)
    for k in range(1, NCL):
        a, b = (P == k) & R, (T == k) & R
        if b.sum() < 30: continue
        r[NAMES[k]] = dict(dice=round(float(2 * (a & b).sum() / (a.sum() + b.sum())), 3), surface_mm=round(surface_mm(a, b), 2))
    e = v['eye']
    if e.sum() > 30: r['eye_vs_expert'] = round(float(2 * ((P == 6) & e).sum() / ((P == 6).sum() + e.sum())), 3)
    J = v['jaw'] & R
    if (J & (T == 5)).sum() > 30: a, b = (P == 5) & J, (T == 5) & J; r['jaw_bone'] = dict(dice=round(float(2 * (a & b).sum() / (a.sum() + b.sum())), 3), surface_mm=round(surface_mm(a, b), 2))
    return r

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    import torch
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0; os.makedirs(W, exist_ok=True); STEPS = int(os.environ.get('STEPS', '4000'))
    zp = hf_hub_download('gardianofthedarkness/hanseg-ct-mr', 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.')
    cases = sorted({n.split('/')[-2] for n in zipfile.ZipFile(zp).namelist() if n.endswith('_IMG_MR_T1.nrrd')})
    if limit: cases = cases[:limit]
    try: q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 16
    nw = max(1, min(int(cpus), int(ram // 5))); log(len(cases), 'people,', nw, 'workers')
    import SimpleITK as sitk; sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    with Pool(nw, _init, (zp,), maxtasksperchild=5) as p: s1 = dict(p.imap_unordered(stage1, cases))
    bad = {c: e for c, e in s1.items() if e}; good = [c for c in cases if c not in bad]; log('registered', len(good), 'failed', bad)
    out = 'mri_tissue' + os.environ.get('OUT_TAG', '') + ('_smoke' if limit else ''); os.makedirs(out, exist_ok=True); rows = []
    for f in range(5):
        test = [c for i, c in enumerate(good) if i % 5 == f]; tr = [c for c in good if c not in test]
        if not test or not tr: continue
        net = train(tr, STEPS); torch.save(net.state_dict(), '%s/segresnet_fold%d.pt' % (out, f))
        for c in test:
            r = score(c, predict(net, np.load('%s/%s.npz' % (W, c))['M'])); r['fold'] = f; rows.append(r); log('fold', f, c, {k: v for k, v in r.items() if k not in ('case', 'fold')})
        del net; torch.cuda.empty_cache()
    summ = {}
    for k in NAMES[1:] + ['jaw_bone']:
        ds = [r[k]['dice'] for r in rows if k in r]; ss = [r[k]['surface_mm'] for r in rows if k in r and np.isfinite(r[k]['surface_mm'])]
        if ds: summ[k] = dict(dice_median=round(float(np.median(ds)), 3), dice_p10=round(float(np.percentile(ds, 10)), 3), surface_mm_median=round(float(np.median(ss)), 2) if ss else None, people=len(ds))
    ev = [r['eye_vs_expert'] for r in rows if 'eye_vs_expert' in r]
    if ev: summ['eye_vs_expert'] = dict(dice_median=round(float(np.median(ev)), 3), people=len(ev))
    json.dump(dict(summary=summ, per_person=rows, failed=bad, steps_per_fold=STEPS), open('%s/mri_tissue.json' % out, 'w'), indent=1)
    log('SUMMARY ' + json.dumps(summ))
    if not limit:                                                                            # one model on all heads, for other MRIs (IXI)
        net = train(good, STEPS); torch.save(net.state_dict(), '%s/segresnet_all.pt' % out)
    for a in range(5):
        try: api.upload_folder(folder_path=out, path_in_repo=out, repo_id=repo, repo_type='dataset', commit_message='MRI to tissue (test 1)'); break
        except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
    log('SUMMARY done')
