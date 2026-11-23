# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK", "nibabel", "torch", "monai==1.5.1", "einops"]
# ///
"""H3b: does an existing head-MRI tissue segmenter (GRACE, lab-smile, weights GRACE_MONAI150.pth from its v1.0.1
release; non-commercial research use only, not redistributed) agree with the same person's CT? HaN-Seg, 42 adults
with CT and T1 MRI (CC BY-ND 4.0, private use). Numbers only, no images (faces).

1. CPU: each person's MRI rigidly registered to their CT (mutual information, two starting alignments, best metric
   kept), CT resampled to 1.5 mm; body and the region outside the cranial cavity (face, scalp, neck) from CT.
2. GPU: GRACE on the raw MRI exactly as its test.py does: 1 mm spacing, RAS, intensities scaled from [0, 255] to
   [0, 1] after mapping the scan's 0.5-99.5th percentiles to [0, 255] (GRACE expects 8-bit-range input; this
   mapping is our assumption), UNETR (feature 16, hidden 768, mlp 3072, 12 heads, instance norm, residual blocks),
   64^3 sliding window, overlap 0.8, argmax.
3. CPU: labels resampled onto the CT grid with the registration (nearest). GRACE's label numbers are not published in
   text, so each label is identified from data: its share of the region, the median CT HU under it, and its overlap
   with HaN-Seg's expert eye and mandible masks. Then groups (bone, fat, muscle/lean, air) are formed from the CT
   medians and compared with CT classes by Dice.
args: <repo> [limit]"""
import os, sys, json, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def otsu(v, bins=256):
    v = np.asarray(v, float); h, e = np.histogram(v, bins); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; tie = np.flatnonzero(s >= np.nanmax(s) * (1 - 1e-9)); return float(c[tie].mean())
W = 'work'
_Z = None
def _init(zp):
    global _Z; _Z = zipfile.ZipFile(zp)
def find(names, c, suf): return [n for n in names if n.endswith('%s/%s_%s' % (c, c, suf))][0]

def stage1(case):
    """registration and CT-side maps"""
    import SimpleITK as sitk
    from scipy.ndimage import median_filter, label, binary_fill_holes, binary_dilation, binary_erosion
    try:
        names = _Z.namelist()
        def read(n, suffix):
            p = '%s/%s_%s.nrrd' % (W, case, suffix); open(p, 'wb').write(_Z.read(n)); im = sitk.ReadImage(p); os.remove(p); return im
        ct = sitk.Cast(read(find(names, case, 'IMG_CT.nrrd'), 'ct'), sitk.sitkFloat32); mr = sitk.Cast(read(find(names, case, 'IMG_MR_T1.nrrd'), 'mr'), sitk.sitkFloat32)
        sitk.WriteImage(mr, '%s/%s_mr.nii.gz' % (W, case))
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
        sitk.WriteTransform(best[1], '%s/%s.tfm' % (W, case)); sitk.WriteImage(ref, '%s/%s_ref.nii.gz' % (W, case))
        fov = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(mr * 0 + 1, sitk.sitkFloat32), ref, best[1], sitk.sitkNearestNeighbor, 0.0)) > 0.5
        C = median_filter(sitk.GetArrayFromImage(ctr), 3); lab, n = label(C > -500); body = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        for z in range(body.shape[0]): body[z] = binary_fill_holes(body[z])
        bone_t = otsu(C[body & (C > 150)]); bone = binary_dilation(C > bone_t, iterations=1); cav = np.zeros_like(bone)
        for z in range(bone.shape[0]): cav[z] = binary_fill_holes(bone[z]) & ~bone[z]
        cl, cn = label(cav); cav = cl == (np.bincount(cl.ravel())[1:].argmax() + 1) if cn else cav
        region = body & ~binary_dilation(cav, iterations=2) & binary_erosion(fov, iterations=2)
        oars = {}
        for key, suf in (('eye', ('OAR_Eye_AL', 'OAR_Eye_AR', 'OAR_Eye_PL', 'OAR_Eye_PR')), ('mandible', ('OAR_Bone_Mandible',))):
            m = np.zeros(C.shape, bool)
            for s_ in suf:
                try:
                    im = read(find(names, case, s_ + '.seg.nrrd'), 'oar'); m |= sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(im, sitk.sitkUInt8), ref, sitk.Transform(), sitk.sitkNearestNeighbor, 0)) > 0
                except Exception: pass
            oars[key] = m
        np.savez_compressed('%s/%s_ct.npz' % (W, case), C=C.astype(np.int16), region=region, eye=oars['eye'], mandible=oars['mandible'], bone_t=bone_t, metric=best[0])
        return case, None
    except Exception as e: return case, '%s %s' % (type(e).__name__, str(e)[:200])

def stage2(cases, dev='cuda'):
    """GRACE on each raw MRI, saved as a label NIfTI in the MRI's 1 mm RAS frame."""
    import torch, nibabel as nib
    from monai.networks.nets import UNETR
    from monai.inferers import sliding_window_inference
    from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityRanged, EnsureTyped
    import urllib.request
    wp = 'GRACE_MONAI150.pth'
    if not os.path.exists(wp): urllib.request.urlretrieve('https://github.com/lab-smile/GRACE/releases/download/v1.0.1/GRACE_MONAI150.pth', wp)
    model = UNETR(in_channels=1, out_channels=12, img_size=(64, 64, 64), feature_size=16, hidden_size=768, mlp_dim=3072, num_heads=12, norm_name='instance', res_block=True, dropout_rate=0.0).to(dev)
    sd = torch.load(wp, map_location='cpu', weights_only=False); sd = sd.get('state_dict', sd) if isinstance(sd, dict) else sd
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}; log('GRACE load', model.load_state_dict(sd, strict=True)); model.eval()
    tf = Compose([LoadImaged(keys=['image']), EnsureChannelFirstd(keys=['image']), Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'), Orientationd(keys=['image'], axcodes='RAS', labels=None),
                  ScaleIntensityRanged(keys=['image'], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True), EnsureTyped(keys=['image'])])
    for c in cases:
        raw = nib.load('%s/%s_mr.nii.gz' % (W, c)); a = np.asanyarray(raw.dataobj).astype(np.float32); lo, hi = np.percentile(a[a > 0], [0.5, 99.5])
        nib.save(nib.Nifti1Image(np.clip((a - lo) / (hi - lo), 0, 1) * 255.0, raw.affine), '%s/%s_mr8.nii.gz' % (W, c))
        d = tf({'image': '%s/%s_mr8.nii.gz' % (W, c)}); x = d['image'][None].to(dev)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16): y = sliding_window_inference(x, (64, 64, 64), 4, model, overlap=0.8)
        lab = torch.argmax(y, 1)[0].cpu().numpy().astype(np.uint8)
        nib.save(nib.Nifti1Image(lab, np.asarray(d['image'].meta['affine'])), '%s/%s_grace.nii.gz' % (W, c)); log('GRACE', c, 'labels', np.bincount(lab.ravel(), minlength=12).tolist())

def stage3(case):
    import SimpleITK as sitk
    try:
        d = np.load('%s/%s_ct.npz' % (W, case)); C = d['C'].astype(np.float32); R = d['region']; bone_t = float(d['bone_t'])
        ref = sitk.ReadImage('%s/%s_ref.nii.gz' % (W, case)); tf = sitk.ReadTransform('%s/%s.tfm' % (W, case))
        G = sitk.GetArrayFromImage(sitk.Resample(sitk.ReadImage('%s/%s_grace.nii.gz' % (W, case)), ref, tf, sitk.sitkNearestNeighbor, 0)).astype(np.int16)
        out = dict(case=case, metric=float(d['metric']), region_voxels=int(R.sum()), labels={})
        for k in range(12):
            m = (G == k) & R
            if m.sum() < 20: continue
            out['labels'][k] = dict(share=float(m.sum() / R.sum()), ct_hu_median=float(np.median(C[m])), ct_hu_iqr=[float(np.percentile(C[m], 25)), float(np.percentile(C[m], 75))],
                                    eye_overlap=float((m & d['eye']).sum() / max(d['eye'].sum(), 1)), mandible_overlap=float((m & d['mandible']).sum() / max(d['mandible'].sum(), 1)))
        out['eye_voxels'] = int(d['eye'].sum()); out['mandible_voxels'] = int(d['mandible'].sum())
        np.savez_compressed('%s/%s_pair.npz' % (W, case), G=G.astype(np.uint8), C=C.astype(np.int16), R=R, bone_t=bone_t, eye=d['eye'], mandible=d['mandible'])
        return out
    except Exception as e: return dict(case=case, failed='%s %s' % (type(e).__name__, str(e)[:200]))

def grouped_scores(rows, ok_cases):
    """name GRACE labels from the data (median CT HU pooled over people), then Dice of groups against CT classes."""
    pooled = {}
    for r in rows:
        for k, v in r.get('labels', {}).items(): pooled.setdefault(int(k), []).append(v)
    ident = {}
    for k, vs in pooled.items():
        hu = float(np.median([v['ct_hu_median'] for v in vs])); eye = float(np.median([v['eye_overlap'] for v in vs])); man = float(np.median([v['mandible_overlap'] for v in vs])); share = float(np.median([v['share'] for v in vs]))
        ident[k] = dict(ct_hu_median=round(hu, 1), share=round(share, 4), eye_overlap=round(eye, 3), mandible_overlap=round(man, 3))
    return ident

def dice_with_groups(case, groups):
    d = np.load('%s/%s_pair.npz' % (W, case)); G, C, R, bt = d['G'], d['C'].astype(np.float32), d['R'], float(d['bone_t'])
    ct = dict(air=(C < -500) & R, fat=(C >= -190) & (C <= -30) & R, lean=(C > -30) & (C <= bt) & R, bone=(C > bt) & R)
    dice = lambda a, b: float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
    out = {}
    for g, ks in groups.items():
        if not ks or g.startswith('_'): continue
        m = np.isin(G, ks) & R; out[g] = dice(m, ct[g])
    eye_lab = groups.get('_eye', [])
    if eye_lab: out['eye_vs_expert'] = dice(np.isin(G, eye_lab) & R, d['eye'] & R)
    return out

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0; os.makedirs(W, exist_ok=True)
    zp = hf_hub_download('gardianofthedarkness/hanseg-ct-mr', 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.')
    cases = sorted({n.split('/')[-2] for n in zipfile.ZipFile(zp).namelist() if n.endswith('_IMG_MR_T1.nrrd')})
    if limit: cases = cases[:limit]
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 16
    nw = max(1, min(int(cpus), int(ram // 5))); log(len(cases), 'people,', nw, 'workers')
    import SimpleITK as sitk; sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    with Pool(nw, _init, (zp,), maxtasksperchild=5) as p: s1 = dict(p.imap_unordered(stage1, cases))
    bad = {c: e for c, e in s1.items() if e}; good = [c for c in cases if c not in bad]; log('registered', len(good), 'failed', bad)
    stage2(good)
    with Pool(nw) as p: rows = list(p.imap_unordered(stage3, good))
    ok = [r for r in rows if 'failed' not in r]; ident = grouped_scores(ok, [r['case'] for r in ok])
    # groups from the data: the label most overlapping the expert eyes; bone = CT median above the bone threshold scale; fat, lean, air by HU ranges
    bt = float(np.median([float(np.load('%s/%s_pair.npz' % (W, r['case']))['bone_t']) for r in ok]))
    eye_k = max(ident, key=lambda k: ident[k]['eye_overlap']) if ident else None
    groups = dict(bone=[k for k, v in ident.items() if v['ct_hu_median'] > bt and k != eye_k], fat=[k for k, v in ident.items() if -190 <= v['ct_hu_median'] <= -30 and k != eye_k],
                  lean=[k for k, v in ident.items() if -30 < v['ct_hu_median'] <= bt and k not in (eye_k, 0)], air=[k for k, v in ident.items() if v['ct_hu_median'] < -500 and k != 0])
    groups['_eye'] = [eye_k] if eye_k is not None and ident[eye_k]['eye_overlap'] > max(v['eye_overlap'] for k, v in ident.items() if k != eye_k) else []
    per = [dict(case=r['case'], metric=r['metric'], **dice_with_groups(r['case'], groups)) for r in ok]
    summ = {g: dict(median=round(float(np.median([p[g] for p in per if g in p])), 3), p10_p90=[round(float(np.percentile([p[g] for p in per if g in p], q)), 3) for q in (10, 90)]) for g in ('bone', 'fat', 'lean', 'air', 'eye_vs_expert') if any(g in p for p in per)}
    res = dict(people=len(cases), registered=len(good), scored=len(ok), label_identity=ident, groups=groups, dice=summ, bone_threshold=bt)
    os.makedirs('h3b', exist_ok=True); json.dump(dict(summary=res, per_person=per, rows=rows, failed=bad), open('h3b/h3b.json', 'w'), indent=1)
    api.upload_folder(folder_path='h3b', path_in_repo='h3b' + ('_smoke' if limit else ''), repo_id='gardianofthedarkness/hanseg-ct-mr', repo_type='dataset', commit_message='H3b')
    log('SUMMARY ' + json.dumps(res))
