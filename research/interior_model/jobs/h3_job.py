# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK"]
# ///
"""H3: can tissue labels made from T1 MRI match the same person's CT? HaN-Seg (42 adults, CT + T1 MRI; CC BY-ND 4.0,
private use only). Numbers only: no images, since these are faces.

Per person: CT and MR resampled to 1.5 mm in the CT frame; MR rigidly registered to CT (Mattes mutual information,
three resolutions). Body = largest piece of CT > -500 HU, holes filled per slice; scored only where the MR field of
view covers the body. CT classes (3x3x3 median): dark (< -190: air, and nothing else in the head), fat (-190..-30),
lean (-29..150 and up to the bone threshold), bone (> the case's own Otsu above lean). MR classes: bias-field
corrected (N4), then three-class Otsu on the body's intensities: dark (bone or air), lean, fat. CT 'dark' for the
comparison is bone or air, the same physical group MR cannot split.
Scores: Dice per class; facial soft-tissue depth (distance from skin to the nearest CT bone, or MR dark, for skin
voxels in front of the face: anterior third of the body, below the eyes' level is not separated here) as median
absolute difference; registration check = Dice of the two body masks. args: <repo> [limit]"""
import os, sys, io, json, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def otsu(v, bins=256):
    v = np.asarray(v, float); h, e = np.histogram(v, bins); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; tie = np.flatnonzero(s >= np.nanmax(s) * (1 - 1e-9)); return float(c[tie].mean())
def otsu3(v, bins=128):
    """two thresholds maximising between-class variance (three classes), exhaustive over the histogram."""
    h, e = np.histogram(v, bins); c = (e[:-1] + e[1:]) / 2; p = h / h.sum(); P = np.cumsum(p); S = np.cumsum(p * c); mu = S[-1]; best = (-1, 0, 0)
    for i in range(1, bins - 1):
        for j in range(i + 1, bins):
            w0, w1, w2 = P[i - 1], P[j - 1] - P[i - 1], 1 - P[j - 1]
            if min(w0, w1, w2) <= 0: continue
            m0, m1, m2 = S[i - 1] / w0, (S[j - 1] - S[i - 1]) / w1, (mu - S[j - 1]) / w2
            b = w0 * (m0 - mu) ** 2 + w1 * (m1 - mu) ** 2 + w2 * (m2 - mu) ** 2
            if b > best[0]: best = (b, e[i], e[j])
    return best[1], best[2]

def one(arg):
    import SimpleITK as sitk
    from scipy.ndimage import median_filter, label, binary_fill_holes, distance_transform_edt, binary_erosion
    case, ct_bytes, mr_bytes = arg
    try:
        def read(b, suffix):
            p = '/tmp/%s_%s.nrrd' % (case, suffix); open(p, 'wb').write(b); im = sitk.ReadImage(p); os.remove(p); return im
        ct = sitk.Cast(read(ct_bytes, 'ct'), sitk.sitkFloat32); mr = sitk.Cast(read(mr_bytes, 'mr'), sitk.sitkFloat32)
        sp = 1.5; size = [int(round(s * z / sp)) for s, z in zip(ct.GetSize(), ct.GetSpacing())]
        ref = sitk.Image(size, sitk.sitkFloat32); ref.SetSpacing([sp] * 3); ref.SetOrigin(ct.GetOrigin()); ref.SetDirection(ct.GetDirection())
        ctr = sitk.Resample(ct, ref, sitk.Transform(), sitk.sitkLinear, -1024.0)
        # MR bias correction on a coarse copy, then rigid registration to CT
        mr_small = sitk.Shrink(mr, [max(1, int(round(sp / s))) for s in mr.GetSpacing()]); mmask = sitk.OtsuThreshold(mr_small, 0, 1, 200)
        n4 = sitk.N4BiasFieldCorrectionImageFilter(); n4.SetMaximumNumberOfIterations([30, 20, 10]); n4.Execute(mr_small, mmask)
        logbias = sitk.Cast(n4.GetLogBiasFieldAsImage(mr), sitk.sitkFloat32); mrc = sitk.Cast(mr / sitk.Exp(logbias), sitk.sitkFloat32)
        best = None
        for mode in (sitk.CenteredTransformInitializerFilter.GEOMETRY, sitk.CenteredTransformInitializerFilter.MOMENTS):   # several starts; keep the best metric
            init = sitk.CenteredTransformInitializer(ctr, mrc, sitk.Euler3DTransform(), mode)
            R = sitk.ImageRegistrationMethod(); R.SetMetricAsMattesMutualInformation(50); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.1, 1)
            R.SetInterpolator(sitk.sitkLinear); R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 400); R.SetOptimizerScalesFromPhysicalShift()
            R.SetShrinkFactorsPerLevel([8, 4, 2, 1]); R.SetSmoothingSigmasPerLevel([3, 2, 1, 0]); R.SetInitialTransform(init, inPlace=False)
            t_ = R.Execute(ctr, mrc); m_ = R.GetMetricValue()
            if best is None or m_ < best[0]: best = (m_, t_)
        tf = best[1]; mrr = sitk.Resample(mrc, ref, tf, sitk.sitkLinear, 0.0); fov = sitk.Resample(sitk.Cast(mr * 0 + 1, sitk.sitkFloat32), ref, tf, sitk.sitkNearestNeighbor, 0.0)
        C = median_filter(sitk.GetArrayFromImage(ctr), 3); M = median_filter(sitk.GetArrayFromImage(mrr), 3); F = sitk.GetArrayFromImage(fov) > 0.5
        lab, n = label(C > -500); body = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        for z in range(body.shape[0]): body[z] = binary_fill_holes(body[z])
        body &= binary_erosion(F, iterations=2)
        bone_t = otsu(C[body & (C > 150)]) if (body & (C > 150)).sum() > 100 else 300.0
        ct_cls = np.select([C < -190, C <= -30, C <= bone_t], [0, 1, 2], 3)                   # 0 air, 1 fat, 2 lean, 3 bone
        ctg = np.where(ct_cls == 3, 0, ct_cls)                                                # dark = air or bone
        # compare outside the skull only (face, scalp, neck): inside, the brain dominates MR brightness
        from scipy.ndimage import binary_dilation
        bone = binary_dilation(ct_cls == 3, iterations=1); cav = np.zeros_like(bone)
        for z in range(bone.shape[0]): cav[z] = binary_fill_holes(bone[z]) & ~bone[z]
        cl, cn = label(cav); cav = cl == (np.bincount(cl.ravel())[1:].argmax() + 1) if cn else cav   # the cranial cavity: the largest enclosed space
        body &= ~binary_dilation(cav, iterations=2)
        mrb = M[body]; t1, t2 = otsu3(mrb); mr_cls = np.select([M < t1, M < t2], [0, 2], 1)    # T1: dark < lean < fat
        mr_body = (lab > 0) & False
        mlab, mn = label((M > t1 * 0.5) & F); mbody = mlab == (np.bincount(mlab.ravel())[1:].argmax() + 1) if mn else F
        for z in range(mbody.shape[0]): mbody[z] = binary_fill_holes(mbody[z])
        dice = lambda a, b: float(2 * (a & b).sum() / max(a.sum() + b.sum(), 1))
        out = dict(case=case, body_voxels=int(body.sum()), registration_body_dice=dice(body, mbody & binary_erosion(F, iterations=2)),
                   dice_fat=dice((ctg == 1) & body, (mr_cls == 1) & body), dice_lean=dice((ctg == 2) & body, (mr_cls == 2) & body), dice_dark=dice((ctg == 0) & body, (mr_cls == 0) & body),
                   share_ct=dict(dark=float(((ctg == 0) & body).sum() / body.sum()), fat=float(((ctg == 1) & body).sum() / body.sum()), lean=float(((ctg == 2) & body).sum() / body.sum())),
                   share_mr=dict(dark=float(((mr_cls == 0) & body).sum() / body.sum()), fat=float(((mr_cls == 1) & body).sum() / body.sum()), lean=float(((mr_cls == 2) & body).sum() / body.sum())))
        # facial soft-tissue depth: skin voxels in the anterior third of the body, distance to bone (CT) or to dark-inside (MR)
        skin = body & ~binary_erosion(body)
        y = np.argwhere(body)[:, 1]; ant = np.zeros_like(body); a_lim = np.percentile(y, 33)
        ant[:, :int(a_lim), :] = True                                                         # array axes (z, y, x); smaller y = anterior in LPS-stored CT
        inner = binary_erosion(body, iterations=3)
        d_ct = distance_transform_edt(~((ct_cls == 3) & inner), sampling=sp); d_mr = distance_transform_edt(~((mr_cls == 0) & inner), sampling=sp)
        sel = skin & ant & (d_ct < 40)
        if sel.sum() > 50: out.update(face_depth_ct_mm=float(np.median(d_ct[sel])), face_depth_mr_mm=float(np.median(d_mr[sel])), face_depth_abs_diff_mm=float(np.median(np.abs(d_ct[sel] - d_mr[sel]))))
        return out
    except Exception as e: return dict(case=case, failed='%s %s' % (type(e).__name__, str(e)[:200]))

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    zp = hf_hub_download('gardianofthedarkness/hanseg-ct-mr', 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.'); Z = zipfile.ZipFile(zp)
    cases = sorted({n.split('/')[-2] for n in Z.namelist() if n.endswith('_IMG_MR_T1.nrrd')})
    if limit: cases = cases[:limit]
    def find(c, suf): return [n for n in Z.namelist() if n.endswith('%s/%s_%s' % (c, c, suf))][0]
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 16
    nw = max(1, min(int(cpus), int(ram // 4))); log(len(cases), 'people,', nw, 'workers')
    os.environ.setdefault('SITK_NUMBER_OF_THREADS', '1')
    import SimpleITK as sitk; sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    args = ((c, Z.read(find(c, 'IMG_CT.nrrd')), Z.read(find(c, 'IMG_MR_T1.nrrd'))) for c in cases)
    with Pool(nw, maxtasksperchild=5) as p: rows = list(p.imap_unordered(one, args))
    ok = [r for r in rows if 'failed' not in r]; log('ok', len(ok), 'failed', [r for r in rows if 'failed' in r][:3])
    med = lambda k: round(float(np.median([r[k] for r in ok if r.get(k) is not None])), 3) if ok else None
    pct = lambda k: [round(float(np.percentile([r[k] for r in ok if r.get(k) is not None], q)), 3) for q in (10, 90)] if ok else None
    res = dict(people=len(rows), ok=len(ok), **{k: dict(median=med(k), p10_p90=pct(k)) for k in ('registration_body_dice', 'dice_fat', 'dice_lean', 'dice_dark', 'face_depth_ct_mm', 'face_depth_mr_mm', 'face_depth_abs_diff_mm')})
    os.makedirs('h3', exist_ok=True); json.dump(dict(summary=res, rows=rows), open('h3/h3.json', 'w'), indent=1)
    api.upload_folder(folder_path='h3', path_in_repo='h3' + ('_smoke' if limit else ''), repo_id='gardianofthedarkness/hanseg-ct-mr', repo_type='dataset', commit_message='H3')
    log('SUMMARY ' + json.dumps(res))
