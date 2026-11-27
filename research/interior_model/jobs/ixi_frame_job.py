# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK"]
# ///
"""IXI step 2c (CPU, HF Jobs): place each labelled IXI head in MARROW's head frame and write it as a training head.
The frame is defined by the ear canals and eyes (face_frames.py), but the MRI labeller cannot find the ear canals. So the
frame comes from fitting the head itself: a reference head is built in the frame from the CT training heads (per voxel,
the most common of skull / mandible / eyes / brain at 1.5 mm), and each subject's labels are registered rigidly to it
(mutual information on the label images, two starting alignments, best kept). The registration's map from frame to
subject gives the subject's frame.
Accuracy check first, on CT heads with a known frame (validation and test heads, which are not in the reference): the same
fit on their own labels; origin error (mm) and axis error (degrees) are reported, so the frame's error is measured.
Output, per IXI subject, <repo>/ixi_headframe_v1/<id>.npz: labels in the frame window at 1 mm (compartment ids as game.py;
compartments in DROP and voxels outside the MRI's view = 255 unknown; outside the MRI body = 0), as face_v1_headframe.
env: COMMIT, DROP (comma list of compartments that failed the step 1 gate), LIMIT (smoke), WORKERS. args: <repo (ixi-t1-t2)>"""
import os, sys, json, time, urllib.request, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
SP = 1.5; KEYS = ('skull', 'mandible', 'eyes', 'brain'); G = {}

def composite(cls, names):
    """compartment ids -> 0 / 1 skull / 2 mandible / 3 eyes / 4 brain"""
    out = np.zeros(cls.shape, np.uint8)
    for i, k in enumerate(KEYS): out[cls == names.index(k)] = i + 1
    return out

def fit(sub_img):
    """rigid map from the reference (frame mm) to the subject (its physical space); returns (transform, metric)"""
    import SimpleITK as sitk
    ref = G['ref']; best = None
    for mode in (sitk.CenteredTransformInitializerFilter.MOMENTS, sitk.CenteredTransformInitializerFilter.GEOMETRY):
        init = sitk.CenteredTransformInitializer(ref, sub_img, sitk.Euler3DTransform(), mode)
        R = sitk.ImageRegistrationMethod(); R.SetMetricAsMattesMutualInformation(16); R.SetMetricSamplingStrategy(R.RANDOM); R.SetMetricSamplingPercentage(0.2, 1)
        R.SetInterpolator(sitk.sitkNearestNeighbor); R.SetOptimizerAsRegularStepGradientDescent(2.0, 1e-3, 300); R.SetOptimizerScalesFromPhysicalShift()
        R.SetShrinkFactorsPerLevel([4, 2, 1]); R.SetSmoothingSigmasPerLevel([2, 1, 0]); R.SetInitialTransform(init, inPlace=False)
        t_ = R.Execute(sitk.Cast(ref, sitk.sitkFloat32), sitk.Cast(sub_img, sitk.sitkFloat32)); m_ = R.GetMetricValue()
        if best is None or m_ < best[1]: best = (t_, m_)
    return best

def frame_of(T):
    """origin and axes (rows) of the frame in subject space from the map frame -> subject"""
    o = np.array(T.TransformPoint((0.0, 0.0, 0.0))); ax = np.stack([np.array(T.TransformPoint(tuple(e))) - o for e in np.eye(3)]); return o, ax

def ct_check(case):
    import SimpleITK as sitk
    try:
        z = np.load(G['ct_files'][case]); L, aff = z['labels'], z['affine']; cls = G['LUT'][L]
        im = sitk.GetImageFromArray(composite(cls, G['names']).transpose(2, 1, 0).copy())   # face_v1 arrays are (i, j, k) with the affine (LPS, step 1 found)
        sp = np.linalg.norm(aff[:3, :3], axis=0); im.SetSpacing(sp.tolist()); im.SetOrigin(aff[:3, 3].tolist()); im.SetDirection((aff[:3, :3] / sp).flatten().tolist())
        T, m = fit(im); o, ax = frame_of(T); h = G['ct_index'][case]; o_true, R_true = np.array(h['origin_mm']), np.array(h['axes'])
        ang = [float(np.degrees(np.arccos(np.clip(ax[i] @ R_true[i], -1, 1)))) for i in range(3)]
        return case, dict(origin_mm=round(float(np.linalg.norm(o - o_true)), 2), axis_deg=[round(a, 2) for a in ang], metric=round(m, 4))
    except Exception as e: return case, dict(error='%s %s' % (type(e).__name__, str(e)[:200]))

def ixi_one(path):
    import SimpleITK as sitk
    sid = os.path.basename(path)[:-4]
    try:
        z = np.load(path); lab = z['labels']; names = G['mri_names']
        cls = lab.copy(); cls[(lab == names.index('canal_left')) | (lab == names.index('canal_right'))] = names.index('airway')   # canals are airway in the compartments
        im = sitk.GetImageFromArray(composite(cls, names)); im.SetSpacing(z['spacing'].tolist()); im.SetOrigin(z['origin'].tolist()); im.SetDirection(z['direction'].tolist())
        T, m = fit(im)
        lo, hi = np.array(G['lo']), np.array(G['hi']); shape = (hi - lo).astype(int)
        out_ref = sitk.Image([int(s) for s in shape], sitk.sitkUInt8); out_ref.SetSpacing([1.0] * 3); out_ref.SetOrigin((lo + 0.5).tolist())
        src = sitk.GetImageFromArray(cls.astype(np.uint8)); src.CopyInformation(im)
        Lf = sitk.GetArrayFromImage(sitk.Resample(src, out_ref, T, sitk.sitkNearestNeighbor, 255)).transpose(2, 1, 0)   # back to (x, y, z) frame order
        inview = sitk.GetArrayFromImage(sitk.Resample(sitk.Cast(src * 0 + 1, sitk.sitkUInt8), out_ref, T, sitk.sitkNearestNeighbor, 0)).transpose(2, 1, 0) > 0
        Lf = Lf.astype(np.uint8); Lf[~inview] = 255
        for d in G['drop']: Lf[Lf == G['names'].index(d)] = 255
        np.savez_compressed('ixi_headframe_v1/%s.npz' % sid, labels=Lf)
        o, ax = frame_of(T)
        return sid, dict(metric=round(m, 4), origin=o.round(2).tolist(), axes=ax.round(5).tolist(), body_ml=round(float(((Lf > 0) & (Lf < 255)).sum()) / 1000, 1),
                         unknown_share=round(float((Lf == 255).mean()), 3))
    except Exception as e: return sid, dict(error='%s %s' % (type(e).__name__, str(e)[:200]))

def _init(g): G.update(g)

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from scipy.ndimage import zoom
    import SimpleITK as sitk
    api = HfApi(); repo = sys.argv[1]; user = api.whoami()['name']; os.makedirs('ixi_headframe_v1', exist_ok=True)
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/game.py' % os.environ.get('COMMIT', 'main'), 'game.py')
    sys.path.insert(0, '.'); import game
    REPOS = {'ts': user + '/totalsegmentator-ct-v201', 'hanseg': user + '/hanseg-ct-mr'}
    idx = json.load(open(hf_hub_download(REPOS['ts'], 'face_v1_headframe/index.json', repo_type='dataset', local_dir='ct')))
    hs = sorted(c for c, v in idx['heads'].items() if v['source'] == 'hanseg'); split = {c: v.get('split') for c, v in idx['heads'].items() if v['source'] == 'ts'}
    split.update({c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)})
    train = [c for c in idx['heads'] if split.get(c) == 'train']; held = [c for c in idx['heads'] if split.get(c) in ('val', 'test')]
    for src, r in REPOS.items():
        sel = [c for c in idx['heads'] if idx['heads'][c]['source'] == src]
        snapshot_download(r, repo_type='dataset', allow_patterns=['face_v1_headframe/%s.npz' % c for c in sel] + ['face_v1/%s.npz' % c for c in sel if c in held], local_dir='ct/' + src, max_workers=16)
    lo, hi = np.array(idx['window_lo_mm']), np.array(idx['window_hi_mm'])
    # reference head: per voxel the most common of the four parts over the training heads, at 1.5 mm in the frame
    acc = None
    for c in train:
        L = np.load('ct/%s/face_v1_headframe/%s.npz' % (idx['heads'][c]['source'], c))['labels']; comp = composite(game.LUT[L], game.GNAMES)
        oh = np.stack([zoom((comp == k).astype(np.float32), 1 / SP, order=1) for k in range(5)])
        acc = oh if acc is None else acc + oh
    refa = acc.argmax(0).astype(np.uint8)                                                   # (x, y, z) in the frame
    ref = sitk.GetImageFromArray(refa.transpose(2, 1, 0).copy()); ref.SetSpacing([SP] * 3); ref.SetOrigin((lo + SP / 2).tolist())
    log('reference head from', len(train), 'CT training heads; parts (voxels)', np.bincount(refa.ravel(), minlength=5).tolist())
    drop = [d for d in os.environ.get('DROP', '').split(',') if d]
    mri_names = game.GNAMES + ['canal_left', 'canal_right']
    g = dict(ref=None, LUT=game.LUT, names=game.GNAMES, mri_names=mri_names, lo=lo.tolist(), hi=hi.tolist(), drop=drop, ct_index=idx['heads'],
             ct_files={c: 'ct/%s/face_v1/%s.npz' % (idx['heads'][c]['source'], c) for c in held})
    nw = int(os.environ.get('WORKERS', os.cpu_count() or 4)); sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    sitk.WriteImage(ref, 'ref.nrrd')
    def _init2(g):
        G.update(g); G['ref'] = sitk.ReadImage('ref.nrrd')
    with Pool(nw, _init2, (g,)) as p: chk = dict(p.imap_unordered(ct_check, [c for c in held if os.path.exists(g['ct_files'][c])]))
    ok = [v for v in chk.values() if 'error' not in v]
    summ_chk = dict(heads=len(ok), errors=len(chk) - len(ok), origin_mm_median=float(np.median([v['origin_mm'] for v in ok])) if ok else None,
                    origin_mm_p90=float(np.percentile([v['origin_mm'] for v in ok], 90)) if ok else None,
                    axis_deg_median=float(np.median([max(v['axis_deg']) for v in ok])) if ok else None)
    log('SUMMARY frame check on CT heads (known frames):', json.dumps(summ_chk))
    files = api.list_repo_files(repo, repo_type='dataset')
    labelled = sorted(f.split('/')[-1][:-4] for f in files if f.startswith('ixi_label_v1/') and f.endswith('.npz'))
    if os.environ.get('LIMIT'): labelled = labelled[:int(os.environ['LIMIT'])]
    res = {}
    for i in range(0, len(labelled), 100):
        part = labelled[i:i + 100]
        d = snapshot_download(repo, repo_type='dataset', allow_patterns=['ixi_label_v1/%s.npz' % s for s in part], local_dir='lab', max_workers=16)
        with Pool(nw, _init2, (g,)) as p: res.update(dict(p.imap_unordered(ixi_one, [os.path.join(d, 'ixi_label_v1', s + '.npz') for s in part])))
        for a in range(5):
            try: api.upload_folder(folder_path='ixi_headframe_v1', path_in_repo='ixi_headframe_v1', repo_id=repo, repo_type='dataset', commit_message='IXI head frame batch'); break
            except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
        for f in os.listdir('ixi_headframe_v1'): os.remove(os.path.join('ixi_headframe_v1', f))
        log(min(i + 100, len(labelled)), 'of', len(labelled))
    bad = {k: v for k, v in res.items() if 'error' in v}
    json.dump(dict(window_lo_mm=lo.tolist(), window_hi_mm=hi.tolist(), voxel_mm=1.0, unknown=255, names=game.GNAMES, dropped=drop, frame_check=summ_chk, frame_check_per_head=chk,
                   heads={k: dict(v, source='ixi', split='train') for k, v in res.items() if 'error' not in v}, failed=bad), open('index.json', 'w'), indent=1)
    api.upload_file(path_or_fileobj='index.json', path_in_repo='ixi_headframe_v1/index.json', repo_id=repo, repo_type='dataset', commit_message='IXI head frame index')
    log('SUMMARY placed', len(res) - len(bad), 'failed', len(bad)); log('SUMMARY done')
