# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK", "nibabel", "TotalSegmentator", "matplotlib"]
# ///
"""PDDCA's 48 full-face CTs with the named face labels (face_v1 classes), in the head frame, so they can join MARROW S1
training (user 2026-11-28: "label them and use them for training"). GPU (Colab). Same recipe as the HaN-Seg heads
(face_set_job.one_hanseg + face_frames.py):
  1. TotalSegmentator's three head tasks (craniofacial_structures, head_glands_cavities, head_muscles) and its general
     model for brain, trachea, spinal cord and C1-C7 (painted only over plain tissue, as the HaN-Seg repair).
  2. Brightness classes 0-5 under them (body mask, fat, lean, gas, dense), as labels_job.
  3. PDDCA's expert outlines replace the equivalent class (mandible, parotids, submandibular glands, optic nerves);
     brainstem and chiasm are added as the expert-only classes. Teeth and lenses are never overwritten.
  4. Head frame from the ear canals and eyes (face_frames.frame), labels resampled at 1 mm into the shared window of
     face_v1_headframe/index.json; outside the scan = unknown (255).
QC per head (JSON): TotalSegmentator against the expert outline (Dice) for mandible, parotids, submandibular glands, so a
left/right mix-up or a weak class shows; compartment volumes. Private pictures (real people) of 6 heads to
<repo>/face_v1_headframe_s1/check/ only.
Output: <repo>/face_v1_headframe_s1/<case>.npz (labels) + index_pddca_s1.json (splits kept from index_pddca.json).
env: LIMIT (smoke), PAR (CTs at once, default 2), FREPO (repo with face_v1_headframe/index.json), COMMIT. args: <pddca repo>"""
import os, sys, json, time, zipfile, subprocess, urllib.request, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
W = '/content/pd' if os.path.isdir('/content') else '/data/pd'
HEAD_TASKS = ['craniofacial_structures', 'head_glands_cavities', 'head_muscles']
TOTAL_ROI = ['brain', 'trachea', 'spinal_cord'] + ['vertebrae_C%d' % i for i in range(1, 8)]
TOTAL_TO = {'brain': 'brain', 'trachea': 'trachea', 'spinal_cord': 'SpinalCord', **{'vertebrae_C%d' % i: 'vertebrae_C%d' % i for i in range(1, 8)}}
EXPERT = {1: 'mandible', 3: 'parotid_gland_left', 4: 'parotid_gland_right', 5: 'submandibular_gland_left', 6: 'submandibular_gland_right',
          7: 'optic_nerve_left', 8: 'optic_nerve_right'}                   # PDDCA labels_map -> face class it replaces
EXPERT_ONLY = {2: 'Brainstem', 9: 'OpticChiasm'}

def segment(ctp, outp):
    """all TotalSegmentator runs for one CT, in its own process (memory)"""
    from totalsegmentator.python_api import totalsegmentator
    import nibabel as nib
    out = {}
    for t in HEAD_TASKS: out[t] = np.asarray(totalsegmentator(ctp, None, task=t, quiet=True, nr_thr_resamp=1, nr_thr_saving=1).dataobj).astype(np.uint8)
    out['total'] = np.asarray(totalsegmentator(ctp, None, task='total', roi_subset=TOTAL_ROI, quiet=True, nr_thr_resamp=1, nr_thr_saving=1).dataobj).astype(np.uint8)
    np.savez_compressed(outp, **out)

def dice(a, b): s = a.sum() + b.sum(); return round(float(2 * (a & b).sum() / s), 3) if s else None

def build(case, ctp, mkp, segp, ids, lo, hi, cmap):
    import SimpleITK as sitk
    from scipy.ndimage import map_coordinates
    sys.path.insert(0, '.'); import face_set_job as fs, face_frames as ff
    im = sitk.ReadImage(ctp); A = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32)          # (x, y, z) index, as nibabel
    mk = sitk.Resample(sitk.ReadImage(mkp), im, sitk.Transform(), sitk.sitkNearestNeighbor, 0); E = sitk.GetArrayFromImage(mk).transpose(2, 1, 0)
    S = dict(np.load(segp)); D = np.array(im.GetDirection()).reshape(3, 3) * np.array(im.GetSpacing()); aff = np.eye(4); aff[:3, :3] = D; aff[:3, 3] = im.GetOrigin()
    B = fs.body_mask(A); Lb = fs.basic_labels(A, B); L = Lb.copy(); del A
    inv = {t: {v: int(k) for k, v in cmap[t].items()} for t in HEAD_TASKS + ['total']}
    for t in HEAD_TASKS:
        for n, k in inv[t].items():
            if n != 'head' and n in ids: L[(S[t] == k) & B] = ids[n]
    for n, to in TOTAL_TO.items():
        if n in inv['total']: L[(S['total'] == inv['total'][n]) & B & np.isin(L, [1, 2, 3, 4, 5])] = ids[to]
    qc = {}
    for k, n in EXPERT.items():
        m = (E == k) & B
        if not m.any(): continue
        ts_m = L == ids[n]; qc[n] = dice(ts_m, m)
        L[ts_m & ~m] = Lb[ts_m & ~m]
        protect = np.isin(L, [ids['eye_lens_left'], ids['eye_lens_right'], ids['teeth_lower'], ids['teeth_upper']])
        L[m & ~protect] = ids[n]
    for k, n in EXPERT_ONLY.items():
        m = (E == k) & B
        if m.any(): L[m] = ids[n]
    f = ff.frame(L, aff, ids)
    if f is None: return None, 'landmark missing (ear canal or eye)', qc
    o, R = f; shape = (hi - lo).astype(int)
    g = np.stack(np.meshgrid(*[np.arange(n) + 0.5 for n in shape], indexing='ij'), -1).reshape(-1, 3) + lo
    idx = (g @ R + o - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T
    Lf = map_coordinates(L, idx.T, order=0, mode='constant', cval=255).reshape(shape).astype(np.uint8)
    np.savez_compressed('%s/out/%s.npz' % (W, case), labels=Lf)
    return dict(source='pddca', labeller='expert+totalsegmentator', native_mm=[round(float(s), 3) for s in im.GetSpacing()], origin_mm=o.round(3).tolist(), axes=R.round(5).tolist(),
                body_voxels_1mm=int(((Lf > 0) & (Lf < 255)).sum()), outside_scan_share=round(float((Lf == 255).mean()), 3), ts_vs_expert_dice=qc,
                expert_present=sorted(EXPERT.get(k) or EXPERT_ONLY[k] for k in np.unique(E).tolist() if k)), None, qc

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--one': segment(sys.argv[2], sys.argv[3]); sys.exit(0)
if __name__ == '__main__':
    try: import totalsegmentator, SimpleITK
    except ImportError: subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'TotalSegmentator', 'SimpleITK', 'nibabel'], check=True)
    from concurrent.futures import ThreadPoolExecutor
    import threading
    from huggingface_hub import HfApi, hf_hub_download
    from totalsegmentator.map_to_binary import class_map
    repo = sys.argv[1]; api = HfApi(); user = api.whoami()['name']; C_ = os.environ.get('COMMIT', 'main')
    LIMIT = int(os.environ.get('LIMIT', '0')); PAR = int(os.environ.get('PAR', '2')); frepo = os.environ.get('FREPO', user + '/totalsegmentator-ct-v201')
    for f in ('jobs/face_set_job.py', 'face_frames.py', 'game.py'):
        urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C_, f), os.path.basename(f))
    sys.path.insert(0, '.'); import game
    fi = json.load(open(hf_hub_download(frepo, 'face_v1_headframe/index.json', repo_type='dataset'))); ids = fi['ids']; lo, hi = np.array(fi['window_lo_mm']), np.array(fi['window_hi_mm'])
    split = {c: v['split'] for c, v in json.load(open(hf_hub_download(repo, 'face_v1_headframe/index_pddca.json', repo_type='dataset')))['heads'].items()}
    Z = zipfile.ZipFile(hf_hub_download(repo, 'raw/data-part001.zip', repo_type='dataset', local_dir=W + '/raw'))
    cts = sorted(n for n in Z.namelist() if n.startswith('Images/') and n.endswith('.nii.gz'))
    if LIMIT: cts = cts[:: max(1, len(cts) // LIMIT)][:LIMIT]
    os.makedirs(W + '/out', exist_ok=True); import torch; log('gpu', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none', len(cts), 'CTs,', PAR, 'at a time')
    cmap = {t: class_map[t] for t in HEAD_TASKS + ['total']}
    index, failed = {}, {}; zl, il = threading.Lock(), threading.Lock()
    def one(n):
        case = 'pd_' + os.path.basename(n)[:-7]; ctp, mkp, sp = '%s/%s.nii.gz' % (W, case), '%s/%s_mask.nii.gz' % (W, case), '%s/%s_seg.npz' % (W, case)
        try:
            with zl: open(ctp, 'wb').write(Z.read(n)); open(mkp, 'wb').write(Z.read(n.replace('Images/', 'Masks/')))
            t1 = time.time(); q = subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--one', ctp, sp], capture_output=True, text=True, timeout=3600)
            if q.returncode != 0: m, why, qc = None, 'segment exit %d: %s' % (q.returncode, q.stderr.strip()[-300:]), {}
            else: m, why, qc = build(case, ctp, mkp, sp, ids, lo, hi, cmap)
        except Exception as e: m, why = None, '%s %s' % (type(e).__name__, str(e)[:200])
        for f_ in (ctp, mkp, sp):
            if os.path.exists(f_): os.remove(f_)
        with il:
            if m is None: failed[case] = why
            else: m['split'] = split.get(case, 'train'); index[case] = m
        log(case, ('%.0fs' % (time.time() - t1)) if m else '', m['ts_vs_expert_dice'] if m else why)
    one(cts[0])                                                                             # alone first: downloads the weights once
    with ThreadPoolExecutor(PAR) as ex: list(ex.map(one, cts[1:]))
    # compartment volumes and private pictures
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(W + '/out/check', exist_ok=True); pal = np.array([(0, 0, 0), (0.98, 0.84, 0.45), (0.93, 0.89, 0.78), (0.85, 0.75, 0.55), (1, 1, 0.92), (0.6, 0.9, 1.0), (0.92, 0.62, 0.70),
                                                               (0.80, 0.22, 0.20), (0.62, 0.15, 0.30), (0.95, 0.40, 0.45), (0.30, 0.60, 0.90), (0.70, 0.70, 0.62), (0.60, 0.55, 0.45), (0.72, 0.45, 0.40)] + [(0.3, 0.3, 0.3)] * 242)
    pics = sorted(index)[:: max(1, len(index) // 6)][:6]
    for c in sorted(index):
        G = game.compartments(np.load('%s/out/%s.npz' % (W, c))['labels']); index[c]['compartment_ml'] = {n: round(float((G == i).sum()) / 1000, 1) for i, n in enumerate(game.GNAMES) if i}
        if c in pics:
            ctr = (-lo).astype(int); fig, axs = plt.subplots(1, 3, figsize=(12, 4.5))
            for a, (sl, t) in zip(axs, [((ctr[0], slice(None), slice(None)), 'sagittal'), ((slice(None), slice(None), ctr[2]), 'axial'), ((slice(None), ctr[1], slice(None)), 'coronal')]):
                a.imshow(pal[G[sl]].transpose(1, 0, 2), origin='lower'); a.set_title(t); a.axis('off')
            fig.suptitle(c); plt.tight_layout(); plt.savefig('%s/out/check/%s.png' % (W, c), dpi=70); plt.close(fig)
    qn = lambda k: [v['ts_vs_expert_dice'].get(k) for v in index.values() if v['ts_vs_expert_dice'].get(k) is not None]
    summ = dict(cts=len(cts), done=len(index), failed=failed, ts_vs_expert_median={k: round(float(np.median(qn(k))), 3) for k in EXPERT.values() if qn(k)},
                compartment_ml_median={n: round(float(np.median([v['compartment_ml'][n] for v in index.values()])), 1) for n in game.GNAMES[1:]} if index else {})
    json.dump(dict(heads=index, failed=failed, ids=ids, window_lo_mm=lo.tolist(), window_hi_mm=hi.tolist(), summary=summ), open('%s/out/index_pddca_s1.json' % W, 'w'), indent=1)
    dest = 'face_v1_headframe_s1' + ('_smoke' if LIMIT else '')
    for a in range(5):
        try: api.upload_folder(folder_path=W + '/out', path_in_repo=dest, repo_id=repo, repo_type='dataset', commit_message='PDDCA heads with named face labels in the head frame (private)'); break
        except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
    log('SUMMARY ' + json.dumps(summ)); log('SUMMARY done')
