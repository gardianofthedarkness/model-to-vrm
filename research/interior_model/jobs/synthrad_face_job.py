# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK", "nibabel", "TotalSegmentator"]
# ///
"""SynthRAD2025 head-and-neck CTs as tissue-type heads in the head frame (plan v8 section 14), in one pass on one card.
Per CT (read straight from the private copy of the training archive):
  1. TotalSegmentator's licensed 'head_glands_cavities' model (the only model run): eyes and lenses, auditory canals.
     Several CTs run at the same time on the card (each in its own process).
  2. Tissue from CT brightness, as the face set's plain classes: outside 0, fat under the skin 1 (fat connected to the
     outside), other fat 2, lean 3, air 4, bone 5; eyes and lenses = the eye class; the space inside the skull (filled from
     bone, as in H3b) = the organ class (brain). No muscle, bone or gland models: names come later from Z-Anatomy fits.
  3. Head frame from the ear canals and eyes exactly as face_frames.py; labels resampled at 1 mm into the existing window
     of face_v1_headframe (index.json), outside the scan = unknown (255).
Output: <repo>/face_v1_headframe/<case>.npz and index_synthrad.json (heads marked tissue_only; split by index: every tenth
test, the next tenth validation). Numbers only leave as JSON. env: LIMIT, PAR (CTs at once, default 3), TOTALSEG_LICENSE.
SOURCE=pddca reads PDDCA (raw/data-part001.zip, Images/*.nii.gz) instead; output index_<source>.json.
args: <synthrad or pddca repo> <repo holding face_v1_headframe/index.json>"""
import os, sys, json, time, zipfile, subprocess, numpy as np
from concurrent.futures import ThreadPoolExecutor
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
W = '/data/w'
SRC = os.environ.get('SOURCE', 'synthrad')                                                     # synthrad | pddca

def worker(case, ctp, outp):
    """one CT: segmentation (GPU) in this separate process"""
    from totalsegmentator.python_api import totalsegmentator
    import nibabel as nib
    seg = totalsegmentator(ctp, None, task='head_glands_cavities', quiet=True, device='gpu', nr_thr_resamp=1, nr_thr_saving=1)
    np.savez_compressed(outp, seg=np.asarray(seg.dataobj).astype(np.uint8))

def build(case, ctp, segp, ids, lo, hi, inv):
    import SimpleITK as sitk
    from scipy.ndimage import median_filter, label, binary_fill_holes, binary_dilation, map_coordinates
    im = sitk.ReadImage(ctp); A = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32); S = np.load(segp)['seg']
    D = np.array(im.GetDirection()).reshape(3, 3) * np.array(im.GetSpacing()); org = np.array(im.GetOrigin()); aff = np.eye(4); aff[:3, :3] = D; aff[:3, 3] = org
    C = median_filter(A, 3); lab, n = label(C > -500); body = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    for z in range(body.shape[2]): body[:, :, z] = binary_fill_holes(body[:, :, z])
    L = np.zeros(C.shape, np.uint8); fat = body & (C >= -190) & (C <= -30); fl, _ = label(fat | ~body); oid = np.bincount(fl[~body].ravel()).argmax() if (~body).any() else -1
    L[body & (C > -30) & (C <= 150)] = 3; L[body & (C < -500)] = 4; L[body & (C > 150)] = 5; L[fat] = 2; L[fat & (fl == oid)] = 1
    bone = binary_dilation(C > 300, iterations=1); cav = np.zeros_like(bone)
    for z in range(bone.shape[2]): cav[:, :, z] = binary_fill_holes(bone[:, :, z]) & ~bone[:, :, z]
    cl, cn = label(cav)
    if cn: L[(cl == (np.bincount(cl.ravel())[1:].argmax() + 1)) & body & (L != 5)] = ids['brain']
    for nm in ('eye_left', 'eye_right', 'eye_lens_left', 'eye_lens_right'):
        if nm in inv: L[(S == inv[nm]) & body] = ids['eye_left']
    cent = {}
    for nm in ('auditory_canal_left', 'auditory_canal_right', 'eye_left', 'eye_right'):
        ii = np.argwhere(S == inv[nm]) if nm in inv else np.zeros((0, 3))
        if len(ii) < 10: return None, 'landmark missing: ' + nm
        cent[nm] = aff[:3, :3] @ ii.mean(0) + aff[:3, 3]
    o = (cent['auditory_canal_left'] + cent['auditory_canal_right']) / 2; x = cent['auditory_canal_right'] - cent['auditory_canal_left']; x /= np.linalg.norm(x)
    y = (cent['eye_left'] + cent['eye_right']) / 2 - o; y -= x * (x @ y); y /= np.linalg.norm(y); R = np.stack([x, y, np.cross(x, y)])
    shape = (hi - lo).astype(int); g = np.stack(np.meshgrid(*[np.arange(n) + 0.5 for n in shape], indexing='ij'), -1).reshape(-1, 3) + lo
    idx = (g @ R + o - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T
    Lf = map_coordinates(L, idx.T, order=0, mode='constant', cval=255).reshape(shape).astype(np.uint8)
    np.savez_compressed('%s/out/%s.npz' % (W, case), labels=Lf)
    known = [0, 1, 2, 3, 4, 5, ids['eye_left'], ids['brain']]
    return dict(source=SRC, tissue_only=True, native_mm=[round(float(s), 3) for s in im.GetSpacing()], known=known, origin_mm=o.round(3).tolist(), axes=R.round(5).tolist(),
                body_voxels_1mm=int(((Lf > 0) & (Lf < 255)).sum()), outside_scan_share=round(float((Lf == 255).mean()), 3), eye_ml=round(float((Lf == ids['eye_left']).sum()) / 1000, 1)), None

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--one': worker(sys.argv[2], sys.argv[3], sys.argv[4]); sys.exit(0)
if __name__ == '__main__':
    from huggingface_hub import HfApi, hf_hub_download
    from totalsegmentator.map_to_binary import class_map
    repo, frepo = sys.argv[1:3]; api = HfApi(); LIMIT = int(os.environ.get('LIMIT', '0')); PAR = int(os.environ.get('PAR', '3'))
    key = os.environ.get('TOTALSEG_LICENSE')
    if key: r = subprocess.run(['totalseg_set_license', '-l', key], capture_output=True, text=True); log('licence set:', r.returncode == 0)
    fi = json.load(open(hf_hub_download(frepo, 'face_v1_headframe/index.json', repo_type='dataset'))); ids = fi['ids']; lo, hi = np.array(fi['window_lo_mm']), np.array(fi['window_hi_mm'])
    inv = {v: int(k) for k, v in class_map['head_glands_cavities'].items()}; log('window', lo.tolist(), hi.tolist(), 'model classes', sorted(inv))
    if SRC == 'pddca':
        Z = zipfile.ZipFile(hf_hub_download(repo, 'raw/data-part001.zip', repo_type='dataset', local_dir='/data/sr')); cts = sorted(n for n in Z.namelist() if n.startswith('Images/') and n.endswith('.nii.gz'))
    else:
        Z = zipfile.ZipFile(hf_hub_download(repo, 'raw/14918214/synthRAD2025_Task1_Train.zip', repo_type='dataset', local_dir='/data/sr')); cts = sorted(n for n in Z.namelist() if n.endswith('/ct.mha') and '/HN/' in n)
    if LIMIT: cts = cts[:: max(1, len(cts) // LIMIT)][:LIMIT]
    os.makedirs(W + '/out', exist_ok=True); log(len(cts), 'head-and-neck CTs,', PAR, 'at a time')
    cases = [('pd_' + os.path.basename(n)[:-7], n) if SRC == 'pddca' else ('sr_' + n.rsplit('/', 2)[-2], n) for n in cts]                                     # each CT is unpacked just before its turn and deleted after (memory)
    import threading; zlock = threading.Lock()
    def unpack(case, n):
        import SimpleITK as sitk
        m = '%s/%s.mha' % (W, case)
        if n.endswith('.nii.gz'):
            with zlock: open('%s/%s.nii.gz' % (W, case), 'wb').write(Z.read(n))
            return '%s/%s.nii.gz' % (W, case)
        with zlock: open(m, 'wb').write(Z.read(n))
        p = '%s/%s.nii.gz' % (W, case); sitk.WriteImage(sitk.ReadImage(m), p); os.remove(m); return p   # TotalSegmentator reads NIfTI, not .mha
    index, failed = {}, {}; ilock = threading.Lock()
    def one_case(cp):
        """unpack, segment (own process), build, log, delete: nothing waits for the other cases"""
        case, n = cp
        try:
            p = unpack(case, n); sp_ = '%s/%s_seg.npz' % (W, case)
            q = subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--one', case, p, sp_], capture_output=True, text=True, timeout=1800)
            if q.returncode != 0: m, why = None, ('out of memory (killed)' if q.returncode in (-9, 137) else 'exit %d: %s' % (q.returncode, q.stderr.strip()[-200:]))
            else: m, why = build(case, p, sp_, ids, lo, hi, inv)
        except Exception as e: m, why = None, '%s %s' % (type(e).__name__, str(e)[:150])
        for f_ in ('%s/%s.nii.gz' % (W, case), '%s/%s_seg.npz' % (W, case)):
            if os.path.exists(f_): os.remove(f_)
        with ilock:
            if m is None: failed[case] = why
            else: index[case] = m
        log(case, m if m else why)
    if cases: one_case(cases[0])                                                                  # the first CT alone: it downloads the model weights once
    with ThreadPoolExecutor(PAR) as ex: list(ex.map(one_case, cases[1:]))
    for i, c in enumerate(sorted(index)): index[c]['split'] = 'test' if i % 10 == 0 else 'val' if i % 10 == 1 else 'train'
    json.dump(dict(heads=index, failed=failed), open('%s/out/index_%s.json' % (W, SRC), 'w'), indent=1)
    log('SUMMARY', json.dumps(dict(cts=len(cts), done=len(index), failed=len(failed), eye_ml_median=float(np.median([m['eye_ml'] for m in index.values()])) if index else None,
                                    outside_scan_median=float(np.median([m['outside_scan_share'] for m in index.values()])) if index else None)))
    dest = 'face_v1_headframe' + ('_smoke' if LIMIT else '')
    for a in range(5):
        try: api.upload_folder(folder_path=W + '/out', path_in_repo=dest, repo_id=repo, repo_type='dataset', commit_message=SRC + ' heads in head frame (tissue types)'); break
        except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
    log('SUMMARY done')
