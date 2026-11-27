# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK", "pandas", "xlrd"]
# ///
"""IXI step 2a (CPU, HF Jobs; user 2026-11-27: CPU-only work goes to HF): every IXI T1 put into the input form the
MRI-to-compartment labeller (mri_game_job.py) was trained on, so the GPU step only runs the network.
Per subject: the T1 reoriented to LPS (the HaN-Seg CT grids the labeller saw are axial LPS) and resampled to 1.5 mm
isotropic (linear); the body from the MRI itself (Otsu level inside the head, largest piece, holes filled per slice);
brightness scaled as in training: 0.5th-99.5th percentile inside the body -> 0..1, clipped. Age and sex from IXI.xls;
subjects under 18 or without a recorded age are listed and left out (IXI is an adult study, the check is cheap).
Output <repo>/ixi_prep_v1/<id>.npz: M (float16), body (packed bits), shape, spacing, origin, direction; index.json with
age, sex, scanner site and failures. Uploads in batches, so a restart skips subjects already done.
env: LIMIT (smoke), WORKERS. args: <repo (ixi-t1-t2)>"""
import os, sys, json, time, tarfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
OUT = 'ixi_prep_v1'; SP = 1.5

def otsu(v, bins=256):
    v = np.asarray(v, float); h, e = np.histogram(v, bins); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; tie = np.flatnonzero(s >= np.nanmax(s) * (1 - 1e-9)); return float(c[tie].mean())

def prep(path):
    import SimpleITK as sitk
    from scipy.ndimage import label, binary_fill_holes, binary_opening
    sid = os.path.basename(path).split('-T1')[0]
    try:
        im = sitk.DICOMOrient(sitk.Cast(sitk.ReadImage(path), sitk.sitkFloat32), 'LPS')
        size = [int(round(s * z / SP)) for s, z in zip(im.GetSize(), im.GetSpacing())]
        ref = sitk.Image(size, sitk.sitkFloat32); ref.SetSpacing([SP] * 3); ref.SetOrigin(im.GetOrigin()); ref.SetDirection(im.GetDirection())
        M = sitk.GetArrayFromImage(sitk.Resample(im, ref, sitk.Transform(), sitk.sitkLinear, 0.0))
        b = M > otsu(M[M > 0]) * 0.5 if (M > 0).any() else M > 0                       # half the Otsu level: dark fat and skin stay in the body
        b = binary_opening(b, iterations=1); lab, n = label(b)
        if not n: return sid, None, 'empty'
        b = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
        for z in range(b.shape[0]): b[z] = binary_fill_holes(b[z])
        lo, hi = np.percentile(M[b], [0.5, 99.5]); Mn = np.clip((M - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float16)
        np.savez_compressed('%s/%s.npz' % (OUT, sid), M=Mn, body=np.packbits(b), shape=np.array(b.shape), spacing=np.array([SP] * 3),
                            origin=np.array(ref.GetOrigin()), direction=np.array(ref.GetDirection()))
        return sid, dict(shape=list(b.shape), body_ml=round(float(b.sum()) * SP ** 3 / 1000, 1)), None
    except Exception as e: return sid, None, '%s %s' % (type(e).__name__, str(e)[:200])

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    import pandas as pd
    api = HfApi(); repo = sys.argv[1]; os.makedirs(OUT, exist_ok=True); os.makedirs('t1', exist_ok=True)
    meta = pd.read_excel(hf_hub_download(repo, 'raw/IXI.xls', repo_type='dataset', local_dir='.'))
    meta = {int(r['IXI_ID']): dict(age=None if pd.isna(r.get('AGE')) else float(r['AGE']), sex=None if pd.isna(r.get('SEX_ID (1=m, 2=f)')) else int(r['SEX_ID (1=m, 2=f)'])) for _, r in meta.iterrows()}
    tp = hf_hub_download(repo, 'raw/IXI-T1.tar', repo_type='dataset', local_dir='.')
    with tarfile.open(tp) as tf: tf.extractall('t1', filter='data')
    files = sorted(os.path.join('t1', f) for f in os.listdir('t1') if f.endswith('.nii.gz'))
    keep, skipped = [], {}
    for f in files:
        sid = os.path.basename(f).split('-T1')[0]; num = int(sid[3:6]); m = meta.get(num, {})
        if m.get('age') is None: skipped[sid] = 'no recorded age'
        elif m['age'] < 18: skipped[sid] = 'under 18'
        else: keep.append(f)
    done = {f.split('/')[-1][:-4] for f in api.list_repo_files(repo, repo_type='dataset') if f.startswith(OUT + '/') and f.endswith('.npz')}
    todo = [f for f in keep if os.path.basename(f).split('-T1')[0] not in done]
    if os.environ.get('LIMIT'): todo = todo[:int(os.environ['LIMIT'])]
    log(len(files), 'T1 scans;', len(keep), 'adults with an age;', len(skipped), 'left out;', len(done), 'already done;', len(todo), 'to do')
    nw = int(os.environ.get('WORKERS', os.cpu_count() or 4)); res, fail = {}, {}
    def flush():
        for a in range(5):
            try: api.upload_folder(folder_path=OUT, path_in_repo=OUT, repo_id=repo, repo_type='dataset', allow_patterns=['*.npz'], commit_message='IXI prep batch'); break
            except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
        for f in os.listdir(OUT):
            if f.endswith('.npz'): os.remove(os.path.join(OUT, f))
    with Pool(nw) as p:
        for i, (sid, info, err) in enumerate(p.imap_unordered(prep, todo)):
            if err: fail[sid] = err
            else: res[sid] = dict(info, **meta.get(int(sid[3:6]), {}), site=sid.split('-')[1] if '-' in sid else None)
            if (i + 1) % 50 == 0: log(i + 1, 'of', len(todo)); flush()
    flush()
    old = {}
    try: old = json.load(open(hf_hub_download(repo, OUT + '/index.json', repo_type='dataset', local_dir='prev')))['subjects']
    except Exception: pass
    json.dump(dict(subjects=dict(old, **res), failed=fail, left_out=skipped, voxel_mm=SP), open('index.json', 'w'), indent=1)
    api.upload_file(path_or_fileobj='index.json', path_in_repo=OUT + '/index.json', repo_id=repo, repo_type='dataset', commit_message='IXI prep index')
    log('SUMMARY prepared', len(res), 'failed', len(fail), fail); log('SUMMARY done')
