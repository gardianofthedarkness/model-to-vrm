# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "nibabel"]
# ///
"""Which adult MRI scans (TotalSegmentator MRI v3.0.0) reach the ankle or the wrist? CPU only.
Adults = recorded age >= 18 (scans without an age are excluded). Per scan: image size, voxel size (slice thickness),
study type, sequence; the femur and humerus labels' far ends (towards knee and elbow, found as the end away from the
hip bone or scapula), and how far the image continues past them along the long axis (mm). Dedicated limb scans
(knee, lower leg, ankle, wrist, forearm, hand) are reported by study type with their size. Counts: reaching the ankle
(at least 380 mm past the femur's far end, or an ankle or lower-leg study) and the wrist (at least 250 mm past the
humerus's far end, or a wrist, forearm or hand study). args: <repo of the MRI set>"""
import os, sys, csv, json, gzip, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
_Z = None; _N = None
def _init(zp):
    global _Z, _N; _Z = zipfile.ZipFile(zp); _N = set(_Z.namelist())
def ends(L, P, axis):
    """far end of bone L along `axis` (index), away from the proximal bone P; None if absent."""
    if not L.any() or not P.any(): return None
    idx = np.flatnonzero(L.any(tuple(a for a in range(3) if a != axis))); pc = np.flatnonzero(P.any(tuple(a for a in range(3) if a != axis))).mean()
    return (int(idx.max()), 1) if idx.mean() > pc else (int(idx.min()), -1)
def one(arg):
    import nibabel as nib
    case, img_name = arg
    try:
        pre = img_name.rsplit('/', 1)[0] + '/'
        im = nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(img_name))); shp = im.shape[:3]; sp = np.linalg.norm(im.affine[:3, :3], axis=0)
        axis = int(np.argmax(np.array(shp) * sp))                                             # the scan's long axis
        rd = lambda n: (np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(pre + 'segmentations/%s.nii.gz' % n))).dataobj) > 0) if (pre + 'segmentations/%s.nii.gz' % n) in _N else None
        out = dict(case=case, shape=[int(x) for x in shp], voxel_mm=[round(float(x), 2) for x in sp], long_axis_mm=round(float(shp[axis] * sp[axis]), 1))
        for limb, prox, key in (('femur', 'hip', 'past_knee_mm'), ('humerus', 'scapula', 'past_elbow_mm')):
            best = None
            for side in ('left', 'right'):
                L, P = rd('%s_%s' % (limb, side)), rd('%s_%s' % (prox, side))
                if L is None or P is None: continue
                e = ends(L, P, axis)
                if e is None: continue
                beyond = ((shp[axis] - 1 - e[0]) if e[1] == 1 else e[0]) * sp[axis]; best = beyond if best is None else max(best, beyond)
            out[key] = None if best is None else round(float(best), 1)
        return out
    except Exception as e: return dict(case=case, failed='%s %s' % (type(e).__name__, str(e)[:150]))

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); repo = sys.argv[1]
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/Totalsegmentator_dataset_v300__meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    raw = [f for f in api.list_repo_files(repo, repo_type='dataset') if f.startswith('raw/')]; log('raw files', raw)
    zp = hf_hub_download(repo, raw[0], repo_type='dataset', local_dir='.'); names = zipfile.ZipFile(zp).namelist()
    imgs = {}
    for n in names:
        if n.endswith('.nii.gz') and '/segmentations/' not in n:
            case = n.rsplit('/', 2)[-2]
            if case in meta: imgs.setdefault(case, n)
    adults = sorted(c for c in imgs if meta[c].get('age') and float(meta[c]['age']) >= 18); log('images', len(imgs), 'adults with age', len(adults), 'example', next(iter(imgs.values())) if imgs else None)
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    with Pool(max(1, int(cpus)), _init, (zp,), maxtasksperchild=50) as p: rows = list(p.imap_unordered(one, [(c, imgs[c]) for c in adults]))
    for r in rows:
        m = meta[r['case']]; r.update(study=m.get('study_type'), sequence=m.get('scanning_sequence'), slice_thickness=m.get('slice_thickness'), field=m.get('magnetic_field_strength'), age=m.get('age'), sex=m.get('gender'), split=m.get('split'))
    ok = [r for r in rows if 'failed' not in r]
    st = lambda r, ks: any(k in (r.get('study') or '') for k in ks)
    ankle = [r for r in ok if (r.get('past_knee_mm') or 0) >= 380 or st(r, ('ankle', 'lower leg'))]
    wrist = [r for r in ok if (r.get('past_elbow_mm') or 0) >= 250 or st(r, ('wrist', 'forearm', 'hand'))]
    def desc(rs):
        th = [float(r['slice_thickness']) for r in rs if r.get('slice_thickness') not in (None, '')]
        return dict(people=len(rs), by_study={s: sum(1 for r in rs if r['study'] == s) for s in sorted({r['study'] for r in rs})}, slice_thickness_median=float(np.median(th)) if th else None,
                    in_plane_mm_median=float(np.median([min(r['voxel_mm']) for r in rs])) if rs else None, train=sum(1 for r in rs if r['split'] == 'train'), test=sum(1 for r in rs if r['split'] == 'test'))
    res = dict(adults_scanned=len(ok), failed=len(rows) - len(ok), reaching_ankle=desc(ankle), reaching_wrist=desc(wrist),
               past_knee_mm_quantiles=np.percentile([r['past_knee_mm'] for r in ok if r.get('past_knee_mm') is not None], [50, 75, 90, 95, 99]).round(0).tolist() if any(r.get('past_knee_mm') is not None for r in ok) else None)
    os.makedirs('mri_census', exist_ok=True); json.dump(dict(summary=res, rows=rows), open('mri_census/limb_coverage.json', 'w'), indent=1)
    api.upload_folder(folder_path='mri_census', path_in_repo='census/limb_coverage', repo_id=repo, repo_type='dataset', commit_message='limb coverage census')
    log('SUMMARY ' + json.dumps(res))
