# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel"]
# ///
"""Where does CT brightness vary? Pelvis region of training-split adults (the stage A window), inside the body.
Classes by brightness after a 3x3x3 median (so noise does not change a voxel's class): gas < -500; fat -190..-30;
lean -29..150; dense 150..Otsu-bone; bone > Otsu-bone (the scan set's own Otsu above lean). Per voxel HU = class mean
over all people + person's class offset + voxel residual. Reports the share of total variance:
  between classes (tissue type), between people within a class (density), and within a person and class
  (texture + noise), with noise = raw minus 3x3x3 median. Also cortical-wall thickness of the femoral shaft below the
  lesser trochanter where the femur label allows (for the blur bound in the proposal). args: <repo> [n people]"""
import os, sys, csv, json, gzip, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
CL = ['gas', 'fat', 'lean', 'dense', 'bone']
_Z = None
def _init(zp):
    global _Z; _Z = zipfile.ZipFile(zp)
def one(arg):
    import nibabel as nib
    from scipy.ndimage import median_filter, label as cc, binary_fill_holes, binary_erosion, distance_transform_edt
    pre, case, bone_t = arg
    try:
        rd = lambda n: np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(pre + n))).dataobj)
        CT = rd('ct.nii.gz').astype(np.float32); HIP = np.zeros(CT.shape, bool)
        for b in ('hip_left', 'hip_right', 'sacrum'): HIP |= rd('segmentations/%s.nii.gz' % b) > 0
        zs = np.flatnonzero(HIP.any((0, 1))); sub = CT[:, :, zs.min():zs.max() + 1]; H = HIP[:, :, zs.min():zs.max() + 1]
        body = sub > -500; M = np.zeros(sub.shape, bool)
        for z in range(sub.shape[2]):
            lab, _ = cc(body[:, :, z]); ids = np.unique(lab[H[:, :, z]]); ids = ids[ids > 0]
            if len(ids): M[:, :, z] = binary_fill_holes(np.isin(lab, ids))
        S = median_filter(sub, 3); v = sub[M]; s = S[M]
        c = np.select([s < -500, (s >= -190) & (s <= -30), (s > -30) & (s <= 150), (s > 150) & (s <= bone_t), s > bone_t], [0, 1, 2, 3, 4], -1)
        out = dict(case=case, n=int(M.sum()), noise_sd=float(np.std(v - s)))
        for k, name in enumerate(CL):
            m = c == k; out[name] = dict(n=int(m.sum()), mean=float(v[m].mean()) if m.any() else None, var=float(v[m].var()) if m.sum() > 1 else None,
                                         noise_var=float(np.var(v[m] - s[m])) if m.sum() > 1 else None)
        # femoral cortex thickness: shaft slices of each femur label (the lowest quarter present), area of bone / outer perimeter
        th = []
        for side in ('left', 'right'):
            try: F = rd('segmentations/femur_%s.nii.gz' % side) > 0
            except KeyError: continue
            fz = np.flatnonzero(F.any((0, 1)))
            if len(fz) < 20: continue
            sp = np.linalg.norm(nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(pre + 'ct.nii.gz'))).affine[:3, :3], axis=0)
            for z in fz[:max(1, len(fz) // 4)]:
                sl = CT[:, :, z]; f = F[:, :, z]; bone = (median_filter(sl, 3) > bone_t) & f
                outer = binary_fill_holes(bone); per = (outer & ~binary_erosion(outer)).sum() * sp[0]
                if per > 0 and bone.sum() > 10 and (outer & ~bone).sum() > 5: th.append(bone.sum() * sp[0] * sp[1] / per)
        out['cortex_mm'] = float(np.median(th)) if th else None
        return out
    except Exception as e: return dict(case=case, failed='%s %s' % (type(e).__name__, str(e)[:150]))

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); repo = sys.argv[1]; npeople = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    ext = list(csv.DictReader(open(hf_hub_download(repo, 'census/label_extents.csv', repo_type='dataset'))))
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    by = {}
    for e in ext: by.setdefault(e['case'], {})[e['label']] = e
    whole = lambda e: bool(e) and e['present'] == '1' and not any(e[k] == '1' for k in ('cut_inf', 'cut_sup', 'cut_side', 'cut_ap'))
    sel = sorted(c for c, d in by.items() if whole(d.get('hip_left')) and whole(d.get('hip_right')) and whole(d.get('sacrum')) and meta[c]['split'] == 'train' and not (meta[c].get('age') and float(meta[c]['age']) < 18))
    sel = list(np.random.default_rng(0).choice(sel, min(npeople, len(sel)), replace=False))
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); names = [i.filename for i in zipfile.ZipFile(zp).infolist() if i.filename.endswith('/ct.nii.gz')]
    pre = {n[:-len('ct.nii.gz')].rstrip('/').rsplit('/', 1)[-1]: n[:-len('ct.nii.gz')] for n in names}
    bone_t = 517.5                                                                            # the stage A scan set's own Otsu bone threshold (logged by that run)
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 16
    nw = max(1, min(int(cpus), int(ram // 4))); log(len(sel), 'people,', nw, 'workers')
    with Pool(nw, _init, (zp,), maxtasksperchild=10) as p: rows = list(p.imap_unordered(one, [(pre[c], c, bone_t) for c in sel]))
    ok = [r for r in rows if 'failed' not in r]; log('ok', len(ok), 'failed', [r for r in rows if 'failed' in r][:3])
    # variance decomposition, voxel-weighted
    N = sum(r['n'] for r in ok); grand = sum(r[c]['mean'] * r[c]['n'] for r in ok for c in CL if r[c]['n']) / N
    cm = {c: sum(r[c]['mean'] * r[c]['n'] for r in ok if r[c]['n']) / max(sum(r[c]['n'] for r in ok), 1) for c in CL}
    between_class = sum(r[c]['n'] * (cm[c] - grand) ** 2 for r in ok for c in CL if r[c]['n']) / N
    between_people = sum(r[c]['n'] * (r[c]['mean'] - cm[c]) ** 2 for r in ok for c in CL if r[c]['n']) / N
    within = sum(r[c]['n'] * r[c]['var'] for r in ok for c in CL if r[c]['n'] > 1) / N
    noise = sum(r[c]['n'] * r[c]['noise_var'] for r in ok for c in CL if r[c]['n'] > 1) / N
    tot = between_class + between_people + within
    res = dict(people=len(ok), voxels=int(N), class_mean_hu={c: round(cm[c], 1) for c in CL}, class_share={c: round(sum(r[c]['n'] for r in ok) / N, 4) for c in CL},
               variance_hu2=dict(total=round(tot, 1), between_classes=round(between_class, 1), between_people_within_class=round(between_people, 1), within_person_and_class=round(within, 1), of_which_noise=round(noise, 1)),
               share=dict(between_classes=round(between_class / tot, 4), between_people_within_class=round(between_people / tot, 4), within_person_and_class=round(within / tot, 4), noise=round(noise / tot, 4)),
               per_class_sd_between_people={c: round(float(np.std([r[c]['mean'] for r in ok if r[c]['n']])), 1) for c in CL},
               per_class_sd_within={c: round(float(np.sqrt(np.average([r[c]['var'] for r in ok if r[c]['n'] > 1], weights=[r[c]['n'] for r in ok if r[c]['n'] > 1]))), 1) for c in CL},
               noise_sd_median=round(float(np.median([r['noise_sd'] for r in ok])), 1),
               femur_cortex_mm=dict(median=round(float(np.median([r['cortex_mm'] for r in ok if r['cortex_mm']])), 2), n=sum(1 for r in ok if r['cortex_mm']), p10=round(float(np.percentile([r['cortex_mm'] for r in ok if r['cortex_mm']], 10)), 2), p90=round(float(np.percentile([r['cortex_mm'] for r in ok if r['cortex_mm']], 90)), 2)))
    os.makedirs('var', exist_ok=True); json.dump(dict(summary=res, rows=rows), open('var/variance.json', 'w'), indent=1)
    api.upload_folder(folder_path='var', path_in_repo='variance_check', repo_id=repo, repo_type='dataset', commit_message='variance check')
    log('SUMMARY ' + json.dumps(res))
