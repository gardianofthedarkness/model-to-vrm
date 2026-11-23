# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "matplotlib"]
# ///
"""Does tissue brightness (HU) within one tissue type follow age, sex, build and scanner? CPU only.

Per adult scan, where the structure is present (labels are TotalSegmentator's; each label is eroded by 2 voxels so
edges and partial volume are left out; medians):
  bone_L1      spongy bone of the L1 vertebral body (the label's core)
  bone_femhead the top tenth of each femur label (femoral head), core
  muscle       gluteus maximus, iliopsoas and autochthon (back) muscles, left and right, core
  fat_sc       subcutaneous fat at the L3 level: voxels within 30 mm of the skin and inside the fat range
  aorta        aorta core (contrast agent marker)
Build, from the outline only, at L3: body cross-section area (cm2) and fat share of the outer 30 mm.
Facts: age, sex, tube voltage (kVp), manufacturer.
Outputs: per-person table and correlations (Spearman) and linear fits (R2 from facts, from facts + build, with and
without scanner terms), plus a scatter figure. args: <repo> [limit]"""
import os, sys, io, csv, json, gzip, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
FAT = (-190, -30)
MUSCLES = [m + s for m in ('gluteus_maximus_', 'iliopsoas_', 'autochthon_') for s in ('left', 'right')]
WANT = ['vertebrae_L1', 'vertebrae_L3', 'femur_left', 'femur_right', 'aorta'] + MUSCLES
_Z = None; _NAMES = None
def _init(zp):
    global _Z, _NAMES; _Z = zipfile.ZipFile(zp); _NAMES = set(_Z.namelist())
def core(L):
    from scipy.ndimage import binary_erosion
    return binary_erosion(L, iterations=2)
def measure(arg):
    import nibabel as nib
    from scipy.ndimage import label, binary_fill_holes, distance_transform_edt
    pre, case = arg
    try:
        rd = lambda n: np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(pre + n))).dataobj)
        img = nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(pre + 'ct.nii.gz'))); CT = np.asanyarray(img.dataobj).astype(np.float32); sp = np.linalg.norm(img.affine[:3, :3], axis=0)
        out = dict(case=case); muscle = []
        for name in WANT:
            f = 'segmentations/%s.nii.gz' % name
            if pre + f not in _NAMES: continue
            L = rd(f) > 0
            if not L.any(): continue
            if name == 'vertebrae_L1': c = core(L); out['bone_L1'] = float(np.median(CT[c])) if c.sum() > 50 else None
            elif name == 'vertebrae_L3': out['_l3z'] = np.flatnonzero(L.any((0, 1)))
            elif name.startswith('femur_'):
                p = np.argwhere(L); z = p[:, 2]; top = z >= z.max() - 0.1 * (z.max() - z.min()); H = np.zeros_like(L); H[tuple(p[top].T)] = True; c = core(H)
                if c.sum() > 50: out.setdefault('_fh', []).append(float(np.median(CT[c])))
            elif name == 'aorta': c = core(L); out['aorta'] = float(np.median(CT[c])) if c.sum() > 50 else None
            else:
                c = core(L)
                if c.sum() > 50: muscle.append(CT[c][::7])
            del L
        if '_fh' in out: out['bone_femhead'] = float(np.mean(out.pop('_fh')))
        if muscle: out['muscle'] = float(np.median(np.concatenate(muscle)))
        zs = out.pop('_l3z', None)
        if zs is not None and len(zs):
            z0 = int(np.median(zs)); sl = CT[:, :, max(z0 - 2, 0):z0 + 3]
            B = sl > -500; lab, n = label(B[..., B.shape[2] // 2])
            if n:
                M2 = binary_fill_holes(lab == np.bincount(lab.ravel())[1:].argmax() + 1); d = distance_transform_edt(M2, sampling=sp[:2])
                band = M2 & (d <= 30); s2 = sl[..., sl.shape[2] // 2]; fat = band & (s2 >= FAT[0]) & (s2 <= FAT[1])
                out['body_area_cm2'] = float(M2.sum() * sp[0] * sp[1] / 100); out['outer_fat_share'] = float(fat.sum() / max(band.sum(), 1))
                out['fat_sc'] = float(np.median(s2[fat])) if fat.sum() > 50 else None
        return out
    except Exception as e: return dict(case=case, failed='%s %s' % (type(e).__name__, str(e)[:150]))

def analyse(rows, outdir):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    num = lambda r, k: float(r[k]) if r.get(k) not in (None, '') else np.nan
    res = {}
    targets = ['bone_L1', 'bone_femhead', 'muscle', 'fat_sc']
    preds = dict(age='age', male='male', kvp='kvp', aorta_contrast='aorta', body_area='body_area_cm2', outer_fat_share='outer_fat_share')
    for t in targets:
        y = np.array([num(r, t) for r in rows]); res[t] = dict(n=int(np.isfinite(y).sum()), median=float(np.nanmedian(y)), spearman={})
        for nm, k in preds.items():
            x = np.array([num(r, k) for r in rows]); m = np.isfinite(x) & np.isfinite(y)
            if m.sum() > 30: rho, pv = spearmanr(x[m], y[m]); res[t]['spearman'][nm] = dict(rho=round(float(rho), 3), p=float(pv), n=int(m.sum()))
        def r2(cols):                                                                         # least squares with intercept, 5-fold cross-validated R2
            X = np.column_stack([np.array([num(r, preds[c]) for r in rows]) for c in cols]); m = np.isfinite(X).all(1) & np.isfinite(y)
            if m.sum() < 60: return None
            X, Y = np.c_[np.ones(m.sum()), X[m]], y[m]; idx = np.random.default_rng(0).permutation(len(Y)); fold = np.array_split(idx, 5); pred = np.empty_like(Y)
            for f in fold:
                tr = np.setdiff1d(idx, f); b = np.linalg.lstsq(X[tr], Y[tr], rcond=None)[0]; pred[f] = X[f] @ b
            return dict(r2=round(float(1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()), 3), n=int(m.sum()), resid_sd=round(float(np.std(Y - pred)), 1), sd=round(float(np.std(Y)), 1))
        res[t]['fit'] = {'scanner only (kVp, contrast)': r2(['kvp', 'aorta_contrast']), 'facts (age, sex)': r2(['age', 'male']), 'facts + scanner': r2(['age', 'male', 'kvp', 'aorta_contrast']),
                         'outline only (area, outer fat)': r2(['body_area', 'outer_fat_share']), 'facts + scanner + outline': r2(['age', 'male', 'kvp', 'aorta_contrast', 'body_area', 'outer_fat_share'])}
    f, a = plt.subplots(2, 3, figsize=(15, 9))
    sx = lambda r: 'tab:red' if r.get('male') == 0 else 'tab:blue'
    for x_, (xk, yk, t) in zip(a.ravel(), (('age', 'bone_L1', 'L1 spongy bone'), ('age', 'bone_femhead', 'femoral head'), ('age', 'muscle', 'muscle'),
                                         ('outer_fat_share', 'muscle', 'muscle vs outer fat share'), ('body_area_cm2', 'fat_sc', 'subcutaneous fat vs body area'), ('kvp', 'bone_L1', 'L1 bone vs tube voltage'))):
        pts = [(num(r, xk), num(r, yk), sx(r)) for r in rows]; pts = [p for p in pts if np.isfinite(p[0]) and np.isfinite(p[1])]
        if pts: x_.scatter([p[0] for p in pts], [p[1] for p in pts], s=5, c=[p[2] for p in pts], alpha=.5)
        x_.set_xlabel(xk); x_.set_ylabel(yk + ' (HU)'); x_.set_title(t + '  (blue men, red women)', fontsize=10)
    f.tight_layout(); f.savefig(os.path.join(outdir, 'density_vs_facts.png'), dpi=110); plt.close(f)
    return res

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); names = [i.filename for i in zipfile.ZipFile(zp).infolist() if i.filename.endswith('/ct.nii.gz')]
    pre = {n[:-len('ct.nii.gz')].rstrip('/').rsplit('/', 1)[-1]: n[:-len('ct.nii.gz')] for n in names}
    cases = sorted(c for c in pre if c in meta and not (meta[c].get('age') and float(meta[c]['age']) < 18))
    if limit: cases = cases[:limit]
    def limits():
        try:
            q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per) if q != 'max' else os.cpu_count()
        except Exception: cpus = os.cpu_count() or 4
        try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
        except Exception: ram = 16
        return cpus, ram
    cpus, ram = limits(); nw = max(1, min(int(cpus), int(ram // 4))); log('%d scans, %d workers (container cpus %.0f, ram %.0f GB)' % (len(cases), nw, cpus, ram))
    rows = []
    with Pool(nw, _init, (zp,), maxtasksperchild=25) as p:
        for k, r in enumerate(p.imap_unordered(measure, [(pre[c], c) for c in cases])):
            m = meta[r['case']]; r.update(age=float(m['age']) if m.get('age') else None, male={'m': 1, 'f': 0}.get(m.get('gender')), kvp=float(m['kvp']) if m.get('kvp') else None, maker=m.get('manufacturer')); rows.append(r)
            if (k + 1) % 100 == 0: log('measured', k + 1)
    os.makedirs('dens', exist_ok=True); res = analyse([r for r in rows if 'failed' not in r], 'dens'); res['failed'] = [r for r in rows if 'failed' in r]
    keys = sorted({k for r in rows for k in r}); b = io.StringIO(); w = csv.DictWriter(b, keys); w.writeheader(); w.writerows(rows); open('dens/per_person.csv', 'w').write(b.getvalue())
    json.dump(res, open('dens/density_summary.json', 'w'), indent=1)
    api.upload_folder(folder_path='dens', path_in_repo='density_check' + ('_smoke' if limit else ''), repo_id=repo, repo_type='dataset', commit_message='density check')
    log('SUMMARY ' + json.dumps({t: dict(n=v['n'], median=v['median'], fit=v['fit'], spearman={k: x['rho'] for k, x in v['spearman'].items()}) for t, v in res.items() if t != 'failed'}))
