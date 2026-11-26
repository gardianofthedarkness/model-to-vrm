# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK"]
# ///
"""Are SynthRAD's head-and-neck faces intact? For a sample of CTs: in the top 140 mm of the body (the head, above the
shoulders), how far the most forward point at the midline lies ahead of the points 30 mm to each side, per axial slice;
an intact face has a nose 15-30 mm ahead, a removed face has none. Also the same for the body mask SynthRAD ships, and
whether the CT outside that mask is blanked. Numbers only, CPU. env: PER. args: <repo>"""
import os, sys, json, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def nose(B, sp, D):
    """max over head slices of (midline front - max(front 30 mm left, right)), mm; B indexed (x, y, z) in LPS"""
    ii = np.argwhere(B); top = ii[:, 2].max() if D[2, 2] > 0 else ii[:, 2].min(); zs = int(140 / sp[2])
    zr = range(max(0, top - zs), top + 1) if D[2, 2] > 0 else range(top, min(B.shape[2], top + zs + 1))
    fwd = -1 if D[1, 1] > 0 else 1; best = -1e9
    for z in zr:
        sl = B[:, :, z]
        if not sl.any(): continue
        xs = np.argwhere(sl.any(1)).ravel(); xm = (xs.min() + xs.max()) // 2; off = int(30 / sp[0])
        def f(x):
            c = np.flatnonzero(sl[x]); return (fwd * c).max() if len(c) else -1e9
        best = max(best, (f(xm) - max(f(max(0, xm - off)), f(min(sl.shape[0] - 1, xm + off)))) * sp[1])
    return float(best)
if __name__ == '__main__':
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    from huggingface_hub import HfApi, hf_hub_download
    repo = sys.argv[1]; api = HfApi(); PER = int(os.environ.get('PER', '20'))
    Z = zipfile.ZipFile(hf_hub_download(repo, 'raw/14918214/synthRAD2025_Task1_Train.zip', repo_type='dataset', local_dir='/data/sr'))
    names = Z.namelist(); cts = sorted(n for n in names if n.endswith('/ct.mha') and '/HN/' in n); rows = []
    for n in cts[:: max(1, len(cts) // PER)][:PER]:
        d = n.rsplit('/', 1)[0]; r = dict(patient=d.split('/')[-1])
        try:
            open('ct.mha', 'wb').write(Z.read(n)); im = sitk.ReadImage('ct.mha'); A = sitk.GetArrayFromImage(im).transpose(2, 1, 0); sp = im.GetSpacing(); D = np.array(im.GetDirection()).reshape(3, 3)
            B = ndi.binary_fill_holes(ndi.binary_opening(A > -300, iterations=2)); lab, _ = ndi.label(B); B = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
            r['nose_mm_ct'] = round(nose(B, sp, D), 1)
            if d + '/mask.mha' in names:
                open('m.mha', 'wb').write(Z.read(d + '/mask.mha')); Mk = sitk.GetArrayFromImage(sitk.ReadImage('m.mha')).transpose(2, 1, 0) > 0
                r['nose_mm_mask'] = round(nose(Mk, sp, D), 1); out = A[~Mk]; r['ct_outside_mask_blank'] = bool(out.size == 0 or np.percentile(out, 99) <= -900)
        except Exception as e: r['error'] = '%s %s' % (type(e).__name__, str(e)[:150])
        rows.append(r); log(r)
    ok = [r for r in rows if 'error' not in r]
    summ = dict(sampled=len(rows), nose_ct_median_mm=float(np.median([r['nose_mm_ct'] for r in ok])) if ok else None, nose_over_10mm=sum(r['nose_mm_ct'] > 10 for r in ok))
    log('SUMMARY', json.dumps(summ)); os.makedirs('counts', exist_ok=True); json.dump(dict(summary=summ, rows=rows), open('counts/synthrad_nose_check.json', 'w'), indent=1)
    api.upload_folder(folder_path='counts', path_in_repo='census', repo_id=repo, repo_type='dataset', commit_message='SynthRAD face check'); log('SUMMARY done')
