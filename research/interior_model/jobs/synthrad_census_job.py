# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "SimpleITK", "TotalSegmentator"]
# ///
"""Face coverage of SynthRAD2025's head-and-neck MRI/CT pairs (private copy, CC BY-NC 4.0), on a sample. Reads the
training archive from <repo>/raw/14918214 inside the job, takes each head-and-neck patient's CT, and records numbers only:
extent, whether the face has a nose (a removed face has none), and whether skull, mandible, upper teeth and maxillary
sinuses are present and whole (TotalSegmentator's licensed craniofacial model). Also checks that an MRI is present.
env: PER (patients, default 20), TOTALSEG_LICENSE (secret). args: <repo>"""
import os, sys, json, time, zipfile, subprocess, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
WATCH = ['skull', 'mandible', 'teeth_upper', 'sinus_maxillary']

def one(nii, outj):
    """worker in its own process: extent, nose, craniofacial structures."""
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    im = sitk.ReadImage(nii); sp = im.GetSpacing(); sz = im.GetSize(); nsz = [max(1, int(round(a * b / 1.5))) for a, b in zip(sz, sp)]
    im = sitk.Resample(im, nsz, sitk.Transform(), sitk.sitkLinear, im.GetOrigin(), (1.5, 1.5, 1.5), im.GetDirection(), -1024, sitk.sitkInt16)
    sitk.WriteImage(im, 'w/ct15.nii.gz'); A = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32); del im
    B = ndi.binary_fill_holes(ndi.binary_opening(A > -300, iterations=2)); lab, _ = ndi.label(B); B = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    ii = np.argwhere(B); lo, hi = ii.min(0), ii.max(0); r = dict(extent_mm=[round(float((h - l + 1) * 1.5)) for l, h in zip(lo, hi)], body_cut_bottom=bool(lo[2] == 0))
    seg = np.asarray(totalsegmentator('w/ct15.nii.gz', None, task='craniofacial_structures', quiet=True, device='gpu', nr_thr_resamp=1, nr_thr_saving=1).dataobj)
    inv = {v: k for k, v in class_map['craniofacial_structures'].items()}
    for n in WATCH:
        if n not in inv: continue
        m = seg == inv[n]
        if m.sum() < 20: r[n] = 'absent'; continue
        jj = np.argwhere(m); a, b = jj.min(0), jj.max(0); r[n] = 'cut' if bool((a == 0).any() or (b == np.array(seg.shape) - 1).any()) else 'whole'
    json.dump(r, open(outj, 'w'))


def nose_mm(nii):
    """how far the most forward point of the lower half of the face lies ahead of the points 30 mm to each side"""
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    im = sitk.ReadImage(nii); sp = im.GetSpacing(); A = sitk.GetArrayFromImage(im).transpose(2, 1, 0)
    B = ndi.binary_fill_holes(ndi.binary_opening(A > -300, iterations=2)); lab, _ = ndi.label(B); B = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    ii = np.argwhere(B); lo, hi = ii.min(0), ii.max(0); xm = (lo[0] + hi[0]) // 2; off = int(30 / sp[0])
    D = np.array(im.GetDirection()).reshape(3, 3); sgn = -1 if D[1, 1] > 0 else 1                    # LPS: larger y index is posterior when the y axis points +y
    def front(x):                                                                             # most forward body point in this sagittal column, as a forward coordinate
        v = [np.flatnonzero(B[x, :, z]) for z in range(lo[2], hi[2] + 1) if B[x, :, z].any()]
        return max((sgn * a).max() for a in v) if v else -1e9
    return float((front(xm) - max(front(max(0, xm - off)), front(min(B.shape[0] - 1, xm + off)))) * sp[1])

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--one': one(sys.argv[2], sys.argv[3]); sys.exit(0)
if __name__ == '__main__':
    from huggingface_hub import HfApi, hf_hub_download
    repo = sys.argv[1]; api = HfApi(); PER = int(os.environ.get('PER', '20'))
    key = os.environ.get('TOTALSEG_LICENSE')
    if key: r = subprocess.run(['totalseg_set_license', '-l', key], capture_output=True, text=True); log('licence set:', r.returncode == 0)
    zp = hf_hub_download(repo, 'raw/14918214/synthRAD2025_Task1_Train.zip', repo_type='dataset', local_dir='/data/sr'); Z = zipfile.ZipFile(zp)
    names = Z.namelist(); cts = sorted(n for n in names if n.endswith('/ct.mha') and '/HN/' in n)
    regions = sorted({n.split('/')[-3] for n in names if n.endswith('/ct.mha')}); log('regions', regions, 'head-and-neck patients', len(cts))
    rows = []
    for n in cts[:: max(1, len(cts) // PER)][:PER]:
        d = n.rsplit('/', 1)[0]; r = dict(patient=d.split('/')[-1], centre=d.split('/')[-1][3:4], mr=(d + '/mr.mha') in names)
        try:
            os.makedirs('w', exist_ok=True); open('w/ct.mha', 'wb').write(Z.read(n))
            r['nose_ahead_mm'] = round(nose_mm('w/ct.mha'), 1)
            q = subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--one', 'w/ct.mha', 'w/r.json'], capture_output=True, text=True, timeout=1500)
            if q.returncode == 0: r.update(json.load(open('w/r.json')))
            else: r['error'] = 'out of memory (killed)' if q.returncode in (-9, 137) else 'worker exit %d: %s' % (q.returncode, q.stderr.strip()[-200:])
        except Exception as e: r['error'] = '%s %s' % (type(e).__name__, str(e)[:150])
        rows.append(r); log({k: v for k, v in r.items() if k != 'patient'})
    ok = [r for r in rows if 'error' not in r]
    summ = dict(hn_patients=len(cts), sampled=len(rows), failed=len(rows) - len(ok), with_mr=sum(r['mr'] for r in rows), jaw_whole=sum(r.get('mandible') == 'whole' for r in ok),
                jaw_cut=sum(r.get('mandible') == 'cut' for r in ok), upper_teeth=sum(r.get('teeth_upper') in ('whole', 'cut') for r in ok), skull_whole=sum(r.get('skull') == 'whole' for r in ok),
                nose_present=sum((r.get('nose_ahead_mm') or 0) > 10 for r in ok), median_extent_mm=[float(np.median([r['extent_mm'][i] for r in ok])) for i in range(3)] if ok else None)
    log('SUMMARY', json.dumps(summ))
    os.makedirs('counts', exist_ok=True); json.dump(dict(summary=summ, rows=rows), open('counts/synthrad_hn_census.json', 'w'), indent=1)
    for a in range(5):
        try: api.upload_folder(folder_path='counts', path_in_repo='census', repo_id=repo, repo_type='dataset', commit_message='SynthRAD head-and-neck census'); break
        except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
    log('SUMMARY done')
