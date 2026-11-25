# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "SimpleITK", "TotalSegmentator", "dcm2niix"]
# ///
"""Face coverage of CQ500 head CTs (plan v8 section 14.1), on a sample. Downloads one archive of the CQ500 copy on
Hugging Face (neurosentry/cq500) inside the job; per patient takes the series with the most slices, and records numbers
only: the head's extent from top to bottom, whether the face has a nose, and from TotalSegmentator's licensed
craniofacial model whether skull, mandible, upper teeth and maxillary sinuses are present, and whole or cut by the scan's
edge. No images leave the job; nothing is kept after it. env: ARCHIVES (comma list, default qct01), PER (patients per
archive, default 12), TOTALSEG_LICENSE (secret). args: <repo for the counts>"""
import os, sys, json, time, glob, shutil, subprocess, tarfile, numpy as np
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

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--one': one(sys.argv[2], sys.argv[3]); sys.exit(0)
if __name__ == '__main__':
    from huggingface_hub import HfApi, hf_hub_download
    repo = sys.argv[1]; api = HfApi(); PER = int(os.environ.get('PER', '12')); out = {}
    key = os.environ.get('TOTALSEG_LICENSE')
    if key: r = subprocess.run(['totalseg_set_license', '-l', key], capture_output=True, text=True); log('licence set:', r.returncode == 0)
    for arc in os.environ.get('ARCHIVES', 'qct01').split(','):
        p = hf_hub_download('neurosentry/cq500', arc + '.tar', repo_type='dataset', local_dir='/data/cq'); log(arc, 'downloaded', round(os.path.getsize(p) / 1e9, 2), 'GB')
        d = '/data/cq/x_' + arc; os.makedirs(d, exist_ok=True)
        with tarfile.open(p) as t: t.extractall(d)
        os.remove(p)                                                                            # nothing is kept: the archive goes as soon as it is unpacked
        pats = sorted({os.path.relpath(os.path.dirname(f), d).split(os.sep)[0] for f in glob.glob(d + '/**/*', recursive=True) if os.path.isfile(f)})
        log(arc, 'patients', len(pats)); rows = []
        for pt in pats[:PER]:
            r = dict(patient=pt)
            try:
                shutil.rmtree('w', ignore_errors=True); os.makedirs('w/nii', exist_ok=True)
                subprocess.run(['dcm2niix', '-z', 'y', '-o', 'w/nii', '-f', '%s_%%s' % 'ct', os.path.join(d, pt)], capture_output=True)
                cands = [(f, os.path.getsize(f)) for f in glob.glob('w/nii/*.nii.gz')]
                if not cands: r['error'] = 'no series converted'; rows.append(r); continue
                nii = max(cands, key=lambda x: x[1])[0]; r['series'] = os.path.basename(nii)
                q = subprocess.run([sys.executable, '-u', os.path.abspath(__file__), '--one', nii, 'w/r.json'], capture_output=True, text=True, timeout=1500)
                if q.returncode == 0: r.update(json.load(open('w/r.json')))
                else: r['error'] = 'out of memory (killed)' if q.returncode in (-9, 137) else 'worker exit %d: %s' % (q.returncode, q.stderr.strip()[-200:])
            except Exception as e: r['error'] = '%s %s' % (type(e).__name__, str(e)[:150])
            rows.append(r); log(arc, {k: v for k, v in r.items() if k != 'patient'})
        shutil.rmtree(d, ignore_errors=True)
        ok = [r for r in rows if 'error' not in r]
        out[arc] = dict(patients=len(pats), sampled=len(rows), failed=len(rows) - len(ok),
                        jaw_whole=sum(r.get('mandible') == 'whole' for r in ok), jaw_cut=sum(r.get('mandible') == 'cut' for r in ok),
                        upper_teeth=sum(r.get('teeth_upper') in ('whole', 'cut') for r in ok), maxillary_sinus_whole=sum(r.get('sinus_maxillary') == 'whole' for r in ok),
                        skull_whole=sum(r.get('skull') == 'whole' for r in ok), median_extent_mm=[float(np.median([r['extent_mm'][i] for r in ok])) for i in range(3)] if ok else None, rows=rows)
        log('SUMMARY', arc, json.dumps({k: v for k, v in out[arc].items() if k != 'rows'}))
    os.makedirs('counts', exist_ok=True); json.dump(out, open('counts/cq500_face_census.json', 'w'), indent=1)
    for a in range(5):
        try: api.upload_folder(folder_path='counts', path_in_repo='census', repo_id=repo, repo_type='dataset', commit_message='CQ500 face coverage census'); break
        except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
    log('SUMMARY done')
