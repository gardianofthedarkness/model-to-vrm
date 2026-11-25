# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "SimpleITK", "TotalSegmentator", "idc-index", "dcm2niix"]
# ///
"""Census of open CT collections in NCI's Imaging Data Commons (public buckets, CC BY): how many scans could join the
interior model's head and pelvis sets. For a sample of CT series per collection (adults where an age is recorded):
download, convert, run TotalSegmentator's open 'total' model in fast mode on a few bones, and record whether the skull and
the pelvis (both hip bones, sacrum, both femoral heads) are in view and not cut by the scan's edge. Numbers only: no
images leave the job. env: COLLECTIONS (comma list), PER (series per collection, default 12). args: <repo>"""
import os, sys, json, time, glob, shutil, subprocess, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
ROI = ['skull', 'brain', 'hip_left', 'hip_right', 'sacrum', 'femur_left', 'femur_right', 'vertebrae_C1', 'vertebrae_L5']
if __name__ == '__main__':
    import nibabel as nib
    from huggingface_hub import HfApi
    from idc_index import index
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    repo = sys.argv[1]; api = HfApi(); c = index.IDCClient(); PER = int(os.environ.get('PER', '12'))
    cols = os.environ.get('COLLECTIONS', 'fdg_pet_ct_lesions,psma_pet_ct_lesions,ct_colonography,eay131,cmb_lca').split(',')
    inv = {v: k for k, v in class_map['total'].items()}; out = {}
    for col in cols:
        df = c.sql_query("""SELECT SeriesInstanceUID, PatientID, PatientAge, PatientSex, BodyPartExamined, SeriesDescription, instanceCount, series_size_MB FROM index
                            WHERE collection_id='%s' AND Modality='CT' AND instanceCount >= 60""" % col)
        age = df.PatientAge.str.extract(r'(\d+)')[0].astype(float); adult = df[(age >= 18) | age.isna()]
        tot = dict(series=int(len(df)), patients=int(df.PatientID.nunique()), adult_or_unknown_patients=int(adult.PatientID.nunique()), age_recorded=int(age.notna().sum()), gb=round(float(df.series_size_MB.sum()) / 1024, 1))
        smp = adult.drop_duplicates('PatientID').sample(min(PER, adult.PatientID.nunique()), random_state=0); rows = []
        log(col, tot, 'sample', len(smp))
        for _, s in smp.iterrows():
            d = 'dl/%s' % s.SeriesInstanceUID; shutil.rmtree('dl', ignore_errors=True); os.makedirs(d, exist_ok=True); r = dict(series=s.SeriesInstanceUID[-12:], age=s.PatientAge, sex=s.PatientSex, part=s.BodyPartExamined)
            try:
                c.download_from_selection(seriesInstanceUID=[s.SeriesInstanceUID], downloadDir=d, quiet=True)
                subprocess.run(['dcm2niix', '-z', 'y', '-o', 'dl', '-f', 'ct', d], capture_output=True)
                nii = sorted(glob.glob('dl/ct*.nii.gz'), key=os.path.getsize)[-1]; img = nib.load(nii); sp = np.abs(np.diag(img.affine)[:3])
                r['extent_mm'] = [round(float(n * q)) for n, q in zip(img.shape[:3], sp)]
                seg = np.asarray(totalsegmentator(nii, None, fast=True, roi_subset=ROI, quiet=True, device='gpu').dataobj)
                for n in ROI:
                    m = seg == inv[n]
                    if not m.any(): r[n] = 'absent'; continue
                    ii = np.argwhere(m); lo, hi = ii.min(0), ii.max(0); cut = bool(lo[2] == 0 or hi[2] == seg.shape[2] - 1 or lo[0] == 0 or hi[0] == seg.shape[0] - 1 or lo[1] == 0 or hi[1] == seg.shape[1] - 1)
                    r[n] = 'cut' if cut else 'whole'
            except Exception as e: r['error'] = '%s %s' % (type(e).__name__, str(e)[:120])
            rows.append(r); log(col, {k: v for k, v in r.items() if k not in ('series',)})
        ok = [r for r in rows if 'error' not in r]
        pel = sum(all(r.get(n) == 'whole' for n in ('hip_left', 'hip_right', 'sacrum')) for r in ok); head = sum(r.get('skull') == 'whole' for r in ok)
        out[col] = dict(totals=tot, sampled=len(rows), failed=len(rows) - len(ok), pelvis_whole=pel, skull_whole=head, rows=rows)
        log('SUMMARY', col, json.dumps({k: v for k, v in out[col].items() if k != 'rows'}))
    os.makedirs('census_idc', exist_ok=True); json.dump(out, open('census_idc/idc_census.json', 'w'), indent=1)
    for a in range(5):
        try: api.upload_folder(folder_path='census_idc', path_in_repo='census_idc', repo_id=repo, repo_type='dataset', commit_message='IDC census'); break
        except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(120)
    log('SUMMARY done')
