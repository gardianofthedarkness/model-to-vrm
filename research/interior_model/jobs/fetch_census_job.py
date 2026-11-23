"""Runs ON a Hugging Face CPU Job.  Modes:
  zenodo <record> <repo> <readme-key>   fetch a Zenodo record's zip (resume + MD5), store unchanged in a private repo, then census
  tsct-regions <repo>                   read the TotalSegmentator CT zip already in our private repo, exact per-label extents + region counts
Only numbers and file lists are written: no images, meshes or renderings."""
# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "requests", "numpy", "nibabel", "pynrrd", "pandas", "xlrd"]
# ///
import os, sys, io, csv, json, gzip, time, hashlib, zipfile, collections, requests, numpy as np
from multiprocessing import Pool
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def put(data, name, repo):
    api.upload_file(path_or_fileobj=data.encode() if isinstance(data, str) else data, path_in_repo=name, repo_id=repo, repo_type='dataset')

READMES = {
 'hanseg': """---
license: cc-by-nc-nd-4.0
---
# HaN-Seg (private copy, unchanged)
Unchanged copy of `{key}` from Zenodo record 7442914 (https://zenodo.org/records/7442914), MD5 {md5}.
Podobnik G., Strojan P., Peterlin P., Ibragimov B., Vrtovec T., "HaN-Seg: The head and neck organ-at-risk CT & MR
segmentation dataset", Medical Physics 2023, doi:10.1002/mp.16197 (the source requires citing this paper).
Licence: CC BY-NC-ND 4.0 (https://creativecommons.org/licenses/by-nc-nd/4.0/). Non-commercial; no adapted material may be
shared. Private research copy, no changes made. `census/` holds derived counts only (kept private).
""",
 'tsmri': """---
license: cc-by-4.0
---
# TotalSegmentator MRI dataset v3.0.0 (private copy, unchanged)
Unchanged copy of `{key}` from Zenodo record 22688334 (https://zenodo.org/records/22688334), MD5 {md5}.
Created by the Research and Analysis department, University Hospital Basel (Wasserthal J., Akinci D'Antonoli T. et al.);
the source asks users to cite the TotalSegmentator MRI paper. Licence: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
No changes made. Contains paediatric cases (see meta.csv); `census/` holds derived counts only.
""",
}

def fetch_zenodo(record, repo, key):
    rec = requests.get('https://zenodo.org/api/records/%s' % record, timeout=60).json()
    f = [x for x in rec['files'] if x['key'].endswith('.zip')][0]; url, size, chk = f['links']['self'], int(f['size']), f['checksum']
    log('zenodo %s: %s, %.2f GB, %s, licence %s' % (record, f['key'], size / 1e9, chk, (rec['metadata'].get('license') or {}).get('id')))
    zpath = os.path.abspath(f['key']); got = 0; md5 = hashlib.md5(); step = 1 << 30; nxt = step
    with open(zpath, 'wb') as out:
        for attempt in range(20):
            try:
                with requests.get(url, headers={'Range': 'bytes=%d-' % got} if got else {}, stream=True, timeout=120) as r:
                    r.raise_for_status()
                    if got and r.status_code != 206: sys.exit('server ignored the resume request; stopping')
                    for b in r.iter_content(1 << 22):
                        out.write(b); md5.update(b); got += len(b)
                        if got >= nxt: log('downloaded %.1f / %.1f GB' % (got / 1e9, size / 1e9)); nxt += step
                if got >= size: break
            except Exception as e:
                log('download interrupted at %.2f GB (%s); resuming' % (got / 1e9, type(e).__name__)); time.sleep(10)
    if got != size: sys.exit('download incomplete: %d of %d bytes' % (got, size))
    if chk.startswith('md5:') and md5.hexdigest() != chk[4:]: sys.exit('MD5 mismatch')
    log('download complete, MD5 ok')
    put(READMES[key].format(key=f['key'], md5=chk), 'README.md', repo)
    api.upload_file(path_or_fileobj=zpath, path_in_repo='raw/' + f['key'], repo_id=repo, repo_type='dataset')
    log('uploaded the zip to the private repo'); return zpath

def listing(zpath, repo):
    z = zipfile.ZipFile(zpath); infos = [i for i in z.infolist() if not i.filename.endswith('/')]
    put('\n'.join('%d\t%s' % (i.file_size, i.filename) for i in infos), 'census/file_list.tsv', repo)
    for i in infos:
        if i.filename.lower().endswith(('.csv', '.txt', '.json', '.md', '.xlsx', '.tsv')) and i.file_size < 50e6:
            put(z.read(i.filename), 'census/' + i.filename.replace('/', '__'), repo)
    return z, infos

# ---------- per-mask extents (NIfTI inside a zip), run in worker processes
_Z = None
def _init(zpath):
    global _Z; _Z = zipfile.ZipFile(zpath)
def nii_extent(name):
    import nibabel as nib
    img = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(name))))   # RAS
    m = np.asanyarray(img.dataobj) > 0; zoom = img.header.get_zooms()[:3]
    if not m.any(): return name, dict(present=0)
    ax = [np.flatnonzero(m.any(tuple(j for j in range(3) if j != i))) for i in range(3)]
    return name, dict(present=1, voxels=int(m.sum()), shape='x'.join(map(str, m.shape)), voxel_mm='x'.join('%.2f' % v for v in zoom),
                      cut_inf=int(ax[2][0] == 0), cut_sup=int(ax[2][-1] == m.shape[2] - 1),
                      cut_side=int(ax[0][0] == 0 or ax[0][-1] == m.shape[0] - 1), cut_ap=int(ax[1][0] == 0 or ax[1][-1] == m.shape[1] - 1),
                      length_mm=round(float((ax[2][-1] - ax[2][0] + 1) * zoom[2]), 1))

def label_census(zpath, infos, repo):
    segs = [i.filename for i in infos if '/segmentations/' in i.filename and i.filename.endswith('.nii.gz')]
    log('%d masks to read' % len(segs)); rows = []
    with Pool(os.cpu_count(), _init, (zpath,)) as p:
        for k, (name, d) in enumerate(p.imap_unordered(nii_extent, segs, chunksize=64)):
            parts = name.split('/'); d.update(case=parts[-3], label=parts[-1][:-7]); rows.append(d)
            if (k + 1) % 10000 == 0: log('read %d / %d masks' % (k + 1, len(segs)))
    cols = ['case', 'label', 'present', 'voxels', 'shape', 'voxel_mm', 'cut_inf', 'cut_sup', 'cut_side', 'cut_ap', 'length_mm']
    buf = io.StringIO(); w = csv.DictWriter(buf, cols, restval=''); w.writeheader(); w.writerows(sorted(rows, key=lambda r: (r['case'], r['label'])))
    put(buf.getvalue(), 'census/label_extents.csv', repo)
    by = collections.defaultdict(dict)
    for r in rows: by[r['case']][r['label']] = r
    per_label = {}
    for r in rows:
        s = per_label.setdefault(r['label'], dict(present=0, whole=0))
        if r['present']: s['present'] += 1; s['whole'] += int(not (r['cut_inf'] or r['cut_sup'] or r['cut_side'] or r['cut_ap']))
    return by, per_label

def whole(c, l): r = c.get(l); return bool(r and r['present'] and not (r['cut_inf'] or r['cut_sup'] or r['cut_side'] or r['cut_ap']))
def pres(c, l): r = c.get(l); return bool(r and r['present'])
def regions(by):
    labs = set(l for c in by.values() for l in c)
    def any_of(*names): return [n for n in names if n in labs]
    V = lambda pre, a, b: ['vertebrae_%s%d' % (pre, i) for i in range(a, b + 1)]
    R = {}
    for side in ('left', 'right'):
        R['whole femur ' + side] = lambda c, s=side: whole(c, 'femur_' + s)
        R['whole humerus ' + side] = lambda c, s=side: whole(c, 'humerus_' + s)
        R['whole shoulder girdle ' + side + ' (scapula+clavicle whole, humerus present)'] = lambda c, s=side: whole(c, 'scapula_' + s) and whole(c, 'clavicula_' + s) and pres(c, 'humerus_' + s)
    R['both femurs whole'] = lambda c: whole(c, 'femur_left') and whole(c, 'femur_right')
    R['whole pelvis (both hips + sacrum whole)'] = lambda c: whole(c, 'hip_left') and whole(c, 'hip_right') and whole(c, 'sacrum')
    R['whole thoracolumbar spine (T1..L5 all whole)'] = lambda c: all(whole(c, v) for v in V('T', 1, 12) + V('L', 1, 5))
    R['whole rib cage (24 ribs + sternum whole)'] = lambda c: all(whole(c, 'rib_%s_%d' % (s, i)) for s in ('left', 'right') for i in range(1, 13)) and whole(c, 'sternum')
    R['whole cervical spine (C1..C7 whole)'] = lambda c: all(whole(c, v) for v in V('C', 1, 7))
    R['trunk: neck to pelvis (C7, all T/L, hips, sacrum whole)'] = lambda c: all(whole(c, v) for v in ['vertebrae_C7'] + V('T', 1, 12) + V('L', 1, 5) + ['hip_left', 'hip_right', 'sacrum'])
    R['skull present'] = lambda c: pres(c, 'skull')
    out = {}
    for k, f in R.items():
        try: out[k] = sum(1 for c in by.values() if f(c))
        except Exception: out[k] = None
    return out

# ---------- HaN-Seg face census (nrrd)
def otsu(v):
    h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    return c[np.nanargmax(s[:-1])]
def to_ras(data, hdr):
    """Reorder an nrrd array so axis 0 = +x (left->right? R), 1 = +y (anterior), 2 = +z (superior); returns array and mm per axis."""
    D = np.array(hdr['space directions'], float); sp = hdr.get('space', 'left-posterior-superior')
    flip = np.array([-1, -1, 1]) if sp.startswith('left-posterior') else np.array([1, 1, 1])   # LPS -> RAS
    D = D * flip; order = [int(np.argmax(np.abs(D[:, j]))) for j in range(3)]   # array axis for each world axis
    a = np.transpose(data, order); Dw = D[order]
    for j in range(3):
        if Dw[j, j] < 0: a = np.flip(a, j)
    return a, np.abs(np.diag(Dw))
def hanseg_case(args):
    """Face census for one HaN-Seg CT.  Orientation is checked against the anatomy (mandible in front of the spinal cord,
    brainstem above the mandible) rather than trusted from the header.  Soft tissue in front of the bone is measured along
    the midline from the most anterior bone point, walking forward only through connected tissue (so a separate
    immobilisation mask or couch is not counted)."""
    import nrrd
    ct_path, oar = args
    C, h = nrrd.read(ct_path); C, zoom = to_ras(C.astype(np.float32), h)
    row = dict(case=os.path.basename(os.path.dirname(ct_path)), shape='x'.join(map(str, C.shape)), voxel_mm='x'.join('%.2f' % v for v in zoom))
    def load(name):
        if name not in oar: return None
        M, hm = nrrd.read(oar[name]); M, _ = to_ras(M > 0, hm)
        return M if M.shape == C.shape and M.any() else None
    Mn, Sc, Bs = load('Bone_Mandible'), load('SpinalCord'), load('Brainstem')
    row['oar_on_ct_grid'] = Mn is not None
    cen = lambda M: np.argwhere(M).mean(0)
    if Mn is not None and Sc is not None and cen(Mn)[1] < cen(Sc)[1]: C = C[:, ::-1]; Mn = Mn[:, ::-1]; row['flipped_ap'] = True
    if Mn is not None and Bs is not None and cen(Bs)[2] < cen(Mn)[2]: C = C[:, :, ::-1]; Mn = Mn[:, :, ::-1]; row['flipped_si'] = True
    t_tissue = otsu(np.clip(C[::3, ::3, ::3], -1024, 2000)); B = C > t_tissue
    t_bone = otsu(np.clip(C[B][::50], -200, 2000)); bone = C > t_bone; row.update(tissue_hu=round(float(t_tissue)), bone_hu=round(float(t_bone)))
    if Mn is None:
        row.update(mandible_present=False); return row
    zs = np.flatnonzero(Mn.any((0, 1))); lo, x0 = int(zs[0]), int(round(cen(Mn)[0]))
    band = slice(max(0, x0 - int(10 / zoom[0])), x0 + int(10 / zoom[0]) + 1)
    hi = min(C.shape[2], lo + int(130 / zoom[2]))                     # mandible bottom up to about the brow
    row.update(mandible_present=True, mandible_cut=bool(lo == 0), face_mm_in_view=round(float((hi - lo) * zoom[2]), 1),
               head_top_in_view=not bool(B[:, :, -1].any()))
    prot = []; ant = lat = False
    for zz in range(lo, hi):
        bs, ks = B[band, :, zz].any(0), bone[band, :, zz].any(0)
        if not ks.any(): continue
        y = int(np.flatnonzero(ks)[-1]); n = 0
        while y + 1 < bs.size and bs[y + 1]: y += 1; n += 1
        prot.append(n * zoom[1]); ant |= bool(y + 1 >= bs.size)
        lat |= bool(B[0, :, zz].any() or B[-1, :, zz].any())
    row.update(face_cropped_front=ant, face_cropped_side=lat,
               max_soft_tissue_in_front_of_bone_mm=round(float(max(prot)), 1) if prot else 0.0,
               median_soft_tissue_in_front_of_bone_mm=round(float(np.median(prot)), 1) if prot else 0.0)
    return row
def hanseg_census(zpath, infos, repo):
    root = os.path.abspath('hanseg_x'); zipfile.ZipFile(zpath).extractall(root); log('extracted')
    cts = [os.path.join(root, i.filename) for i in infos if i.filename.endswith('_IMG_CT.nrrd')]
    oars = sorted({i.filename.split('_OAR_')[1].replace('.seg.nrrd', '') for i in infos if '_OAR_' in i.filename})
    jobs = []
    for c in cts:
        d = os.path.dirname(c)
        jobs.append((c, {f.split('_OAR_')[1].replace('.seg.nrrd', ''): os.path.join(d, f) for f in os.listdir(d) if '_OAR_' in f}))
    log('%d CT cases, %d OAR names' % (len(cts), len(oars)))
    with Pool(2, maxtasksperchild=4) as p: rows = p.map(hanseg_case, jobs, chunksize=1)   # full CT volumes: few workers (8 ran out of memory)
    keys = list(dict.fromkeys(k for r in rows for k in r)); buf = io.StringIO(); w = csv.DictWriter(buf, keys, restval=''); w.writeheader(); w.writerows(rows); put(buf.getvalue(), 'census/faces.csv', repo)
    n = lambda f: sum(1 for r in rows if f(r))
    face = lambda r: r['mandible_present'] and not r.get('mandible_cut') and not r.get('face_cropped_front') and not r.get('face_cropped_side')
    s = dict(cases=len(rows), oar_labels=oars, mandible_present=n(lambda r: r['mandible_present']), mandible_not_cut=n(lambda r: r['mandible_present'] and not r.get('mandible_cut')),
             front_not_cropped=n(lambda r: r['mandible_present'] and not r.get('face_cropped_front')), side_not_cropped=n(lambda r: r['mandible_present'] and not r.get('face_cropped_side')),
             head_top_in_view=n(lambda r: r.get('head_top_in_view')), face_in_view=n(face), face_and_head_top=n(lambda r: face(r) and r.get('head_top_in_view')),
             flipped_ap=n(lambda r: r.get('flipped_ap')), flipped_si=n(lambda r: r.get('flipped_si')), oar_on_ct_grid=n(lambda r: r.get('oar_on_ct_grid')),
             face_mm_in_view_percentiles={p: round(float(np.percentile([r.get('face_mm_in_view', 0) for r in rows], p)), 1) for p in (5, 50, 95)},
             max_soft_tissue_mm_percentiles={p: round(float(np.percentile([r.get('max_soft_tissue_in_front_of_bone_mm', 0) for r in rows], p)), 1) for p in (5, 25, 50, 75, 95)},
             median_soft_tissue_mm_percentiles={p: round(float(np.percentile([r.get('median_soft_tissue_in_front_of_bone_mm', 0) for r in rows], p)), 1) for p in (5, 50, 95)},
             suspect_defaced_lt_10mm=n(lambda r: r['mandible_present'] and r.get('max_soft_tissue_in_front_of_bone_mm', 0) < 10))
    return s

IXI_README = """---
license: cc-by-sa-3.0
---
# IXI dataset, T1 + T2 + demographics (private copy, unchanged)
Unchanged copies of IXI-T1.tar, IXI-T2.tar and IXI.xls from https://brain-development.org/ixi-dataset/
(Imperial College London / IXI project). Licence: CC BY-SA 3.0 (https://creativecommons.org/licenses/by-sa/3.0/);
the source asks users to acknowledge it, e.g. by citing that website. Derived material must be shared under the same
licence if distributed. No changes made. `census/` holds derived counts only.
"""
def fetch_url(url, dest):
    got = 0; step = 1 << 30; nxt = step
    with requests.get(url, stream=True, timeout=120, allow_redirects=True) as r:
        r.raise_for_status(); size = int(r.headers.get('content-length', 0))
        with open(dest, 'wb') as out:
            for b in r.iter_content(1 << 22):
                out.write(b); got += len(b)
                if got >= nxt: log('%s: %.1f / %.1f GB' % (os.path.basename(dest), got / 1e9, size / 1e9)); nxt += step
    if size and got != size: sys.exit('%s incomplete: %d of %d' % (url, got, size))
    log('%s complete, %.2f GB (no checksum published)' % (os.path.basename(dest), got / 1e9)); return got
def ixi_face(path):
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path)); A = np.asanyarray(img.dataobj).astype(np.float32); zoom = img.header.get_zooms()[:3]
    B = A > otsu(A[::2, ::2, ::2])
    ys = np.flatnonzero(B.any((0, 2))); zs = np.flatnonzero(B.any((0, 1)))
    return dict(file=os.path.basename(path), shape='x'.join(map(str, A.shape)), voxel_mm='x'.join('%.2f' % v for v in zoom),
                head_touches_front=bool(B[:, -1, :].any()), head_touches_side=bool(B[0].any() or B[-1].any()),
                head_touches_bottom=bool(B[:, :, 0].any()), head_touches_top=bool(B[:, :, -1].any()),
                head_ap_mm=round(float((ys[-1] - ys[0] + 1) * zoom[1]), 1) if ys.size else 0, head_si_mm=round(float((zs[-1] - zs[0] + 1) * zoom[2]), 1) if zs.size else 0)
def ixi(repo):
    import tarfile
    base = 'http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/'
    put(IXI_README, 'README.md', repo); total = 0
    for f in ['IXI.xls', 'IXI-T1.tar', 'IXI-T2.tar']:
        total += fetch_url(base + f, f)
        api.upload_file(path_or_fileobj=f, path_in_repo='raw/' + f, repo_id=repo, repo_type='dataset'); log('uploaded ' + f)
    os.makedirs('t1', exist_ok=True); tarfile.open('IXI-T1.tar').extractall('t1', filter='data')
    files = sorted(os.path.join('t1', x) for x in os.listdir('t1') if x.endswith('.nii.gz'))
    with Pool(os.cpu_count()) as p: rows = p.map(ixi_face, files)
    buf = io.StringIO(); w = csv.DictWriter(buf, list(rows[0])); w.writeheader(); w.writerows(rows); put(buf.getvalue(), 'census/t1_head_extent.csv', repo)
    n = lambda f: sum(1 for r in rows if f(r))
    demo = {}
    try:
        import pandas as pd; d = pd.read_excel('IXI.xls'); demo = dict(rows=len(d), columns=list(map(str, d.columns)))
        if 'AGE' in d: demo['age_min_max'] = [float(d['AGE'].min()), float(d['AGE'].max())]; demo['under_18'] = int((d['AGE'] < 18).sum())
    except Exception as e: demo = dict(error=type(e).__name__)
    return dict(total_gb=round(total / 1e9, 2), t1_volumes=len(rows), t2_in_tar=None, demographics=demo,
                front_not_touching=n(lambda r: not r['head_touches_front']), side_not_touching=n(lambda r: not r['head_touches_side']),
                bottom_not_touching=n(lambda r: not r['head_touches_bottom']),
                head_ap_mm_percentiles={p: round(float(np.percentile([r['head_ap_mm'] for r in rows], p)), 1) for p in (5, 50, 95)},
                head_si_mm_percentiles={p: round(float(np.percentile([r['head_si_mm'] for r in rows], p)), 1) for p in (5, 50, 95)})

if __name__ == '__main__':
    mode = sys.argv[1]
    if mode == 'ixi':
        repo = sys.argv[2]; s = ixi(repo); put(json.dumps(s, indent=1), 'census/summary.json', repo); log('SUMMARY ' + json.dumps(s)); sys.exit(0)
    if mode == 'hanseg-census':   # re-run the census from the copy already in our repo
        repo = sys.argv[2]; zpath = hf_hub_download(repo, 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.'); log('zip fetched from our repo')
        z, infos = listing(zpath, repo); s = hanseg_census(zpath, infos, repo)
        put(json.dumps(s, indent=1), 'census/summary.json', repo); log('SUMMARY ' + json.dumps(s)); sys.exit(0)
    if mode == 'zenodo':
        record, repo, key = sys.argv[2:5]; zpath = fetch_zenodo(record, repo, key); z, infos = listing(zpath, repo)
        log('%d files in the zip' % len(infos))
        if key == 'hanseg': summary = hanseg_census(zpath, infos, repo)
        else:
            by, per_label = label_census(zpath, infos, repo)
            summary = dict(cases=len(by), labels=len(per_label), per_label=per_label, regions=regions(by))
    else:
        repo = sys.argv[2]
        zpath = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); log('zip fetched from our repo')
        z, infos = zipfile.ZipFile(zpath), [i for i in zipfile.ZipFile(zpath).infolist() if not i.filename.endswith('/')]
        by, per_label = label_census(zpath, infos, repo)
        summary = dict(cases=len(by), labels=len(per_label), per_label=per_label, regions=regions(by))
    name = 'census/summary.json' if mode == 'zenodo' else 'census/regions_summary.json'
    put(json.dumps(summary, indent=1), name, repo)
    log('SUMMARY ' + json.dumps({k: v for k, v in summary.items() if k != 'per_label'}))
