"""Runs ON a Hugging Face CPU Job: fetch TotalSegmentator CT v2.0.1 (Zenodo 10047292, CC BY 4.0) from its publisher,
check its MD5, store it unchanged in a private dataset repo, then list its contents and count head cases with a full face.
Only numbers are written (no images, no meshes).  args: <repo id>"""
# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "requests", "numpy", "nibabel"]
# ///
import os, sys, io, csv, json, gzip, time, hashlib, zipfile, requests, numpy as np, nibabel as nib
from huggingface_hub import HfApi
repo = sys.argv[1]; api = HfApi(); t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def put(text, name): api.upload_file(path_or_fileobj=text.encode(), path_in_repo=name, repo_id=repo, repo_type='dataset')

# --- 1. fetch from Zenodo with resume, verify MD5
rec = requests.get('https://zenodo.org/api/records/10047292', timeout=60).json()
f = [x for x in rec['files'] if x['key'].endswith('.zip')][0]; url, size, chk = f['links']['self'], int(f['size']), f['checksum']
log('zenodo: %s, %.2f GB, %s' % (f['key'], size / 1e9, chk))
zpath = os.path.abspath('ts.zip'); got = 0; md5 = hashlib.md5(); step = 1 << 30; nxt = step
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
if chk.startswith('md5:') and md5.hexdigest() != chk[4:]: sys.exit('MD5 mismatch: %s vs %s' % (md5.hexdigest(), chk))
log('download complete, MD5 ok')

# --- 2. store unchanged in the private repo, with attribution
put("""---
license: cc-by-4.0
---
# TotalSegmentator CT dataset v2.0.1 (private copy)
Unchanged copy of `%s` from Zenodo record 10047292 (https://zenodo.org/records/10047292), MD5 %s.
Wasserthal J. et al., "TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images",
Radiology: Artificial Intelligence 2023, doi:10.1148/ryai.230024. Licence: CC BY 4.0
(https://creativecommons.org/licenses/by/4.0/). No changes made to the data. `census/` holds derived counts only.
""" % (f['key'], chk), 'README.md')
api.upload_file(path_or_fileobj=zpath, path_in_repo='raw/' + f['key'], repo_id=repo, repo_type='dataset')
log('uploaded the zip to the private repo')

# --- 3. what the zip contains
z = zipfile.ZipFile(zpath); infos = z.infolist()
put('\n'.join('%d\t%s' % (i.file_size, i.filename) for i in infos), 'census/file_list.tsv')
cts = sorted(i.filename for i in infos if i.filename.endswith('/ct.nii.gz')); case_dir = {c.rsplit('/', 1)[0] for c in cts}
labels = sorted({i.filename.rsplit('/', 1)[1][:-7] for i in infos if '/segmentations/' in i.filename and i.filename.endswith('.nii.gz')})
others = sorted(i.filename for i in infos if not any(i.filename.startswith(d + '/') for d in case_dir) and not i.filename.endswith('/'))
log('%d entries, %d cases with ct.nii.gz, %d label names, other files: %s' % (len(infos), len(cts), len(labels), others))
for o in others:
    if o.lower().endswith(('.csv', '.txt', '.json', '.md')): api.upload_file(path_or_fileobj=z.read(o), path_in_repo='census/' + o.split('/')[-1], repo_id=repo, repo_type='dataset')

# --- 4. head census (numbers only)
def vol(name):
    img = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(gzip.decompress(z.read(name))))   # RAS: +y anterior, +z superior
    return np.asanyarray(img.dataobj), img.header.get_zooms()[:3]
def otsu(v):
    h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    return c[np.nanargmax(s[:-1])]
rows = []
for k, d in enumerate(sorted(case_dir)):
    sk = d + '/segmentations/skull.nii.gz'
    if sk not in z.NameToInfo: continue
    S, zoom = vol(sk); S = S > 0
    if not S.any(): continue
    C, _ = vol(d + '/ct.nii.gz'); C = np.asarray(C, np.float32)
    thr = otsu(np.clip(C[::4, ::4, ::4], -1024, 400)); B = C > thr                     # tissue vs air, split from this scan's own histogram
    zs = np.flatnonzero(S.any((0, 1))); lo, hi = zs[0], zs[-1]
    c3 = d + '/segmentations/vertebrae_C3.nii.gz'; chin = bool(c3 in z.NameToInfo and (vol(c3)[0] > 0).any())
    prot = []; ant_crop = lat_crop = False
    for zz in range(lo, hi + 1):
        bs, ss = B[:, :, zz], S[:, :, zz]
        if not ss.any() or not bs.any(): continue
        yb = np.flatnonzero(bs.any(0)); ys = np.flatnonzero(ss.any(0)); prot.append((yb[-1] - ys[-1]) * zoom[1])
        ant_crop |= bool(bs[:, -1].any()); lat_crop |= bool(bs[0, :].any() or bs[-1, :].any())
    rows.append(dict(case=d.split('/')[-1], slices=int(S.shape[2]), voxel_mm=[round(float(v), 2) for v in zoom],
                     skull_cut_bottom=bool(lo == 0), skull_cut_top=bool(hi == S.shape[2] - 1), c3_present=chin,
                     face_cropped_front=ant_crop, face_cropped_side=lat_crop, max_soft_tissue_in_front_of_skull_mm=round(float(max(prot)) if prot else 0.0, 1)))
    if len(rows) % 25 == 0: log('%d head cases so far (%d of %d cases scanned)' % (len(rows), k + 1, len(case_dir)))
prot = np.array([r['max_soft_tissue_in_front_of_skull_mm'] for r in rows]); split = float(otsu(prot)) if len(rows) > 2 else 0.0
for r in rows:
    r['face_tissue_present'] = bool(r['max_soft_tissue_in_front_of_skull_mm'] > split)
    r['full_face'] = bool(not r['skull_cut_bottom'] and r['c3_present'] and not r['face_cropped_front'] and not r['face_cropped_side'] and r['face_tissue_present'])
    r['whole_head'] = bool(r['full_face'] and not r['skull_cut_top'])
summary = dict(cases=len(case_dir), labels=len(labels), head_cases=len(rows), soft_tissue_split_mm=round(split, 1),
               **{key: int(sum(r[key] for r in rows)) for key in ['skull_cut_bottom', 'skull_cut_top', 'c3_present', 'face_cropped_front', 'face_cropped_side', 'face_tissue_present', 'full_face', 'whole_head']},
               soft_tissue_mm_percentiles={p: round(float(np.percentile(prot, p)), 1) for p in (5, 25, 50, 75, 95)} if len(rows) else {})
if rows:
    buf = io.StringIO(); wr = csv.DictWriter(buf, fieldnames=list(rows[0])); wr.writeheader(); wr.writerows(rows); put(buf.getvalue(), 'census/heads.csv')
put(json.dumps(dict(summary=summary, labels=labels, other_files=others), indent=1), 'census/summary.json')
log('SUMMARY ' + json.dumps(summary))
