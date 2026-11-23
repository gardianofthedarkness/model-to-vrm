# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "SimpleITK", "TotalSegmentator"]
# ///
"""Face and head labels on CT with TotalSegmentator's three open head tasks (Apache-2.0 weights):
head_glands_cavities (eyes, lenses, optic nerves, salivary glands, nasal cavity, pharynx, palate, auditory canals),
head_muscles (masseter, temporalis, pterygoids, tongue, digastric), craniofacial_structures (mandible, upper and lower
teeth, skull, head, maxillary and frontal sinuses).
SOURCE=ts      TotalSegmentator CT adults with a whole skull; each scan is cropped to the skull label's box with a
               margin of a fifth of the skull's own size, so the models run on the head only; output on that crop,
               with its offset in the codes_v1 crop.
SOURCE=hanseg  HaN-Seg's 42 head-and-neck CTs (CC BY-ND 4.0, private), cropped to the body (largest piece above -500 HU)
               with its offset in the scan recorded.
Output per scan and task: uint8 label map (the task's own class numbers) plus class names, voxel counts, timing.
EXTRA_CASES (ts only): comma-separated scans added to this shard, e.g. ones that ran out of RAM elsewhere.
One upload per job. args: <repo> <shard i/n> [limit]"""
import os, sys, csv, json, gzip, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
TASKS = ['head_glands_cavities', 'head_muscles', 'craniofacial_structures']
if __name__ == '__main__':
    import nibabel as nib, SimpleITK as sitk
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    from totalsegmentator.python_api import totalsegmentator
    from totalsegmentator.map_to_binary import class_map
    api = HfApi(); repo = sys.argv[1]; si, sn = map(int, sys.argv[2].split('/')); limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0; SRC = os.environ.get('SOURCE', 'ts')
    out_dir = 'head_v1'; os.makedirs(out_dir, exist_ok=True); os.makedirs('tmp', exist_ok=True); stats = {}; items = []
    if SRC == 'ts':
        ext = list(csv.DictReader(open(hf_hub_download(repo, 'census/label_extents.csv', repo_type='dataset'))))
        meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
        whole = lambda e: e['present'] == '1' and not any(e[k] == '1' for k in ('cut_inf', 'cut_sup', 'cut_side', 'cut_ap'))
        cases = sorted(e['case'] for e in ext if e['label'] == 'skull' and whole(e) and not (meta[e['case']].get('age') and float(meta[e['case']]['age']) < 18))[si::sn]
        if limit: cases = cases[:limit]
        cases += [c for c in os.environ.get('EXTRA_CASES', '').split(',') if c and c not in cases]   # scans retried from another shard
        zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); Z = zipfile.ZipFile(zp)
        pre = {n[:-len('ct.nii.gz')].rstrip('/').rsplit('/', 1)[-1]: n[:-len('ct.nii.gz')] for n in Z.namelist() if n.endswith('/ct.nii.gz')}
        snap = snapshot_download(repo, repo_type='dataset', allow_patterns=['codes_v1/%s.json' % c for c in cases])
        for c in cases: items.append(c)
    else:
        zp = hf_hub_download('gardianofthedarkness/hanseg-ct-mr', 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.'); Z = zipfile.ZipFile(zp)
        cases = sorted({n.split('/')[-2] for n in Z.namelist() if n.endswith('_IMG_CT.nrrd')})[si::sn]
        if limit: cases = cases[:limit]
        items = cases
    log('%s shard %s: %d scans' % (SRC, sys.argv[2], len(items)))
    for c in items:
        t1 = time.time(); rec = dict(tasks={})
        try:
            p = 'tmp/%s.nii.gz' % c
            if SRC == 'ts':
                img = nib.Nifti1Image.from_bytes(gzip.decompress(Z.read(pre[c] + 'ct.nii.gz'))); m = json.load(open(os.path.join(snap, 'codes_v1', c + '.json')))
                sk = m['landmarks']['skull']; clo = np.array(m['crop_lo']); lo = np.array(sk['lo']) + clo; hi = np.array(sk['hi']) + clo + 1; mg = ((hi - lo) / 5).astype(int)
                lo = np.maximum(lo - mg, 0); hi = np.minimum(hi + mg, img.shape[:3]); sub = img.slicer[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; nib.save(sub, p)
                rec['offset_in_codes_crop'] = (lo - clo).tolist(); rec['shape'] = list(sub.shape[:3])
            else:
                q = 'tmp/%s.nrrd' % c; open(q, 'wb').write(Z.read([n for n in Z.namelist() if n.endswith('%s/%s_IMG_CT.nrrd' % (c, c))][0])); sitk.WriteImage(sitk.ReadImage(q), p); os.remove(q)
                full = nib.load(p); A = np.asanyarray(full.dataobj); B = A > -500                    # crop to the body: the scan's air is dropped, nothing of the body
                from scipy.ndimage import label as cc
                lab, n_ = cc(B[::2, ::2, ::2]); keep = lab == (np.bincount(lab.ravel())[1:].argmax() + 1) if n_ else lab > 0
                ii = np.argwhere(keep) * 2; lo = np.maximum(ii.min(0) - 2, 0); hi = np.minimum(ii.max(0) + 4, A.shape[:3])
                nib.save(full.slicer[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]], p); rec['offset_in_scan'] = lo.tolist(); rec['full_shape'] = list(A.shape[:3]); del A, B, lab
                rec['shape'] = list(nib.load(p).shape[:3])
            arrays = {}
            for t in TASKS:
                t2 = time.time(); seg = totalsegmentator(p, None, task=t, ml=True, quiet=True, device='gpu'); L = np.asarray(seg.dataobj).astype(np.uint8); cm = class_map[t]
                cnt = np.bincount(L.ravel(), minlength=max(cm) + 1); arrays[t] = L
                rec['tasks'][t] = dict(seconds=round(time.time() - t2, 1), voxels={cm[i]: int(cnt[i]) for i in cm if i < len(cnt) and cnt[i] > 0})
            np.savez_compressed(os.path.join(out_dir, '%s.npz' % c), **arrays); os.remove(p); rec['seconds'] = round(time.time() - t1, 1)
        except Exception as e: rec = 'failed: %s %s' % (type(e).__name__, str(e)[:200])
        stats[c] = rec; log(c, rec if isinstance(rec, str) else rec['seconds'], 's')
    json.dump(dict(source=SRC, shard=sys.argv[2], tasks={t: {str(i): n for i, n in class_map[t].items()} for t in TASKS}, stats=stats), open(os.path.join(out_dir, '%s_shard_%d_of_%d.json' % (SRC, si, sn)), 'w'))
    dest = os.environ.get('HEAD_DIR', 'head_v1'); target = repo if SRC == 'ts' else 'gardianofthedarkness/hanseg-ct-mr'
    for attempt in range(6):
        try: api.upload_folder(folder_path=out_dir, path_in_repo=dest, repo_id=target, repo_type='dataset', commit_message='head labels %s %s' % (SRC, sys.argv[2])); break
        except Exception as e: log('upload retry', attempt, type(e).__name__); time.sleep(600)
    secs = [v['seconds'] for v in stats.values() if isinstance(v, dict)]
    log('SUMMARY ' + json.dumps(dict(source=SRC, shard=sys.argv[2], scans=len(items), failed=sum(isinstance(v, str) for v in stats.values()), median_seconds=float(np.median(secs)) if secs else None)))
