# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "nibabel", "TotalSegmentator"]
# ///
"""Licensed TotalSegmentator 'appendicular_bones' (limb bones beyond femur and humerus: radius, ulna, carpals,
metacarpals, hand phalanges; tibia, fibula, patella, tarsals, metatarsals, foot phalanges) on the CT scans whose
image reaches the wrist (the scan continues at least 250 mm past the end of the humerus; list passed in CASES).
Academic, non-commercial licence; key passed as a job secret, never printed. Output per scan: uint8 map on the
codes_v1 crop with the task's own class numbers, the class names, voxel counts, timing. args: <repo> <shard i/n>"""
import os, sys, csv, json, gzip, time, zipfile, subprocess, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
if __name__ == '__main__':
    import nibabel as nib
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; si, sn = map(int, sys.argv[2].split('/')); limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    key = os.environ['TOTALSEG_LICENSE']; r = subprocess.run(['totalseg_set_license', '-l', key], capture_output=True, text=True); log('licence set:', r.returncode == 0)
    from totalsegmentator.python_api import totalsegmentator
    ext = list(csv.DictReader(open(hf_hub_download(repo, 'census/label_extents.csv', repo_type='dataset'))))
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    by = {}
    for e in ext: by.setdefault(e['case'], {})[e['label']] = e
    whole = lambda e: bool(e) and e['present'] == '1' and not any(e[k] == '1' for k in ('cut_inf', 'cut_sup', 'cut_side', 'cut_ap'))
    cases = sorted(os.environ['CASES'].split(','))[si::sn]
    if limit: cases = cases[:limit]
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); Z = zipfile.ZipFile(zp)
    pre = {n[:-len('ct.nii.gz')].rstrip('/').rsplit('/', 1)[-1]: n[:-len('ct.nii.gz')] for n in Z.namelist() if n.endswith('/ct.nii.gz')}
    snap = snapshot_download(repo, repo_type='dataset', allow_patterns=['codes_v1/%s.json' % c for c in cases])
    out_dir = 'limbs_v1'; os.makedirs(out_dir, exist_ok=True); os.makedirs('tmp', exist_ok=True); stats = {}
    for k, c in enumerate(cases):
        t1 = time.time()
        try:
            p = 'tmp/%s.nii.gz' % c; open(p, 'wb').write(Z.read(pre[c] + 'ct.nii.gz'))
            seg = totalsegmentator(p, None, task='appendicular_bones', ml=True, quiet=True, device='gpu')
            L = np.asanyarray(seg.dataobj); names = None
            lm = getattr(seg, 'extra', {}) if hasattr(seg, 'extra') else {}
            from totalsegmentator.map_to_binary import class_map
            cm = class_map['appendicular_bones']; out = np.asarray(L).astype(np.uint8)
            m = json.load(open(os.path.join(snap, 'codes_v1', c + '.json'))); lo = np.array(m['crop_lo']); shp = np.array(m['crop_shape']); hi = lo + shp
            pad = [(0, max(0, h - n)) for h, n in zip(hi, out.shape)]
            if any(q[1] for q in pad): out = np.pad(out, pad)
            out = out[tuple(slice(a, b) for a, b in zip(lo, hi))]
            np.savez_compressed(os.path.join(out_dir, c + '.npz'), limbs=out); os.remove(p)
            cnt = np.bincount(out.ravel(), minlength=max(cm) + 1); stats[c] = dict(seconds=round(time.time() - t1, 1), voxels={cm[i]: int(cnt[i]) for i in cm if i < len(cnt) and cnt[i] > 0}, classes={str(i): n for i, n in cm.items()})
        except Exception as e: stats[c] = 'failed: %s %s' % (type(e).__name__, str(e)[:200])
        log(c, stats[c] if isinstance(stats[c], str) else stats[c]['seconds'], 's')
    json.dump(dict(shard=sys.argv[2], stats=stats), open(os.path.join(out_dir, 'shard_%d_of_%d.json' % (si, sn)), 'w'))
    for attempt in range(6):
        try: api.upload_folder(folder_path=out_dir, path_in_repo=os.environ.get('LIMBS_DIR', 'limbs_v1'), repo_id=repo, repo_type='dataset', commit_message='tissue shard %s' % sys.argv[2]); break
        except Exception as e: log('upload retry', attempt, type(e).__name__); time.sleep(600)
    secs = [v['seconds'] for v in stats.values() if isinstance(v, dict)]
    have = lambda names: sum(1 for v in stats.values() if isinstance(v, dict) and any(n in v['voxels'] for n in names))
    log('SUMMARY ' + json.dumps(dict(shard=sys.argv[2], scans=len(cases), failed=sum(isinstance(v, str) for v in stats.values()), median_seconds=float(np.median(secs)) if secs else None,
        with_radius_or_ulna=have(['radius', 'ulna']), with_carpals=have(['carpal']), with_hand_bones=have(['metacarpal', 'phalanges_hand']), with_tibia_or_fibula=have(['tibia', 'fibula']), with_foot_bones=have(['tarsal', 'metatarsal', 'phalanges_feet']))))
