# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel"]
# ///
"""Label volumes v1 for every adult CT scan (design v7 section 3), on exactly the crop and 1.5 mm grid of codes_v1.

Per voxel (uint8):
  0      outside the body
  1      subcutaneous fat: fat (-190..-30 HU after a 3x3x3 median) connected to the outside through fat only
  2      other fat (visceral, intermuscular, marrow fat not inside a bone label)
  3      other lean (-29..150 HU, no structure label)
  4      gas inside the body (< -500 HU)
  5      dense unlabelled (> 150 HU, no structure label: calcification, contrast, unlabelled bone)
  10+    TotalSegmentator structures, id = 10 + index in the sorted list of its 117 label names (written to label_ids.json)
A structure label always wins over a brightness class. Body mask and crop are recomputed with the encoding job's own
function, and checked against the stored crop shape. Licensed tissue labels (tissue_types) come in a later version.
args: <repo> <shard i/n> [limit]"""
import os, sys, csv, json, gzip, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def otsu(v):
    v = np.asarray(v, float); h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; tie = np.flatnonzero(s >= np.nanmax(s) * (1 - 1e-9)); return float(c[tie].mean())
def body_mask(CT):                                                                         # identical to encode_job.body_mask
    from scipy.ndimage import label, binary_fill_holes
    sub = CT[::4, ::4, ::4]; bt = otsu(np.clip(sub, -1024, 200)); B = CT > bt
    lab, n = label(B)
    if n == 0: return B, bt
    keep = np.bincount(lab.ravel())[1:].argmax() + 1; M = lab == keep
    for z in range(M.shape[2]): M[:, :, z] = binary_fill_holes(M[:, :, z])
    return M, bt
_Z = None; _NAMES = None; IDS = None
def _init(zp, ids):
    global _Z, _NAMES, IDS; _Z = zipfile.ZipFile(zp); _NAMES = set(_Z.namelist()); IDS = ids
def one(arg):
    import nibabel as nib
    from scipy.ndimage import median_filter, label
    pre, case, meta, out_dir = arg
    try:
        rd = lambda n: np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(pre + n))).dataobj)
        CT = rd('ct.nii.gz').astype(np.float32); M, _ = body_mask(CT)
        lo = np.array(meta['crop_lo']); shp = np.array(meta['crop_shape']); hi = lo + shp
        pad = [(0, max(0, h - n)) for h, n in zip(hi, CT.shape)]
        if any(p[1] for p in pad): CT = np.pad(CT, pad, constant_values=-1000); M = np.pad(M, pad)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi)); T = CT[sl]; B = M[sl]; del CT, M
        S = median_filter(T, 3); L = np.zeros(T.shape, np.uint8)
        fat = B & (S >= -190) & (S <= -30)
        conn, _ = label(fat | ~B); outside_id = np.bincount(conn[~B].ravel()).argmax() if (~B).any() else -1
        sc = fat & (conn == outside_id)
        L[B & (S > -30) & (S <= 150)] = 3; L[B & (S < -500)] = 4; L[B & (S > 150)] = 5; L[fat] = 2; L[sc] = 1
        files = sorted(x for x in _NAMES if x.startswith(pre + 'segmentations/') and x.endswith('.nii.gz'))
        for n in files:
            name = n.rsplit('/', 1)[-1][:-len('.nii.gz')]
            Lab = np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(n))).dataobj) > 0
            if any(p[1] for p in pad): Lab = np.pad(Lab, pad)
            Lab = Lab[sl] & B
            if Lab.any(): L[Lab] = IDS[name]
            del Lab
        np.savez_compressed(os.path.join(out_dir, case + '.npz'), labels=L)
        counts = np.bincount(L.ravel(), minlength=10 + len(IDS))
        return case, dict(shape=list(L.shape), subcut_fat=int(counts[1]), other_fat=int(counts[2]), lean=int(counts[3]), body=int(B.sum()), structures=int((counts[10:] > 0).sum()))
    except Exception as e: return case, 'failed: %s %s' % (type(e).__name__, str(e)[:160])

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; si, sn = map(int, sys.argv[2].split('/')); limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); allnames = zipfile.ZipFile(zp).namelist()
    names = [n for n in allnames if n.endswith('/ct.nii.gz')]; pre = {n[:-len('ct.nii.gz')].rstrip('/').rsplit('/', 1)[-1]: n[:-len('ct.nii.gz')] for n in names}
    labnames = sorted({n.rsplit('/', 1)[-1][:-len('.nii.gz')] for n in allnames if '/segmentations/' in n and n.endswith('.nii.gz')}); ids = {nm: 10 + i for i, nm in enumerate(labnames)}
    assert 10 + len(ids) <= 255, len(ids)
    cases = sorted(c for c in pre if c in meta and not (meta[c].get('age') and float(meta[c]['age']) < 18))[si::sn]
    if limit: cases = cases[:limit]
    snap = snapshot_download(repo, repo_type='dataset', allow_patterns=['codes_v1/%s.json' % c for c in cases]); cmeta = {c: json.load(open(os.path.join(snap, 'codes_v1', c + '.json'))) for c in cases}
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 16
    nw = max(1, min(int(cpus), int(ram // 5))); log('shard %s: %d scans, %d labels, %d workers' % (sys.argv[2], len(cases), len(ids), nw))
    out_dir = 'labels_v1'; os.makedirs(out_dir, exist_ok=True); stats = {}
    with Pool(nw, _init, (zp, ids), maxtasksperchild=20) as p:
        for k, (c, r) in enumerate(p.imap_unordered(one, [(pre[c], c, cmeta[c], out_dir) for c in cases])):
            stats[c] = r
            if (k + 1) % 25 == 0: log('labelled', k + 1)
    bad = {c: r for c, r in stats.items() if isinstance(r, str)}
    json.dump(dict(ids=ids, classes={0: 'outside', 1: 'subcutaneous fat', 2: 'other fat', 3: 'other lean', 4: 'gas', 5: 'dense unlabelled'}), open(os.path.join(out_dir, 'label_ids.json'), 'w'), indent=1)
    json.dump(dict(shard=sys.argv[2], stats=stats, failed=bad), open(os.path.join(out_dir, 'shard_%d_of_%d.json' % (si, sn)), 'w'))
    for attempt in range(6):
        try: api.upload_folder(folder_path=out_dir, path_in_repo=os.environ.get('LABELS_DIR', 'labels_v1'), repo_id=repo, repo_type='dataset', commit_message='labels shard %s' % sys.argv[2]); break
        except Exception as e: log('upload retry', attempt, type(e).__name__); time.sleep(600)
    log('SUMMARY ' + json.dumps(dict(shard=sys.argv[2], scans=len(cases), failed=len(bad), example=next(iter(stats.items())) if stats else None)))
