# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK"]
# ///
"""How far can TotalSegmentator's head labels be trusted? CPU only.
HaN-Seg's 42 head-and-neck CTs carry expert outlines (CC BY-ND 4.0, private; numbers only). For each structure both
sides have, compare the expert mask with TotalSegmentator's label (job head_v1, on the body crop at offset_in_scan):
Dice, mean and 95th-percentile symmetric surface distance (mm), volume ratio (TS / expert). Pairs:
mandible vs Bone_Mandible (also mandible+lower teeth), eye vs Eye_A+Eye_P, optic nerve, parotid, submandibular gland
(left and right), and tongue's share inside Cavity_Oral. Left/right are also checked crossed, to catch a side swap.
args: <hanseg repo>"""
import os, sys, json, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
PAIRS = [('mandible', 'craniofacial_structures', ['mandible'], ['Bone_Mandible']),
         ('mandible_with_lower_teeth', 'craniofacial_structures', ['mandible', 'teeth_lower'], ['Bone_Mandible'])]
for s, S in (('left', 'L'), ('right', 'R')):
    PAIRS += [('eye_' + s, 'head_glands_cavities', ['eye_' + s], ['Eye_A' + S, 'Eye_P' + S]),
              ('optic_nerve_' + s, 'head_glands_cavities', ['optic_nerve_' + s], ['OpticNrv_' + S]),
              ('parotid_' + s, 'head_glands_cavities', ['parotid_gland_' + s], ['Parotid_' + S]),
              ('submandibular_' + s, 'head_glands_cavities', ['submandibular_gland_' + s], ['Glnd_Submand_' + S])]
CROSS = [('eye', 'head_glands_cavities', 'eye_%s', ['Eye_A%s', 'Eye_P%s']), ('parotid', 'head_glands_cavities', 'parotid_gland_%s', ['Parotid_%s'])]
_Z = None; _T = None
def _init(zp, tasks):
    global _Z, _T; _Z = zipfile.ZipFile(zp); _T = tasks
def surf(m):
    from scipy.ndimage import binary_erosion
    return m & ~binary_erosion(m)
def metrics(a, b, sp):
    from scipy.ndimage import distance_transform_edt
    if not a.any() or not b.any(): return dict(dice=0.0 if (a.any() or b.any()) else None, vol_ratio=None if not b.any() else float(a.sum() / b.sum()))
    ii = np.argwhere(a | b); lo = np.maximum(ii.min(0) - 3, 0); hi = ii.max(0) + 4; sl = tuple(slice(x, y) for x, y in zip(lo, hi)); a, b = a[sl], b[sl]
    sa, sb = surf(a), surf(b); da = distance_transform_edt(~sa, sampling=sp); db = distance_transform_edt(~sb, sampling=sp)
    d = np.concatenate([db[sa], da[sb]])
    return dict(dice=float(2 * (a & b).sum() / (a.sum() + b.sum())), msd_mm=float(d.mean()), hd95_mm=float(np.percentile(d, 95)), vol_ratio=float(a.sum() / b.sum()))
def one(arg):
    import SimpleITK as sitk
    case, files, stat, npz = arg
    try:
        tmp = '/tmp/%s' % case; os.makedirs(tmp, exist_ok=True)
        ctn = [f for f in files if f.endswith('_IMG_CT.nrrd')][0]; p = tmp + '/ct.nrrd'; open(p, 'wb').write(_Z.read(ctn)); ct = sitk.ReadImage(p); os.remove(p)
        sp = np.array(ct.GetSpacing()); shp = ct.GetSize()
        L = np.load(npz); lo = np.array(stat['offset_in_scan'])
        def ts(task, names):
            inv = {v: int(k) for k, v in _T[task].items()}; A = L[task]; m = np.zeros(shp, bool)
            sub = np.isin(A, [inv[n] for n in names]); m[lo[0]:lo[0] + A.shape[0], lo[1]:lo[1] + A.shape[1], lo[2]:lo[2] + A.shape[2]] = sub; return m
        cache = {}
        def ex(name):
            if name in cache: return cache[name]
            f = [f for f in files if f.endswith('_OAR_%s.seg.nrrd' % name)]
            if not f: cache[name] = None; return None
            q = tmp + '/s.nrrd'; open(q, 'wb').write(_Z.read(f[0])); im = sitk.ReadImage(q); os.remove(q)
            if im.GetNumberOfComponentsPerPixel() > 1:
                a = sitk.GetArrayFromImage(im).any(-1).astype(np.uint8); j = sitk.GetImageFromArray(a); j.CopyInformation(im); im = j
            r = sitk.Resample(sitk.Cast(im > 0, sitk.sitkUInt8), ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
            cache[name] = sitk.GetArrayFromImage(r).transpose(2, 1, 0) > 0; return cache[name]
        def exu(names):
            ms = [ex(n) for n in names]; ms = [m for m in ms if m is not None]
            return np.logical_or.reduce(ms) if ms else None
        out = dict(case=case, voxel_mm=sp.round(3).tolist(), pairs={}, crossed={})
        assert list(L['craniofacial_structures'].shape) == stat['shape'], 'shape mismatch'
        for key, task, tn, en in PAIRS:
            e = exu(en)
            if e is not None: out['pairs'][key] = metrics(ts(task, tn), e, sp)
        for key, task, tn, en in CROSS:
            for s, S, o in (('left', 'L', 'R'), ('right', 'R', 'L')):
                e = exu([x % o for x in en])
                if e is not None: out['crossed']['%s_%s_vs_expert_%s' % (key, s, o)] = metrics(ts(task, [tn % s]), e, sp)['dice']
        e = ex('Cavity_Oral'); t = ts('head_muscles', ['tongue'])
        if e is not None and t.any(): out['tongue_share_inside_oral_cavity'] = float((t & e).sum() / t.sum())
        return out
    except Exception as e: return dict(case=case, failed='%s %s' % (type(e).__name__, str(e)[:200]))

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]
    zp = hf_hub_download(repo, 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.'); names = zipfile.ZipFile(zp).namelist()
    snap = snapshot_download(repo, repo_type='dataset', allow_patterns=['head_v1/*'])
    stats, tasks = {}, None
    for f in os.listdir(os.path.join(snap, 'head_v1')):
        if f.endswith('.json'): d = json.load(open(os.path.join(snap, 'head_v1', f))); stats.update(d['stats']); tasks = d['tasks']
    cases = sorted(c for c, s in stats.items() if isinstance(s, dict)); log(len(cases), 'cases')
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 16
    try: q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    nw = max(1, min(int(cpus), int(ram // 6))); log('workers', nw, 'ram GB', round(ram), 'cpus', cpus)
    with Pool(nw, _init, (zp, tasks), maxtasksperchild=4) as p:
        rows = list(p.imap_unordered(one, [(c, [n for n in names if '/%s/' % c in n], stats[c], os.path.join(snap, 'head_v1', c + '.npz')) for c in cases]))
    ok = [r for r in rows if 'failed' not in r]; log('ok', len(ok), 'failed', [r for r in rows if 'failed' in r][:3])
    summ = {}
    for key in [k for k, *_ in PAIRS]:
        v = [r['pairs'][key] for r in ok if key in r['pairs']]
        if not v: continue
        g = lambda m: [x[m] for x in v if x.get(m) is not None]
        summ[key] = dict(people=len(v), **{m: dict(median=round(float(np.median(g(m))), 3), p10=round(float(np.percentile(g(m), 10)), 3), p90=round(float(np.percentile(g(m), 90)), 3)) for m in ('dice', 'msd_mm', 'hd95_mm', 'vol_ratio') if g(m)})
    cr = {}
    for r in ok:
        for k, v in r['crossed'].items(): cr.setdefault(k, []).append(v)
    res = dict(people=len(ok), pairs=summ, crossed_dice_median={k: round(float(np.median(v)), 3) for k, v in cr.items()},
               tongue_share_inside_oral_cavity_median=round(float(np.median([r['tongue_share_inside_oral_cavity'] for r in ok if 'tongue_share_inside_oral_cavity' in r])), 3))
    os.makedirs('agree', exist_ok=True); json.dump(dict(summary=res, rows=rows), open('agree/head_agreement.json', 'w'), indent=1)
    api.upload_folder(folder_path='agree', path_in_repo='head_v1_agreement', repo_id=repo, repo_type='dataset', commit_message='head label agreement vs expert')
    log('SUMMARY ' + json.dumps(res))
