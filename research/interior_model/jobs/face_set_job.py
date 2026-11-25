# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK"]
# ///
"""Face training set face_v1 (design v7 milestone 16, face part). CPU only.

One uint8 label volume per head, on the scan's own grid, cropped to TotalSegmentator's 'head' region, with the
voxel-to-mm affine. Classes (face_ids.json):
  0-5   as labels_v1: outside, subcutaneous fat, other fat, other lean, gas, dense unlabelled (same HU rules, same
        body mask); 6 = another body structure (labels_v1 only) not in this list
  10+   TotalSegmentator head classes (craniofacial without 'head', glands and cavities, muscles)
  then  expert-only HaN-Seg structures (lips, oral cavity, buccal mucosa, brainstem, ...), then labels_v1 neck/brain
        structures (brain, vertebrae C1-C7, trachea)
Where an expert outline has a TotalSegmentator equivalent (mandible, eyes = anterior + posterior, optic nerves,
parotid and submandibular glands) the expert outline replaces TotalSegmentator's inside HaN-Seg (agreement check:
parotid 0.59, optic nerve 0.55 Dice). labels_v1 spinal cord, carotids, thyroid, oesophagus map to the expert classes
of the same organ. Painting order: brightness classes, TotalSegmentator, then expert (or labels_v1) structures.
Per head: which classes are known (unknown ones are ignored by the loss), the labeller (expert / totalsegmentator),
native voxel size. Skull-scan heads are kept only if every TotalSegmentator face class found in all 42 HaN-Seg heads
is also found in them (drops speck skulls and heads cut above or through the face).
Face label repair (2026-11-25): HaN-Seg heads also get brain, trachea and C1-C7 from TotalSegmentator's general model
(head_v1_total in the HaN-Seg repo, written by head_job.py with TASKS=total), painted only over plain tissue.
SOURCE=hanseg | ts.  args: <repo> [limit]"""
import os, sys, csv, json, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
TS_TASKS = ['craniofacial_structures', 'head_glands_cavities', 'head_muscles']
EXPERT_EQUIV = {'mandible': ['Bone_Mandible'], 'eye_left': ['Eye_AL', 'Eye_PL'], 'eye_right': ['Eye_AR', 'Eye_PR'], 'optic_nerve_left': ['OpticNrv_L'],
                'optic_nerve_right': ['OpticNrv_R'], 'parotid_gland_left': ['Parotid_L'], 'parotid_gland_right': ['Parotid_R'],
                'submandibular_gland_left': ['Glnd_Submand_L'], 'submandibular_gland_right': ['Glnd_Submand_R']}
EXPERT_ONLY = ['A_Carotid_L', 'A_Carotid_R', 'Arytenoid', 'Brainstem', 'BuccalMucosa', 'Cavity_Oral', 'Cochlea_L', 'Cochlea_R', 'Cricopharyngeus',
               'Esophagus_S', 'Glnd_Lacrimal_L', 'Glnd_Lacrimal_R', 'Glnd_Thyroid', 'Glottis', 'Larynx_SG', 'Lips', 'OpticChiasm', 'Pituitary', 'SpinalCord']
V1_TO = {'spinal_cord': 'SpinalCord', 'common_carotid_artery_left': 'A_Carotid_L', 'common_carotid_artery_right': 'A_Carotid_R', 'thyroid_gland': 'Glnd_Thyroid', 'esophagus': 'Esophagus_S',
         'brain': 'brain', 'trachea': 'trachea', **{'vertebrae_C%d' % i: 'vertebrae_C%d' % i for i in range(1, 8)}}
V1_NEW = ['brain', 'trachea'] + ['vertebrae_C%d' % i for i in range(1, 8)]
BASIC = {0: 'outside', 1: 'subcutaneous fat', 2: 'other fat', 3: 'other lean', 4: 'gas', 5: 'dense unlabelled', 6: 'other body structure'}
def class_list(ts_tasks):
    names = [n for t in TS_TASKS for _, n in sorted(((int(k), v) for k, v in ts_tasks[t].items())) if n != 'head']
    return {n: 10 + i for i, n in enumerate(names + EXPERT_ONLY + V1_NEW)}
def otsu(v):
    v = np.asarray(v, float); h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; tie = np.flatnonzero(s >= np.nanmax(s) * (1 - 1e-9)); return float(c[tie].mean())
def body_mask(CT):                                                                         # as labels_job.body_mask
    from scipy.ndimage import label, binary_fill_holes
    sub = CT[::4, ::4, ::4]; bt = otsu(np.clip(sub, -1024, 200)); B = CT > bt
    lab, n = label(B)
    if n == 0: return B
    M = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    for z in range(M.shape[2]): M[:, :, z] = binary_fill_holes(M[:, :, z])
    return M
def basic_labels(T, B):                                                                    # as labels_job
    from scipy.ndimage import median_filter, label
    S = median_filter(T, 3); L = np.zeros(T.shape, np.uint8); fat = B & (S >= -190) & (S <= -30)
    conn, _ = label(fat | ~B); oid = np.bincount(conn[~B].ravel()).argmax() if (~B).any() else -1
    L[B & (S > -30) & (S <= 150)] = 3; L[B & (S < -500)] = 4; L[B & (S > 150)] = 5; L[fat] = 2; L[fat & (conn == oid)] = 1; return L
def head_box(H):
    ii = np.argwhere(H); return ii.min(0), ii.max(0) + 1
_Z = None; _G = None
def _init(zp, g):
    global _Z, _G; _Z = zipfile.ZipFile(zp) if zp else None; _G = g

def one_hanseg(arg):
    import SimpleITK as sitk
    case, files, stat, npz, out_dir = arg; ids, tsk = _G['ids'], _G['tasks']
    try:
        tmp = '/tmp/f_%s' % case; os.makedirs(tmp, exist_ok=True)
        p = tmp + '/ct.nrrd'; open(p, 'wb').write(_Z.read([f for f in files if f.endswith('_IMG_CT.nrrd')][0])); ct = sitk.ReadImage(p); os.remove(p)
        A = sitk.GetArrayFromImage(ct).transpose(2, 1, 0).astype(np.float32)                 # (x, y, z) index order, as nibabel saw it
        lo = np.array(stat['offset_in_scan']); TS = np.load(npz); shp = np.array(TS['craniofacial_structures'].shape)
        inv = {t: {v: int(k) for k, v in tsk[t].items()} for t in TS_TASKS}
        hlo, hhi = head_box(TS['craniofacial_structures'] == inv['craniofacial_structures']['head'])
        g0 = lo + hlo; g1 = lo + hhi; sl = tuple(slice(a, b) for a, b in zip(g0, g1)); tsl = tuple(slice(a, b) for a, b in zip(hlo, hhi))
        T = A[sl]; del A
        B = body_mask(T); Lb = basic_labels(T, B); L = Lb.copy()
        for t in TS_TASKS:
            X = TS[t][tsl]
            for n, k in inv[t].items():
                if n != 'head': L[(X == k) & B] = ids[n]
        if case in _G.get('total', {}):                                                       # general-model structures this source never drew (brain, trachea, C1-C7); expert outlines still win
            tf, tmap = _G['total'][case]; X = np.load(tf)['total'][tsl]; tinv = {v: int(k) for k, v in tmap.items()}
            for nm in V1_NEW:
                if nm in tinv: L[(X == tinv[nm]) & B & np.isin(L, [1, 2, 3, 4, 5])] = ids[nm]      # only over plain tissue, never over a head-task structure
            known_total = [ids[nm] for nm in V1_NEW if nm in tinv]
        else: known_total = []
        cache = {}
        def ex(name):
            f = [f for f in files if f.endswith('_OAR_%s.seg.nrrd' % name)]
            if not f: return None
            q = tmp + '/s.nrrd'; open(q, 'wb').write(_Z.read(f[0])); im = sitk.ReadImage(q); os.remove(q)
            if im.GetNumberOfComponentsPerPixel() > 1:
                a = sitk.GetArrayFromImage(im).any(-1).astype(np.uint8); j = sitk.GetImageFromArray(a); j.CopyInformation(im); im = j
            r = sitk.Resample(sitk.Cast(im > 0, sitk.sitkUInt8), ct, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
            return sitk.GetArrayFromImage(r).transpose(2, 1, 0)[sl] > 0
        known = [0, 1, 2, 3, 4, 5] + [ids[n] for t in TS_TASKS for n in inv[t] if n != 'head'] + known_total
        for n, en in EXPERT_EQUIV.items():                                                    # expert replaces the equivalent class
            ms = [m for m in (ex(e) for e in en) if m is not None]
            if not ms: continue
            m = np.logical_or.reduce(ms) & B; old = L == ids[n]
            L[old & ~m] = Lb[old & ~m]
            keep_lens = np.isin(L, [ids.get('eye_lens_left', -1), ids.get('eye_lens_right', -1)])
            L[m & ~keep_lens] = ids[n]
        for n in EXPERT_ONLY:
            m = ex(n)
            if m is not None and m.any(): L[m & B] = ids[n]; known.append(ids[n])
        D = np.array(ct.GetDirection()).reshape(3, 3) * np.array(ct.GetSpacing()); org = np.array(ct.TransformIndexToPhysicalPoint([int(v) for v in g0]))
        aff = np.eye(4); aff[:3, :3] = D; aff[:3, 3] = org                                     # index (in this crop) -> mm, ITK LPS
        np.savez_compressed(os.path.join(out_dir, case + '.npz'), labels=L, affine=aff)
        cnt = np.bincount(L.ravel(), minlength=256)
        return case, dict(source='hanseg', labeller='expert+totalsegmentator', native_mm=[round(float(s), 3) for s in ct.GetSpacing()], shape=list(L.shape), offset_in_scan=g0.tolist(), frame='LPS',
                          known=sorted(set(int(k) for k in known)), present=[int(k) for k in np.flatnonzero(cnt)], body_voxels=int(B.sum()))
    except Exception as e: return case, 'failed: %s %s' % (type(e).__name__, str(e)[:200])

def one_ts(arg):
    case, v1npz, headnpz, stat, cm, out_dir = arg; ids, tsk, v1ids, need = _G['ids'], _G['tasks'], _G['v1ids'], _G['need']
    try:
        TS = np.load(headnpz); inv = {t: {v: int(k) for k, v in tsk[t].items()} for t in TS_TASKS}
        found = {n for t in TS_TASKS for n, k in inv[t].items() if (TS[t] == k).any()}
        missing = sorted(set(need) - found)
        if missing: return case, 'skipped: face classes missing %s' % missing[:6]
        V = np.load(v1npz)['labels']; off = np.array(stat['offset_in_codes_crop']); shp = np.array(TS['craniofacial_structures'].shape)
        H = TS['craniofacial_structures'] == inv['craniofacial_structures']['head']; hlo, hhi = head_box(H)
        a0 = np.maximum(off + hlo, 0); a1 = np.minimum(off + hhi, V.shape); b0 = a0 - off; b1 = a1 - off
        L = V[tuple(slice(a, b) for a, b in zip(a0, a1))].copy(); B = L > 0
        v1inv = {v: k for k, v in v1ids.items()}; struct = L >= 10; newL = L.copy()
        newL[struct] = 6
        for nm, tgt in V1_TO.items():
            if nm in v1ids: newL[L == v1ids[nm]] = ids[tgt]
        L = newL; known = [0, 1, 2, 3, 4, 5, 6] + [ids[n] for t in TS_TASKS for n in inv[t] if n != 'head'] + sorted({ids[t] for t in V1_TO.values()})
        for t in TS_TASKS:
            X = TS[t][tuple(slice(a, b) for a, b in zip(b0, b1))]
            for n, k in inv[t].items():
                if n != 'head': L[(X == k) & B] = ids[n]
        aff = np.array(cm['affine']); aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ (np.array(cm['crop_lo']) + a0)   # index (in this crop) -> mm, nibabel RAS
        np.savez_compressed(os.path.join(out_dir, case + '.npz'), labels=L, affine=aff)
        cnt = np.bincount(L.ravel(), minlength=256)
        return case, dict(source='ts', labeller='totalsegmentator', native_mm=cm['spacing'], shape=list(L.shape), offset_in_codes_crop=a0.tolist(), frame='RAS',
                          known=sorted(set(known)), present=[int(k) for k in np.flatnonzero(cnt)], body_voxels=int(B.sum()), split=cm.get('split'))
    except Exception as e: return case, 'failed: %s %s' % (type(e).__name__, str(e)[:200])

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0; SRC = os.environ.get('SOURCE', 'hanseg')
    HS, TSR = 'gardianofthedarkness/hanseg-ct-mr', 'gardianofthedarkness/totalsegmentator-ct-v201'
    hs = snapshot_download(HS, repo_type='dataset', allow_patterns=['head_v1/*.json'] + (['head_v1/*.npz', 'head_v1_total/*'] if SRC == 'hanseg' else []))
    tot = {}                                                                                 # face label repair: TotalSegmentator 'total' labels on HaN-Seg (brain, trachea, vertebrae ...)
    if os.path.isdir(os.path.join(hs, 'head_v1_total')):
        for f in os.listdir(os.path.join(hs, 'head_v1_total')):
            if f.endswith('.json'): d = json.load(open(os.path.join(hs, 'head_v1_total', f))); tot.update({c: (os.path.join(hs, 'head_v1_total', c + '.npz'), d['tasks']['total']) for c, s_ in d['stats'].items() if isinstance(s_, dict)})
    log('general-model labels for', len(tot), 'HaN-Seg heads')
    hstats, tasks = {}, None
    for f in os.listdir(os.path.join(hs, 'head_v1')):
        if f.endswith('.json'): d = json.load(open(os.path.join(hs, 'head_v1', f))); hstats.update(d['stats']); tasks = d['tasks']
    ids = class_list(tasks)
    # face classes found by TotalSegmentator in every HaN-Seg head: the completeness test for skull-scan heads
    inv = {t: {v: int(k) for k, v in tasks[t].items()} for t in TS_TASKS}
    hs_ok = [s for s in hstats.values() if isinstance(s, dict)]
    need = sorted(n for t in TS_TASKS for n in inv[t] if n != 'head' and all(n in s['tasks'][t]['voxels'] for s in hs_ok))
    log('classes', len(ids), 'completeness classes', need)
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 16
    try: q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    out_dir = 'face_v1'; os.makedirs(out_dir, exist_ok=True); stats = {}
    if SRC == 'hanseg':
        zp = hf_hub_download(HS, 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.'); names = zipfile.ZipFile(zp).namelist()
        cases = sorted(c for c, s in hstats.items() if isinstance(s, dict))
        if limit: cases = cases[:limit]
        nw = max(1, min(int(cpus), int(ram // 7))); log('hanseg', len(cases), 'workers', nw)
        with Pool(nw, _init, (zp, dict(ids=ids, tasks=tasks, total=tot)), maxtasksperchild=4) as p:
            for c, r in p.imap_unordered(one_hanseg, [(c, [n for n in names if '/%s/' % c in n], hstats[c], os.path.join(hs, 'head_v1', c + '.npz'), out_dir) for c in cases]): stats[c] = r; log(c, r if isinstance(r, str) else r['shape'])
        target = HS
    else:
        ts = snapshot_download(TSR, repo_type='dataset', allow_patterns=['head_v1/*', 'labels_v1/label_ids.json'])
        tstats = {}
        for f in os.listdir(os.path.join(ts, 'head_v1')):
            if f.endswith('.json'): tstats.update(json.load(open(os.path.join(ts, 'head_v1', f)))['stats'])
        cases = sorted(c for c, s in tstats.items() if isinstance(s, dict))
        if limit: cases = cases[:limit]
        ts2 = snapshot_download(TSR, repo_type='dataset', allow_patterns=['labels_v1/%s.npz' % c for c in cases] + ['codes_v1/%s.json' % c for c in cases])
        v1ids = json.load(open(os.path.join(ts, 'labels_v1', 'label_ids.json')))['ids']
        nw = max(1, min(int(cpus), int(ram // 4))); log('ts', len(cases), 'workers', nw)
        with Pool(nw, _init, (None, dict(ids=ids, tasks=tasks, v1ids=v1ids, need=need)), maxtasksperchild=10) as p:
            args = [(c, os.path.join(ts2, 'labels_v1', c + '.npz'), os.path.join(ts, 'head_v1', c + '.npz'), tstats[c], json.load(open(os.path.join(ts2, 'codes_v1', c + '.json'))), out_dir) for c in cases]
            for c, r in p.imap_unordered(one_ts, args): stats[c] = r; log(c, r if isinstance(r, str) else r['shape'])
        target = TSR
    json.dump(dict(ids=ids, basic=BASIC, expert_equivalents=EXPERT_EQUIV, v1_mapping=V1_TO, completeness_classes=need), open(os.path.join(out_dir, 'face_ids.json'), 'w'), indent=1)
    json.dump(dict(source=SRC, stats=stats), open(os.path.join(out_dir, 'face_%s%s.json' % (SRC, '_smoke' if limit else '')), 'w'), indent=1)
    dest = 'face_v1' + ('_smoke' if limit else '')
    for attempt in range(6):
        try: api.upload_folder(folder_path=out_dir, path_in_repo=dest, repo_id=target, repo_type='dataset', commit_message='face set %s' % SRC); break
        except Exception as e: log('upload retry', attempt, type(e).__name__); time.sleep(600)
    ok = [r for r in stats.values() if isinstance(r, dict)]
    log('SUMMARY ' + json.dumps(dict(source=SRC, scans=len(stats), kept=len(ok), skipped=sum(isinstance(r, str) and r.startswith('skipped') for r in stats.values()),
                                     failed=[c for c, r in stats.items() if isinstance(r, str) and r.startswith('failed')][:5], classes=len(ids))))
