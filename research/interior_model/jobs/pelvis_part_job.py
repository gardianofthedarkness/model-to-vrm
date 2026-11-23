# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy"]
# ///
"""Milestone 15 (Tokens), pelvis part: every adult pelvis put into its own pelvis frame, built from its bones. CPU only.

People: adults whose hip bones and sacrum are whole in the scan (label_extents: present, not cut).
Pelvis frame (per person, from labels_v1 centroids; no fixed numbers):
  origin  midpoint of the two femoral heads   x  left -> right femoral head (patient right)
  z       towards the centroid of vertebrae L4-L5, made orthogonal to x (superior, along the lower spine)
  y       z cross x (anterior)
Height range: the hip bones' and sacrum's own extent in the frame (as milestone 10's window). The window is the union,
over all people, of the body inside that range, rounded up to the 12 mm token. Classes: milestone 12's (licensed tissue
types + named pelvic bones and muscles + 'organ'). Labels are resampled at 1.5 mm (nearest) then reduced to 3 mm by
majority, outside the scan = 255. Output: uint8 labels in the frame, frame, split. args: <repo> [limit]"""
import os, sys, csv, json, time, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
TOKEN_MM = 12; UNK = 255
NAMED = [('hip', ['hip_left']), ('hip_r', ['hip_right']), ('sacrum', ['sacrum']), ('femur_l', ['femur_left']), ('femur_r', ['femur_right']),
         ('gluteus_max', ['gluteus_maximus_left', 'gluteus_maximus_right']), ('gluteus_med', ['gluteus_medius_left', 'gluteus_medius_right']),
         ('gluteus_min', ['gluteus_minimus_left', 'gluteus_minimus_right']), ('iliopsoas', ['iliopsoas_left', 'iliopsoas_right']),
         ('autochthon', ['autochthon_left', 'autochthon_right']), ('lumbar', ['vertebrae_L4', 'vertebrae_L5'])]
BASE = ['outside', 'subcut_fat', 'torso_fat', 'inter_fat', 'muscle_other', 'lean_other', 'gas', 'dense']
CLASSES = BASE + [n for n, _ in NAMED] + ['organ']; K = len(CLASSES); NB = len(BASE); ORGAN = K - 1; ix = {n: i for i, n in enumerate(CLASSES)}
def remap(L, ids, tissue):                                                                  # as a2b_job.remap with LABELSRC=tissue
    named = np.zeros(256, np.uint8)
    for k, (_, names) in enumerate(NAMED):
        for n in names:
            if n in ids: named[ids[n]] = NB + k
    isnamed = named[L] > 0; out = np.full(L.shape, ORGAN, np.uint8); basic = L < 10
    out[basic & np.isin(L, [1, 2])] = ix['torso_fat']; out[basic & (L == 3)] = ix['lean_other']; out[basic & (L == 4)] = ix['gas']; out[basic & (L == 5)] = ix['dense']; out[L == 0] = 0
    out[basic & (tissue == 1)] = ix['subcut_fat']; out[basic & (tissue == 2)] = ix['torso_fat']; out[basic & (tissue == 3)] = ix['muscle_other']; out[basic & (tissue == 4)] = ix['inter_fat']; out[L == 0] = 0
    out[isnamed] = named[L][isnamed]; return out
def down2(L):                                                                               # majority over 2x2x2, unknown only if all unknown
    X, Y, Z = (n // 2 * 2 for n in L.shape); b = L[:X, :Y, :Z].reshape(X // 2, 2, Y // 2, 2, Z // 2, 2).transpose(0, 2, 4, 1, 3, 5).reshape(X // 2, Y // 2, Z // 2, 8)
    cnt = np.stack([(b == k).sum(-1) for k in range(K)], -1); out = (K - 1 - np.argmax(cnt[..., ::-1], -1)).astype(np.uint8)
    out[(b == UNK).all(-1)] = UNK; return out
_A = None
def _init(a): global _A; _A = a
def load(c):
    d = _A['dirs']; m = json.load(open(os.path.join(d['codes'], c + '.json'))); L = np.load(os.path.join(d['labels'], c + '.npz'))['labels']
    T = np.load(os.path.join(d['tissue'], c + '.npz'))['tissue']; aff = np.array(m['affine']); aff[:3, 3] = aff[:3, 3] + aff[:3, :3] @ np.array(m['crop_lo'])
    return m, remap(L, _A['ids'], T), aff
def frame_of(c):
    try:
        m, L, aff = load(c); ids = _A['ids']
        def cen(names):
            ii = np.argwhere(np.isin(L, [ix[n] for n in names]))
            if not len(ii): raise ValueError('no %s label' % '/'.join(names))
            return aff[:3, :3] @ ii.mean(0) + aff[:3, 3]
        lm = m['landmarks']; fh = {s: aff[:3, :3] @ np.array(lm['femoral_head_%s' % s]['centroid']) + aff[:3, 3] for s in ('left', 'right')}
        o = (fh['left'] + fh['right']) / 2; x = fh['right'] - fh['left']; x /= np.linalg.norm(x)
        z = cen(['lumbar']) - o; z -= x * (x @ z); z /= np.linalg.norm(z); y = np.cross(z, x); R = np.stack([x, y, z])   # z towards L4-L5, y anterior
        bones = np.argwhere(np.isin(L, [ix['hip'], ix['hip_r'], ix['sacrum']]))[::7]; qb = (bones @ aff[:3, :3].T + aff[:3, 3] - o) @ R.T
        z0, z1 = qb[:, 2].min(), qb[:, 2].max(); body = np.argwhere(L[::2, ::2, ::2] > 0) * 2; q = (body @ aff[:3, :3].T + aff[:3, 3] - o) @ R.T
        q = q[(q[:, 2] >= z0) & (q[:, 2] <= z1)]
        return c, dict(origin=o.tolist(), axes=R.tolist(), lo=q.min(0).tolist(), hi=q.max(0).tolist())
    except Exception as e: return c, 'failed: %s %s' % (type(e).__name__, str(e)[:160])
def resample(arg):
    from scipy.ndimage import map_coordinates
    c, f = arg
    try:
        m, L, aff = load(c); o, R = np.array(f['origin']), np.array(f['axes']); lo, shape = np.array(_A['lo']), np.array(_A['shape15'])
        L15 = np.empty(shape, np.uint8); Ai = np.linalg.inv(aff[:3, :3]); gx, gy = np.meshgrid(*[np.arange(n) * 1.5 + 0.75 + lo[a] for a, n in enumerate(shape[:2])], indexing='ij')
        for k in range(shape[2]):                                                            # one slab at a time keeps memory small
            g = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, k * 1.5 + 0.75 + lo[2])], 1)
            L15[:, :, k] = map_coordinates(L, (((g @ R + o) - aff[:3, 3]) @ Ai.T).T, order=0, mode='constant', cval=UNK).reshape(shape[:2])
        # height range: outside the person's own bone range is not part of the pelvis window
        zz = lo[2] + np.arange(shape[2]) * 1.5 + 0.75; L15[:, :, (zz < f['z0']) | (zz > f['z1'])] = UNK
        L3 = down2(L15); np.savez_compressed(os.path.join('pelvis_v1', c + '.npz'), labels=L3)
        return c, dict(body_voxels_3mm=int(((L3 > 0) & (L3 < UNK)).sum()), unknown_share=round(float((L3 == UNK).mean()), 3))
    except Exception as e: return c, 'failed: %s %s' % (type(e).__name__, str(e)[:160])

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    ext = list(csv.DictReader(open(hf_hub_download(repo, 'census/label_extents.csv', repo_type='dataset'))))
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    whole = {}
    for e in ext:
        if e['label'] in ('hip_left', 'hip_right', 'sacrum'):
            ok = e['present'] == '1' and not any(e[k] == '1' for k in ('cut_inf', 'cut_sup', 'cut_side', 'cut_ap')); whole.setdefault(e['case'], []).append(ok)
    cases = sorted(c for c, v in whole.items() if len(v) == 3 and all(v) and c in meta and not (meta[c].get('age') and float(meta[c]['age']) < 18))
    if limit: cases = cases[:limit]
    snap = snapshot_download(repo, repo_type='dataset', allow_patterns=[p % c for c in cases for p in ('labels_v1/%s.npz', 'tissue_v1/%s.npz', 'codes_v1/%s.json')] + ['labels_v1/label_ids.json'], max_workers=16)
    ids = json.load(open(os.path.join(snap, 'labels_v1', 'label_ids.json')))['ids']
    dirs = dict(codes=os.path.join(snap, 'codes_v1'), labels=os.path.join(snap, 'labels_v1'), tissue=os.path.join(snap, 'tissue_v1'))
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 16
    try: q, per = open('/sys/fs/cgroup/cpu.max').read().split(); cpus = int(q) / int(per)
    except Exception: cpus = os.cpu_count()
    nw = max(1, min(int(cpus), int(ram // 3))); log(len(cases), 'pelvises, workers', nw)
    with Pool(nw, _init, (dict(ids=ids, dirs=dirs),), maxtasksperchild=50) as p: frames = dict(p.imap_unordered(frame_of, cases))
    good = {c: f for c, f in frames.items() if isinstance(f, dict)}; log('frames', len(good), 'failed', [v for v in frames.values() if isinstance(v, str)][:3])
    lo = np.min([f['lo'] for f in good.values()], 0); hi = np.max([f['hi'] for f in good.values()], 0)
    lo = np.floor(lo / TOKEN_MM) * TOKEN_MM; hi = np.ceil(hi / TOKEN_MM) * TOKEN_MM; shape15 = ((hi - lo) / 1.5).astype(int)
    for f in good.values(): f['z0'], f['z1'] = f['lo'][2], f['hi'][2]
    log('window mm', lo.tolist(), hi.tolist(), '3 mm grid', (shape15 // 2).tolist())
    os.makedirs('pelvis_v1', exist_ok=True)
    with Pool(nw, _init, (dict(ids=ids, dirs=dirs, lo=lo.tolist(), shape15=shape15.tolist()),), maxtasksperchild=50) as p: res = dict(p.imap_unordered(resample, list(good.items())))
    index = {c: dict(**good[c], **r, split=meta[c]['split'], age=meta[c].get('age'), sex=meta[c].get('gender')) for c, r in res.items() if isinstance(r, dict)}
    json.dump(dict(classes=CLASSES, unknown=UNK, voxel_mm=3.0, window_lo_mm=lo.tolist(), window_hi_mm=hi.tolist(), people=index,
                   failed={c: r for c, r in list(frames.items()) + list(res.items()) if isinstance(r, str)}), open('pelvis_v1/index.json', 'w'), indent=1)
    dest = 'pelvis_v1' + ('_smoke' if limit else '')
    for attempt in range(6):
        try: api.upload_folder(folder_path='pelvis_v1', path_in_repo=dest, repo_id=repo, repo_type='dataset', commit_message='pelvis part in pelvis frame'); break
        except Exception as e: log('upload retry', attempt, type(e).__name__); time.sleep(600)
    log('SUMMARY ' + json.dumps(dict(people=len(index), failed=len(cases) - len(index), grid_3mm=(shape15 // 2).tolist(), split={s: sum(1 for v in index.values() if v['split'] == s) for s in ('train', 'val', 'test')})))
