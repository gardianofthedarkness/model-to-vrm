# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK"]
# ///
"""IXI brightness normalisation (CPU, HF Jobs; user 2026-11-27: "normalize first", before any contrastive training).
The step-1 pictures showed the MRI labeller (trained on HaN-Seg T1) calling IXI brain skull or soft tissue on some HH and
Guys heads: a contrast gap. Two standard steps, no training:
1. N4 bias-field correction (SimpleITK) of every prepared IXI T1 inside its body: removes the slow brightness drift across
   the head that makes one tissue look like another.
2. Histogram standardisation (Nyul & Udupa 2000): the IXI head's brightness deciles (1st, 10th..90th, 99th percentile)
   are mapped piecewise-linearly onto the mean deciles of the HaN-Seg T1 heads, as the labeller saw them (0.5-99.5
   percentile scaling inside the body). Deciles are taken over the part of the head both sets cover: from the top of the
   head down by the median IXI head length (from the data); HaN-Seg heads whose top is cut off are left out of the reference.
Output <ixi repo>/ixi_norm_v1/<id>.npz: M (float16, same grid as ixi_prep_v1) + reference.json (reference deciles, IXI
head length, per-subject deciles before and after). Private repos only (real people).
env: LIMIT (smoke), WORKERS, HANSEG_REPO (default <user>/hanseg-ct-mr). args: <ixi repo (ixi-t1-t2)>"""
import os, sys, json, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
OUT = 'ixi_norm_v1'; SP = 1.5; Q = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]          # Nyul-Udupa decile landmarks

def otsu(v, bins=256):                                                                     # as ixi_prep_job.py
    v = np.asarray(v, float); h, e = np.histogram(v, bins); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; tie = np.flatnonzero(s >= np.nanmax(s) * (1 - 1e-9)); return float(c[tie].mean())

def body_of(M):                                                                            # as ixi_prep_job.py
    from scipy.ndimage import label, binary_fill_holes, binary_opening
    b = M > otsu(M[M > 0]) * 0.5; b = binary_opening(b, iterations=1); lab, n = label(b)
    b = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    for z in range(b.shape[0]): b[z] = binary_fill_holes(b[z])
    return b

def top_slab(body, length_mm):
    """body voxels from the top of the head (LPS grid: last axis-0 index with body is the top) down by length_mm"""
    zs = np.flatnonzero(body.any((1, 2))); top = zs.max(); keep = np.zeros_like(body); lo = max(0, top - int(round(length_mm / SP)) + 1)
    keep[lo:top + 1] = body[lo:top + 1]; return keep, top == body.shape[0] - 1

def hanseg_deciles(args):
    zp, n, L = args
    import SimpleITK as sitk
    try:
        z = zipfile.ZipFile(zp); p = 'tmp_%s.nrrd' % os.getpid(); open(p, 'wb').write(z.read(n)); im = sitk.ReadImage(p); os.remove(p)
        im = sitk.DICOMOrient(sitk.Cast(im, sitk.sitkFloat32), 'LPS')
        size = [int(round(s * zz / SP)) for s, zz in zip(im.GetSize(), im.GetSpacing())]
        ref = sitk.Image(size, sitk.sitkFloat32); ref.SetSpacing([SP] * 3); ref.SetOrigin(im.GetOrigin()); ref.SetDirection(im.GetDirection())
        M = sitk.GetArrayFromImage(sitk.Resample(im, ref, sitk.Transform(), sitk.sitkLinear, 0.0)); b = body_of(M)
        lo, hi = np.percentile(M[b], [0.5, 99.5]); Mn = np.clip((M - lo) / max(hi - lo, 1e-6), 0, 1)
        slab, cut = top_slab(b, L)
        return n, None if cut else np.percentile(Mn[slab], Q), 'top of head cut off by the field of view' if cut else None
    except Exception as e: return n, None, '%s %s' % (type(e).__name__, str(e)[:200])

def n4(sid_path):
    import SimpleITK as sitk
    sid, path = sid_path
    try:
        z = np.load(path); sh = tuple(z['shape']); M = z['M'].astype(np.float32); b = np.unpackbits(z['body'])[:np.prod(sh)].reshape(sh).astype(bool)
        im = sitk.GetImageFromArray(M + 1e-3); mk = sitk.GetImageFromArray(b.astype(np.uint8))
        sh2 = sitk.Shrink(im, [2] * 3); mk2 = sitk.Shrink(mk, [2] * 3)                    # bias field fitted at 3 mm, applied at 1.5 mm
        f = sitk.N4BiasFieldCorrectionImageFilter(); f.Execute(sh2, mk2)
        logb = sitk.GetArrayFromImage(f.GetLogBiasFieldAsImage(im)); C = M / np.exp(logb)
        lo, hi = np.percentile(C[b], [0.5, 99.5]); C = np.clip((C - lo) / max(hi - lo, 1e-6), 0, 1)
        return sid, C.astype(np.float32), b
    except Exception as e: return sid, None, '%s %s' % (type(e).__name__, str(e)[:200])

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    import SimpleITK as sitk; sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    api = HfApi(); repo = sys.argv[1]; user = api.whoami()['name']; hrepo = os.environ.get('HANSEG_REPO', user + '/hanseg-ct-mr')
    nw = int(os.environ.get('WORKERS', os.cpu_count() or 4)); os.makedirs(OUT, exist_ok=True)
    idx = json.load(open(hf_hub_download(repo, 'ixi_prep_v1/index.json', repo_type='dataset', local_dir='in')))['subjects']
    subs = sorted(idx)
    if os.environ.get('LIMIT'): subs = subs[:int(os.environ['LIMIT'])]
    d = snapshot_download(repo, repo_type='dataset', allow_patterns=['ixi_prep_v1/%s.npz' % s for s in subs], local_dir='in', max_workers=16)
    log(len(subs), 'IXI subjects;', nw, 'workers')
    # 1. N4 on every IXI head (kept in memory as float16 until the reference exists)
    C, B, fail = {}, {}, {}
    with Pool(nw) as p:
        for i, (sid, c, b) in enumerate(p.imap_unordered(n4, [(s, os.path.join(d, 'ixi_prep_v1', s + '.npz')) for s in subs])):
            if c is None: fail[sid] = b
            else: C[sid] = c.astype(np.float16); B[sid] = np.packbits(b)
            if (i + 1) % 50 == 0: log('N4', i + 1, 'of', len(subs))
    log('N4 done', len(C), 'failed', fail)
    shp = {s: tuple(np.load(os.path.join(d, 'ixi_prep_v1', s + '.npz'))['shape']) for s in C}
    unb = lambda s: np.unpackbits(B[s])[:np.prod(shp[s])].reshape(shp[s]).astype(bool)
    # 2. shared region: IXI head length from the top of the head (median over IXI, data)
    lens = [(lambda zs: (zs.max() - zs.min() + 1) * SP)(np.flatnonzero(unb(s).any((1, 2)))) for s in C]; L = float(np.median(lens)); log('IXI median head length mm', L)
    # 3. HaN-Seg reference deciles over the same slab
    zp = hf_hub_download(hrepo, 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.')
    mrs = sorted(n for n in zipfile.ZipFile(zp).namelist() if n.endswith('_IMG_MR_T1.nrrd'))
    if os.environ.get('LIMIT'): mrs = mrs[:max(3, int(os.environ['LIMIT']))]
    ref_q, ref_used, ref_left = [], [], {}
    with Pool(min(nw, 8)) as p:
        for n, q, why in p.imap_unordered(hanseg_deciles, [(zp, n, L) for n in mrs]):
            case = n.split('/')[-2]
            if q is None: ref_left[case] = why; continue
            ref_q.append(q); ref_used.append(case)
    R = np.mean(ref_q, 0); log('reference from', len(ref_used), 'HaN-Seg heads; left out', ref_left); log('reference deciles', np.round(R, 3).tolist())
    # 4. map each IXI head's slab deciles onto the reference; write
    per = {}
    def flush():
        for a in range(5):
            try: api.upload_folder(folder_path=OUT, path_in_repo=OUT, repo_id=repo, repo_type='dataset', allow_patterns=['*.npz'], commit_message='IXI normalised batch'); break
            except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
        for f in os.listdir(OUT):
            if f.endswith('.npz'): os.remove(os.path.join(OUT, f))
    for i, s in enumerate(sorted(C)):
        c = C[s].astype(np.float32); b = unb(s); slab, _ = top_slab(b, L); q = np.percentile(c[slab], Q)
        x = np.concatenate([[0.0], q, [1.0]]); y = np.concatenate([[0.0], R, [1.0]])          # ends pinned at the 0..1 scale ends
        x = np.maximum.accumulate(x) + np.arange(len(x)) * 1e-6                                # strictly increasing for interpolation
        Mo = np.clip(np.interp(c, x, y), 0, 1).astype(np.float16)
        old = np.load(os.path.join(d, 'ixi_prep_v1', s + '.npz'))['M'].astype(np.float32)
        per[s] = dict(before=np.round(np.percentile(old[slab], Q), 3).tolist(), after_n4=np.round(q, 3).tolist(), after=np.round(np.percentile(Mo.astype(np.float32)[slab], Q), 3).tolist())
        np.savez_compressed('%s/%s.npz' % (OUT, s), M=Mo); del C[s]
        if (i + 1) % 50 == 0: log('written', i + 1); flush()
    flush()
    gap = lambda k: float(np.median([np.abs(np.array(v[k]) - R).mean() for v in per.values()]))
    summ = dict(subjects=len(per), failed=fail, ixi_head_length_mm=L, reference_heads=ref_used, reference_left_out=ref_left, reference_deciles=np.round(R, 4).tolist(),
                median_decile_gap_to_reference=dict(before=round(gap('before'), 4), after_n4=round(gap('after_n4'), 4), after=round(gap('after'), 4)))
    json.dump(dict(summary=summ, deciles=Q, per_subject=per), open('reference.json', 'w'), indent=1)
    api.upload_file(path_or_fileobj='reference.json', path_in_repo=OUT + '/reference.json', repo_id=repo, repo_type='dataset', commit_message='IXI normalisation reference')
    log('SUMMARY ' + json.dumps({k: v for k, v in summ.items() if k not in ('reference_heads',)})); log('SUMMARY done')
