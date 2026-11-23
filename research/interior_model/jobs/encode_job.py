# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "torch", "monai==1.4.0"]
# ///
"""Encode whole CT scans once with the frozen MAISI autoencoder (NV-Generate-CT, research use only) and save what
every later training run needs, so nothing is encoded twice.

Per adult scan (TotalSegmentator v2.0.1, official split kept):
  code      float16 (4, X/4, Y/4, Z/4): posterior mean of the body crop (HU clipped to [-1000, 1000], scaled to [0, 1];
            outside the body set to air, so tables and blankets are not coded). Long scans are encoded in overlapping
            blocks along z; each block keeps its middle.
  mask      the body (largest 3D piece above the scan's own Otsu air/body threshold, holes filled per slice), packed bits.
  sdf       signed distance to the body surface on the code grid (cm, positive inside), float16. This is the condition.
  frame     body-frame numbers measured from the surface only: centroid and half-extents of the mask per axis, and per
            code slice the body's width and depth (for alignment by positional encoding, no resampling).
  landmarks for scoring only, never as input: centroid and box of every labelled structure present, and the centre of
            each femoral head (the femur label's highest part, the top tenth of its length).
  facts     age, sex, split.
Parallel: CPU workers prepare scans while the GPU encodes; the scan list is split into shards run as separate jobs.
One upload per job (the Hub allows 256 commits an hour).
args: <repo> <shard i/n> [limit]"""
import os, sys, io, csv, json, gzip, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def otsu(v):
    v = np.asarray(v, float); h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; tie = np.flatnonzero(s >= np.nanmax(s) * (1 - 1e-9)); return float(c[tie].mean())

def body_mask(CT):
    from scipy.ndimage import label, binary_fill_holes
    sub = CT[::4, ::4, ::4]; bt = otsu(np.clip(sub, -1024, 200)); B = CT > bt
    lab, n = label(B)
    if n == 0: return B, bt
    keep = np.bincount(lab.ravel())[1:].argmax() + 1; M = lab == keep
    for z in range(M.shape[2]): M[:, :, z] = binary_fill_holes(M[:, :, z])
    return M, bt

def prepare(CT, aff, labels, f=4):
    """CPU part for one scan. labels: iterable of (name, loader) pairs, each loader returning that label's full-size
    bool array; read one at a time after the crop is known, so only one label is in memory. Returns arrays/numbers."""
    from scipy.ndimage import distance_transform_edt
    sp = np.linalg.norm(aff[:3, :3], axis=0)
    M, bt = body_mask(CT); ii = np.argwhere(M)
    if not len(ii): raise ValueError('no body')
    lo = ii.min(0); hi = ii.max(0) + 1; hi = lo + ((hi - lo + 15) // 16) * 16                   # multiple of 16 (the autoencoder trims other sizes)
    pad = [(0, max(0, h - n)) for h, n in zip(hi, CT.shape)]
    if any(p[1] for p in pad): CT = np.pad(CT, pad, constant_values=-1000); M = np.pad(M, pad)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi)); T = np.clip(CT[sl], -1000, 1000).astype(np.float32); Mc = M[sl]; T[~Mc] = -1000
    d = (distance_transform_edt(Mc, sampling=sp) - distance_transform_edt(~Mc, sampling=sp)) / 10.0
    G = [n // f for n in Mc.shape]; sdf = d.reshape(G[0], f, G[1], f, G[2], f).mean((1, 3, 5)).astype(np.float16)
    mc = np.argwhere(Mc); cen = mc.mean(0); half = (mc.max(0) - mc.min(0) + 1) / 2.0
    wz = np.array([[np.ptp(np.flatnonzero(Mc[:, :, z].any(1))) + 1 if Mc[:, :, z].any() else 0, np.ptp(np.flatnonzero(Mc[:, :, z].any(0))) + 1 if Mc[:, :, z].any() else 0] for z in range(0, Mc.shape[2], f)], np.float32)
    lm = {}; del CT, M, d
    for name, get in (labels.items() if isinstance(labels, dict) else labels):
        L = get() if callable(get) else get
        Lc = L[sl] if not any(p_[1] for p_ in pad) else np.pad(L, pad)[sl]; del L
        p = np.argwhere(Lc)
        if not len(p): continue
        lm[name] = dict(centroid=p.mean(0).round(2).tolist(), lo=p.min(0).tolist(), hi=p.max(0).tolist(), voxels=int(len(p)))
        if name.startswith('femur_'):                                                        # femoral head: the top tenth of the femur label
            z = p[:, 2]; top = p[z >= z.max() - 0.1 * (z.max() - z.min())]; lm[name.replace('femur', 'femoral_head')] = dict(centroid=top.mean(0).round(2).tolist())
    return dict(T=T.astype(np.int16), mask=Mc, sdf=sdf, crop_lo=lo.tolist(), crop_shape=list(Mc.shape), spacing=sp.round(4).tolist(), affine=aff.tolist(), body_threshold=bt,
                frame=dict(centroid_vox=cen.round(2).tolist(), half_extent_vox=half.round(2).tolist(), slice_width_depth_vox=wz), landmarks=lm)

def encode_blocks(T, enc, block=128, overlap=32, f=4):
    """T (X,Y,Z) HU, sides multiples of 16. enc maps a [0,1] array (X,Y,z) to a code (4,X/4,Y/4,z/4)."""
    U = (T + 1000) / 2000.0; Z = U.shape[2]
    if Z <= block: return enc(U)
    starts = list(range(0, Z - block + 1, block - 2 * overlap))
    if starts[-1] + block < Z: starts.append(Z - block)
    out = None
    for i, s in enumerate(starts):
        c = enc(U[:, :, s:s + block])
        if out is None: out = np.zeros(c.shape[:3] + (Z // f,), np.float32)
        a = 0 if i == 0 else overlap // f; b = block // f if i == len(starts) - 1 else (block - overlap) // f
        out[..., s // f + a:s // f + b] = c[..., a:b]
    return out

def container_limits():
    """the job container's own CPU and memory limits (cgroup v2, then v1), not the host's."""
    cpus = None; ram = None
    try:
        q, per = open('/sys/fs/cgroup/cpu.max').read().split()
        if q != 'max': cpus = int(q) / int(per)
    except Exception: pass
    if cpus is None:
        try:
            q = int(open('/sys/fs/cgroup/cpu/cpu.cfs_quota_us').read()); per = int(open('/sys/fs/cgroup/cpu/cpu.cfs_period_us').read())
            if q > 0: cpus = q / per
        except Exception: pass
    if cpus is None: cpus = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else (os.cpu_count() or 4)
    for f in ('/sys/fs/cgroup/memory.max', '/sys/fs/cgroup/memory/memory.limit_in_bytes'):
        try:
            v = open(f).read().strip()
            if v != 'max' and int(v) < 1 << 60: ram = int(v) / 1e9; break
        except Exception: pass
    if ram is None: ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / 1e9
    return cpus, ram

_Z = None; _NAMES = None
def _init(zp):
    global _Z, _NAMES; _Z = zipfile.ZipFile(zp); _NAMES = set(_Z.namelist())
def load_and_prepare(arg):
    import nibabel as nib
    pre, case = arg
    try:
        rd = lambda n: nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(pre + n)))
        ct = rd('ct.nii.gz'); CT = np.asanyarray(ct.dataobj).astype(np.float32)
        files = sorted(x for x in _NAMES if x.startswith(pre + 'segmentations/') and x.endswith('.nii.gz'))
        labels = ((n.rsplit('/', 1)[-1][:-len('.nii.gz')], (lambda n=n: np.asanyarray(nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(n))).dataobj) > 0)) for n in files)
        r = prepare(CT, ct.affine, labels); del CT
        import resource; r['peak_rss_gb'] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2)
        return case, r
    except Exception as e: return case, 'failed: %s %s' % (type(e).__name__, str(e)[:160])

def save_case(out_dir, case, r, code, facts):
    np.savez_compressed(os.path.join(out_dir, case + '.npz'), code=code.astype(np.float16), mask=np.packbits(r['mask']), sdf=r['sdf'], slice_width_depth_vox=r['frame']['slice_width_depth_vox'])
    meta = {k: r[k] for k in ('crop_lo', 'crop_shape', 'spacing', 'affine', 'body_threshold', 'landmarks')}
    meta['frame'] = {k: v for k, v in r['frame'].items() if k != 'slice_width_depth_vox'}; meta['code_shape'] = list(code.shape); meta.update(facts)
    json.dump(meta, open(os.path.join(out_dir, case + '.json'), 'w'))

def maisi_encoder(dev):
    import torch
    from huggingface_hub import hf_hub_download
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
    net = AutoencoderKlMaisi(spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4, num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2], norm_num_groups=32, norm_eps=1e-6,
                             attention_levels=[False, False, False], with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False, use_checkpointing=False, use_convtranspose=False,
                             norm_float16=True, num_splits=4, dim_split=1)
    net.load_state_dict(torch.load(hf_hub_download('nvidia/NV-Generate-CT', 'models/autoencoder_v1.pt'), map_location='cpu', weights_only=False), strict=True); net = net.to(dev).eval()
    def enc(U):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            return net.encode(torch.tensor(U, dtype=torch.float32, device=dev)[None, None])[0].float()[0].cpu().numpy()
    return enc

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); repo = sys.argv[1]; si, sn = map(int, sys.argv[2].split('/')); limit = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); names = [i.filename for i in zipfile.ZipFile(zp).infolist() if i.filename.endswith('/ct.nii.gz')]
    pre = {n[:-len('ct.nii.gz')].rstrip('/').rsplit('/', 1)[-1]: n[:-len('ct.nii.gz')] for n in names}
    cases = sorted(c for c in pre if c in meta and not (meta[c].get('age') and float(meta[c]['age']) < 18))[si::sn]
    if os.environ.get('LARGEST'):                                                             # smoke on the biggest scans (worst-case memory)
        import nibabel as nib
        zz = zipfile.ZipFile(zp); size = {c: zz.getinfo(pre[c] + 'ct.nii.gz').file_size for c in cases}; cases = sorted(cases, key=lambda c: -size[c])
    if limit: cases = cases[:limit]
    log('shard %d/%d: %d scans' % (si, sn, len(cases)))
    enc = maisi_encoder('cuda'); out_dir = 'codes_v1'; os.makedirs(out_dir, exist_ok=True); bad = {}
    cpus, ram_gb = container_limits(); per_worker_gb = float(os.environ.get('WORKER_GB', '8'))
    nw = max(1, min(int(cpus) - 2, int(ram_gb // per_worker_gb))); log('workers', nw, 'container cpus %.1f' % cpus, 'ram GB %.0f' % ram_gb, '(host shows %d cpus)' % (os.cpu_count() or 0))
    peaks = []
    with Pool(nw, _init, (zp,), maxtasksperchild=20) as pool:                     # CPU workers prepare while the GPU encodes
        for k, (case, r) in enumerate(pool.imap_unordered(load_and_prepare, [(pre[c], c) for c in cases])):
            if isinstance(r, str): bad[case] = r; log(case, r); continue
            peaks.append(r.pop('peak_rss_gb', None)); code = encode_blocks(r['T'].astype(np.float32), enc)
            save_case(out_dir, case, r, code, dict(age=meta[case].get('age'), sex=meta[case].get('gender'), split=meta[case].get('split')))
            if (k + 1) % 20 == 0 or k + 1 == len(cases): log('encoded %d / %d' % (k + 1, len(cases)))
    pk = [p_ for p_ in peaks if p_]; log('worker peak RAM GB: median %.1f max %.1f' % (np.median(pk), max(pk)) if pk else 'no peaks')
    json.dump(dict(shard=sys.argv[2], scans=len(cases), failed=bad, worker_peak_ram_gb=max(pk) if pk else None), open(os.path.join(out_dir, 'shard_%d_of_%d.json' % (si, sn)), 'w'))
    for attempt in range(6):
        try: api.upload_folder(folder_path=out_dir, path_in_repo=os.environ.get('CODES_DIR', 'codes_v1'), repo_id=repo, repo_type='dataset', commit_message='codes shard %s' % sys.argv[2]); break
        except Exception as e: log('upload retry', attempt, type(e).__name__); time.sleep(600)
    log('SUMMARY ' + json.dumps(dict(shard=sys.argv[2], scans=len(cases), failed=len(bad))))
