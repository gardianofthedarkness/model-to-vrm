# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK", "scikit-image"]
# ///
"""Head labels in the new format (plan v8 14.5, user decision 2026-11-26) for HaN-Seg and SynthRAD centre A, in each
head's existing head frame and the shared 1 mm window of face_v1_headframe. CPU. Per head <case>.npz:
  field   raw CT brightness interpolated (linear) into the frame, HU, float16; outside the body -1000
  eye     eyes and lenses as a fraction 0-255 (HaN-Seg: labelling model's masks interpolated; SynthRAD: its head-frame eye label)
  subcut  fat connected to the outside (fat under the skin), packed bits, from the 1 mm field
  unk     unknown, packed bits: outside the scan; never-measured fill (SynthRAD only: exactly the scan minimum, opened 3^3: its defacing wedge); dental metal (saturated voxels) and its streaks (projection-interpolation
          streak map above the noise far from the metal); bone-bright voxels found only in metal slices
  hard    the old labels for the same head (median filter, hard classes, nearest neighbour; eye 16; unknown 255): the baseline
Tissue types are read from the field at use time: air < -500, fat -190..-30, lean -30..150, bone > 150.
env: LIMIT, WORKERS, REDO (source to rebuild). Uploads every 10 heads (<case>.npz + <case>.json) and on restart skips heads already uploaded. args: <repo for output> (hanseg-ct-mr) <synthrad repo>"""
import os, sys, json, time, zipfile, numpy as np
from multiprocessing import Pool
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
G = {}
def _init(g):
    G.update(g); G['zh'] = zipfile.ZipFile(g['zh_path']); G['zs'] = zipfile.ZipFile(g['zs_path'])

def streak_unknown(A, body):
    """metal and its streaks on the CT's own grid (A indexed x, y, z)"""
    from scipy import ndimage as ndi
    from skimage.transform import radon, iradon
    top = float(A.max()); metal = A >= top - 1e-3; lb, n = ndi.label(metal)
    if n == 0: return np.zeros(A.shape, bool), 0
    sz = np.bincount(lb.ravel())[1:]; metal = np.isin(lb, np.flatnonzero(sz >= 8) + 1)
    if metal.sum() < 30: return np.zeros(A.shape, bool), 0
    zs = np.flatnonzero(metal.any((0, 1))); streak = np.zeros(A.shape, np.float32)
    for z in zs:
        S = A[:, :, z]; M = metal[:, :, z]; nn = max(S.shape); pad = [(0, nn - S.shape[0]), (0, nn - S.shape[1])]
        s = ndi.zoom(np.pad(S, pad, constant_values=-1000.0) + 1000.0, 0.5, order=1); m = ndi.zoom(np.pad(M, pad).astype(np.float32), 0.5, order=1) > 0.05
        th = np.linspace(0.0, 180.0, max(s.shape), endpoint=False); P = radon(s, theta=th, circle=False); Pm = radon(m.astype(np.float32), theta=th, circle=False) > 1e-3; Pc = P.copy()
        for j in range(P.shape[1]):
            b = Pm[:, j]
            if b.any() and (~b).sum() > 2: Pc[b, j] = np.interp(np.flatnonzero(b), np.flatnonzero(~b), P[~b, j])
        dd = iradon(Pc, theta=th, circle=False, output_size=s.shape[0]) - iradon(P, theta=th, circle=False, output_size=s.shape[0])
        d = ndi.zoom(dd, (nn / dd.shape[0], nn / dd.shape[1]), order=1)[:S.shape[0], :S.shape[1]]   # back to the exact padded size, then the slice
        streak[:, :, z] = np.abs(d)
    aff = np.zeros(A.shape, bool); aff[:, :, zs] = True; far = ndi.distance_transform_edt(~metal) > 60
    noise = float(np.percentile(streak[far & aff & body], 99)) if (far & aff & body).any() else 1e9
    unk = ((streak > noise) | metal) & body
    bl, _ = ndi.label(A > 150); clean = np.unique(bl[:, :, ~aff[0, 0]]); unk |= (bl > 0) & ~np.isin(bl, clean) & body
    return unk, len(zs)

def one(arg):
    import SimpleITK as sitk
    from scipy import ndimage as ndi
    case, src = arg
    try:
        W = '/tmp/hf_%s' % case; os.makedirs(W, exist_ok=True)
        if src == 'hanseg':
            Z = G['zh']; n = [x for x in Z.namelist() if x.endswith('%s/%s_IMG_CT.nrrd' % (case, case))][0]; p = W + '/ct.nrrd'
        else:
            Z = G['zs']; n = G['sr_paths'][case]; p = W + '/ct.mha'
        open(p, 'wb').write(Z.read(n)); im = sitk.ReadImage(p); os.remove(p)
        A = sitk.GetArrayFromImage(im).transpose(2, 1, 0).astype(np.float32); sp = np.array(im.GetSpacing()); D = np.array(im.GetDirection()).reshape(3, 3); org = np.array(im.GetOrigin())
        body = ndi.binary_fill_holes(ndi.median_filter(A, 3) > -500)
        unk, nmetal = streak_unknown(A, body)
        # never measured: SynthRAD's defacing wedge and field-of-view padding are set to exactly the scan minimum, while its real
        # air is noisy around -1000 (checked 2026-11-26). Not for HaN-Seg: its scanner clamps real air to the minimum too.
        fill = ndi.binary_opening(A <= A.min(), np.ones((3, 3, 3), bool)) if src in G['fill_sources'] else np.zeros(A.shape, bool)
        unk |= fill
        # old-style hard labels on the CT grid (as face_set_job.basic_labels)
        S = ndi.median_filter(A, 3); L = np.zeros(A.shape, np.uint8); fat = body & (S >= -190) & (S <= -30); fl, _ = ndi.label(fat | ~body); oid = np.bincount(fl[~body].ravel()).argmax()
        L[body & (S > -30) & (S <= 150)] = 3; L[body & (S < -500)] = 4; L[body & (S > 150)] = 5; L[fat] = 2; L[fat & (fl == oid)] = 1
        m = G['frames'][case]; o, R = np.array(m['origin_mm']), np.array(m['axes']); lo, hi = G['lo'], G['hi']; shape = (hi - lo).astype(int)
        g = np.stack(np.meshgrid(*[np.arange(k) + 0.5 for k in shape], indexing='ij'), -1).reshape(-1, 3) + lo
        idx = ((g @ R + o - org) @ np.linalg.inv(D * sp).T).T                                    # frame mm -> CT index (x, y, z)
        inside = np.all((idx >= -0.5) & (idx <= np.array(A.shape)[:, None] - 0.5), 0).reshape(shape)
        F = ndi.map_coordinates(np.where(body, A, -1000.0), idx, order=1, mode='nearest').reshape(shape).astype(np.float32)
        U = (ndi.map_coordinates(unk.astype(np.float32), idx, order=1, mode='constant', cval=0).reshape(shape) > 0.25) | ~inside
        H = ndi.map_coordinates(L, idx, order=0, mode='constant', cval=255).reshape(shape).astype(np.uint8)
        if src == 'hanseg':
            T = np.load(os.path.join(G['hv'], case + '.npz'))['head_glands_cavities']; inv = G['inv']; off = np.array(G['meta'][case]['offset_in_scan'])
            em = np.zeros(A.shape, np.float32); t = np.isin(T, [inv[k] for k in ('eye_left', 'eye_right', 'eye_lens_left', 'eye_lens_right')]); s = t.shape; em[off[0]:off[0] + s[0], off[1]:off[1] + s[1], off[2]:off[2] + s[2]] = t
            E = ndi.map_coordinates(em, idx, order=1, mode='constant', cval=0).reshape(shape)
        else:
            E = (np.load(os.path.join(G['srh'], case + '.npz'))['labels'] == G['eye_id']).astype(np.float32)
        H[E > 0.5] = G['eye_id']; H[U & ~inside] = 255; H[U] = 255
        bodyF = F > -500; airF = F < -500
        al, _ = ndi.label(airF | ~bodyF); outer = np.isin(al, np.unique(al[(~bodyF) & ~inside | (F <= -999)])) & (al > 0)
        fatF = (F >= -190) & (F <= -30) & ~outer; fl2, _ = ndi.label(fatF | outer); sub = fatF & np.isin(fl2, np.unique(fl2[outer])) & (fl2 > 0)
        np.savez_compressed(os.path.join(G['out'], case + '.npz'), field=F.astype(np.float16), eye=np.clip(E * 255, 0, 255).astype(np.uint8), subcut=np.packbits(sub), unk=np.packbits(U), hard=H, shape=np.array(shape))
        return case, dict(source=src, fill_ml=round(float(fill.sum() * np.prod(sp)) / 1000, 1), metal_slices=int(nmetal), unknown_ml=round(float(U[inside].sum()) / 1000, 1), outside_scan=round(float((~inside).mean()), 3), eye_ml=round(float((E > 0.5).sum()) / 1000, 1))
    except Exception as e: return case, 'failed: %s %s' % (type(e).__name__, str(e)[:200])

if __name__ == '__main__':
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    repo, srepo = sys.argv[1:3]; api = HfApi(); LIMIT = int(os.environ.get('LIMIT', '0'))
    fi = json.load(open(hf_hub_download('gardianofthedarkness/totalsegmentator-ct-v201', 'face_v1_headframe/index.json', repo_type='dataset')))
    ids = fi['ids']; lo, hi = np.array(fi['window_lo_mm']), np.array(fi['window_hi_mm'])
    hsf = {c: v for c, v in fi['heads'].items() if v['source'] == 'hanseg'}
    srx = json.load(open(hf_hub_download(srepo, 'face_v1_headframe/index_synthrad.json', repo_type='dataset')))['heads']; srA = {c: v for c, v in srx.items() if c[6] == 'A'}
    zh = hf_hub_download('gardianofthedarkness/hanseg-ct-mr', 'raw/HaN-Seg.zip', repo_type='dataset'); zs = hf_hub_download(srepo, 'raw/14918214/synthRAD2025_Task1_Train.zip', repo_type='dataset')   # normal cache (large files)
    for f_ in (zh, zs): log(f_, os.path.exists(f_), round(os.path.getsize(f_) / 1e9, 2) if os.path.exists(f_) else 0, 'GB')
    hv = snapshot_download('gardianofthedarkness/hanseg-ct-mr', repo_type='dataset', allow_patterns=['head_v1/*'], local_dir='/data/hv'); hv = os.path.join(hv, 'head_v1')
    srh = snapshot_download(srepo, repo_type='dataset', allow_patterns=['face_v1_headframe/sr_*.npz'], local_dir='/data/srh'); srh = os.path.join(srh, 'face_v1_headframe')
    meta, tasks = {}, {}
    for f in os.listdir(hv):
        if f.endswith('.json'): d = json.load(open(os.path.join(hv, f))); meta.update(d['stats']); tasks = d['tasks']
    inv = {v: int(k) for k, v in tasks['head_glands_cavities'].items()}
    srnames = zipfile.ZipFile(zs).namelist(); srp = {c: [n for n in srnames if n.endswith('/%s/ct.mha' % c[3:])][0] for c in srA}
    work = [(c, 'hanseg') for c in sorted(hsf)] + [(c, 'synthrad') for c in sorted(srA)]
    if LIMIT: work = work[:LIMIT // 2] + [w for w in work if w[1] == 'synthrad'][:LIMIT - LIMIT // 2]
    out = '/data/out'; os.makedirs(out, exist_ok=True)
    g = dict(zh_path=zh, zs_path=zs, frames=dict(hsf, **srA), lo=lo, hi=hi, hv=hv, inv=inv, meta=meta, srh=srh, sr_paths=srp, eye_id=ids['eye_left'], out=out, fill_sources=os.environ.get('FILL_SOURCES', 'synthrad').split(','))
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 30
    nw = int(os.environ.get('WORKERS', max(1, min(os.cpu_count() or 4, int(ram // 8))))); log(len(work), 'heads,', nw, 'workers')
    dest = 'head_fields_v1' + ('_smoke' if LIMIT else '')
    def push(files):
        for a in range(5):
            try: api.upload_folder(folder_path=out, path_in_repo=dest, repo_id=repo, repo_type='dataset', allow_patterns=files, commit_message='head fields v1 (%d heads)' % (len(files) // 2)); return
            except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
    stats = {}
    try: have = [f for f in api.list_repo_files(repo, repo_type='dataset') if f.startswith(dest + '/') and f.endswith('.json') and not f.endswith('index.json')]
    except Exception: have = []
    REDO = os.environ.get('REDO', '')                                                       # a source to rebuild even if uploaded
    for f in have:
        c = os.path.basename(f)[:-5]; r_ = json.load(open(hf_hub_download(repo, f, repo_type='dataset')))
        if REDO not in ('all', r_.get('source')): stats[c] = r_
    work = [w for w in work if w[0] not in stats]; log('resume:', len(stats), 'heads already uploaded,', len(work), 'to do')
    pend = []
    with Pool(nw, _init, (g,), maxtasksperchild=4) as p:
        for c, r in p.imap_unordered(one, work):
            stats[c] = r; log(c, r)
            if isinstance(r, dict): json.dump(r, open(os.path.join(out, c + '.json'), 'w')); pend += [c + '.npz', c + '.json']
            if len(pend) >= 20: push(pend); [os.remove(os.path.join(out, f)) for f in pend]; pend = []
    if pend: push(pend); [os.remove(os.path.join(out, f)) for f in pend]
    ok = {c: r for c, r in stats.items() if isinstance(r, dict)}
    for i, c in enumerate(sorted(ok)): ok[c]['split'] = fi['heads'][c].get('split') if c in fi['heads'] and fi['heads'][c].get('split') else srx.get(c, {}).get('split') or ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train')
    hs_sorted = sorted(c for c in ok if ok[c]['source'] == 'hanseg')
    for i, c in enumerate(hs_sorted): ok[c]['split'] = 'test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train'   # HaN-Seg split as in every earlier run
    json.dump(dict(window_lo_mm=lo.tolist(), window_hi_mm=hi.tolist(), voxel_mm=1.0, eye_id=ids['eye_left'], levels=dict(air=-500, fat=[-190, -30], lean=[-30, 150], bone=150),
                   heads=ok, failed={c: r for c, r in stats.items() if not isinstance(r, dict)}), open(os.path.join(out, 'index.json'), 'w'), indent=1)
    log('SUMMARY', json.dumps(dict(heads=len(work), ok=len(ok), failed=len(stats) - len(ok), metal_heads=sum(r['metal_slices'] > 0 for r in ok.values()))))
    push(['index.json'])
    log('SUMMARY done')
