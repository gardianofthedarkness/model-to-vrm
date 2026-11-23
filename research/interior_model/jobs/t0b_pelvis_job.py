"""T0b, pelvis: can the layered atlas reproduce a real body's skin and fascia when its bones are held where the CT shows
them?  Runs ON a private Hugging Face CPU Job.  Numbers only: no images are written.  args: <repo id> [limit]

Per person (the same 408-person selection as T0a):
  bones     each atlas pelvic bone (hip bones, sacrum, femurs) is placed on that bone's CT label surface: first one
            similarity for the whole pelvis from the three bone centroids (Umeyama 1991), then an affine per bone by ICP
            (Besl and McKay 1992) from the CT label surface to the atlas surface (the CT femur is often cut, so the
            matches run from the CT side; each match is weighted by Cauchy weights with the median match distance as
            scale, so missing parts do not pull). Bone error = distance from the CT label surface to the placed atlas bone.
  skin      the atlas's own skin around the pelvis is carried by the placed bones: each skin vertex blends the bones'
            affine maps with weights proportional to 1 / distance to each bone in the atlas (the atlas's own geometry).
  measure   along each carried skin vertex's normal, in the CT: the skin (the outermost body-to-air crossing near the
            vertex), the subcutaneous fat run, the fascia stand-in (first lean run of at least two samples inside the fat),
            as in T0a. These are the targets.
  layers    the layered state on the atlas skin: a skin offset field s (CT skin minus carried atlas skin, along the
            normal) and a subcutaneous thickness field t (skin to fascia). Each is fitted as a smooth field on the atlas
            skin mesh: min ||M(x - y)||^2 + lambda ||L x||^2 (L the mesh graph Laplacian, M the measured vertices).
            lambda is chosen by spatial cross-validation: the skin vertices are split into blocks by height (the levels
            of T0a: between consecutive landmarks), each block held out in turn; lambda from a log grid around the
            data's own scale (the median degree of L times the variance of y), the one with the least held-out error.
  report    held-out error of skin and fascia (the representation's error where it was not fitted), against T0a's
            measurement repeatability; bone error; crossings (triangles of the fascia surface whose normal flips against
            the skin surface's: neighbouring offset lines that cross).
Outputs (private repo, numbers only): t0b_pelvis/per_person.csv, summary.json."""
# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel"]
# ///
import os, sys, io, csv, json, gzip, time, zipfile, collections, numpy as np
from multiprocessing import Pool
FAT, LEAN = (-190, -30), (-29, 150)
BONES = ('hip_left', 'hip_right', 'sacrum', 'femur_left', 'femur_right')
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def otsu(v):
    v = np.asarray(v, float); h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    return float(c[np.nanargmax(s[:-1])])
_Z = None; ATL = None
def _init(zp, ap):
    global _Z, ATL; _Z = zipfile.ZipFile(zp); ATL = dict(np.load(ap))
def load(name):
    import nibabel as nib
    img = nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(name))); return np.asanyarray(img.dataobj), img.affine

def surface_points(mask, aff):
    """centres of the label's boundary voxels, in metres (world)."""
    from scipy.ndimage import binary_erosion
    m = np.asarray(mask) > 0; b = m & ~binary_erosion(m); ijk = np.argwhere(b)
    return ((aff @ np.c_[ijk, np.ones(len(ijk))].T)[:3].T) / 1000.0

def umeyama(A, B):
    ma, mb = A.mean(0), B.mean(0); U, D, Vt = np.linalg.svd((B - mb).T @ (A - ma) / len(A)); S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / ((A - ma) ** 2).sum(1).mean(); return s, R, mb - s * R @ ma

def wumeyama(A, B, w):
    """weighted similarity B ~ s R A + t."""
    w = w / w.sum(); ma, mb = w @ A, w @ B; U, D, Vt = np.linalg.svd(((B - mb) * w[:, None]).T @ (A - ma)); S = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / (w @ ((A - ma) ** 2).sum(1)); return np.c_[s * R, mb - s * R @ ma]

def affine_icp(Pct, Vat, M0, both=False):
    """atlas -> CT. Matches run from each CT point to the nearest placed atlas point, Cauchy-weighted with the median
    match distance as scale (robust M-estimation). First a similarity (turn, shift, one size) until it settles, then a
    full affine from there: an affine fitted from a poor start shears into a wrong minimum, a similarity does not."""
    from scipy.spatial import cKDTree
    X = np.c_[Vat, np.ones(len(Vat))]; M = M0.copy()
    for stage in ('similarity', 'affine'):
        prev = np.inf
        for it in range(100):
            Y = X @ M.T; d, j = cKDTree(Y).query(Pct); Aa, Bb, dd = Vat[j], Pct, d
            if both:                                                   # a whole bone on both sides: matches also from each atlas point to the CT, or the atlas can grow round the CT unpunished
                d2, j2 = cKDTree(Pct).query(Y); Aa, Bb, dd = np.r_[Aa, Vat], np.r_[Bb, Pct[j2]], np.r_[d, d2]
            sc = max(np.median(dd), 1e-6); w = 1 / (1 + (dd / sc) ** 2)
            if stage == 'similarity': M = wumeyama(Aa, Bb, w)
            else: M = np.linalg.lstsq(np.c_[Aa, np.ones(len(Aa))] * np.sqrt(w)[:, None], Bb * np.sqrt(w)[:, None], rcond=None)[0].T
            cost = float(np.median(dd))
            if prev - cost < 1e-3 * cost: break                        # a round gains less than a thousandth of the median match: settled
            prev = cost
    Y = X @ M.T; d = cKDTree(Y).query(Pct)[0]; return M, d

def normals(V, F):
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); N = np.zeros_like(V)
    for k in range(3): np.add.at(N, F[:, k], n)
    return N / np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)

def laplacian(nv, F):
    from scipy.sparse import coo_matrix, diags, identity
    E = np.r_[F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]; A = coo_matrix((np.ones(len(E)), (E[:, 0], E[:, 1])), shape=(nv, nv)).tocsr(); A = ((A + A.T) > 0).astype(float)
    deg = np.asarray(A.sum(1)).ravel(); return (identity(nv) - diags(1 / np.maximum(deg, 1)) @ A).tocsr(), float(np.median(deg))

def smooth_fit(L, idx, y, lam):
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    n = L.shape[0]; m = np.zeros(n); m[idx] = 1; b = np.zeros(n); b[idx] = y
    return spsolve((diags(m) + lam * (L.T @ L)).tocsc(), b)

def measure_case(case):
    from scipy.spatial import cKDTree
    from scipy.ndimage import map_coordinates
    CT, aff = load('%s/ct.nii.gz' % case); CT = np.asarray(CT, np.float32); inv = np.linalg.inv(aff)
    lab = {}
    for b in BONES:
        try: lab[b] = np.asarray(load('%s/segmentations/%s.nii.gz' % (case, b))[0]) > 0
        except KeyError: pass
    if not all(k in lab and lab[k].any() for k in ('hip_left', 'hip_right', 'sacrum')): return dict(case=case, failed='labels')
    Pct = {b: surface_points(m, aff) for b, m in lab.items() if m.any()}
    # whole-pelvis similarity from the three pelvic bones' centroids
    Ac = np.array([ATL['bone_%s_V' % b].astype(float).mean(0) for b in ('hip_left', 'hip_right', 'sacrum')]); Bc = np.array([Pct[b].mean(0) for b in ('hip_left', 'hip_right', 'sacrum')])
    s, R, t = umeyama(Ac, Bc); M0 = np.c_[s * R, t]
    Maff, berr = {}, {}
    for b, P in Pct.items():
        Va = ATL['bone_%s_V' % b].astype(float); M, d = affine_icp(P, Va, M0, both=not b.startswith('femur')); Maff[b] = M; berr[b] = d    # hip bones and sacrum are whole by selection; the femur is often cut
    # carry the atlas skin
    SV = ATL['skin_V'].astype(float); SF = ATL['skin_F']; W = []
    for b in Maff: W.append(1 / np.maximum(cKDTree(ATL['bone_%s_V' % b].astype(float)).query(SV)[0], 1e-6))
    W = np.array(W).T; W /= W.sum(1, keepdims=True); X = np.c_[SV, np.ones(len(SV))]
    C = sum(W[:, [k]] * (X @ Maff[b].T) for k, b in enumerate(Maff)); N = normals(C, SF)
    # sample along each normal, in the CT (metres -> voxels)
    vox = float(abs(np.linalg.det(aff[:3, :3])) ** (1 / 3)) / 1000.0; step = vox / 2
    reach = float(np.median(np.linalg.norm(C - C.mean(0), axis=1)))                     # half the section's own size: far enough inward and outward
    tt = np.arange(-reach, reach, step)                                                # as far outward as inward: the carried atlas skin may lie well inside or outside the body
    body_t = otsu(np.clip(CT[:, :, CT.shape[2] // 2], -1024, 200)); out = []; why = collections.Counter()
    # the patient, not the table or blanket: the 3D piece of body that holds the sacrum, with gas inside it filled per slice
    from scipy.ndimage import label as cc3, binary_fill_holes
    comp, _ = cc3(CT > body_t); ids = comp[lab['sacrum']]; ids = ids[ids > 0]; PAT = comp == int(np.bincount(ids).argmax())
    for z_ in range(PAT.shape[2]): PAT[:, :, z_] = binary_fill_holes(PAT[:, :, z_])
    PATf = PAT.astype(np.float32); del comp
    for i in range(len(C)):
        P = C[i][None] + tt[:, None] * N[i][None]; ijk = (inv[:3, :3] @ (P.T * 1000.0) + inv[:3, 3:4])
        hu = map_coordinates(CT, ijk, order=1, mode='constant', cval=np.nan)
        fin = np.isfinite(hu); bodyr = fin & (map_coordinates(PATf, ijk, order=0, mode='constant', cval=0) > 0.5); outer = np.flatnonzero(bodyr)
        if not len(outer): why['no_body'] += 1; out.append(None); continue
        k = int(outer[-1])
        if k >= len(tt) - 1: why['body_to_end'] += 1; out.append(None); continue        # the body reaches the outer end: no skin crossing
        if not fin[k + 1]: why['outside_scan'] += 1; out.append(None); continue        # the scan ends at the skin: the true skin is not seen
        if not fin[:k + 1].all(): why['gap_inside'] += 1; out.append(None); continue    # the line leaves the scanned volume inside the body
        r = dict(s=tt[k])                                                              # CT skin offset along the normal (metres)
        seg = hu[:k + 1][::-1]; Fm = (seg >= FAT[0]) & (seg <= FAT[1]); Lm = (seg >= LEAN[0]) & (seg <= LEAN[1])
        f2 = np.flatnonzero(Fm[1:] & Fm[:-1])
        if len(f2):                                                                    # lean runs inward of the first fat: (start, length) in samples
            j0 = int(f2[0]); x = np.r_[0, Lm[j0:].astype(int), 0]; d = np.diff(x); a = np.flatnonzero(d == 1); b = np.flatnonzero(d == -1)
            r['runs'] = [(j0 + u, v - u) for u, v in zip(a, b)]
        out.append(r)
    # the deep fascia is where the subcutaneous fat meets muscle: the first lean run that is muscle-thick, not a septum or the membranous
    # layer inside the fat. 'Thick' is split from 'thin' by Otsu over this person's own lean-run lengths (log scale)
    LR = np.log([n for r in out if r is not None for _, n in r.get('runs', [])] or [1.0]); cut = np.exp(otsu(LR)) if np.ptp(LR) > 0 else 0.0
    why['lean_run_cut_mm'] = round(float(cut * step * 1000), 2)
    for r in out:
        if r is None: continue
        th = [u for u, n in r.pop('runs', []) if n > cut]
        if th: r['t'] = (th[0] - 0.5) * step                                           # skin to fascia along the normal
    s_idx = np.array([i for i, r in enumerate(out) if r is not None]); s_y = np.array([out[i]['s'] for i in s_idx])
    t_idx = np.array([i for i, r in enumerate(out) if r is not None and 't' in r]); t_y = np.array([out[i]['t'] for i in t_idx]) if len(t_idx) else np.zeros(0)
    why['measured'] = len(s_idx); why['with_fascia'] = len(t_idx)
    return dict(case=case, why=dict(why), C=C.astype(np.float32), N=N.astype(np.float32), s_idx=s_idx, s_y=s_y, t_idx=t_idx, t_y=t_y,
                bone_err={b: (float(np.median(d)), float(np.percentile(d, 90))) for b, d in berr.items()}, scale=float(s), z=C[:, 2])

def fit_and_score(r, L, degm):
    """smooth fields with lambda by height-block cross-validation; held-out errors."""
    res = {}
    zq = np.quantile(r['z'], np.linspace(0, 1, 6)); blk = np.clip(np.searchsorted(zq, r['z'], side='right') - 1, 0, 4)     # five height bands of equal vertex count
    for key, idx, y in (('skin', r['s_idx'], r['s_y']), ('fascia', r['t_idx'], r['t_y'])):
        if len(idx) < 50: res[key] = None; continue
        base = degm * float(np.var(y)) if np.var(y) > 0 else 1.0; grid = base * np.logspace(-4, 8, 25)
        cv = []
        for lam in grid:
            e = []
            for b in range(5):
                tr = blk[idx] != b
                if tr.sum() < 20 or (~tr).sum() < 5: continue
                x = smooth_fit(L, idx[tr], y[tr], lam); e.append(np.abs(x[idx[~tr]] - y[~tr]))
            cv.append(np.median(np.concatenate(e)) if e else np.inf)
        k = int(np.argmin(cv)); lam = grid[k]; x = smooth_fit(L, idx, y, lam)
        # line-level noise: raw value difference between mesh neighbours that were both measured (/sqrt 2 = one line's spread)
        v = np.full(L.shape[0], np.nan); v[idx] = y; Lc = L.tocoo(); off = Lc.row != Lc.col; dn = np.abs(v[Lc.row[off]] - v[Lc.col[off]]); dn = dn[np.isfinite(dn)]
        # interpolation held-out: five random vertex folds, same lambda grid
        fold = np.random.default_rng(0).integers(0, 5, len(idx)); cvr = []
        for lam_ in grid:
            e = [np.abs(smooth_fit(L, idx[fold != b], y[fold != b], lam_)[idx[fold == b]] - y[fold == b]) for b in range(5)]
            cvr.append(np.median(np.concatenate(e)))
        res[key] = dict(heldout_median_mm=1000 * float(cv[k]), fit_median_mm=1000 * float(np.median(np.abs(x[idx] - y))), lambda_rel=float(lam / base), coverage=float(len(idx) / L.shape[0]), field=x,
                        neighbour_diff_mm=1000 * float(np.median(dn)) / np.sqrt(2) if len(dn) else None, heldout_random_mm=1000 * float(min(cvr)), value_median_mm=1000 * float(np.median(y)))
    return res

if __name__ == '__main__':
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0; shard = sys.argv[3] if len(sys.argv) > 3 else '0/1'; si, sn = map(int, shard.split('/'))
    def put(data, name): api.upload_file(path_or_fileobj=data.encode() if isinstance(data, str) else data, path_in_repo=name, repo_id=repo, repo_type='dataset')
    ext = list(csv.DictReader(open(hf_hub_download(repo, 'census/label_extents.csv', repo_type='dataset'))))
    meta = {r['image_id']: r for r in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    by = collections.defaultdict(dict)
    for r in ext: by[r['case']][r['label']] = r
    whole = lambda r: bool(r) and r['present'] == '1' and not any(r[k] == '1' for k in ('cut_inf', 'cut_sup', 'cut_side', 'cut_ap'))
    sel = sorted(c for c, d in by.items() if whole(d.get('hip_left')) and whole(d.get('hip_right')) and whole(d.get('sacrum')))
    age = lambda c: float(meta[c]['age']) if meta.get(c, {}).get('age') else None
    sel = [c for c in sel if not (age(c) is not None and age(c) < 18)]
    if limit: sel = sel[:limit]
    sel = sel[si::sn]; tag = '' if sn == 1 else '_shard%d' % si
    log('%d people' % len(sel))
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); ap = hf_hub_download(repo, 'atlas/atlas_pelvis.npz', repo_type='dataset', local_dir='.'); log('zip and atlas ready')
    first = [i.filename for i in zipfile.ZipFile(zp).infolist() if i.filename.endswith('/ct.nii.gz')][0]
    pre = first[:-len('/ct.nii.gz')].rsplit('/', 1)[0] + '/' if first.count('/') > 1 else ''
    A = dict(np.load(ap)); L, degm = laplacian(len(A['skin_V']), A['skin_F'])
    rows = []; crossings = []; rawbuf = io.BytesIO(); RAW = zipfile.ZipFile(rawbuf, 'w')
    with Pool(os.cpu_count() or 4, _init, (zp, ap)) as p:
        for k, r in enumerate(p.imap_unordered(measure_case, [pre + c for c in sel])):
            c = r['case'].rsplit('/', 1)[-1]
            if 'failed' in r: rows.append(dict(case=c, failed=r['failed'])); continue
            f = fit_and_score(r, L, degm); row = dict(case=c, sex=meta.get(c, {}).get('gender', ''), scale=round(r['scale'], 3), **{'n_' + k_: v_ for k_, v_ in r['why'].items()})
            for b, (m50, m90) in r['bone_err'].items(): row['bone_%s_median_mm' % b] = round(1000 * m50, 2); row['bone_%s_p90_mm' % b] = round(1000 * m90, 2)
            for key in ('skin', 'fascia'):
                if f[key] is None: continue
                for q in ('heldout_median_mm', 'fit_median_mm', 'lambda_rel', 'coverage', 'neighbour_diff_mm', 'heldout_random_mm', 'value_median_mm'): row['%s_%s' % (key, q)] = round(f[key][q], 4) if f[key][q] is not None else None
            if f['skin'] is not None and f['fascia'] is not None:                       # crossings: fascia surface triangles that flip against the skin surface
                C = r['C'].astype(float); N = r['N'].astype(float); S = C + f['skin']['field'][:, None] * N; Fa = S - f['fascia']['field'][:, None] * N; SF = A['skin_F']
                ns = np.cross(S[SF[:, 1]] - S[SF[:, 0]], S[SF[:, 2]] - S[SF[:, 0]]); nf = np.cross(Fa[SF[:, 1]] - Fa[SF[:, 0]], Fa[SF[:, 2]] - Fa[SF[:, 0]])
                row['fascia_triangles_flipped'] = round(float(((ns * nf).sum(1) < 0).mean()), 4)
                nc = np.cross(C[SF[:, 1]] - C[SF[:, 0]], C[SF[:, 2]] - C[SF[:, 0]]); row['skin_triangles_flipped'] = round(float(((ns * nc).sum(1) < 0).mean()), 4)
                np.savez_compressed(bb := io.BytesIO(), s_idx=r['s_idx'], s_y=r['s_y'].astype(np.float32), t_idx=r['t_idx'], t_y=r['t_y'].astype(np.float32), C=r['C'], N=r['N']); RAW.writestr('%s.npz' % c, bb.getvalue())
            rows.append(row)
            if (k + 1) % 10 == 0 or k + 1 == len(sel): log('fitted %d / %d' % (k + 1, len(sel)))
    keys = sorted({k for r in rows for k in r}, key=lambda k: (k != 'case', k)); b = io.StringIO(); w = csv.DictWriter(b, keys); w.writeheader(); w.writerows(rows)
    RAW.close(); os.makedirs('upload/t0b_pelvis', exist_ok=True)                                # one commit per shard (the Hub allows 256 commits an hour)
    open('upload/t0b_pelvis/per_person%s.csv' % tag, 'w').write(b.getvalue()); open('upload/t0b_pelvis/raw%s.zip' % tag, 'wb').write(rawbuf.getvalue())
    ok = [r for r in rows if 'failed' not in r]
    def med(k): v = [r[k] for r in ok if r.get(k) is not None]; return dict(median=round(float(np.median(v)), 3), p90=round(float(np.percentile(v, 90)), 3), n=len(v)) if v else None
    out = dict(people=len(rows), failed=len(rows) - len(ok), **{k: med(k) for k in ('skin_heldout_median_mm', 'skin_fit_median_mm', 'skin_coverage', 'fascia_heldout_median_mm', 'fascia_fit_median_mm', 'fascia_coverage', 'fascia_triangles_flipped', 'skin_triangles_flipped', 'skin_neighbour_diff_mm', 'skin_heldout_random_mm', 'skin_value_median_mm', 'fascia_neighbour_diff_mm', 'fascia_heldout_random_mm', 'fascia_value_median_mm', 'scale')},
               bone={b: med('bone_%s_median_mm' % b) for b in BONES})
    open('upload/t0b_pelvis/summary%s.json' % tag, 'w').write(json.dumps(out, indent=1))
    for attempt in range(6):
        try: api.upload_folder(folder_path='upload', repo_id=repo, repo_type='dataset', commit_message='t0b pelvis%s' % tag); break
        except Exception as e: log('upload retry', attempt, type(e).__name__); time.sleep(600)
    log('SUMMARY ' + json.dumps(out))
