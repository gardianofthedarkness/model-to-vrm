# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "scikit-image", "trimesh", "torch", "matplotlib"]
# ///
"""V6 first runs on one thigh (CT only, no network).

Subject: the first long femur whose twelve T0a planes all passed every check (t0a_v2/thigh_planes.csv), so the choice
is the data's, not ours.  Region: the thigh body around that femur (per axial slice, the body piece holding the femur),
cropped to its box; everything else in the box set to air.

(1) Round trip.  Export the cropped CT as NIfTI with its voxel->world affine and the thigh skin as a mesh in world mm
    (marching cubes of the thigh mask).  A separate process reopens both from disk only, cuts random oblique planes
    through the thigh centre, and measures the distance from the mesh's cut line to the volume's skin edge in that
    plane.  Control: the same with the mesh mirrored left-right about the volume centre - it must fail, or the test
    cannot see an axis error.
(2) One-person Gaussian fit.  The field V(x) = b + sum_i w_i exp(-1/2 (x-mu_i)^T Sigma_i^-1 (x-mu_i)) is fitted to the
    voxels of context axial slices only (every k-th slice), then read on the withheld axial slices and on random
    oblique planes.  Baseline: linear interpolation between the same context slices.  k = 1 (all slices) measures what
    the Gaussian count can hold at all.  Gaussian counts are set as voxels-per-Gaussian of the thigh.
    Scores on withheld planes, inside the thigh: HU error; bone edge and skin edge distance (mm, mean of both ways,
    thresholds by Otsu as in T0a); fat Dice (fat HU range as in T0a).
Outputs (private repo only): t1_thigh/summary.json, t1_thigh/*.png (stay in the private repo).
"""
import os, sys, io, csv, json, gzip, time, zipfile, subprocess, numpy as np
FAT, LEAN = (-190, -30), (-29, 150)
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def otsu(v):
    v = np.asarray(v, float); h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    return float(c[np.nanargmax(s[:-1])])

# ---------------------------------------------------------------- shared helpers
def plane_grid(centre, normal, half, step):
    """points of a square plane (world mm), its in-plane step, and its shape."""
    n = normal / np.linalg.norm(normal); a = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0]); a /= np.linalg.norm(a); b = np.cross(n, a)
    u = np.arange(-half, half + step / 2, step); U, Vv = np.meshgrid(u, u, indexing='ij')
    return centre + U[..., None] * a + Vv[..., None] * b, a, b

def sample(vol, aff, P, order=1, cval=np.nan):
    from scipy.ndimage import map_coordinates
    inv = np.linalg.inv(aff); ijk = inv[:3, :3] @ P.reshape(-1, 3).T + inv[:3, 3:4]
    return map_coordinates(vol, ijk, order=order, mode='constant', cval=cval).reshape(P.shape[:-1])

def edge_dist(A, B, px):
    """mean of both ways boundary distance (mm) between 2D masks A, B with pixel size px; None if either is empty."""
    from scipy.ndimage import binary_erosion, distance_transform_edt
    if not A.any() or not B.any(): return None
    ea = A & ~binary_erosion(A); eb = B & ~binary_erosion(B)
    da = distance_transform_edt(~eb, sampling=px)[ea]; db = distance_transform_edt(~ea, sampling=px)[eb]
    return float((da.mean() + db.mean()) / 2)

def main_piece(M):
    """largest connected piece of a 2D mask, holes filled (stray specks outside the body are not skin)."""
    from scipy.ndimage import label, binary_fill_holes
    lab, n = label(M)
    return binary_fill_holes(lab == (np.bincount(lab.ravel())[1:].argmax() + 1)) if n else M

def dice(A, B):
    s = A.sum() + B.sum(); return float(2 * (A & B).sum() / s) if s else None

# ---------------------------------------------------------------- (1) round trip, reopen side (separate process)
def reopen(folder):
    import nibabel as nib, trimesh
    img = nib.load(os.path.join(folder, 'thigh_ct.nii.gz')); V = np.asanyarray(img.dataobj).astype(np.float32); aff = img.affine
    mesh = trimesh.load(os.path.join(folder, 'thigh_skin.ply'), process=False); meta = json.load(open(os.path.join(folder, 'meta.json')))
    c = np.array(meta['centre_mm']); bt = meta['body_threshold']; step = float(np.min(np.linalg.norm(aff[:3, :3], axis=0)))
    half = 0.5 * float(np.linalg.norm(np.array(V.shape) * np.linalg.norm(aff[:3, :3], axis=0)))
    rng = np.random.default_rng(1); res = {}
    for name in ('as_exported', 'control_mirrored_about_scanner_midline', 'control_voxel_units_not_mm'):
        M = mesh.copy()
        if name == 'control_mirrored_about_scanner_midline': M.vertices[:, 0] = -M.vertices[:, 0]           # a left-right error
        if name == 'control_voxel_units_not_mm':                                                            # a units error
            inv = np.linalg.inv(aff); M.vertices = (inv[:3, :3] @ M.vertices.T + inv[:3, 3:4]).T + aff[:3, 3]
        d = []
        for p in range(6):
            n = rng.normal(size=3); n /= np.linalg.norm(n)
            G, a, b = plane_grid(c, n, half, step); S = sample(V, aff, G, cval=-1024)
            body = S > bt
            from scipy.ndimage import binary_erosion, distance_transform_edt
            edge = body & ~binary_erosion(body)
            if not edge.any(): continue
            dt = distance_transform_edt(~edge, sampling=step)
            seg = trimesh.intersections.mesh_plane(M, n, c)
            if not len(seg): d.append(np.array([np.inf])); continue
            Q = seg.reshape(-1, 3) - c; ij = np.c_[Q @ a, Q @ b] / step + (G.shape[0] - 1) / 2
            ij = np.round(ij).astype(int); ok = (ij >= 0).all(1) & (ij < G.shape[0]).all(1)
            d.append(np.where(ok, dt[np.clip(ij[:, 0], 0, G.shape[0] - 1), np.clip(ij[:, 1], 0, G.shape[1] - 1)], np.inf))
        d = np.concatenate(d) if d else np.array([np.inf]); fin = np.isfinite(d)
        res[name] = dict(median_mm=float(np.median(d[fin])) if fin.any() else None, p95_mm=float(np.percentile(d[fin], 95)) if fin.any() else None,
                         off_volume_fraction=float(1 - fin.mean()), voxel_mm=step)
    print('ROUNDTRIP ' + json.dumps(res), flush=True)

def reopen_asset(folder):
    """separate process: read the native Gaussian asset from disk, choose oblique planes, evaluate the field by its
    written definition (every Gaussian within 5 sigma, no boxes, no voxels), and save the planes for scoring."""
    from scipy.spatial import cKDTree
    A = np.load(os.path.join(folder, 'gaussians.npz')); d = json.load(open(os.path.join(folder, 'gaussians.json')))
    mu, sg, q, w, b = A['mu_mm'].astype(float), A['sigma_mm'].astype(float), A['quat_wxyz'].astype(float), A['w_hu'].astype(float), float(A['b_hu'])
    qw, qx, qy, qz = q.T
    R = np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy), 2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
                  2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], 1).reshape(-1, 3, 3)
    c = np.array(d['centre_mm']); half = d['half_mm']; step = d['step_mm']; rng = np.random.default_rng(3); out = {}
    for pidx in range(3):
        n = rng.normal(size=3); n /= np.linalg.norm(n); G = plane_grid(c, n, half, step)[0]; P = G.reshape(-1, 3); val = np.full(len(P), b)
        tree = cKDTree(P); near = tree.query_ball_point(mu, 5 * sg.max(1))
        for i, idx in enumerate(near):
            if not idx: continue
            loc = ((P[idx] - mu[i]) @ R[i]) / sg[i]; val[idx] += w[i] * np.exp(-0.5 * (loc * loc).sum(1))
        out['points_%d' % pidx] = G.astype(np.float32); out['value_%d' % pidx] = val.reshape(G.shape[:2]).astype(np.float32)
    np.savez_compressed(os.path.join(folder, 'asset_planes.npz'), **out); print('ASSET_REOPENED', flush=True)

# ---------------------------------------------------------------- (2) Gaussian field
def fit_gaussians(T, sp, fitmask, inside, n_g, k, dev, seed=0, max_iter=1500, lr=0.03, mem_entries=6e7):
    """T: target HU (X,Y,Z) on the crop grid; sp: voxel size mm; fitmask: voxels that may be used; inside: thigh mask
    (initial centres drawn there).  Gaussians live in mm; each is splatted into a fixed box of voxels round its centre.
    Box half-sizes: in-plane from the mean Gaussian spacing, through-plane at least the context gap k."""
    import torch
    g = torch.Generator(device='cpu').manual_seed(seed)
    vol_mm3 = inside.sum() * np.prod(sp); spacing = (vol_mm3 / n_g) ** (1 / 3)               # mean distance between Gaussian centres (mm)
    R = np.maximum(np.ceil(1.5 * spacing / sp), 1).astype(int); R[2] = max(R[2], k)             # box half-size in voxels
    smax = torch.tensor(R * sp / 3.0, device=dev, dtype=torch.float32)                         # 3 sigma fits in the box
    idx = np.argwhere(inside); pick = idx[torch.randint(len(idx), (n_g,), generator=g).numpy()]
    mu = torch.tensor((pick + torch.rand(n_g, 3, generator=g).numpy() - 0.5) * sp, device=dev, dtype=torch.float32, requires_grad=True)
    s0 = torch.full((n_g, 3), spacing, device=dev); s0 = torch.minimum(s0, smax * 0.99)
    raw_s = torch.logit(s0 / smax).clone().requires_grad_(True)
    q = torch.zeros(n_g, 4, device=dev); q[:, 0] = 1; q.requires_grad_(True)
    Tt = torch.tensor(T, device=dev, dtype=torch.float32) / 1000.0
    b = torch.tensor(-1.0, device=dev, requires_grad=True)                                     # air
    w = (Tt[tuple(torch.tensor(pick.T, device=dev))] - (-1.0)).clone().mul_(0.5).requires_grad_(True)
    shape = torch.tensor(T.shape, device=dev); spt = torch.tensor(sp, device=dev, dtype=torch.float32)
    rx, ry, rz = [torch.arange(-r, r + 1, device=dev) for r in R]
    O = torch.stack(torch.meshgrid(rx, ry, rz, indexing='ij'), -1).reshape(-1, 3)
    M = torch.tensor(fitmask, device=dev); nfit = M.sum()
    from torch.utils.checkpoint import checkpoint
    CH = max(1, int(2e7 // len(O)))                                                            # Gaussians per checkpointed chunk (~2e7 box entries)
    big = n_g * len(O) > mem_entries                                                           # checkpoint only when the boxes would not fit in memory
    def render():
        out = 0
        for j in range(0, n_g, CH):
            sl = slice(j, j + CH); out = out + (checkpoint(part, mu[sl], raw_s[sl], q[sl], w[sl], use_reentrant=False) if big else part(mu[sl], raw_s[sl], q[sl], w[sl]))
        return out.view(*T.shape) + b
    def part(mu, raw_s, q, w):
        c = torch.round(mu.detach() / spt).long(); I = c[:, None, :] + O[None]
        ok = ((I >= 0) & (I < shape)).all(-1)
        d = I.float() * spt - mu[:, None, :]
        qq = q / q.norm(dim=1, keepdim=True); qw, qx, qy, qz = qq.unbind(1)
        Rm = torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
                          2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
                          2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], 1).view(-1, 3, 3)
        s = smax * torch.sigmoid(raw_s)
        loc = torch.einsum('nkj,nji->nki', d, Rm) / s[:, None, :]                                # coordinates in each Gaussian's own axes
        val = w[:, None] * torch.exp(-0.5 * (loc * loc).sum(-1))
        flat = (I[..., 0] * shape[1] + I[..., 1]) * shape[2] + I[..., 2]
        return torch.zeros(int(shape.prod()), device=dev).index_add_(0, flat[ok], val[ok])
    opt = torch.optim.Adam([{'params': [mu], 'lr': float(spacing) * lr}, {'params': [raw_s, q, w, b], 'lr': lr}])
    best, last, hist = np.inf, np.inf, []
    for it in range(max_iter):
        opt.zero_grad(); V = render(); loss = (V - Tt).abs()[M].sum() / nfit; loss.backward(); opt.step()
        hist.append(loss.item())
        if it % 100 == 99:                                                                     # stop when 100 steps gain under 0.1% of the loss
            if last - hist[-1] < 1e-3 * hist[-1]: break
            last = hist[-1]
    with torch.no_grad():
        V = render().cpu().numpy() * 1000.0
        params = dict(mu_mm=mu.cpu().numpy(), sigma_mm=(smax * torch.sigmoid(raw_s)).cpu().numpy(), quat_wxyz=(q / q.norm(dim=1, keepdim=True)).cpu().numpy(),
                      w_hu=w.cpu().numpy() * 1000.0, b_hu=np.float32(b.item() * 1000.0))
    return V, params, dict(iterations=it + 1, final_loss_hu=hist[-1] * 1000, box_voxels=[int(r) for r in R], spacing_mm=float(spacing))

def interp_z(T, ctx):
    """linear interpolation along z between context slices (nearest context at the ends)."""
    z = np.arange(T.shape[2]); out = np.empty_like(T)
    for zz in z:
        j = np.searchsorted(ctx, zz)
        if j == 0: out[..., zz] = T[..., ctx[0]]
        elif j >= len(ctx): out[..., zz] = T[..., ctx[-1]]
        else:
            a, c = ctx[j - 1], ctx[j]; t = (zz - a) / (c - a) if c > a else 0; out[..., zz] = (1 - t) * T[..., a] + t * T[..., c]
    return out

def score(P, T, inside, px, bt, bone_t):
    """P, T: 2D predicted and true HU on a plane; inside: thigh pixels."""
    if inside.sum() == 0: return None
    return dict(hu_mae=float(np.abs(P - T)[inside].mean()),
                bone_edge_mm=edge_dist((P > bone_t) & inside, (T > bone_t) & inside, px),
                skin_edge_mm=edge_dist(main_piece(P > bt), main_piece(T > bt), px),
                fat_dice=dice((P >= FAT[0]) & (P <= FAT[1]) & inside, (T >= FAT[0]) & (T <= FAT[1]) & inside))

def agg(rows):
    out = {}
    for key in ('hu_mae', 'bone_edge_mm', 'skin_edge_mm', 'fat_dice'):
        v = [r[key] for r in rows if r and r.get(key) is not None]; out[key] = round(float(np.median(v)), 3) if v else None
    return out

def run_fit_experiments(T, aff, thigh, dev, ratios=(256, 64, 32), ks=(1, 2, 4, 8), log=print):
    sp = np.linalg.norm(aff[:3, :3], axis=0); bt = otsu(np.clip(T, -1024, 200)); hi = T[thigh & (T > LEAN[1])]; bone_t = otsu(hi) if len(hi) else 300.0
    nvox = int(thigh.sum()); res = dict(voxel_mm=[round(float(x), 3) for x in sp], thigh_voxels=nvox, body_threshold=bt, bone_threshold=bone_t, runs=[])
    centre = (np.argwhere(thigh).mean(0) * sp); half = 0.5 * float(np.linalg.norm(np.array(T.shape) * sp)); step = float(sp.min())
    rng = np.random.default_rng(2); normals = [v / np.linalg.norm(v) for v in rng.normal(size=(6, 3))]
    affc = np.diag(list(sp) + [1.0])                                                           # crop-local frame in mm
    def oblique(V):
        out = []
        for n in normals:
            G = plane_grid(centre, n, half, step)[0]; out.append(sample(V, affc, G, cval=-1024.0))
        return out
    Tob = oblique(T); Iob = [sample(thigh.astype(np.float32), affc, plane_grid(centre, n, half, step)[0], order=0, cval=0) > 0.5 for n in normals]
    for k in ks:
        ctx = np.arange(0, T.shape[2], k); held = np.setdiff1d(np.arange(T.shape[2]), ctx)
        fitmask = np.zeros(T.shape, bool); fitmask[..., ctx] = True
        base = interp_z(T, ctx) if k > 1 else None
        for r in ratios:
            n_g = max(nvox // r, 1)
            Vg, params, info = fit_gaussians(T, sp, fitmask, thigh, n_g, k, dev); log('k=%d  gaussians=%d  %s' % (k, n_g, info))
            row = dict(context_every=k, gaussians=n_g, voxels_per_gaussian=r, **info)
            if k == 1:
                res['_asset'] = (params, Vg)                                                   # the last k=1 fit (most Gaussians) is the exported asset
                row['representation_all_slices'] = agg([score(Vg[..., z], T[..., z], thigh[..., z], sp[:2], bt, bone_t) for z in range(T.shape[2])])
            else:
                row['withheld_axial'] = agg([score(Vg[..., z], T[..., z], thigh[..., z], sp[:2], bt, bone_t) for z in held])
                row['oblique'] = agg([score(P, Tq, I, (step, step), bt, bone_t) for P, Tq, I in zip(oblique(Vg), Tob, Iob)])
            res['runs'].append(row)
        if base is not None:
            res['runs'].append(dict(context_every=k, method='linear interpolation between context slices',
                                    withheld_axial=agg([score(base[..., z], T[..., z], thigh[..., z], sp[:2], bt, bone_t) for z in held]),
                                    oblique=agg([score(P, Tq, I, (step, step), bt, bone_t) for P, Tq, I in zip(oblique(base), Tob, Iob)])))
            log('k=%d interpolation %s' % (k, res['runs'][-1]))
    return res

# ---------------------------------------------------------------- main
if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--reopen': reopen(sys.argv[2]); sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == '--reopen-asset': reopen_asset(sys.argv[2]); sys.exit()
    import torch, nibabel as nib
    from huggingface_hub import HfApi, hf_hub_download
    from scipy.ndimage import label as cc, binary_fill_holes
    api = HfApi(); repo = sys.argv[1]
    def put(data, name): api.upload_file(path_or_fileobj=data.encode() if isinstance(data, str) else data, path_in_repo=name, repo_id=repo, repo_type='dataset')
    planes = list(csv.DictReader(open(hf_hub_download(repo, 't0a_v2/thigh_planes.csv', repo_type='dataset'))))
    flags = ('legs_touching', 'other_bone_in_thigh', 'metal')
    byk = {}
    for p in planes: byk.setdefault((p['case'], p['side']), []).append(p)
    clean = [k_ for k_, v in byk.items() if not any(p.get(f) == 'True' for p in v for f in flags) and v[0]['contrast'] == 'False']
    case, side = clean[0]; log('subject', case, side, 'from', len(clean), 'clean femurs')
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); Z = zipfile.ZipFile(zp)
    names = [i.filename for i in Z.infolist() if i.filename.endswith('%s/ct.nii.gz' % case)]; pre = names[0][:-len('ct.nii.gz')]
    rd = lambda n: nib.Nifti1Image.from_bytes(gzip.decompress(Z.read(pre + n)))
    ct = rd('ct.nii.gz'); CT = np.asanyarray(ct.dataobj).astype(np.float32); aff = ct.affine
    FEM = np.asanyarray(rd('segmentations/femur_%s.nii.gz' % side).dataobj) > 0
    bt0 = otsu(np.clip(CT[:, :, CT.shape[2] // 2], -1024, 200)); body = CT > bt0
    zs = np.flatnonzero(FEM.any((0, 1))); thigh = np.zeros(CT.shape, bool)
    for z in zs:                                                                               # the body piece holding the femur, slice by slice
        lab, _ = cc(body[:, :, z]); ids = np.unique(lab[FEM[:, :, z]]); ids = ids[ids > 0]
        if len(ids): thigh[:, :, z] = binary_fill_holes(np.isin(lab, ids))
    ii = np.argwhere(thigh); lo = ii.min(0); hi = ii.max(0) + 1
    T = CT[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy(); TH = thigh[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; T[~TH] = -1024.0
    affc = aff.copy(); affc[:3, 3] = aff[:3, :3] @ lo + aff[:3, 3]; del CT, body, thigh
    log('thigh crop', T.shape, 'voxel mm', np.round(np.linalg.norm(aff[:3, :3], axis=0), 3).tolist())
    # (1) round trip: export, then reopen in another process
    import trimesh
    from skimage.measure import marching_cubes
    os.makedirs('export', exist_ok=True)
    nib.save(nib.Nifti1Image(T.astype(np.int16), affc), 'export/thigh_ct.nii.gz')
    vv, ff, _, _ = marching_cubes(np.pad(TH, 1).astype(np.float32), 0.5); vv = vv - 1
    trimesh.Trimesh(vv @ affc[:3, :3].T + affc[:3, 3], ff[:, ::-1], process=False).export('export/thigh_skin.ply')
    ctr = (affc @ np.r_[np.argwhere(TH).mean(0), 1])[:3]
    json.dump(dict(centre_mm=ctr.tolist(), body_threshold=otsu(np.clip(T, -1024, 200))), open('export/meta.json', 'w'))
    rt = subprocess.run([sys.executable, os.path.abspath(__file__), '--reopen', 'export'], capture_output=True, text=True); print(rt.stdout[-3000:], rt.stderr[-3000:], flush=True)
    roundtrip = json.loads(rt.stdout.split('ROUNDTRIP ', 1)[1]) if 'ROUNDTRIP ' in rt.stdout else dict(error=rt.stderr[-500:])
    log('round trip', roundtrip)
    # (2) Gaussian fits
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'; log('device', dev)
    res = run_fit_experiments(T, affc, TH, dev, log=log)
    # (3) the native Gaussian asset: export, reopen elsewhere, cut planes chosen there, score against the real CT
    params, Vg = res.pop('_asset'); sp = np.linalg.norm(affc[:3, :3], axis=0); local = np.diag(list(sp) + [1.0])
    np.savez_compressed('export/gaussians.npz', **params)
    json.dump(dict(field='V(x) = b_hu + sum_i w_hu[i] * exp(-0.5 * |diag(1/sigma_mm[i]) R(quat_wxyz[i])^T (x - mu_mm[i])|^2), x in mm in the crop frame',
                   crop_to_world=affc.tolist(), voxel_mm=sp.tolist(), centre_mm=(np.argwhere(TH).mean(0) * sp).tolist(),
                   half_mm=0.5 * float(np.linalg.norm(np.array(T.shape) * sp)), step_mm=float(sp.min())), open('export/gaussians.json', 'w'))
    ra = subprocess.run([sys.executable, os.path.abspath(__file__), '--reopen-asset', 'export'], capture_output=True, text=True); print(ra.stdout[-500:], ra.stderr[-2000:], flush=True)
    asset = dict(error=ra.stderr[-300:])
    if 'ASSET_REOPENED' in ra.stdout:
        AP = np.load('export/asset_planes.npz'); rows, gap = [], []
        for i in range(3):
            G = AP['points_%d' % i].astype(float); Pv = AP['value_%d' % i]; Tq = sample(T, local, G, cval=-1024.0); I = sample(TH.astype(np.float32), local, G, order=0, cval=0) > 0.5
            rows.append(score(Pv, Tq, I, (float(sp.min()),) * 2, res['body_threshold'], res['bone_threshold'])); gap.append(float(np.abs(Pv - sample(Vg, local, G, cval=-1024.0))[I].mean()))
        asset = dict(gaussians=int(len(params['w_hu'])), oblique_against_ct=agg(rows), hu_gap_to_in_process_render=round(float(np.mean(gap)), 2))
    log('asset reopen', asset)
    out = dict(subject=dict(case=case, side=side, chosen_from_clean=len(clean)), crop_shape=list(T.shape), roundtrip=roundtrip, asset_reopen=asset, **res)
    put(json.dumps(out, indent=1), 't1_thigh/summary.json'); log('SUMMARY ' + json.dumps(out))
