# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "scikit-image", "trimesh", "torch", "matplotlib"]
# ///
"""v6.4 item 1, the 3D reload fix, built on the helpers of the V6 first runs on one thigh (CT only, no network).

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


# ================================================================ v6.4 item 1: the 3D reload fix
KERNEL = 'f(x) = w * exp(-q/2) * max(0, 1 - q/9)^2,  q = |diag(1/sigma) R^T (x - mu)|^2  (zero for q >= 9)'

def fit_v2(T, sp, fitmask, inside, n_g, dev, seed=0, max_iter=1500, lr=0.03, offgrid=True, z_shift=True):
    """Gaussians with (1) the written kernel above, drawn whole (the box always holds the kernel's support),
    (2) no axis thinner than half a voxel, (3) every other step fitted on a grid shifted by a random sub-voxel offset
    against the CT interpolated there (z_shift False keeps the shift in-plane, for fits to given slices only)."""
    import torch, torch.nn.functional as Fn
    g = torch.Generator(device='cpu').manual_seed(seed); torch.manual_seed(seed)
    spacing = (inside.sum() * np.prod(sp) / n_g) ** (1 / 3)
    R = np.maximum(np.ceil(1.5 * spacing / sp), 1).astype(int) + 1                              # +1 voxel: centre rounding and the sub-voxel shift
    smin = 0.5 * float(sp.max()); smax = (float((R * sp).min()) - 1.0 * float(sp.max())) / 3.0    # support (3 sigma) inside the box after both half-voxel offsets
    assert smax > smin, (smin, smax)
    idx = np.argwhere(inside); pick = idx[torch.randint(len(idx), (n_g,), generator=g).numpy()]
    mu = torch.tensor((pick + torch.rand(n_g, 3, generator=g).numpy() - 0.5) * sp, device=dev, dtype=torch.float32, requires_grad=True)
    s0 = float(np.clip(spacing, smin * 1.01, smax * 0.99))
    raw_s = torch.full((n_g, 3), float(np.log((s0 - smin) / (smax - s0))), device=dev, requires_grad=True)
    q = torch.zeros(n_g, 4, device=dev); q[:, 0] = 1; q.requires_grad_(True)
    Tt = torch.tensor(T, device=dev, dtype=torch.float32) / 1000.0
    b = torch.tensor(-1.0, device=dev, requires_grad=True)
    w = (Tt[tuple(torch.tensor(pick.T, device=dev))] + 1.0).clone().mul_(0.5).requires_grad_(True)
    shape = torch.tensor(T.shape, device=dev); spt = torch.tensor(sp, device=dev, dtype=torch.float32)
    O = torch.stack(torch.meshgrid(*[torch.arange(-r, r + 1, device=dev) for r in R], indexing='ij'), -1).reshape(-1, 3)
    M = torch.tensor(fitmask, device=dev); nfit = M.sum()
    from torch.utils.checkpoint import checkpoint
    CH = max(1, int(2e7 // len(O)))
    sig = lambda rs: smin + (smax - smin) * torch.sigmoid(rs)
    def part(mu, raw_s, q, w, delta):
        c = torch.round(mu.detach() / spt).long(); I = c[:, None, :] + O[None]; ok = ((I >= 0) & (I < shape)).all(-1)
        d = (I.float() + delta) * spt - mu[:, None, :]
        qq = q / q.norm(dim=1, keepdim=True); qw, qx, qy, qz = qq.unbind(1)
        Rm = torch.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy), 2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
                          2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], 1).view(-1, 3, 3)
        loc = torch.einsum('nkj,nji->nki', d, Rm) / sig(raw_s)[:, None, :]; qd = (loc * loc).sum(-1)
        val = w[:, None] * torch.exp(-0.5 * qd) * torch.clamp(1 - qd / 9.0, min=0) ** 2
        flat = (I[..., 0] * shape[1] + I[..., 1]) * shape[2] + I[..., 2]
        return torch.zeros(int(shape.prod()), device=dev).index_add_(0, flat[ok], val[ok])
    def render(delta):
        out = 0
        for j in range(0, n_g, CH):
            sl = slice(j, j + CH); out = out + checkpoint(part, mu[sl], raw_s[sl], q[sl], w[sl], delta, use_reentrant=False)
        return out.view(*T.shape) + b
    def shifted_target(delta):                                                                   # CT read at grid + delta (trilinear), voxel units
        Xs = [torch.arange(n, device=dev, dtype=torch.float32) + delta[a] for a, n in enumerate(T.shape)]
        G = torch.stack(torch.meshgrid(*Xs, indexing='ij'), -1); G = 2 * G / (shape.float() - 1) - 1   # normalised, order (x,y,z)
        return Fn.grid_sample(Tt[None, None], G.flip(-1)[None], mode='bilinear', padding_mode='border', align_corners=True)[0, 0]
    zero = torch.zeros(3, device=dev)
    opt = torch.optim.Adam([{'params': [mu], 'lr': float(spacing) * lr}, {'params': [raw_s, q, w, b], 'lr': lr}])
    last, hist = np.inf, []
    for it in range(max_iter):
        opt.zero_grad()
        if offgrid and it % 2:
            delta = torch.rand(3, device=dev) - 0.5
            if not z_shift: delta[2] = 0
            loss = (render(delta) - shifted_target(delta)).abs()[M].sum() / nfit
        else: loss = (render(zero) - Tt).abs()[M].sum() / nfit
        loss.backward(); opt.step(); hist.append(loss.item())
        if it % 100 == 99:
            if last - np.mean(hist[-100:]) < 1e-3 * np.mean(hist[-100:]): break
            last = np.mean(hist[-100:])
    with torch.no_grad():
        V = render(zero).cpu().numpy() * 1000.0
        params = dict(mu_mm=mu.cpu().numpy(), sigma_mm=sig(raw_s).cpu().numpy(), quat_wxyz=(q / q.norm(dim=1, keepdim=True)).cpu().numpy(), w_hu=w.cpu().numpy() * 1000.0, b_hu=np.float32(b.item() * 1000.0))
        probe = torch.rand(3, device=dev) - 0.5; Vp = render(probe).cpu().numpy() * 1000.0          # the same field on a shifted grid, for the reload check
    return V, params, dict(iterations=it + 1, final_loss_hu=float(np.mean(hist[-50:])) * 1000, box_voxels=R.tolist(), sigma_min_mm=smin, sigma_max_mm=smax), (probe.cpu().numpy(), Vp)

def eval_file(folder, P):
    """separate process side: the field at points P (mm, crop frame) from the saved file and its written kernel only."""
    from scipy.spatial import cKDTree
    A = np.load(os.path.join(folder, 'gaussians.npz')); meta = json.load(open(os.path.join(folder, 'gaussians.json'))); assert meta['kernel'] == KERNEL
    mu, sg, q, w = A['mu_mm'].astype(np.float64), A['sigma_mm'].astype(np.float64), A['quat_wxyz'].astype(np.float64), A['w_hu'].astype(np.float64)
    qw, qx, qy, qz = q.T
    Rm = np.stack([1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy), 2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
                   2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy)], 1).reshape(-1, 3, 3)
    val = np.full(len(P), float(A['b_hu'])); tree = cKDTree(P)
    for i, idx in enumerate(tree.query_ball_point(mu, 3 * sg.max(1) * 1.0001)):
        if not idx: continue
        loc = ((P[idx] - mu[i]) @ Rm[i]) / sg[i]; qd = (loc * loc).sum(1); val[idx] += w[i] * np.exp(-0.5 * qd) * np.clip(1 - qd / 9, 0, None) ** 2
    return val

def cortex_scores(P, Tq, F, px, bone_t):
    """femur wall on one plane: outer-edge and inner-edge distance (mm, both ways) and wall-thickness difference
    (area / outer perimeter). F: femur label on the plane, grown by its own wall so the predicted wall can be wider."""
    from scipy.ndimage import binary_dilation, binary_fill_holes, binary_erosion
    if F.sum() < 10: return None
    near = binary_dilation(F, iterations=3)
    def parts(V):
        bone = (V > bone_t) & near; outer = binary_fill_holes(bone); inner = outer & ~bone
        per = (outer & ~binary_erosion(outer)).sum() * px[0]; th = bone.sum() * px[0] * px[1] / per if per else np.nan
        return outer, inner, th
    po, pi, pt = parts(P); to, ti, tt = parts(Tq)
    return dict(outer_edge_mm=edge_dist(po, to, px), inner_edge_mm=edge_dist(pi, ti, px) if pi.any() and ti.any() else None,
                wall_thickness_err_mm=float(abs(pt - tt)) if np.isfinite(pt) and np.isfinite(tt) else None)

def agg2(rows, keys):
    out = {}
    for key in keys:
        v = [r[key] for r in rows if r and r.get(key) is not None]; out[key] = round(float(np.median(v)), 3) if v else None
    return out

if __name__ == '__main__' and len(sys.argv) > 1 and sys.argv[1] == '--eval-file':
    P = np.load(sys.argv[3]); np.save(sys.argv[4], eval_file(sys.argv[2], P)); print('EVAL_OK', flush=True); sys.exit()

def reload_experiments(T, TH, FM, sp, dev, n_g, max_iter=1500, work='export2', log=print):
    from scipy.ndimage import map_coordinates
    bt = otsu(np.clip(T, -1024, 200)); bone_t = otsu(T[TH & (T > LEAN[1])]); px = (float(sp[0]), float(sp[1])); local = np.diag(list(sp) + [1.0])
    res = dict(kernel=KERNEL, gaussians=n_g, bone_threshold=bone_t, runs=[])
    centre = np.argwhere(FM).mean(0) * sp; half = 0.5 * float(np.linalg.norm(np.array(T.shape) * sp)); step = float(sp.min())
    normals = [v / np.linalg.norm(v) for v in np.random.default_rng(2).normal(size=(6, 3))]
    obl = [plane_grid(centre, n, half, step)[0] for n in normals]
    Tob = [sample(T, local, G, cval=-1024.0) for G in obl]; Iob = [sample(TH.astype(np.float32), local, G, order=0, cval=0) > .5 for G in obl]; Fob = [sample(FM.astype(np.float32), local, G, order=0, cval=0) > .5 for G in obl]
    def scores_axial(V, zs):
        a = agg([score(V[..., z], T[..., z], TH[..., z], px, bt, bone_t) for z in zs]); c = agg2([cortex_scores(V[..., z], T[..., z], FM[..., z], px, bone_t) for z in zs], ('outer_edge_mm', 'inner_edge_mm', 'wall_thickness_err_mm')); return {**a, **c}
    def scores_obl(planes):
        a = agg([score(P, Tq, I, (step, step), bt, bone_t) for P, Tq, I in zip(planes, Tob, Iob)]); c = agg2([cortex_scores(P, Tq, F, (step, step), bone_t) for P, Tq, F in zip(planes, Tob, Fob)], ('outer_edge_mm', 'inner_edge_mm', 'wall_thickness_err_mm')); return {**a, **c}
    # (1) capacity with the fixes, and the exact reload test
    V, params, info, (probe, Vp) = fit_v2(T, sp, np.ones(T.shape, bool), TH, n_g, dev, max_iter=max_iter); log('capacity fit', info)
    os.makedirs(work, exist_ok=True); np.savez(os.path.join(work, 'gaussians.npz'), **params)
    json.dump(dict(kernel=KERNEL, frame='crop, mm', crop_to_world='see export/thigh_ct.nii.gz affine'), open(os.path.join(work, 'gaussians.json'), 'w'))
    rng = np.random.default_rng(5); vox = np.argwhere(TH); vox = vox[rng.choice(len(vox), min(200000, len(vox)), replace=False)]
    Pgrid = (vox + probe) * sp; trained = Vp[tuple(vox.T)]
    Pobl = np.concatenate([G.reshape(-1, 3) for G in obl]); np.save(os.path.join(work, 'pts.npy'), np.concatenate([Pgrid, Pobl]))
    r = subprocess.run([sys.executable, os.path.abspath(__file__), '--eval-file', work, os.path.join(work, 'pts.npy'), os.path.join(work, 'vals.npy')], capture_output=True, text=True)
    if 'EVAL_OK' not in r.stdout: raise RuntimeError(r.stderr[-2000:])
    vals = np.load(os.path.join(work, 'vals.npy')); fv = vals[:len(Pgrid)]; fo = vals[len(Pgrid):]
    dif = np.abs(fv - trained); n = 0; planes_file = []
    for G in obl: m = G.shape[0] * G.shape[1]; planes_file.append(fo[n:n + m].reshape(G.shape[:2])); n += m
    res['reload'] = dict(points=int(len(Pgrid)), max_abs_hu=float(dif.max()), mean_abs_hu=float(dif.mean()), passed=bool(dif.max() < 0.05))
    res['runs'].append(dict(name='capacity, all slices, fixes', **info, all_slices=scores_axial(V, range(T.shape[2])), oblique_from_file=scores_obl(planes_file)))
    log('reload', res['reload']); log(res['runs'][-1])
    # (2) Astra's control: the same Gaussians fitted to the volume interpolated from every k-th slice; (3) the fixes alone, given slices only
    for k in (4, 8):
        ctx = np.arange(0, T.shape[2], k); held = np.setdiff1d(np.arange(T.shape[2]), ctx); B = interp_z(T, ctx)
        res['runs'].append(dict(name='interpolation, every %d' % k, withheld_axial=scores_axial(B, held), oblique=scores_obl([sample(B, local, G, cval=-1024.0) for G in obl])))
        Vc, _, ic, _ = fit_v2(B, sp, np.ones(T.shape, bool), TH, n_g, dev, max_iter=max_iter); log('control k=%d' % k, ic)
        res['runs'].append(dict(name='Gaussians fitted to the interpolated volume, every %d' % k, **ic, withheld_axial=scores_axial(Vc, held), oblique=scores_obl([sample(Vc, local, G, cval=-1024.0) for G in obl])))
        fm = np.zeros(T.shape, bool); fm[..., ctx] = True
        Vs, _, isp, _ = fit_v2(T, sp, fm, TH, n_g, dev, max_iter=max_iter, z_shift=False); log('given slices only k=%d' % k, isp)
        res['runs'].append(dict(name='Gaussians with the fixes, given slices only, every %d' % k, **isp, withheld_axial=scores_axial(Vs, held), oblique=scores_obl([sample(Vs, local, G, cval=-1024.0) for G in obl])))
        for rr in res['runs'][-3:]: log(rr['name'], rr.get('withheld_axial'))
    # the interpolation and CT references for the capacity oblique scores (bone walls read the same way)
    res['ct_self_check_oblique'] = scores_obl(Tob)
    return res

if __name__ == '__main__':
    import torch, nibabel as nib
    from huggingface_hub import HfApi, hf_hub_download
    from scipy.ndimage import label as cc, binary_fill_holes
    api = HfApi(); repo = sys.argv[1]
    planes = list(csv.DictReader(open(hf_hub_download(repo, 't0a_v2/thigh_planes.csv', repo_type='dataset'))))
    byk = {}
    for p in planes: byk.setdefault((p['case'], p['side']), []).append(p)
    clean = [k_ for k_, v in byk.items() if not any(p.get(f) == 'True' for p in v for f in ('legs_touching', 'other_bone_in_thigh', 'metal')) and v[0]['contrast'] == 'False']
    case, side = clean[0]; log('subject', case, side)
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); Z = zipfile.ZipFile(zp)
    pre = [i.filename for i in Z.infolist() if i.filename.endswith('%s/ct.nii.gz' % case)][0][:-len('ct.nii.gz')]
    rd = lambda n: nib.Nifti1Image.from_bytes(gzip.decompress(Z.read(pre + n)))
    ct = rd('ct.nii.gz'); CT = np.asanyarray(ct.dataobj).astype(np.float32); aff = ct.affine
    FEM = np.asanyarray(rd('segmentations/femur_%s.nii.gz' % side).dataobj) > 0
    bt0 = otsu(np.clip(CT[:, :, CT.shape[2] // 2], -1024, 200)); body = CT > bt0; zs = np.flatnonzero(FEM.any((0, 1))); thigh = np.zeros(CT.shape, bool)
    for z in zs:
        lab, _ = cc(body[:, :, z]); ids = np.unique(lab[FEM[:, :, z]]); ids = ids[ids > 0]
        if len(ids): thigh[:, :, z] = binary_fill_holes(np.isin(lab, ids))
    ii = np.argwhere(thigh); lo = ii.min(0); hi = ii.max(0) + 1
    T = CT[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]].copy(); TH = thigh[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; T[~TH] = -1024.0
    FM = FEM[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]; sp = np.linalg.norm(aff[:3, :3], axis=0); del CT, body, thigh
    res = reload_experiments(T, TH, FM, sp, 'cuda', int(TH.sum()) // 64, log=log); res['subject'] = [case, side]
    api.upload_file(path_or_fileobj=json.dumps(res, indent=1).encode(), path_in_repo='t1_thigh/reload_v2.json', repo_id=repo, repo_type='dataset')
    log('SUMMARY ' + json.dumps(res))
