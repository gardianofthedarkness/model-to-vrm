# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel", "torch", "monai==1.4.0", "matplotlib"]
# ///
"""Pictures for v6.4 E1a: can NVIDIA's frozen MAISI CT autoencoder (NV-Generate-CT, models/autoencoder_v1.pt, research use only)
hold our anatomy? No training. Each crop is encoded to the posterior mean and decoded, then scored like the Gaussian
capacity test, on every axial slice.

Crops: the same thigh as the Gaussian runs (s0783 left, chosen by the T0a checks) and the first 10 adults with a
whole pelvis (the T0b selection, in sorted order). MAISI's intensity rule: HU clipped to [-1000, 1000] and scaled to
[0, 1]; sides padded to a multiple of 16. Long crops are passed in overlapping blocks along z, keeping each block's
middle. Scores compare against the CT clipped the same way (the autoencoder never sees HU above 1000; the bone
threshold, set by Otsu as before, lies below that).
Outputs (private repo): e1/maisi_capacity.json."""
import os, sys, io, csv, json, gzip, time, zipfile, numpy as np
FAT, LEAN = (-190, -30), (-29, 150)
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def otsu(v):
    v = np.asarray(v, float); h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    s = s[:-1]; best = np.nanmax(s); tie = np.flatnonzero(s >= best * (1 - 1e-9))    # an empty gap between classes ties: take its middle
    return float(c[tie].mean())
def edge_dist(A, B, px):
    from scipy.ndimage import binary_erosion, distance_transform_edt
    if not A.any() or not B.any(): return None
    ea = A & ~binary_erosion(A); eb = B & ~binary_erosion(B)
    return float((distance_transform_edt(~eb, sampling=px)[ea].mean() + distance_transform_edt(~ea, sampling=px)[eb].mean()) / 2)
def main_piece(M):
    from scipy.ndimage import label, binary_fill_holes
    lab, n = label(M); return binary_fill_holes(lab == (np.bincount(lab.ravel())[1:].argmax() + 1)) if n else M
def dice(A, B):
    s = A.sum() + B.sum(); return float(2 * (A & B).sum() / s) if s else None
def score(P, T, inside, px, bt, bone_t):
    if inside.sum() == 0: return None
    return dict(hu_mae=float(np.abs(P - T)[inside].mean()), bone_edge_mm=edge_dist((P > bone_t) & inside, (T > bone_t) & inside, px),
                skin_edge_mm=edge_dist(main_piece(P > bt), main_piece(T > bt), px),
                fat_dice=dice((P >= FAT[0]) & (P <= FAT[1]) & inside, (T >= FAT[0]) & (T <= FAT[1]) & inside))
def cortex_scores(P, Tq, F, px, bone_t):
    from scipy.ndimage import binary_dilation, binary_fill_holes, binary_erosion
    if F.sum() < 10: return None
    near = binary_dilation(F, iterations=3)
    def parts(V):
        bone = (V > bone_t) & near; outer = binary_fill_holes(bone); inner = outer & ~bone
        per = (outer & ~binary_erosion(outer)).sum() * px[0]; return outer, inner, (bone.sum() * px[0] * px[1] / per if per else np.nan)
    po, pi, pt = parts(P); to, ti, tt = parts(Tq)
    return dict(outer_edge_mm=edge_dist(po, to, px), inner_edge_mm=edge_dist(pi, ti, px) if pi.any() and ti.any() else None,
                wall_thickness_err_mm=float(abs(pt - tt)) if np.isfinite(pt) and np.isfinite(tt) else None)
def agg(rows, keys):
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r and r.get(k) is not None]; out[k] = round(float(np.median(v)), 3) if v else None
    return out
KEYS = ('hu_mae', 'bone_edge_mm', 'skin_edge_mm', 'fat_dice'); CKEYS = ('outer_edge_mm', 'inner_edge_mm', 'wall_thickness_err_mm')

def roundtrip(T, model_fn, block=128, overlap=32):
    """T: HU volume. Returns the decoded HU volume. model_fn maps a (1,1,X,Y,Z) [0,1] tensor-like array to the same
    shape. Blocks along z of `block` slices overlapping by `overlap`, each keeping its middle; sides padded to x16 (the network trims other sizes)."""
    U = (np.clip(T, -1000, 1000) + 1000) / 2000.0
    pad = [(0, (-n) % 16) for n in U.shape]; Up = np.pad(U, pad, mode='edge'); Z = Up.shape[2]; out = np.zeros_like(Up)
    if Z <= block: out = model_fn(Up)
    else:
        starts = list(range(0, Z - block + 1, block - 2 * overlap));
        if starts[-1] + block < Z: starts.append(Z - block)
        for i, s in enumerate(starts):
            R = model_fn(Up[:, :, s:s + block]); a = 0 if i == 0 else overlap; b = block if i == len(starts) - 1 else block - overlap
            out[:, :, s + a:s + b] = R[:, :, a:b]
    out = out[:T.shape[0], :T.shape[1], :T.shape[2]]
    return out * 2000.0 - 1000.0

def evaluate(T, TH, FM, sp, model_fn):
    Tc = np.clip(T, -1000, 1000); D = roundtrip(T, model_fn); px = (float(sp[0]), float(sp[1]))
    bt = otsu(np.clip(Tc, -1024, 200)); bone_t = otsu(Tc[TH & (Tc > LEAN[1])])
    zs = [z for z in range(T.shape[2]) if TH[..., z].any()]
    r = agg([score(D[..., z], Tc[..., z], TH[..., z], px, bt, bone_t) for z in zs], KEYS)
    if FM is not None: r.update(agg([cortex_scores(D[..., z], Tc[..., z], FM[..., z], px, bone_t) for z in zs], CKEYS))
    r.update(voxels=int(np.prod(T.shape)), code_numbers=int(np.prod([-(-n // 4) for n in T.shape]) * 4), bone_threshold=bone_t)
    return r

def maisi_fn(dev):
    import torch
    from huggingface_hub import hf_hub_download
    from monai.apps.generation.maisi.networks.autoencoderkl_maisi import AutoencoderKlMaisi
    net = AutoencoderKlMaisi(spatial_dims=3, in_channels=1, out_channels=1, latent_channels=4, num_channels=[64, 128, 256], num_res_blocks=[2, 2, 2],
                             norm_num_groups=32, norm_eps=1e-6, attention_levels=[False, False, False], with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False,
                             use_checkpointing=False, use_convtranspose=False, norm_float16=True, num_splits=4, dim_split=1)
    sd = torch.load(hf_hub_download('nvidia/NV-Generate-CT', 'models/autoencoder_v1.pt'), map_location='cpu', weights_only=False)
    for key in ('state_dict', 'autoencoder', 'model'):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict): sd = sd[key]
    missing = net.load_state_dict(sd, strict=True); log('MAISI autoencoder loaded', missing)
    net = net.to(dev).eval()
    def fn(U):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            x = torch.tensor(U[None, None], dtype=torch.float32, device=dev); z_mu, _ = net.encode(x); y = net.decode(z_mu)
        return y.float()[0, 0].cpu().numpy()
    return fn


def figs(T, TH, FM, sp, D, tag, outdir):
    """pictures of one crop: CT, MAISI round trip, difference, fat masks; axial at the femur's middle and coronal."""
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    Tc = np.clip(T, -1000, 1000); W = dict(cmap='gray', vmin=-250, vmax=1000, interpolation='nearest')
    zf = np.flatnonzero(FM.any((0, 1))) if FM is not None and FM.any() else np.flatnonzero(TH.any((0, 1)))
    zc = int(np.median(zf)); jc = int(np.argwhere(FM[:, :, zc]).mean(0)[1]) if FM is not None and FM[:, :, zc].any() else T.shape[1] // 2
    ax_ = lambda V: V[:, :, zc].T; co_ = lambda V: V[:, jc, :].T[::-1]
    fat = lambda V: (V >= FAT[0]) & (V <= FAT[1]) & TH
    ext_a = [0, T.shape[0] * sp[0], T.shape[1] * sp[1], 0]; ext_c = [0, T.shape[0] * sp[0], 0, T.shape[2] * sp[2]]
    f, a = plt.subplots(2, 4, figsize=(17, 9.5), gridspec_kw=dict(height_ratios=[1, 1.4]))
    for row, (cut, ext) in enumerate(((ax_, ext_a), (co_, ext_c))):
        a[row, 0].imshow(cut(Tc), extent=ext, **W); a[row, 0].set_title('CT')
        a[row, 1].imshow(cut(D), extent=ext, **W); a[row, 1].set_title('MAISI encode + decode')
        im = a[row, 2].imshow(cut(D - Tc), extent=ext, cmap='coolwarm', vmin=-200, vmax=200, interpolation='nearest'); a[row, 2].set_title('decoded minus CT (HU)')
        fc, fd = cut(fat(Tc)), cut(fat(D)); rgb = np.zeros(fc.shape + (3,)); rgb[fc & fd] = (.85, .85, .85); rgb[fc & ~fd] = (.2, .45, 1); rgb[~fc & fd] = (1, .35, .2)
        a[row, 3].imshow(rgb, extent=ext, interpolation='nearest'); a[row, 3].set_title('fat: both (grey), CT only (blue), decoded only (red)', fontsize=9)
    f.colorbar(im, ax=a[:, 2], fraction=.03); f.suptitle('%s: axial at the dashed height (top), coronal through the femur (bottom)' % tag)
    for x in a[1]: x.axhline((T.shape[2] - zc) * sp[2], color='c', lw=.6, ls='--')
    f.savefig(os.path.join(outdir, '%s.png' % tag), dpi=120, bbox_inches='tight'); plt.close(f)
    # zoom on the femur wall and the fat layer next to it
    c = np.argwhere(FM[:, :, zc]).mean(0).astype(int) if FM is not None and FM[:, :, zc].any() else np.array(T.shape[:2]) // 2
    h = int(40 / sp[0]); sl = (slice(max(c[0] - h, 0), c[0] + h), slice(max(c[1] - h, 0), c[1] + h))
    f, a = plt.subplots(1, 2, figsize=(10, 5))
    for x, (V, t) in zip(a, ((Tc, 'CT'), (D, 'decoded'))): x.imshow(V[sl[0], sl[1], zc].T, **W); x.set_title('%s, 8 cm square round the femur' % t, fontsize=10); x.set_xticks([]); x.set_yticks([])
    f.savefig(os.path.join(outdir, '%s_zoom.png' % tag), dpi=130, bbox_inches='tight'); plt.close(f)

if __name__ == '__main__':
    import nibabel as nib
    from huggingface_hub import HfApi, hf_hub_download
    from scipy.ndimage import label as cc, binary_fill_holes
    api = HfApi(); repo = sys.argv[1]; fn = maisi_fn('cuda'); os.makedirs('figs_e1', exist_ok=True)
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); Z = zipfile.ZipFile(zp)
    names = [i.filename for i in Z.infolist() if i.filename.endswith('/ct.nii.gz')]
    def loader(case):
        pre = [n for n in names if n.endswith('%s/ct.nii.gz' % case)][0][:-len('ct.nii.gz')]
        return lambda n: nib.Nifti1Image.from_bytes(gzip.decompress(Z.read(pre + n)))
    def body_crop(CT, KEEP, zs):
        bt0 = otsu(np.clip(CT[:, :, CT.shape[2] // 2], -1024, 200)); body = CT > bt0; M = np.zeros(CT.shape, bool)
        for z in zs:
            lab, _ = cc(body[:, :, z]); ids = np.unique(lab[KEEP[:, :, z]]); ids = ids[ids > 0]
            if len(ids): M[:, :, z] = binary_fill_holes(np.isin(lab, ids))
        ii = np.argwhere(M); lo = ii.min(0); hi = ii.max(0) + 1; sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        T = CT[sl].copy(); TH = M[sl]; T[~TH] = -1024.0; return T, TH, sl
    for case, side, kind in (('s0783', 'left', 'thigh'), ('s0038', None, 'pelvis'), ('s0019', None, 'pelvis')):
        rd = loader(case); ct = rd('ct.nii.gz'); CT = np.asanyarray(ct.dataobj).astype(np.float32); sp = np.linalg.norm(ct.affine[:3, :3], axis=0)
        if kind == 'thigh':
            FEM = np.asanyarray(rd('segmentations/femur_%s.nii.gz' % side).dataobj) > 0; T, TH, sl = body_crop(CT, FEM, np.flatnonzero(FEM.any((0, 1)))); FM = FEM[sl]
        else:
            HIP = np.zeros(CT.shape, bool); FMp = np.zeros(CT.shape, bool)
            for b in ('hip_left', 'hip_right', 'sacrum'): HIP |= np.asanyarray(rd('segmentations/%s.nii.gz' % b).dataobj) > 0
            for b in ('femur_left', 'femur_right'):
                try: FMp |= np.asanyarray(rd('segmentations/%s.nii.gz' % b).dataobj) > 0
                except KeyError: pass
            T, TH, sl = body_crop(CT, HIP, np.flatnonzero(HIP.any((0, 1)))); FM = FMp[sl]
        D = roundtrip(T, fn); figs(T, TH, FM, sp, D, '%s_%s' % (kind, case), 'figs_e1'); log('drew', case)
    api.upload_folder(folder_path='figs_e1', path_in_repo='e1/figs', repo_id=repo, repo_type='dataset', commit_message='e1 pictures'); log('SUMMARY done')
