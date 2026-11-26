"""Local check of the MRI-to-tissue model (test 1, run v2, trained on 42 HaN-Seg pairs) on IXI T1 heads, which come from
other scanners. Each MRI is prepared as in training: resampled to 1.5 mm on axes aligned with the patient (LPS), arrays in
(z, y, x) order, brightness scaled by the 0.5-99.5th percentiles inside the head. Reports tissue volumes, the eyes
(pieces and sizes) and writes two slice pictures per head (kept local: real people). args: <model .pt> <out dir> <nii>..."""
import os, sys, json, numpy as np, torch, SimpleITK as sitk
from scipy import ndimage as ndi
from monai.networks.nets import SegResNet
from monai.inferers import sliding_window_inference
NAMES = ['outside', 'fat_under_skin', 'fat_other', 'lean', 'air', 'bone', 'eye']
COL = np.array([[1, 1, 1], [0.98, 0.84, 0.45], [0.9, 0.7, 0.3], [0.62, 0.42, 0.40], [0.30, 0.60, 0.90], [0.93, 0.89, 0.78], [0.2, 0.9, 0.3]])
mf, out = sys.argv[1:3]; files = sys.argv[3:]; os.makedirs(out, exist_ok=True); dev = 'cuda'
net = SegResNet(spatial_dims=3, in_channels=1, out_channels=7, init_filters=16).to(dev); net.load_state_dict(torch.load(mf, map_location=dev)); net.eval()
rows = []
for f in files:
    im = sitk.Cast(sitk.ReadImage(f), sitk.sitkFloat32); sz = np.array(im.GetSize()); corners = np.array([im.TransformContinuousIndexToPhysicalPoint([float(a), float(b), float(c)]) for a in (0, sz[0]) for b in (0, sz[1]) for c in (0, sz[2])])
    lo, hi = corners.min(0), corners.max(0); sp = 1.5; size = [int(np.ceil((h - l) / sp)) for l, h in zip(lo, hi)]
    ref = sitk.Image(size, sitk.sitkFloat32); ref.SetSpacing([sp] * 3); ref.SetOrigin(lo.tolist()); ref.SetDirection(np.eye(3).ravel().tolist())
    M = sitk.GetArrayFromImage(sitk.Resample(im, ref, sitk.Transform(), sitk.sitkLinear, 0.0))          # (z, y, x), LPS axes as in training
    th = np.percentile(M[M > 0], 50) * 0.3 if (M > 0).any() else 0; B = ndi.binary_fill_holes(ndi.binary_opening(M > th, iterations=2)); lab, _ = ndi.label(B); B = lab == (np.bincount(lab.ravel())[1:].argmax() + 1)
    a, b = np.percentile(M[B], [0.5, 99.5]); X = np.clip((M - a) / max(b - a, 1e-6), 0, 1).astype(np.float32)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
        P = sliding_window_inference(torch.tensor(X[None, None], device=dev), (96, 96, 96), 1, net, overlap=0.5).float().argmax(1)[0].cpu().numpy().astype(np.uint8)
    vol = {NAMES[k]: round(float((P == k).sum()) * sp ** 3 / 1000, 1) for k in range(1, 7)}
    el, n = ndi.label(P == 6); sizes = sorted((np.bincount(el.ravel())[1:] * sp ** 3 / 1000).round(1).tolist(), reverse=True)[:4]
    r = dict(file=os.path.basename(f), ml=vol, eye_pieces=int(n), eye_piece_ml=sizes); rows.append(r); print(json.dumps(r), flush=True)
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    ez = np.argwhere(P == 6); zc = int(np.median(ez[:, 0])) if len(ez) else P.shape[0] // 2; xc = int(np.argwhere(B)[:, 2].mean())
    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    for r_, (img, seg, t) in enumerate(((X[:, :, xc], P[:, :, xc], 'side view, middle'), (X[zc], P[zc], 'top-down, through the eyes'))):
        ax[r_, 0].imshow(img, cmap='gray', origin='lower'); ax[r_, 0].set_title('MRI, ' + t); ax[r_, 1].imshow(COL[seg], origin='lower'); ax[r_, 1].set_title('tissue from MRI, ' + t)
        for a_ in ax[r_]: a_.axis('off')
    plt.tight_layout(); plt.savefig(os.path.join(out, os.path.basename(f).replace('.nii.gz', '.png')), dpi=60); plt.close()
json.dump(rows, open(os.path.join(out, 'ixi_tissue_local.json'), 'w'), indent=1)
