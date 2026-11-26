"""Dental metal and its streaks in CT, found from the data (user request 2026-11-26: remove them so the model does not
learn them).
  metal    voxels at the scan's ceiling value (the scanner saturates on metal: HaN-Seg heads 01, 03, 05 all hold 2.6-7.2 ml
           pinned at 3000 HU near the teeth), kept only where such voxels are a real block, not a lone voxel
  streaks  metal streaks run inside the slice plane, so an affected slice disagrees with the clean slices above and below.
           For each slice holding metal: the value expected there is interpolated (along z) from the nearest clean slices
           on either side; a body voxel is a streak if it differs from that by more than the 99th percentile of the same
           difference measured on clean slices (the normal slice-to-slice change, from the scan itself, not a set number).
Returns the artifact mask (metal, streaks, one voxel widened in-plane). Downstream: artifact voxels are unknown, except
where the labelling model says tooth or jaw bone (that identity stays).
detect(A, body) -> (mask, info); A in HU indexed (x, y, z) with z the slice axis."""
import numpy as np
from scipy import ndimage as ndi

def detect(A, body):
    top = float(A.max()); metal = A >= top - 1e-3
    lab, n = ndi.label(metal); sizes = np.bincount(lab.ravel())[1:]
    if n == 0 or sizes.max() < 8: return np.zeros(A.shape, bool), dict(metal_voxels=0, slices=0)
    metal = np.isin(lab, np.flatnonzero(sizes >= 8) + 1)
    zs = np.flatnonzero(metal.any((0, 1))); bad = np.zeros(A.shape[2], bool); bad[zs] = True
    clean = np.flatnonzero(~bad)
    # normal slice-to-slice change: |slice - mean of neighbours| on clean slices with clean neighbours, inside the body
    ok = [z for z in clean if 0 < z < A.shape[2] - 1 and not bad[z - 1] and not bad[z + 1]]
    samp = ok[:: max(1, len(ok) // 40)]
    diffs = np.concatenate([np.abs(A[:, :, z] - 0.5 * (A[:, :, z - 1] + A[:, :, z + 1]))[body[:, :, z]] for z in samp]) if samp else np.array([0.0])
    thr = float(np.percentile(diffs, 99))
    mask = metal.copy()
    for z in zs:
        lo = clean[clean < z]; hi = clean[clean > z]
        if len(lo) and len(hi): z0, z1 = lo.max(), hi.min(); w = (z - z0) / (z1 - z0); R = (1 - w) * A[:, :, z0] + w * A[:, :, z1]
        elif len(lo): R = A[:, :, lo.max()]
        else: R = A[:, :, hi.min()]
        mask[:, :, z] |= body[:, :, z] & (np.abs(A[:, :, z] - R) > thr)
    mask = ndi.binary_dilation(mask, structure=np.ones((3, 3, 1), bool)) & body
    return mask, dict(metal_voxels=int(metal.sum()), slices=int(len(zs)), threshold_hu=round(thr, 1), artifact_voxels=int(mask.sum()))
