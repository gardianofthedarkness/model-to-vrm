"""Metal artifact reduction by linear interpolation in projection space (Kalender et al. 1987), slice by slice, with
scikit-image's Radon transform. For each slice that holds metal (voxels at the scan's saturation ceiling):
  1. forward-project the slice (radon), and the metal mask (which rays pass through metal)
  2. replace the metal rays in each projection by linear interpolation from the unaffected rays on either side
  3. reconstruct (filtered back projection); outside the metal, this is the slice without streaks
  4. put the metal voxels back as they were (they are identified by the labelling model as tooth or filling)
The difference between the original and corrected slice shows where the streaks were. Works on a downsampled square crop
for speed, and adds the correction (corrected - original) back at full resolution.
correct(A, metal) -> corrected A (HU), streak magnitude (HU); A indexed (x, y, z), z the slice axis."""
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import radon, iradon

def correct(A, metal, down=2):
    out = A.astype(np.float32).copy(); streak = np.zeros(A.shape, np.float32)
    zs = np.flatnonzero(metal.any((0, 1)))
    for z in zs:
        S = A[:, :, z].astype(np.float32); M = metal[:, :, z]
        n = max(S.shape); pad = [(0, n - S.shape[0]), (0, n - S.shape[1])]; Sp = np.pad(S, pad, constant_values=-1000.0); Mp = np.pad(M, pad)
        s = ndi.zoom(Sp + 1000.0, 1 / down, order=1); m = ndi.zoom(Mp.astype(np.float32), 1 / down, order=1) > 0.05   # attenuation-like (air = 0)
        th = np.linspace(0.0, 180.0, max(s.shape), endpoint=False)
        P = radon(s, theta=th, circle=False); Pm = radon(m.astype(np.float32), theta=th, circle=False) > 1e-3
        Pc = P.copy()
        for j in range(P.shape[1]):                                                            # each projection angle: interpolate across metal rays
            bad = Pm[:, j]
            if bad.any() and (~bad).sum() > 2: Pc[bad, j] = np.interp(np.flatnonzero(bad), np.flatnonzero(~bad), P[~bad, j])
        rec_o = iradon(P, theta=th, circle=False, output_size=s.shape[0]); rec_c = iradon(Pc, theta=th, circle=False, output_size=s.shape[0])
        delta = ndi.zoom(rec_c - rec_o, down, order=1)[:S.shape[0], :S.shape[1]]               # the correction, at full resolution
        delta[M] = 0.0                                                                         # metal voxels keep their value (identity from the label)
        out[:, :, z] = S + delta; streak[:, :, z] = np.abs(delta)
    return out, streak
