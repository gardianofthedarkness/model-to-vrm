"""Bone support over the body surface as statistics over people: the HIT dataset's 398 whole-body MRI scans (bone, lean
tissue, fat labelled; each with a body-model surface of 6,890 corresponding points fitted to it; Keller et al., CVPR 2024,
Max Planck licence: non-commercial research only, no redistribution).
  E:\\splat\\.venv\\Scripts\\python.exe hit_support_stats.py <hit dataset dir> <out dir>
  bone label   only the long bones and the pelvis are labelled bone (upper arm, forearm, pelvis, thigh, shin): no skull,
               spine, ribs, hands or feet. So depth to bone is meaningful on the limbs and hips only.
  per person   every surface point, in its nearest scan slice (voxels 2.2 x 2.2 mm in the slice, slices 20 mm apart, so
               distances are measured inside the slice): (a) fat layer = distance to the nearest tissue that is not fat -
               valid over the whole body; (b) distance to the nearest bone voxel of that slice. Both as shares of the
               person's height (head top to heels on the fitted surface): proportions, not millimetres.
  statistics   per surface point over people: median and robust spread (1.4826 x median absolute deviation of the log
               share), for everyone and per sex, and the share of people with a value there. A thin layer that varies
               little between people marks a dependable firm area.
The .gz files are pickles: they are read with an unpickler that only builds arrays and plain containers.
Writes <out>/hit_support.npz (per point: depth/fat/lean medians and spreads, mean surface, faces) and prints a summary."""
import sys, os, glob, gzip, pickle, io, numpy as np
from scipy.spatial import cKDTree
from multiprocessing import Pool
HEAD_TOP, HEELS = 411, (3458, 6858)                                # SMPL surface point ids commonly used for stature (checked in main: top-most of the head, lowest of the feet)
class Safe(pickle.Unpickler):
    OK = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', '_reconstruct'),
          ('numpy._core.multiarray', 'scalar'), ('collections', 'OrderedDict'), ('numpy.core.numeric', '_frombuffer'), ('numpy._core.numeric', '_frombuffer')}
    def find_class(self, module, name):
        if (module, name) in self.OK: return super().find_class(module, name)
        raise pickle.UnpicklingError('blocked %s.%s' % (module, name))
def load(p):
    with gzip.open(p, 'rb') as f: return Safe(io.BytesIO(f.read())).load()
def one(p):
    d = load(p); seg = d['mri_seg']; res = d['resolution'][0]; cen = d['center'][0]; V = np.asarray(d['smpl_dict']['verts_free'], float); L = d['mri_labels']
    sl = np.clip(np.rint(V[:, 2] / res[2]).astype(int), 0, seg.shape[0] - 1); q = np.c_[V[:, 0] / res[0] + cen[0], V[:, 1] / res[1] + cen[1]]   # each surface point in its nearest slice (voxel units;
    bone = np.full(len(V), np.nan); fat = np.full(len(V), np.nan)                                                                                  # the grid was checked against the scan's own body contour)
    for k in np.unique(sl):
        m = sl == k; S = seg[k]; b = np.argwhere(S == L['BONE']); t = np.argwhere((S != L['AT']) & (S != L['NO']))
        if len(b): bone[m] = cKDTree(b).query(q[m])[0] * res[0]                                                                                    # in-slice distance to the nearest bone voxel
        if len(t): fat[m] = cKDTree(t).query(q[m])[0] * res[0]                                                                                     # in-slice distance to the nearest tissue that is not fat
    H = float(V[HEAD_TOP, 2] - V[list(HEELS), 2].mean())
    return d['gender'], H, (bone / H).astype(np.float32), (fat / H).astype(np.float32), V.astype(np.float32), np.asarray(d['smpl_dict']['faces']), float(res[2])
if __name__ == '__main__':
    D, OD = sys.argv[1:3]; os.makedirs(OD, exist_ok=True); files = sorted(glob.glob(os.path.join(D, '*', '*', '*.gz')))
    with Pool(8) as pool: R = [r for r in pool.map(one, files, chunksize=4) if r is not None]
    sex = np.array([r[0] for r in R]); H = np.array([r[1] for r in R]); DP = np.array([r[2] for r in R]); FT = np.array([r[3] for r in R]); Vm = np.mean([r[4] for r in R], 0); F = R[0][5]
    print('HS %d people (%d female, %d male); height median %.3f m (5%%..95%% %.3f..%.3f); slice spacing %s mm' % (len(R), (sex == 'female').sum(), (sex == 'male').sum(), np.median(H), *np.percentile(H, [5, 95]),
          sorted(set(round(1000 * r[6]) for r in R))))
    V0 = R[0][4]; print('HS stature points on the first person: head top z %.3f (surface max z %.3f incl. raised arms), heels z %.3f, %.3f (surface min z %.3f)' % (V0[HEAD_TOP, 2], V0[:, 2].max(), V0[HEELS[0], 2], V0[HEELS[1], 2], V0[:, 2].min()))
    def stats(X):                                                   # per point over the people who have a value there (a slice without bone gives none)
        l = np.log(np.maximum(X, res_floor)); med = np.nanmedian(l, 0); return np.exp(med), 1.4826 * np.nanmedian(np.abs(l - med), 0), np.isfinite(X).mean(0)
    res_floor = 0.0011 / float(np.median(H))                        # half an in-slice voxel: a point closer than the grid can resolve
    out = {'faces': F, 'verts_mean': Vm, 'height_median': float(np.median(H)), 'n': len(R)}
    for nm, m in (('all', np.ones(len(R), bool)), ('female', sex == 'female'), ('male', sex == 'male')):
        for key, X in (('depth', DP), ('fat', FT)):
            out['%s_%s_med' % (key, nm)], out['%s_%s_spread' % (key, nm)], out['%s_%s_have' % (key, nm)] = stats(X[m])
    np.savez(os.path.join(OD, 'hit_support.npz'), **out); Hm = out['height_median']
    for key, lab in (('fat', 'fat layer thickness (whole body)'), ('depth', 'depth to the nearest long bone / pelvis (only where most people have bone in the slice)')):
        md, sp, hv = out[key + '_all_med'], out[key + '_all_spread'], out[key + '_all_have']; ok = hv > 0.9
        print('HS %s: %d of %d points covered; median %.1f mm, 5%%..95%% of points %.1f..%.1f mm (at height %.2f m); spread between people median %.0f%%, 5%%..95%% %.0f..%.0f%%' % (
            lab, ok.sum(), len(ok), 1000 * Hm * np.median(md[ok]), *(1000 * Hm * np.percentile(md[ok], [5, 95])), Hm, 100 * np.median(sp[ok]), *(100 * np.percentile(sp[ok], [5, 95]))))
