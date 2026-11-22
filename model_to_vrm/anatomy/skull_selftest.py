"""Hidden-skull test of the face-to-skull placement: the atlas's own head, where the true skull is known.
  E:\\splat\\.venv\\Scripts\\python.exe skull_selftest.py <fits dir with atlas_warp.npz> <atlas head dir> <face_sim.npz of a model>
Fitting the atlas skull to the undeformed atlas face would be the identity (same template), so the head - skin and skull
together - is first deformed by a known map, and only the deformed SKIN is shown to the pipeline:
  case similar   the model's own similarity (rotation, size, shift from its face points): tests the detector's noise
  case affine    the model's own affine map (different stretch per axis: its proportions): tests what one size cannot follow
Then, exactly as on a model: zoomed views of the deformed skin (part_views.py), face points (zoom_joints3d.py), pairs with
the atlas's face points (brows, nose, eyes, mouth), skull placed by (a) one similarity, (b) one affine map of the pairs.
Error = distance of every skull vertex to its true place (the known map of the atlas skull), per bone, in mm and as a
share of the head's height. Writes <atlas head dir>/selftest.json."""
import sys, os, re, json, subprocess, numpy as np, trimesh
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segments import SEG
FD, AH, FSIM = sys.argv[1:4]; HERE = os.path.dirname(os.path.abspath(__file__)); BL = r'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe'
W = np.load(os.path.join(FD, 'atlas_warp.npz')); V0 = W['V0'].astype(np.float64); OB = W['obj']; names = [str(x) for x in W['names']]; sysn = [str(x) for x in W['system']]
SV = np.load(os.path.join(AH, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); SF = np.load(os.path.join(AH, 'parts', 'orig_normals.npz'))['F']
LA = np.load(os.path.join(AH, 'parts_zoom', 'head', 'points3d.npz'))['X'][:68]; bx = json.load(open(os.path.join(AH, 'parts_zoom', 'boxes.json')))['head']
S = np.load(FSIM); R, s, t, M = S['R'], float(S['s']), S['t'], S['M']
hb = [i for i, n in enumerate(names) if sysn[i] == 'bone' and re.search(SEG['head']['bones'], n)]; hm = np.isin(OB, hb)
CASES = {'similar': lambda X: X @ (s * R).T + t, 'affine': lambda X: np.c_[X, np.ones(len(X))] @ M}
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U, S_, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R_ = U @ D @ Vt; s_ = np.trace(np.diag(S_) @ D) / (Xc ** 2).sum(1).mean(); return R_, s_, my - s_ * R_ @ mx
out = {}
for case, T in CASES.items():
    D = os.path.join(AH, 'selftest_' + case); os.makedirs(os.path.join(D, 'parts'), exist_ok=True); os.makedirs(os.path.join(D, 'parts_zoom'), exist_ok=True)
    V = T(SV); N = np.array(trimesh.Trimesh(V, SF, process=False).vertex_normals)
    np.savez(os.path.join(D, 'parts', 'orig_colours.npz'), V=V); np.savez(os.path.join(D, 'parts', 'orig_normals.npz'), N=N, F=SF)
    trimesh.Trimesh(np.c_[V[:, 0], V[:, 2], -V[:, 1]], SF, process=False).export(os.path.join(D, 'head.glb'))
    cn = np.array(bx['centre']); hf = np.array(bx['half']); cr = T(cn + hf * np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]))   # the head box's corners, deformed
    json.dump({'head': {'centre': ((cr.min(0) + cr.max(0)) / 2).tolist(), 'half': ((cr.max(0) - cr.min(0)) / 2).tolist()}}, open(os.path.join(D, 'parts_zoom', 'boxes.json'), 'w'))
    np.save(os.path.join(D, 'parts_zoom', 'head_vids.npy'), np.arange(len(V)))
    subprocess.run([BL, '-b', '-P', os.path.join(HERE, 'part_views.py'), '--', os.path.join(D, 'head.glb'), os.path.join(D, 'parts_zoom'), 'head'], check=True, capture_output=True)
    subprocess.run([sys.executable, '-W', 'ignore', os.path.join(HERE, 'zoom_joints3d.py'), os.path.join(D, 'parts_zoom'), 'head', 'clay'], check=True, capture_output=True)
    LE = np.load(os.path.join(D, 'parts_zoom', 'head', 'points3d.npz'))['X'][:68]; sel = np.array([i for i in range(17, 68) if np.isfinite(LA[i]).all() and np.isfinite(LE[i]).all()])
    la, le = LA[sel], LE[sel]; truth = T(V0[hm]); H = float(np.ptp(T(SV[np.abs(SV - cn).max(1) < hf.max()])[:, 2])); det = np.linalg.norm(le - T(la), axis=1)   # detection noise: the same point found twice
    Ru, su, tu = umeyama(la, le); Ma = np.linalg.lstsq(np.c_[la, np.ones(len(la))], le, rcond=None)[0]
    res = {'pairs': int(len(sel)), 'head_height_mm': 1000 * H, 'detected_vs_true_landmark_mm': {'median': 1000 * float(np.median(det)), 'p90': 1000 * float(np.percentile(det, 90))}}
    for fit, P in (('one_similarity', V0[hm] @ (su * Ru).T + tu), ('one_affine', np.c_[V0[hm], np.ones(hm.sum())] @ Ma)):
        e = np.linalg.norm(P - truth, axis=1); ob = OB[hm]; pb = {names[b]: 1000 * float(np.median(e[ob == b])) for b in np.unique(ob)}
        res[fit] = {'skull_error_mm': {'median': 1000 * float(np.median(e)), 'p90': 1000 * float(np.percentile(e, 90)), 'max': 1000 * float(e.max())}, 'share_of_head_height': float(np.median(e) / H),
                    'worst_bones_mm': dict(sorted(pb.items(), key=lambda z: -z[1])[:6]), 'best_bones_mm': dict(sorted(pb.items(), key=lambda z: z[1])[:3]),
                    'named_bones_mm': {k: pb[k] for k in ('Frontal bone', 'Maxilla.r', 'Zygomatic bone.r', 'Mandible', 'Occipital bone', 'Parietal bone.r') if k in pb}}
        print('ST %s / %s: skull error median %.2f mm, p90 %.2f, max %.2f (%.1f%% of the head height %.0f mm); %s' % (case, fit, res[fit]['skull_error_mm']['median'], res[fit]['skull_error_mm']['p90'],
              res[fit]['skull_error_mm']['max'], 100 * res[fit]['share_of_head_height'], 1000 * H, {k: round(v, 1) for k, v in res[fit]['named_bones_mm'].items()}), flush=True)
    if case == 'similar': res['scale_true_vs_found'] = [s, float(su)]
    print('ST %s: %d pairs; detected vs true landmark median %.2f mm (p90 %.2f)' % (case, len(sel), res['detected_vs_true_landmark_mm']['median'], res['detected_vs_true_landmark_mm']['p90']), flush=True); out[case] = res
json.dump(out, open(os.path.join(AH, 'selftest.json'), 'w'), indent=1)
