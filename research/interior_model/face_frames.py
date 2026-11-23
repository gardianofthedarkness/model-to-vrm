"""Milestone 15 (Tokens), face part: put every face_v1 head into its own head frame, built from its anatomy.

Head frame (per head, from its labels, no fixed numbers):
  origin  midpoint of the two auditory canals' centroids
  x       left canal -> right canal  (patient right)
  y       towards the midpoint of the two eyes, made orthogonal to x  (anterior)
  z       x cross y  (superior); right-handed in either scanner convention (LPS or RAS differ by a rotation)
The window is the union, over all heads, of the body's extent in that frame (read on a subsample), rounded up to the
coarsest token size tested. Labels are resampled to 1 mm in the frame (nearest), outside the scan = unknown (255).
Output per head: uint8 labels in the frame, the frame (origin, axes in the scan's mm), source and split.
args: <data dir> [--upload]"""
import os, sys, json, glob, numpy as np
from scipy.ndimage import map_coordinates
COARSEST_TOKEN_MM = 12          # the largest token size compared in milestone 15 (fixed-grid baseline, design v7)

def centroid_mm(L, k, aff):
    ii = np.argwhere(L == k)
    if not len(ii): return None
    return aff[:3, :3] @ ii.mean(0) + aff[:3, 3]

def frame(L, aff, ids):
    c = {n: centroid_mm(L, ids[n], aff) for n in ('auditory_canal_left', 'auditory_canal_right', 'eye_left', 'eye_right')}
    if any(v is None for v in c.values()): return None
    o = (c['auditory_canal_left'] + c['auditory_canal_right']) / 2
    x = c['auditory_canal_right'] - c['auditory_canal_left']; x /= np.linalg.norm(x)
    y = (c['eye_left'] + c['eye_right']) / 2 - o; y -= x * (x @ y); y /= np.linalg.norm(y)
    z = np.cross(x, y)
    return o, np.stack([x, y, z])          # rows: axes in scan mm

def to_frame(pts_mm, o, R): return (pts_mm - o) @ R.T

if __name__ == '__main__':
    root = sys.argv[1]; upload = '--upload' in sys.argv
    sys.path.insert(0, r'E:\code\model-to-vrm\model_to_vrm\common')
    from vault import get
    from huggingface_hub import snapshot_download, HfApi
    tok = get('huggingface'); api = HfApi(token=tok)
    REPOS = {'hanseg': 'gardianofthedarkness/hanseg-ct-mr', 'ts': 'gardianofthedarkness/totalsegmentator-ct-v201'}
    heads = []
    for src, repo in REPOS.items():
        d = snapshot_download(repo, repo_type='dataset', allow_patterns=['face_v1/*'], local_dir=os.path.join(root, 'raw_' + src), token=tok)
        meta = json.load(open(os.path.join(d, 'face_v1', 'face_%s.json' % src)))['stats']; ids = json.load(open(os.path.join(d, 'face_v1', 'face_ids.json')))['ids']
        for c, m in meta.items():
            if isinstance(m, dict): heads.append((src, c, os.path.join(d, 'face_v1', c + '.npz'), m))
    print(len(heads), 'heads')
    # pass 1: frames and extents
    frames, ext, skipped = {}, [], {}
    for src, c, p, m in heads:
        z = np.load(p); L, aff = z['labels'], z['affine']; f = frame(L, aff, ids)
        if f is None: skipped[c] = 'landmark missing'; continue
        frames[c] = f; ii = np.argwhere(L[::2, ::2, ::2] > 0) * 2
        q = to_frame(ii @ aff[:3, :3].T + aff[:3, 3], *f); ext.append((q.min(0), q.max(0)))
    lo = np.min([e[0] for e in ext], 0); hi = np.max([e[1] for e in ext], 0)
    lo = np.floor(lo / COARSEST_TOKEN_MM) * COARSEST_TOKEN_MM; hi = np.ceil(hi / COARSEST_TOKEN_MM) * COARSEST_TOKEN_MM
    shape = (hi - lo).astype(int); print('window mm', lo.tolist(), hi.tolist(), 'shape', shape.tolist())
    print('per-head extent quantiles (mm): lo', np.percentile([e[0] for e in ext], [0, 50, 100], 0).round().tolist(), 'hi', np.percentile([e[1] for e in ext], [0, 50, 100], 0).round().tolist())
    out = os.path.join(root, 'face_v1_headframe'); os.makedirs(out, exist_ok=True)
    g = np.stack(np.meshgrid(*[np.arange(n) + 0.5 for n in shape], indexing='ij'), -1).reshape(-1, 3) + lo    # frame mm
    index = {}
    for src, c, p, m in heads:
        if c not in frames: continue
        z = np.load(p); L, aff = z['labels'], z['affine']; o, R = frames[c]
        world = g @ R + o; idx = (world - aff[:3, 3]) @ np.linalg.inv(aff[:3, :3]).T
        Lf = map_coordinates(L, idx.T, order=0, mode='constant', cval=255).reshape(shape).astype(np.uint8)
        np.savez_compressed(os.path.join(out, c + '.npz'), labels=Lf)
        index[c] = dict(source=src, split=m.get('split') or 'hanseg', native_mm=m['native_mm'], known=m['known'], origin_mm=o.round(3).tolist(), axes=R.round(5).tolist(),
                        body_voxels_1mm=int(((Lf > 0) & (Lf < 255)).sum()), outside_scan_share=round(float((Lf == 255).mean()), 3))
        print(c, src, index[c]['body_voxels_1mm'], index[c]['outside_scan_share'])
    json.dump(dict(window_lo_mm=lo.tolist(), window_hi_mm=hi.tolist(), voxel_mm=1.0, unknown=255, ids=ids, heads=index, skipped=skipped), open(os.path.join(out, 'index.json'), 'w'), indent=1)
    if upload:
        for src, repo in REPOS.items():
            sel = [c for c, v in index.items() if v['source'] == src]
            api.upload_folder(folder_path=out, path_in_repo='face_v1_headframe', repo_id=repo, repo_type='dataset', allow_patterns=[c + '.npz' for c in sel] + ['index.json'], commit_message='face heads in head frame')
        print('uploaded')
