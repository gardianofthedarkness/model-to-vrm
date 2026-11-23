"""Self-test of the extended classifier job on made-up pelvises, in both label modes and both window modes."""
import sys, os, json, shutil, importlib, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(0); root = 'a2btest'; shutil.rmtree(root, ignore_errors=True); cd, ld, td = root + '/codes', root + '/labels', root + '/tissue'; [os.makedirs(d) for d in (cd, ld, td)]
ids = {'hip_left': 10, 'hip_right': 11, 'sacrum': 12, 'femur_left': 13, 'femur_right': 14, 'gluteus_maximus_left': 15, 'liver': 20}
def person(c):
    a, b = rng.uniform(70, 100), rng.uniform(50, 70); X, Y, Z = np.meshgrid(np.arange(224) - 112, np.arange(160) - 80, np.arange(128), indexing='ij')
    taper = 1 - 0.25 * np.abs(Z - 70) / 70.0; r = np.sqrt((X / (a * taper)) ** 2 + (Y / (b * taper)) ** 2); M = r < 1; L = np.zeros(X.shape, np.uint8); L[M] = 1; fat = 0.15 + 0.2 * (a - 70) / 30; L[r < 1 - fat] = 3
    for sx, i in ((-1, 10), (1, 11)): L[(np.hypot(X - sx * a * .45, Y + 5) < a * .12) & (Z > 40) & (Z < 100)] = i
    for sx, i in ((-1, 13), (1, 14)): L[(np.hypot(X - sx * a * .45, Y + 5) < a * .09) & (Z <= 40)] = i
    L[(np.hypot(X, Y + b * .55) < a * .12) & (Z > 50) & (Z < 95)] = 12; L[~M] = 0
    T = np.zeros(L.shape, np.uint8); T[L == 1] = 1; T[(L == 3) & (r > 0.5)] = 3; T[(L == 3) & (r <= 0.2)] = 2
    np.savez_compressed(f'{ld}/{c}.npz', labels=L); np.savez_compressed(f'{td}/{c}.npz', tissue=T); np.savez_compressed(f'{cd}/{c}.npz', mask=np.packbits(M))
    def bb(i): p = np.argwhere(L == i); return dict(lo=p.min(0).tolist(), hi=p.max(0).tolist(), centroid=p.mean(0).tolist())
    lm = {'hip_left': bb(10), 'hip_right': bb(11), 'sacrum': bb(12), 'femoral_head_left': dict(centroid=np.argwhere(L == 13)[np.argwhere(L == 13)[:, 2] >= 37].mean(0).tolist())}
    json.dump(dict(crop_shape=list(L.shape), landmarks=lm), open(f'{cd}/{c}.json', 'w'))
cases = {'train': ['t%02d' % i for i in range(14)], 'val': ['v0', 'v1'], 'test': ['x0', 'x1']}
[person(c) for v in cases.values() for c in v]
for src, win in (('tissue', 'labels'), ('v1', 'surface')):
    os.environ['LABELSRC'] = src; os.environ['WINDOW'] = win
    import a2b_job as J; J = importlib.reload(J)
    res = J.run(cases, cd, ld, td, ids, 'cuda', 2, figdir=None)
    print(src, win, 'K', J.K, res.get('window_error_mm'), {k: res['test']['median'][k].get('bone_surface_mm') for k in res['test']['median']}, res['test']['model_better'], {k: v for k, v in res['test']['median']['surface_atlas'].items() if 'hip' in k or 'fascia' in k})
