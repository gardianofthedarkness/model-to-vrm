"""Self-test of A2 on made-up pelvises whose bone positions follow the outline (so a model can learn them)."""
import sys, os, json, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import a2_job as J
rng = np.random.default_rng(0); root = 'a2test'; shutil.rmtree(root, ignore_errors=True); cd, ld = root + '/codes', root + '/labels'; os.makedirs(cd); os.makedirs(ld)
ids = {'hip_left': 10, 'hip_right': 11, 'sacrum': 12, 'femur_left': 13, 'femur_right': 14, 'gluteus_maximus_left': 15, 'gluteus_maximus_right': 16, 'iliopsoas_left': 17, 'iliopsoas_right': 18, 'vertebrae_L5': 19, 'liver': 20}
def person(c):
    a, b = rng.uniform(70, 100), rng.uniform(50, 70); X, Y, Z = np.meshgrid(np.arange(224) - 112, np.arange(160) - 80, np.arange(96), indexing='ij')
    r = np.sqrt((X / a) ** 2 + (Y / b) ** 2); M = r < 1; L = np.zeros(X.shape, np.uint8); L[M] = 1; fat = 0.15 + 0.2 * (a - 70) / 30; L[r < 1 - fat] = 3
    for sx, i in ((-1, 10), (1, 11)): L[(np.hypot(X - sx * a * .45, Y + 5) < a * .12) & (Z > 30)] = i
    for sx, i in ((-1, 13), (1, 14)): L[(np.hypot(X - sx * a * .45, Y + 5) < a * .09) & (Z <= 30)] = i
    L[(np.hypot(X, Y + b * .55) < a * .12) & (Z > 40)] = 12; L[~M] = 0
    shp = list(L.shape); np.savez_compressed(f'{ld}/{c}.npz', labels=L); np.savez_compressed(f'{cd}/{c}.npz', mask=np.packbits(M))
    def bb(i): p = np.argwhere(L == i); return dict(lo=p.min(0).tolist(), hi=p.max(0).tolist(), centroid=p.mean(0).tolist())
    lm = {'hip_left': bb(10), 'hip_right': bb(11), 'sacrum': bb(12), 'femoral_head_left': dict(centroid=np.argwhere(L == 13)[np.argwhere(L == 13)[:, 2] >= 27].mean(0).tolist()), 'femoral_head_right': dict(centroid=np.argwhere(L == 14)[np.argwhere(L == 14)[:, 2] >= 27].mean(0).tolist())}
    json.dump(dict(crop_shape=shp, landmarks=lm), open(f'{cd}/{c}.json', 'w'))
cases = {'train': ['t%02d' % i for i in range(20)], 'val': ['v0', 'v1'], 'test': ['x0', 'x1', 'x2']}
[person(c) for v in cases.values() for c in v]
os.makedirs('figsA2_test', exist_ok=True)
res = J.run(cases, cd, ld, ids, 'cuda', 8, figdir='figsA2_test')
print({k: v for k, v in res['test']['median'].items()})
