"""Self-test of stage A with a stand-in autoencoder (average-pool / upsample) on made-up bodies whose bone size and fat
thickness follow the body's size plus person noise."""
import sys, os, json, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stageA_job as J
rng = np.random.default_rng(0); work = 'work_test'; shutil.rmtree(work, ignore_errors=True); os.makedirs(work); os.makedirs('figsA_test', exist_ok=True)
def person(name):
    a, b = rng.uniform(34, 52), rng.uniform(26, 40); nz = int(rng.integers(40, 56)); X, Y, Z = np.meshgrid(np.arange(112) - 56, np.arange(88) - 44, np.arange(nz), indexing='ij')
    r = np.sqrt((X / a) ** 2 + (Y / b) ** 2); M = r < 1; fat = 0.12 + 0.25 * (a - 34) / 18 + rng.normal(0, .03); T = np.full(X.shape, -1000.0); T[M] = -100; T[r < 1 - fat] = 50
    for sx in (-1, 1): T[np.hypot(X - sx * a * .45, Y) < a * .12] = 900
    T += rng.normal(0, 20, T.shape); T[~M] = -1000; ii = np.argwhere(M); lo, hi = ii.min(0), ii.max(0) + 1; sl = tuple(slice(p, q) for p, q in zip(lo, hi))
    np.save('%s/%s_ct.npy' % (work, name), np.clip(T[sl], -1000, 1000).astype(np.int16)); np.save('%s/%s_m.npy' % (work, name), M[sl])
cases = {'train': ['a%02d' % i for i in range(40)], 'val': ['v%d' % i for i in range(6)], 'test': ['t%d' % i for i in range(8)]}
[person(c) for v in cases.values() for c in v]
pool = lambda A: A.reshape(A.shape[0] // 4, 4, A.shape[1] // 4, 4, A.shape[2] // 4, 4).mean((1, 3, 5))
enc = lambda T: np.stack([pool(T) / 1000, pool(np.clip(T, 300, 1000)) / 1000, pool(np.clip(T, -200, 100)) / 300, pool(T) / 1000 * 0 + .1 * rng.normal(size=pool(T).shape)]).astype(np.float32)
dec = lambda c: np.repeat(np.repeat(np.repeat(c[0], 4, 0), 4, 1), 4, 2) * 1000
if __name__ == '__main__':
  res = J.run(cases, work, enc, dec, 'cuda', 4, figdir='figsA_test')
  for s in ('val', 'test'): print(s, {n: {k: (v['median'] if v else None) for k, v in res[s][n].items()} for n in ('model', 'mean_ct', 'nearest_body', 'autoencoder_ceiling')}, 'beats mean', res[s]['model_beats_mean_ct_people'], 'beats nearest', res[s]['model_beats_nearest_body_people'])
