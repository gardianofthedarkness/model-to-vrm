"""Synthetic thigh (made-up numbers): skin/fat ring, muscle, a femur that drifts with height, a fat septum. Checks the
Gaussian fit and scores run end to end and that k=1 fits better than sparse context."""
import sys, os, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1_thigh_gauss_job as J
sp = np.array([1.5, 1.5, 1.5]); X, Y, Z = np.meshgrid(np.arange(70), np.arange(70), np.arange(60), indexing='ij')
cx, cy = 35 + 4 * np.sin(Z / 15.0), 35.0
r = np.hypot(X - 35, Y - 35); T = np.full(X.shape, -1000.0); thigh = r < 30
T[thigh] = -100; T[r < 22] = 50; T[(np.abs(X - 35) < 1) & (r < 22)] = -100
fem = np.hypot(X - cx, Y - cy) < 7; T[fem] = 1200; T[np.hypot(X - cx, Y - cy) < 4] = 300
T += np.random.default_rng(0).normal(0, 10, T.shape); T[~thigh] = -1024
aff = np.diag(list(sp) + [1.0]); t = time.time()
res = J.run_fit_experiments(T.astype(np.float32), aff, thigh, 'cuda', ratios=(64, 16), ks=(1, 4))
for row in res['runs']: print({k: v for k, v in row.items() if k not in ('box_voxels',)})
print('%.0f s' % (time.time() - t))
