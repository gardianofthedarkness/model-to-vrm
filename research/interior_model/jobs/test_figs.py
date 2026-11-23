import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1_figs_job as J
sp = np.array([1.5, 1.5, 1.5]); X, Y, Z = np.meshgrid(np.arange(60), np.arange(56), np.arange(48), indexing='ij')
cx = 30 + 3 * np.sin(Z / 12.0); r = np.hypot(X - 30, Y - 28); T = np.full(X.shape, -1000.0); TH = r < 26
T[TH] = -100; T[r < 19] = 50; FM = np.hypot(X - cx, Y - 28) < 6; T[FM] = 1200; T[np.hypot(X - cx, Y - 28) < 3] = 150; T += np.random.default_rng(0).normal(0, 30, T.shape); T[~TH] = -1024
print(J.make_figs(T.astype(np.float32), TH, FM, sp, 'cuda', 1500, max_iter=40))
