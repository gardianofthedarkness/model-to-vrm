import sys, os, numpy as np
from scipy.ndimage import gaussian_filter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e1_maisi_job as J
sp = np.array([1.5, 1.5, 1.5]); X, Y, Z = np.meshgrid(np.arange(62), np.arange(57), np.arange(301), indexing='ij')
cx = 30 + 3 * np.sin(Z / 40.0); r = np.hypot(X - 30, Y - 28); T = np.full(X.shape, -1000.0); TH = r < 26
T[TH] = -100; T[r < 19] = 50; FM = np.hypot(X - cx, Y - 28) < 6; T[FM] = 1200; T[np.hypot(X - cx, Y - 28) < 3] = 150
T += np.random.default_rng(0).normal(0, 30, T.shape); T[~TH] = -1024
calls = []
def ident(U): calls.append(U.shape); assert all(n % 16 == 0 for n in U.shape); return U
print('identity:', J.evaluate(T, TH, FM, sp, ident), 'blocks', calls)
print('blur 1 voxel:', J.evaluate(T, TH, FM, sp, lambda U: gaussian_filter(U, 1.0)))
