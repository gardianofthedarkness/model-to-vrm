import sys, os, json, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1_thigh_gauss_job as J
sp = np.array([1.5, 1.5, 1.5]); X, Y, Z = np.meshgrid(np.arange(70), np.arange(70), np.arange(60), indexing='ij')
cx = 35 + 4 * np.sin(Z / 15.0); r = np.hypot(X - 35, Y - 35); T = np.full(X.shape, -1000.0); TH = r < 30
T[TH] = -100; T[r < 22] = 50; T[np.hypot(X - cx, Y - 35) < 7] = 1200; T[~TH] = -1024; T = T.astype(np.float32)
V, params, info = J.fit_gaussians(T, sp, np.ones(T.shape, bool), TH, 2600, 1, 'cuda', max_iter=150, mem_entries=2e7)
os.makedirs('export_test', exist_ok=True); np.savez_compressed('export_test/gaussians.npz', **params)
json.dump(dict(centre_mm=(np.argwhere(TH).mean(0) * sp).tolist(), half_mm=0.5 * float(np.linalg.norm(np.array(T.shape) * sp)), step_mm=1.5), open('export_test/gaussians.json', 'w'))
t = time.time(); J.reopen_asset('export_test'); AP = np.load('export_test/asset_planes.npz'); local = np.diag(list(sp) + [1.0])
for i in range(3):
    G = AP['points_%d' % i].astype(float); Pv = AP['value_%d' % i]; I = J.sample(TH.astype(np.float32), local, G, order=0, cval=0) > 0.5
    print('plane', i, 'gap to in-process render %.2f HU' % np.abs(Pv - J.sample(V, local, G, cval=-1024.0))[I].mean(), J.score(Pv, J.sample(T, local, G, cval=-1024.0), I, (1.5, 1.5), -500, 600))
print('%.0f s' % (time.time() - t))
