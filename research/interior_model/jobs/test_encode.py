import sys, os, json, shutil, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import encode_job as J
X, Y, Z = np.meshgrid(np.arange(150), np.arange(120), np.arange(300), indexing='ij')
r = np.hypot((X - 70) / 50, (Y - 60) / 35); CT = np.full(X.shape, -1024.0); CT[r < 1] = -80; CT[r < .7] = 40
fem = np.hypot(X - 50, Y - 60) < 8; CT[fem & (Z < 150)] = 1100; CT[(Y > 110) & (Y < 114)] = 300            # a table under the body
CT += np.random.default_rng(0).normal(0, 15, CT.shape); aff = np.diag([1.5, 1.5, 1.5, 1.0])
labels = dict(femur_left=fem & (Z < 150))
r = J.prepare(CT.astype(np.float32), aff, labels)
calls = []
def enc(U): calls.append(U.shape); assert all(n % 16 == 0 for n in U.shape); return np.stack([U[::4, ::4, ::4]] * 4)
code = J.encode_blocks(r['T'].astype(np.float32), enc)
print('crop', r['crop_shape'], 'code', code.shape, 'blocks', calls[:2], len(calls), 'table kept out:', bool((r['T'] > 200).sum() < fem.sum() * 1.05))
print('landmarks', json.dumps(r['landmarks'])[:300]); print('sdf range', float(r['sdf'].min()), float(r['sdf'].max()))
os.makedirs('codes_test', exist_ok=True); J.save_case('codes_test', 'x', r, code, dict(age='50', sex='f', split='train'))
d = np.load('codes_test/x.npz'); print({k: d[k].shape for k in d.files}, os.path.getsize('codes_test/x.npz') // 1024, 'KB')
# code must equal a direct pooled encode in the kept middles
direct = np.stack([((r['T'].astype(np.float32) + 1000) / 2000)[::4, ::4, ::4]] * 4); print('block stitch max error', float(np.abs(code - direct).max()))
