"""T1, pack integrity (rules in pack_format.md).  Writes <pack>.t1.json with one pass/fail per rule and the numbers.
Rule 4 runs libuipc's own sanity check on the region surfaces that will be split (any region not tied to all its
neighbours), so the verdict is the solver's, not ours.  Rule 6 builds the pack in Newton and counts the tets it keeps.
  E:\\splat\\.venv-uipc\\Scripts\\python.exe t1_pack_check.py <pack.npz> [d_hat m, default 1e-3]
  (rule 6 needs Newton: run again with E:\\splat\\.venv\\Scripts\\python.exe and NEWTON=1)"""
import sys, os, json, io, contextlib, tempfile, numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
PK = sys.argv[1]; DHAT = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-3
P = np.load(PK, allow_pickle=True); X, T, REG = P['X'], P['T'], P['region']; NAMES = [str(s) for s in P['region_names']]
KIND = [str(s) for s in P['region_kind']] if 'region_kind' in P.files else ['muscle'] * len(NAMES)
res = dict(pack=os.path.basename(PK), nodes=int(len(X)), tets=int(len(T)), rules={})
def ok(rule, passed, **kw): res['rules'][rule] = dict(passed=bool(passed), **kw); print('T1 %-26s %s %s' % (rule, 'PASS' if passed else 'FAIL', kw), flush=True)
# 1 signed volumes
D = np.stack([X[T[:, i]] - X[T[:, 0]] for i in (1, 2, 3)], 2); vol = np.linalg.det(D) / 6
ok('1 positive volume', (vol > 0).all() and (vol > 1e-6 * np.median(np.abs(vol))).all(), negative=int((vol <= 0).sum()), degenerate=int((np.abs(vol) <= 1e-6 * np.median(np.abs(vol))).sum()))
# 2 per region: one piece, closed manifold boundary
FACES = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])                  # outward for positive tets
surfs, rep2 = {}, {}
for r, nm in enumerate(NAMES):
    t = T[REG == r]
    if not len(t): rep2[nm] = 'empty'; continue
    e = np.concatenate([t[:, [a, b]] for a in range(4) for b in range(a + 1, 4)])
    nc, lab = connected_components(coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(X), len(X))), directed=False)
    pieces = len(np.unique(lab[t[:, 0]]))
    f = np.concatenate([t[:, k] for k in FACES]); key = np.sort(f, 1); _, inv, cnt = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    bf = f[cnt[inv.ravel()] == 1]; surfs[nm] = bf
    be = np.sort(np.concatenate([bf[:, [0, 1]], bf[:, [1, 2]], bf[:, [2, 0]]]), 1); _, ec = np.unique(be, axis=0, return_counts=True)
    nonman = int((ec != 2).sum()); vcount = np.bincount(bf.ravel(), minlength=len(X));
    rep2[nm] = dict(pieces=int(pieces), boundary_faces=int(len(bf)), nonmanifold_edges=nonman, volume_cm3=round(float(vol[REG == r].sum()) * 1e6, 3))
ok('2 regions closed manifold', all(isinstance(v, dict) and v['pieces'] == 1 and v['nonmanifold_edges'] == 0 for v in rep2.values()), regions=rep2)
# 3 volumes against the source surfaces (reported only)
if 'source_volume_cm3' in P.files:
    sv = json.loads(str(P['source_volume_cm3'])); ok('3 volume vs source', True, change_pct={k: round(100 * (rep2[k]['volume_cm3'] / v - 1), 1) for k, v in sv.items() if k in rep2 and isinstance(rep2[k], dict)})
# 4 libuipc sanity check on the region surfaces (each surface a separate object, as split interfaces will be)
inter = json.loads(str(P['interfaces'])) if 'interfaces' in P.files else []
tied = {(i['a'], i['b']) for i in inter if i['law'] in ('tied', 'attached')} | {(i['b'], i['a']) for i in inter if i['law'] in ('tied', 'attached')}
try:
    import uipc
    from uipc import Engine, World, Scene
    from uipc.geometry import trimesh as uipc_trimesh, label_surface
    from uipc.constitution import ElasticModuli2D
    cfg = Scene.default_config(); cfg['contact']['d_hat'] = DHAT; cfg['gravity'] = [[0.0], [0.0], [0.0]]
    ws = tempfile.mkdtemp(prefix='t1_uipc_'); eng = Engine('cuda', ws, Engine.default_config() | {'extras': {'gui': {'enable': False}}})
    world = World(eng); scene = Scene(cfg); scene.contact_tabular().default_model(0.0, 1e9)
    from uipc.constitution import NeoHookeanShell
    sh = NeoHookeanShell()
    for nm, bf in surfs.items():
        u, inv = np.unique(bf, return_inverse=True); m = uipc_trimesh(X[u].astype(np.float64), inv.reshape(-1, 3).astype(np.int32)); label_surface(m)
        sh.apply_to(m, ElasticModuli2D.youngs_poisson(1e5, 0.3), 1000.0, 1e-4); scene.objects().create(nm).geometries().create(m)
    world.init(scene)                                    # the sanity check prints from C++ and writes its findings under ws
    found = [os.path.relpath(os.path.join(d, f), ws) for d, _, fs in os.walk(ws) for f in fs if f.endswith('.obj')]
    # TODO when interfaces exist: pairs the pack declares tied are allowed to touch
    ok('4 no intersection (uipc)', world.is_valid() and not found, uipc_valid=bool(world.is_valid()), findings=found[:20], d_hat=DHAT, workspace=ws)
except ImportError:
    ok('4 no intersection (uipc)', False, error='run with E:/splat/.venv-uipc')
# 5 fibres
if 'fib' in P.files:
    mus = np.isin(REG, [i for i, k in enumerate(KIND) if k == 'muscle']); n_ = np.linalg.norm(P['fib'][mus], axis=1)
    ok('5 unit fibres in muscle', np.allclose(n_, 1, atol=1e-6), min=float(n_.min()) if len(n_) else None, max=float(n_.max()) if len(n_) else None)
# 6 Newton keeps every tet
if os.environ.get('NEWTON') == '1':
    import warp as wp, newton
    b = newton.ModelBuilder()
    for x in X: b.add_particle(tuple(map(float, x)), (0.0, 0.0, 0.0), 1.0)
    for t in T: b.add_tetrahedron(*map(int, t), k_mu=1e3, k_lambda=1e4, k_damp=0.0)
    m = b.finalize(); ok('6 solver keeps all tets', m.tet_count == len(T), newton_tets=int(m.tet_count), pack_tets=int(len(T)))
json.dump(res, open(PK.replace('.npz', '.t1.json'), 'w'), indent=1, default=str)
print('T1 overall', 'PASS' if all(r['passed'] for r in res['rules'].values()) else 'FAIL')
