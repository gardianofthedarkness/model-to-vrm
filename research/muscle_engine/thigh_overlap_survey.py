"""E-PACK groundwork: how clean are the warped atlas surfaces of the thigh?  For every thigh structure: closed or not,
watertight edges, signed volume; for every pair of muscles (and muscle against bone): the share of A's surface vertices
that lie inside B (generalised winding number > 0.5, Warp).  Decides whether muscles can be meshed one by one or must be
resolved into one labelled volume first.
  E:\\splat\\.venv-sim\\Scripts\\python.exe thigh_overlap_survey.py <anatomy_dense dir> <out json>"""
import sys, os, json, numpy as np, warp as wp
FD, OUT = sys.argv[1:3]
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
MUS = ['Rectus femoris muscle', 'Vastus lateralis muscle', 'Vastus medialis muscle', 'Vastus intermedius muscle', 'Sartorius muscle',
       'Gracilis muscle', 'Long head of biceps femoris', 'Short head of biceps femoris', 'Semitendinosus muscle', 'Semimembranosus muscle',
       'Adductor longus', 'Adductor brevis', 'Adductor magnus', 'Tensor fasciae latae']
BON = ['Femur', 'Patella', 'Tibia', 'Fibula', 'Hip bone']
FAS = ['Fascia lata', 'Lateral femoral intermuscular septum', 'Medial femoral intermuscular septum']
def surf(n, F):
    f = F[OB[F[:, 0]] == nid[n + '.l']]; u, inv = np.unique(f, return_inverse=True); return V[u], inv.reshape(-1, 3).astype(np.int32)
S = {}
for n in MUS + FAS: S[n] = surf(n, R['F_muscle'])
for n in BON: S[n] = surf(n, R['F_bone'])
info = {}
for n, (P, F) in S.items():
    e = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1); _, c = np.unique(e, axis=0, return_counts=True)
    vol = np.einsum('ij,ij->i', P[F[:, 0]], np.cross(P[F[:, 1]], P[F[:, 2]])).sum() / 6
    info[n] = dict(verts=len(P), faces=len(F), boundary_edges=int((c == 1).sum()), nonmanifold_edges=int((c > 2).sum()), signed_volume_cm3=round(vol * 1e6, 2))
wp.init()
@wp.kernel
def wind(m: wp.uint64, q: wp.array(dtype=wp.vec3), out: wp.array(dtype=float)):
    i = wp.tid(); r = wp.mesh_query_point_sign_winding_number(m, q[i], 1.0e6, 2.0, 0.5)
    out[i] = 0.0
    if r.result: out[i] = r.sign
meshes = {}
for n, (P, F) in S.items():
    if n in FAS: continue
    vol = info[n]['signed_volume_cm3']; F2 = F if vol > 0 else F[:, [0, 2, 1]]
    meshes[n] = wp.Mesh(points=wp.array(P.astype(np.float32), dtype=wp.vec3), indices=wp.array(F2.ravel(), dtype=int), support_winding_number=True)
pairs = []
for a in MUS:
    Pa = S[a][0]; q = wp.array(Pa.astype(np.float32), dtype=wp.vec3); o = wp.zeros(len(Pa), dtype=float)
    for b in MUS + BON:
        if a == b: continue
        wp.launch(wind, len(Pa), [meshes[b].id, q, o]); inside = (o.numpy() < 0).mean()          # sign < 0 = inside
        if inside > 0.001: pairs.append(dict(a=a, b=b, share_of_a_verts_inside_b=round(float(inside), 4)))
pairs.sort(key=lambda r: -r['share_of_a_verts_inside_b'])
json.dump(dict(structures=info, overlaps=pairs), open(OUT, 'w'), indent=1)
for n, i in info.items(): print('S %-40s %s' % (n, i))
print('overlapping pairs (share of A inside B > 0.1 %%): %d' % len(pairs))
for r in pairs[:30]: print('O %-32s in %-32s %.1f %%' % (r['a'], r['b'], 100 * r['share_of_a_verts_inside_b']))
