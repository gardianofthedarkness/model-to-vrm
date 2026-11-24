"""Why the knee looks wrong in the fixed flesh run (elf_leg_flesh.py SAVE_NODES=1).
For every clip frame:
  * knee angle (from the calibrated transforms),
  * the patella against the femur: how deep patella vertices sit inside the femur (signed distance, winding number),
    and the patella's distance to the femur surface, since in elf_leg_flesh.py the patella rides rigidly with the shank,
  * flesh damage near the knee: inverted tets and flipped skin triangles within 3 tet edges of the patella, split into front
    (skin side of the patella) and back (the crease), and how many of those inverted tets have pinned nodes on two bone groups.
Writes knee_diagnose.json and a close-up picture of the knee (bones + skin) at chosen frames.
  E:\\splat\\.venv\\Scripts\\python.exe knee_diagnose.py <fits dir> <flesh_nodes dir> <leg_transforms npz> <out dir> [frames ...]"""
import sys, os, json, numpy as np, trimesh, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import cKDTree
FD, ND, TRF, OUT = sys.argv[1:5]; want = [int(x) for x in sys.argv[5:]]
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB, FB = R['V'].astype(np.float64), R['obj'], R['F_bone']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
def bone(n):
    fb = FB[OB[FB[:, 0]] == nid[n]]; u, inv = np.unique(fb, return_inverse=True); F = inv.reshape(-1, 3)
    if trimesh.Trimesh(V[u], F, process=False).volume < 0: F = F[:, [0, 2, 1]]     # the atlas bones are wound inward: make normals point out
    return V[u], F
FemV, FemF = bone('Femur.l'); PatV, PatF = bone('Patella.l'); TibV, TibF = bone('Tibia.l')
fem = trimesh.Trimesh(FemV, FemF, process=False); FemN = fem.vertex_normals; kfem = cKDTree(FemV)
def signed_to_femur(P):
    """distance to the nearest femur vertex, negative when on the inner side of its outward normal (the atlas femur is dense)"""
    d, i = kfem.query(P); return np.where(((P - FemV[i]) * FemN[i]).sum(1) < 0, -d, d)
N = np.load(os.path.join(ND, 'elf_leg_nodes.npz')); Q, X0, T, pin, grp = N['q'], N['X0'], N['T'], N['pin'], N['grp']
TR = np.load(TRF); nf = len(Q)
D = np.load(os.path.join(ND, 'elf_leg_flesh.npz')); SF = D['faces']; S0 = D['rest'].astype(np.float64)
# the knee region: tets within 3 tet edges of the patella; front = nodes farther from the femur's long axis than the patella centre
h = json.load(open(os.path.join(ND, 'elf_leg_flesh.json')))['summary']['tet_edge_mm'] / 1000
pc = PatV.mean(0); cen0 = X0[T].mean(1); near = cKDTree(PatV).query(cen0)[0] < 3 * h
fa = np.linalg.svd(FemV - FemV.mean(0))[2][0]; rad = lambda P: np.linalg.norm((P - FemV.mean(0)) - np.outer((P - FemV.mean(0)) @ fa, fa), axis=1)
front = rad(cen0) > rad(pc[None])[0]
mixed = np.array([len(set(grp[t][pin[t]])) > 1 for t in T])
Dm = np.stack([X0[T[:, 1]] - X0[T[:, 0]], X0[T[:, 2]] - X0[T[:, 0]], X0[T[:, 3]] - X0[T[:, 0]]], 2); Dinv = np.linalg.inv(Dm)
# skin near the knee from the saved skin frames (every 15th) is too coarse; use the tet surface instead
Fc = np.sort(np.vstack([T[:, [0, 1, 2]], T[:, [0, 1, 3]], T[:, [0, 2, 3]], T[:, [1, 2, 3]]]), 1); uf, inv_, cf = np.unique(Fc, axis=0, return_inverse=True, return_counts=True)
rows = []
for f in range(nf):
    Rs, ts = TR['shank_R'][f], TR['shank_t'][f]; Rt, tt = TR['thigh_R'][f], TR['thigh_t'][f]
    P = PatV @ Rs.T + ts; Pf = (P - tt) @ Rt                     # patella in the femur's rest frame
    sd = signed_to_femur(Pf)
    q = Q[f].astype(np.float64); J = np.linalg.det(np.stack([q[T[:, 1]] - q[T[:, 0]], q[T[:, 2]] - q[T[:, 0]], q[T[:, 3]] - q[T[:, 0]]], 2) @ Dinv)
    bad = J <= 0
    rows.append(dict(frame=f, knee_deg=float(TR['knee'][f]),
                     patella_inside_femur_vertices=int((sd < 0).sum()), patella_deepest_mm=float(-1000 * min(0, sd.min())), patella_gap_min_mm=float(1000 * max(0, sd.min())),
                     knee_tets=int(near.sum()), knee_inverted_front=int((bad & near & front).sum()), knee_inverted_back=int((bad & near & ~front).sum()),
                     knee_inverted_mixed_pins=int((bad & near & mixed).sum()), all_inverted=int(bad.sum()), knee_J_min=float(J[near].min())))
J_ = dict(frames=rows, knee_tets=int(near.sum()), knee_tets_front=int((near & front).sum()), knee_mixed_pin_tets=int((near & mixed).sum()), all_mixed_pin_tets=int(mixed.sum()))
os.makedirs(OUT, exist_ok=True); json.dump(J_, open(os.path.join(OUT, 'knee_diagnose.json'), 'w'), indent=1)
worst = int(np.argmax([r['patella_deepest_mm'] for r in rows])); wi = int(np.argmax([r['knee_inverted_front'] + r['knee_inverted_back'] for r in rows]))
print('KD knee region %d tets (%d front of the patella centre), %d with pins on two bone groups (whole leg %d)' % (near.sum(), (near & front).sum(), (near & mixed).sum(), mixed.sum()))
for f in sorted(set([0, worst, wi] + list(range(0, nf, 30)))):
    r = rows[f]; print('KD frame %3d (knee %5.1f deg): patella inside femur %4d verts, deepest %.1f mm | knee inverted front %d back %d (mixed pins %d) | knee J min %.2f' % (
        f, r['knee_deg'], r['patella_inside_femur_vertices'], r['patella_deepest_mm'], r['knee_inverted_front'], r['knee_inverted_back'], r['knee_inverted_mixed_pins'], r['knee_J_min']))
# ---- close-up picture: bones + tet surface coloured by J of its tet, around the knee, from her left and from the front
pick = want or [0, wi]
fig = plt.figure(figsize=(5.2 * len(pick), 10)); light = np.array([-0.4, -0.6, 0.7]); light /= np.linalg.norm(light)
surf = uf[cf == 1]
tet_of = np.repeat(np.arange(len(T)), 4)[np.argsort(inv_, kind='stable')]; first = np.searchsorted(np.sort(inv_), np.arange(len(uf))); stet = tet_of[first][cf == 1]
for j, f in enumerate(pick):
    q = Q[f].astype(np.float64); J = np.linalg.det(np.stack([q[T[:, 1]] - q[T[:, 0]], q[T[:, 2]] - q[T[:, 0]], q[T[:, 3]] - q[T[:, 0]]], 2) @ Dinv)
    Rs, ts = TR['shank_R'][f], TR['shank_t'][f]; Rt, tt = TR['thigh_R'][f], TR['thigh_t'][f]; P = PatV @ Rs.T + ts; c = P.mean(0); hw = 12 * h
    for row, (el, az, lab) in enumerate(((0, 0, 'from her left'), (0, -90, 'from the front'))):
        ax = fig.add_subplot(2, len(pick), row * len(pick) + j + 1, projection='3d', computed_zorder=False)
        for (BV, BF, RR, tt_) in ((FemV, FemF, Rt, tt), (TibV, TibF, Rs, ts), (PatV, PatF, Rs, ts)):
            X = BV @ RR.T + tt_; keep = (np.abs(X[BF].mean(1) - c) < hw).all(1); Fk = BF[keep][::2]
            n = np.cross(X[Fk[:, 1]] - X[Fk[:, 0]], X[Fk[:, 2]] - X[Fk[:, 0]]); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
            col = np.array([0.95, 0.75, 0.2]) if BV is PatV else np.array([0.85, 0.82, 0.74])
            ax.add_collection3d(Poly3DCollection(X[Fk], facecolors=np.clip(col * (0.4 + 0.6 * np.abs(n @ light))[:, None], 0, 1), linewidths=0, zorder=1))
        keep = (np.abs(q[surf].mean(1) - c) < hw).all(1); Fs = surf[keep]; js = J[stet[keep]]
        v = np.clip(np.log2(np.clip(js, 1e-3, 4)) / 2, -1, 1); colr = np.c_[0.75 + 0.25 * np.clip(v, 0, 1), 0.72 * (1 - np.abs(v)), 0.55 + 0.45 * np.clip(-v, 0, 1)]; colr[js <= 0] = [0.05, 0.05, 0.05]
        ax.add_collection3d(Poly3DCollection(q[Fs], facecolors=np.c_[colr, np.full(len(Fs), 0.35)], edgecolors=(0, 0, 0, 0.08), linewidths=0.2, zorder=2))
        ax.set_xlim(c[0] - hw, c[0] + hw); ax.set_ylim(c[1] - hw, c[1] + hw); ax.set_zlim(c[2] - hw, c[2] + hw); ax.set_box_aspect((1, 1, 1)); ax.view_init(el, az); ax.axis('off')
        r = rows[f]; ax.set_title('clip frame %d (t = %.1f s), %s\npatella inside femur: %d verts, %.1f mm deep; knee tets inverted %d' % (
            f, f / 30, lab, r['patella_inside_femur_vertices'], r['patella_deepest_mm'], r['knee_inverted_front'] + r['knee_inverted_back']), fontsize=9)
fig.suptitle('Knee close-up, fixed flesh run: bones (patella yellow) and the flesh surface, translucent, coloured by tet volume change (blue squashed, red swollen, black inverted)', fontsize=10)
fig.tight_layout(); fig.savefig(os.path.join(OUT, 'pic_knee_diagnose.png'), dpi=80); print('wrote', os.path.join(OUT, 'pic_knee_diagnose.png'))
