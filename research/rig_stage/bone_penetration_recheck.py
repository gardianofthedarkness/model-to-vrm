"""Re-measures muscle-node penetration into the bones for saved newton_muscle_active.py runs, with the atlas bones' winding
corrected.  The atlas bones are wound inward (negative signed volume); warp's mesh_query_point_sign_winding_number then
calls every point 'outside', so the runs' own 'inside bone' counts were always 0.  Here each bone mesh is reoriented to
positive volume first.  Surface nodes of every saved frame are tested against the bones posed at that frame (or the held
frame for isometric runs).
  E:\splat\.venv-sim\Scripts\python.exe bone_penetration_recheck.py <fits dir> <leg_transforms npz> <muscle mesh npz> <run npz> [...]
Glued nodes (the attachment patches, which sit on the bone by construction) are counted apart."""
import sys, os, json, numpy as np, warp as wp, trimesh
wp.config.quiet = True
FD, TRF, MESH = sys.argv[1:4]; RUNS = sys.argv[4:]
MM = np.load(MESH); srf, glue = MM['srf'], MM['glue']
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB, FB = R['V'].astype(np.float32), R['obj'], R['F_bone']
nm = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]
BONES = {0: ['Hip bone.l'], 1: ['Femur.l'], 2: ['Tibia.l', 'Fibula.l', 'Patella.l']}
@wp.kernel
def sdist(mid: wp.uint64, P: wp.array(dtype=wp.vec3), d: wp.array(dtype=float)):
    i = wp.tid(); r = wp.mesh_query_point_sign_winding_number(mid, P[i], 1.0)
    if r.result:
        d[i] = r.sign * wp.length(P[i] - wp.mesh_eval_position(mid, r.face, r.u, r.v))
meshes = {}
for sg, names in BONES.items():
    for n in names:
        fb = FB[OB[FB[:, 0]] == nm.index(n)]; u, inv = np.unique(fb, return_inverse=True); F = inv.reshape(-1, 3)
        vol = trimesh.Trimesh(V[u].astype(float), F, process=False).volume
        if vol < 0: F = F[:, [0, 2, 1]]
        meshes.setdefault(sg, []).append((n, vol, wp.Mesh(points=wp.array(V[u], dtype=wp.vec3), indices=wp.array(F.astype(np.int32).ravel(), dtype=int), support_winding_number=True)))
print('BP atlas bone signed volumes (negative = wound inward):', {n: '%.2e' % v for ms in meshes.values() for n, v, _ in ms})
TR = np.load(TRF)
def tf(sg, f):
    if sg == 0: return np.eye(3), np.zeros(3)
    return (TR['thigh_R'][f], TR['thigh_t'][f]) if sg == 1 else (TR['shank_R'][f], TR['shank_t'][f])
for run in RUNS:
    D = np.load(run); S, fr = D['surf'], list(D['frames']); hold = int(D['hold']) if 'hold' in D.files else -1; out = []
    for k, f in enumerate(fr):
        X = S[k]; ok = np.isfinite(X[:, 0]) & (srf >= 0); ok &= glue[np.maximum(srf, 0)] < 0; P = X[ok].astype(np.float64); best = np.full(len(P), np.inf)   # free surface nodes only
        for sg, ms in meshes.items():
            Rb, tb = tf(sg, hold if hold >= 0 else f)
            for n, _, m in ms:
                d = wp.full(len(P), 1e9, dtype=float); wp.launch(sdist, len(P), [m.id, wp.array(((P - tb) @ Rb).astype(np.float32), dtype=wp.vec3), d]); best = np.minimum(best, d.numpy())
        out.append(((best < 0).sum(), -1000 * min(0, best.min()), len(P)))
    w = int(np.argmax([o[1] for o in out]))
    print('BP %s: FREE surface nodes inside bone per saved frame, max %d of %d (%.1f%%), deepest %.1f mm at frame %d; frames with any inside: %d of %d' % (
        os.path.basename(run), max(o[0] for o in out), out[0][2], 100 * max(o[0] for o in out) / out[0][2], out[w][1], fr[w], sum(o[0] > 0 for o in out), len(out)))
