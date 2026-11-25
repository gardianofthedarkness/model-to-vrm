"""E-PACK, GPU route, step 1: a label grid of the left thigh.  Same members, box and sizes as epack_thigh.py (epack_members.py);
the inside tests that took wildmeshing 68 min on the CPU run here on the GPU with Warp (generalised winding number + closest
point, per member only inside its own bounding box).  Each cell takes one label:
  outside the skin or the box -> -1;  inside a bone -> that bone (bones override);  inside one or more muscles -> the muscle
  it is deepest inside (largest distance to that muscle's surface: the overlap goes to the nearer muscle, as resolve_overlaps);
  else fat.
Cell size: the finest member element (volume/area rule), unless given.
Writes <out>/grid_thigh_l.npz: lab (int16, z-y-x... stored x,y,z), origin, h, region_names, region_kind, source_volume_cm3.
  E:\\splat\\.venv-sim\\Scripts\\python.exe epack_grid.py <anatomy_dense dir> <model dir> <out dir> [cell m]"""
import sys, os, json, time, numpy as np, warp as wp
from epack_members import build, say
FD, MD, OUT = sys.argv[1:4]; os.makedirs(OUT, exist_ok=True); t00 = time.time()
B = build(FD, MD); names, system, mem, S, thick = B['names'], B['system'], B['mem'], B['S'], B['thick']
h = float(sys.argv[4]) if len(sys.argv) > 4 else min(thick.values())
lo, hi = B['lo'], B['hi']; n = np.ceil((hi - lo) / h).astype(int); say('grid %s cells of %.2f mm = %.1f M' % (n.tolist(), 1e3 * h, n.prod() / 1e6))
wp.init()
@wp.kernel
def probe(m: wp.uint64, o: wp.vec3, h: float, i0: wp.vec3i, depth: wp.array3d(dtype=float), lab: wp.array3d(dtype=wp.int16),
          me: wp.int16, mode: int):
    # mode 0: skin (cells outside -> -2); 1: bone (inside -> me); 2: muscle (inside and deeper than the best so far -> me)
    a, b, c = wp.tid(); i = a + i0[0]; j = b + i0[1]; k = c + i0[2]
    if lab[i, j, k] == wp.int16(-2): return
    p = o + h * wp.vec3(float(i) + 0.5, float(j) + 0.5, float(k) + 0.5)
    q = wp.mesh_query_point_sign_winding_number(m, p, 1.0e3, 2.0, 0.5)
    if not q.result: return
    if mode == 0:
        if q.sign > 0.0: lab[i, j, k] = wp.int16(-2)
        return
    if q.sign > 0.0: return
    if mode == 1:
        lab[i, j, k] = me; depth[i, j, k] = 1.0e9; return
    d = wp.length(wp.mesh_eval_position(m, q.face, q.u, q.v) - p)
    if d > depth[i, j, k]:
        depth[i, j, k] = d; lab[i, j, k] = me
bones = [i for i in mem if system[i] == 'bone']; muscles = [i for i in mem if system[i] != 'bone']; order = bones + muscles
rnames = [names[i] for i in order] + ['fat']; kind = ['bone'] * len(bones) + ['muscle'] * len(muscles) + ['fat']; FAT = len(order)
lab = wp.full(tuple(n.tolist()), wp.int16(FAT), dtype=wp.int16); depth = wp.zeros(tuple(n.tolist()), dtype=float)
o = wp.vec3(*map(float, lo)); timing = {}
def run(P, G, me, mode):
    P = np.asarray(P, np.float32); a = np.clip(np.floor((P.min(0) - lo) / h).astype(int) - 1, 0, n); b = np.clip(np.ceil((P.max(0) - lo) / h).astype(int) + 1, 0, n)
    if mode == 0: a, b = np.zeros(3, int), n
    if (b <= a).any(): return
    m = wp.Mesh(points=wp.array(P, dtype=wp.vec3), indices=wp.array(np.asarray(G, np.int32).ravel(), dtype=int), support_winding_number=True)
    wp.launch(probe, tuple((b - a).tolist()), [m.id, o, h, wp.vec3i(*map(int, a)), depth, lab, wp.int16(me), mode])
t0 = time.time(); run(B['SP'], B['SG'], -2, 0); wp.synchronize(); timing['skin_s'] = time.time() - t0
# skin is tested with its own winding (outward): sign > 0 = outside.  Bones then muscles, muscles skip bone cells (depth 1e9).
t0 = time.time()
for r, i in enumerate(order): run(*S[i], r, 1 if system[i] == 'bone' else 2)
wp.synchronize(); timing['members_s'] = time.time() - t0
L = lab.numpy(); L[L == -2] = -1
cv = h ** 3 * 1e6; vols = np.bincount(L[L >= 0].ravel(), minlength=len(rnames)) * cv
src = {names[i]: round(abs(B['vol_area'](*S[i])[0]) * 1e6, 3) for i in order}
np.savez_compressed(os.path.join(OUT, 'grid_thigh_l.npz'), lab=L, origin=lo, h=h, region_names=np.array(rnames), region_kind=np.array(kind),
                    source_volume_cm3=json.dumps(src))
timing['total_s'] = time.time() - t00
json.dump(dict(cells=n.tolist(), h=h, timing=timing, volumes_cm3={r: dict(grid=round(float(v), 3), source=src.get(r)) for r, v in zip(rnames, vols)}),
          open(os.path.join(OUT, 'grid_thigh_l.json'), 'w'), indent=1)
say('GPU labels: skin %.1f s, %d members %.1f s, total %.0f s' % (timing['skin_s'], len(order), timing['members_s'], timing['total_s']))
for r, v in zip(rnames, vols): say('  %-42s %8.2f cm3 (atlas %s)' % (r, v, src.get(r, '-')))
