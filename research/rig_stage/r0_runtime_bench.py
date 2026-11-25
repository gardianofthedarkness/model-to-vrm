"""R0: what B costs per frame where a game would run it.  The per-vertex part (jiggle = weights x compartment offsets, bulge =
radius x ratio along the skinned normal) as one Warp kernel on the GPU, the compartment oscillators on the CPU, both timed.
The kernel's cost depends only on the sizes, so it is timed at the leg's size and at a whole-body size.
  E:\\splat\\.venv-sim\\Scripts\\python.exe r0_runtime_bench.py [vertices ...]"""
import sys, time, math, json, numpy as np, warp as wp
wp.init(); NC = 7; NM = 34; SUB = 4
@wp.kernel
def flesh(XA: wp.array(dtype=wp.vec3), NA: wp.array(dtype=wp.vec3), WJ: wp.array2d(dtype=float), Y: wp.array(dtype=wp.vec3),
          RU: wp.array(dtype=float), UM: wp.array(dtype=int), ratio: wp.array(dtype=float), XB: wp.array(dtype=wp.vec3)):
    i = wp.tid(); o = wp.vec3()
    for c in range(NC): o += WJ[i, c] * Y[c]
    XB[i] = XA[i] + o + RU[i] * ratio[UM[i]] * NA[i]
res = {}
for nv in [int(a) for a in sys.argv[1:]] or [49309, 300000]:
    r = np.random.default_rng(0); arr = lambda a, dt=wp.vec3: wp.array(a, dtype=dt)
    XA, NA, XB = arr(r.random((nv, 3), np.float32)), arr(r.random((nv, 3), np.float32)), wp.zeros(nv, dtype=wp.vec3)
    WJ = wp.array(r.random((nv, NC), np.float32), dtype=float); RU = arr(r.random(nv, np.float32), float); UM = arr(r.integers(0, NM, nv).astype(np.int32), int)
    Y = arr(r.random((NC, 3), np.float32)); ratio = arr(r.random(NM, np.float32), float)
    for _ in range(20): wp.launch(flesh, nv, [XA, NA, WJ, Y, RU, UM, ratio, XB])
    wp.synchronize(); n = 500; t0 = time.perf_counter()
    for _ in range(n): Y.assign(r.random((NC, 3), np.float32)); ratio.assign(r.random(NM, np.float32)); wp.launch(flesh, nv, [XA, NA, WJ, Y, RU, UM, ratio, XB])
    wp.synchronize(); gpu = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n): wp.launch(flesh, nv, [XA, NA, WJ, Y, RU, UM, ratio, XB])
    wp.synchronize(); kern = (time.perf_counter() - t0) / n
    x = np.zeros((NC, 3)); v = np.zeros((NC, 3)); Rc = np.repeat(np.eye(3)[None], NC, 0); w = np.full((NC, 1), 2 * math.pi * 15); z = np.full((NC, 1), 0.3); a = np.zeros((NC, 3)); t0 = time.perf_counter()
    for _ in range(n):
        f = np.einsum('cji,j->ci', Rc, np.array([0, 0, -9.81])) - a
        for s in range(SUB): v += (f - 2 * z * w * v - w * w * x) / 240; x += v / 240
    cpu = (time.perf_counter() - t0) / n
    res[nv] = dict(gpu_kernel_ms=round(1e3 * kern, 4), gpu_with_upload_ms=round(1e3 * gpu, 4), cpu_oscillators_numpy_ms=round(1e3 * cpu, 4))
    print('BENCH %7d vertices: GPU kernel %.3f ms, with per-frame upload from python %.3f ms; oscillators (numpy, all compartments) %.3f ms' % (nv, 1e3 * kern, 1e3 * gpu, 1e3 * cpu), flush=True)
print(json.dumps(dict(device=str(wp.get_device()), results=res)))
