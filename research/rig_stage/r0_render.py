"""R0 preview: A (plain skinning) and B (bulge + jiggle) side by side, shaded the same, the camera following the leg; frames
as PNG (pyvista off-screen), then r0_video.py joins them.  Optional third panel: B's offset from A, coloured (mm).
  E:\\splat\\.venv-sim\\Scripts\\python.exe r0_render.py <r0_skin.npz> <frames dir> [view: side|front|back|backside]  env FOCUS=upper (thigh and buttock close-up), FPS=<output rate> [heat: 0|1] [first:last]"""
import sys, os, numpy as np, pyvista as pv
SK, OUT = sys.argv[1:3]; VIEW = sys.argv[3] if len(sys.argv) > 3 else 'side'; HEAT = len(sys.argv) > 4 and sys.argv[4] == '1'
D = np.load(SK); A, B, F = D['A'], D['B'], D['F']; n = len(A)
a, b = (int(x) for x in sys.argv[5].split(':')) if len(sys.argv) > 5 else (0, n)
os.makedirs(OUT, exist_ok=True); pv.OFF_SCREEN = True
faces = np.c_[np.full(len(F), 3), F].ravel()
cols = 3 if HEAT else 2; pl = pv.Plotter(off_screen=True, shape=(1, cols), window_size=(560 * cols, 820), border=False)
ext = float(np.ptp(A[0], 0).max()); dirs = {'side': (1, 0, 0), 'front': (0, -1, 0), 'back': (0, 1, 0), 'backside': (0.7, 0.7, 0)}
FPS = float(os.environ.get('FPS', 60)); UP = os.environ.get('FOCUS') == 'upper'     # upper: frame the thigh and buttock (rest height above the median)
sel = D['rest'][:, 2] > np.median(D['rest'][:, 2]) if UP else slice(None); DIST = 1.15 if UP else 1.9
meshes = []
for k, (X, title) in enumerate(((A, 'A  plain skinning'), (B, 'B  bulge + tension-aware jiggle'))):
    pl.subplot(0, k); m = pv.PolyData(X[a].astype(np.float64), faces); meshes.append(m)
    pl.add_mesh(m, color='#d9d4cc', smooth_shading=True, specular=0.25, specular_power=20); pl.add_text(title, font_size=11, position='upper_left')
if HEAT:
    pl.subplot(0, 2); h = pv.PolyData(B[a].astype(np.float64), faces); h['offset mm'] = 1e3 * np.linalg.norm(B[a] - A[a], axis=1); meshes.append(h)
    pl.add_mesh(h, scalars='offset mm', cmap='magma', clim=[0, 6], smooth_shading=True); pl.add_text('B - A (mm)', font_size=11, position='upper_left')
pl.link_views()
for i in range(a, b):
    c = A[i][sel].mean(0); d = np.array(dirs[VIEW], float); d /= np.linalg.norm(d)
    for k, m in enumerate(meshes):
        m.points = (B if k else A)[i].astype(np.float64)
        if HEAT and k == 2: m['offset mm'] = 1e3 * np.linalg.norm(B[i] - A[i], axis=1)
    pl.camera.focal_point = c; pl.camera.position = c + DIST * ext * d; pl.camera.up = (0, 0, 1)
    pl.add_text('frame %d  (%.2f s clip time)' % (i, i / FPS), font_size=9, position='lower_right', name='clock')
    pl.screenshot(os.path.join(OUT, 'f%04d.png' % i))
pl.close(); print('wrote', b - a, 'frames to', OUT)
