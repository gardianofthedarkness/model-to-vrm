"""Renders of the carved subparts for review: python carve_show.py <carve.npz> <orig_colours.npz> <out_dir>
Rows (front, back, head): the original (AI texture), colour subparts in distinct colours, colour subparts in their own
median colour, dents (blue) and bumps (orange) on grey."""
import sys, os, subprocess, numpy as np, cv2, colorsys
K, OC, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]; os.makedirs(out, exist_ok=True)
cr, dr, br, med = K['colour_region'], K['dent_region'], K['bump_region'], K['colour_median']; part = OC['part']
rng = np.random.default_rng(1); pal = np.array([colorsys.hsv_to_rgb(rng.random(), 0.55 + 0.4 * rng.random(), 0.6 + 0.4 * rng.random()) for _ in range(cr.max() + 1)])
sets = {'regions': pal[cr], 'median': med[cr], 'shape': np.where((dr >= 0)[:, None], [0.15, 0.35, 0.95], np.where((br >= 0)[:, None], [1.0, 0.55, 0.1], [0.82, 0.82, 0.82]))}
np.savez(os.path.join(out, 'parts.npz'), full_part=part)
B = r'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe'; script = r'E:\figures\rikka\tripo\transfer\pi_ws\tools\_blender_flat.py'
for name, cols in sets.items():
    np.save(os.path.join(out, name + '.npy'), cols.astype(np.float32))
    r = subprocess.run([B, '-b', '-P', script, '--', r'E:\figures\rikka\tripo\tripo_parts_orig.glb', os.path.join(out, 'parts.npz'), os.path.join(out, name + '.npy'), os.path.join(out, name)], capture_output=True, text=True)
    print(name, 'ok' if 'BF done' in r.stdout else r.stdout[-500:])
T = r'E:\figures\rikka\tripo\transfer\pi_ws\renders'
row = lambda pre: np.hstack([cv2.imread('%s_%s.png' % (pre, v)) for v in ('front', 'back', 'head')])
im = np.vstack([row(os.path.join(T, 'tripo'))] + [row(os.path.join(out, n)) for n in ('regions', 'median', 'shape')])
cv2.imwrite(os.path.join(out, 'carve_review.jpg'), cv2.resize(im, None, fx=0.45, fy=0.45), [cv2.IMWRITE_JPEG_QUALITY, 88])
hc = lambda pre: cv2.imread('%s_head.png' % pre)[150:950]
cv2.imwrite(os.path.join(out, 'carve_head.jpg'), np.hstack([hc(os.path.join(T, 'tripo')), hc(os.path.join(out, 'regions')), hc(os.path.join(out, 'median')), hc(os.path.join(out, 'shape'))]), [cv2.IMWRITE_JPEG_QUALITY, 88])
print('review images written')
