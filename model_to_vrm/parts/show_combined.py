"""Render the combined parts: python show_combined.py <pi_tree dir>
Big parts in their own median colour (muted); each SAM 3 detail concept in a strong distinct colour; legend.
Writes combined_review.jpg (front/back/head) and combined_head.jpg (head close-up)."""
import sys, os, json, subprocess, numpy as np, cv2, colorsys
W = sys.argv[1]; L = json.load(open(os.path.join(W, 'part_list.json'))); reg = np.load(os.path.join(W, 'parts.npz'))['region']
OC = np.load(r'E:\figures\rikka\tripo\transfer\orig_colours.npz')
lut = np.zeros((reg.max() + 1, 3))
for b in L['colour_regions']: lut[b['id']] = np.array(b['colour']) * 0.6 + 0.4 * np.mean(b['colour'])     # muted
cons = sorted({d['concept'] for d in L['sam3_details']}); pal = {c: colorsys.hsv_to_rgb(k / len(cons), 1, 1) for k, c in enumerate(cons)}
for d in L['sam3_details']: lut[d['id']] = pal[d['concept']]
od = os.path.join(W, 'renders'); np.save(os.path.join(od, 'comb.npy'), lut[reg].astype(np.float32)); np.savez(os.path.join(od, 'parts.npz'), full_part=OC['part'])
B = r'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe'
subprocess.run([B, '-b', '-P', r'E:\figures\rikka\tripo\transfer\pi_ws\tools\_blender_flat.py', '--', r'E:\figures\rikka\tripo\tripo_parts_orig.glb',
                os.path.join(od, 'parts.npz'), os.path.join(od, 'comb.npy'), os.path.join(od, 'comb')], capture_output=True)
T = r'E:\figures\rikka\tripo\transfer\pi_ws\renders'; row = lambda pre: np.hstack([cv2.imread('%s_%s.png' % (pre, v)) for v in ('front', 'back', 'head')])
im = np.vstack([row(os.path.join(T, 'tripo')), row(os.path.join(od, 'comb'))]); leg = np.full((im.shape[0], 330, 3), 255, np.uint8)
for k, c in enumerate(cons):
    y = 40 + 44 * k; cv2.rectangle(leg, (12, y - 22), (48, y + 8), tuple(int(255 * x) for x in pal[c][::-1]), -1); cv2.putText(leg, c, (58, y + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
cv2.imwrite(os.path.join(W, 'combined_review.jpg'), cv2.resize(np.hstack([im, leg]), None, fx=0.5, fy=0.5), [cv2.IMWRITE_JPEG_QUALITY, 88])
h = lambda pre: cv2.imread('%s_head.png' % pre)[150:950]
cv2.imwrite(os.path.join(W, 'combined_head.jpg'), np.hstack([h(os.path.join(T, 'tripo')), h(os.path.join(od, 'comb')), leg[:800]]), [cv2.IMWRITE_JPEG_QUALITY, 88]); print('SC done')
