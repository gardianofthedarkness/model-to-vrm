"""Eye balls from the modelled eye: SAM 3 "eye" on the bare-surface front view of the head gives each eye region on the mesh
(the landmark outline follows the painted eye, which does not sit on the modelled one); a ball is fitted to each region
(least squares, re-weighted), and the painted eye (SAM 3 "eyeball" on the textured view) is compared with it.
  python eye_geo.py <model dir>     needs parts/part_concept_masks.py run on parts_zoom/head for clay_e+00_a000 (eye)
                                    and tex_e+00_a000 (eyeball)
Writes <model dir>/parts_zoom/eye_geo.json and eye_geo_check.jpg."""
import numpy as np, json, cv2, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import sys; R = sys.argv[1]; PZ = R + '/parts_zoom'; MV = np.load(R + '/parts/orig_colours.npz')['V'].astype(float); MF = np.load(R + '/parts/orig_normals.npz')['F']; vid = np.load(PZ + '/head_vids.npy'); V = MV[vid]
edge = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1))); cams = json.load(open(PZ + '/head/cameras.json'))
def lift(view, concept):
    m = np.load(PZ + '/head/masks/%s.npz' % view); res = int(m['res']); S = m['s_' + concept]; M = np.unpackbits(m['m_' + concept], axis=1)[:, :res * res].reshape(-1, res, res).astype(bool); cam = cams[view]
    Mi = np.array(cam['matrix_inv']); c = V @ Mi[:3, :3].T + Mi[:3, 3]; p = np.c_[(c[:, 0] / cam['scale'] + 0.5) * res, (0.5 - c[:, 1] / cam['scale']) * res]; blk = max(1, int(np.ceil(edge / (cam['scale'] / res))))
    key = (p[:, 0] // blk).astype(int) * 100000 + (p[:, 1] // blk).astype(int); o = np.argsort(key); ku, st = np.unique(key[o], return_index=True); zmax = np.maximum.reduceat(c[o, 2], st); near = np.empty(len(V)); near[o] = np.repeat(zmax, np.diff(np.r_[st, len(V)]))
    vis = c[:, 2] >= near - 2 * edge; pi = np.clip(np.rint(p).astype(int), 0, res - 1); out = [np.flatnonzero(vis & M[k][pi[:, 1], pi[:, 0]]) for k in np.argsort(-S)[:2]]; out.sort(key=lambda ii: V[ii, 0].mean()); return out, [M[k] for k in np.argsort(-S)[:2]], p
def fit(P, w=None):
    w = np.ones(len(P)) if w is None else w; A = np.c_[2 * P, np.ones(len(P))] * w[:, None]; sol = np.linalg.lstsq(A, (P ** 2).sum(1) * w, rcond=None)[0]; c = sol[:3]; return c, float(np.sqrt(max(sol[3] + c @ c, 0)))
def robust(P):
    c, r = fit(P)
    for _ in range(50):
        d = np.linalg.norm(P - c, axis=1) - r; s = 1.4826 * np.median(np.abs(d)) + 1e-12; c2, r2 = fit(P, np.sqrt(np.where(np.abs(d) <= s, 1.0, s / np.abs(d)))); done = abs(r2 - r) < 1e-7; c, r = c2, r2
        if done: break
    return c, r, np.linalg.norm(P - c, axis=1) - r
geo, gm, p = lift('clay_e+00_a000', 'eye'); tex, tm, _ = lift('tex_e+00_a000', 'eyeball'); rep = {}
fig, ax = plt.subplots(2, 4, figsize=(20, 9)); clay = cv2.cvtColor(cv2.imread(PZ + '/head/clay_e+00_a000.png'), cv2.COLOR_BGR2RGB); t = np.linspace(0, 2 * np.pi, 200)
for row, nm in enumerate(('right', 'left')):
    G, T = V[geo[row]], V[tex[row]]; c, r, d = robust(G); pc = G - G.mean(0); n = np.linalg.svd(pc)[2][-1]; wid = float(G[:, 0].max() - G[:, 0].min()); hgt = float(G[:, 2].max() - G[:, 2].min()); off = (T.mean(0) - G.mean(0)) * 1000
    rep[nm] = dict(points=len(G), width_mm=1000 * wid, height_mm=1000 * hgt, radius_mm=1000 * r, centre=c.tolist(), centre_behind_mm=1000 * float(c[1] - G[:, 1].mean()), off_ball_mm=1000 * float(np.median(np.abs(d))), off_plane_mm=1000 * float(np.median(np.abs(pc @ n))), painted_minus_modelled_mm=off.tolist())
    print('%-5s modelled eye: %d points, %.1f x %.1f mm; ball radius %.1f mm, centre %.1f mm behind; off the ball %.2f mm, off a plane %.2f mm; painted eye sits %.1f mm across, %.1f mm up from the modelled eye' % (nm, len(G), 1000 * wid, 1000 * hgt, 1000 * r, rep[nm]['centre_behind_mm'], rep[nm]['off_ball_mm'], rep[nm]['off_plane_mm'], off[0], off[2]))
    q = p[np.r_[geo[row], tex[row]]]; lo = (q.min(0) - 45).astype(int); hi = (q.max(0) + 45).astype(int); o = clay.copy()
    for M_, col in ((gm, (255, 200, 0)), (tm, (0, 120, 255))):
        for mm in M_:
            cs, _ = cv2.findContours(mm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); cv2.drawContours(o, cs, -1, col, 1)
    ax[row, 0].imshow(o[lo[1]:hi[1], lo[0]:hi[0]]); ax[row, 0].set_title('%s eye, bare surface: yellow = modelled eye (SAM 3 on the surface), blue = painted eye (SAM 3 on the texture)' % nm, fontsize=8); ax[row, 0].axis('off')
    big = (np.abs(V[:, 0] - G[:, 0].mean()) < wid) & (np.abs(V[:, 2] - G[:, 2].mean()) < wid) & (np.abs(V[:, 1] - G[:, 1].mean()) < wid); Pb = V[big]
    s = ax[row, 1].scatter(Pb[:, 0] * 1000, Pb[:, 2] * 1000, c=-(Pb[:, 1] - np.median(G[:, 1])) * 1000, cmap='coolwarm', s=10, vmin=-3, vmax=3); ax[row, 1].plot(G[:, 0] * 1000, G[:, 2] * 1000, 'k.', ms=2); ax[row, 1].set_aspect('equal'); ax[row, 1].set_title('surface height mm (red = out); black = modelled eye points'); plt.colorbar(s, ax=ax[row, 1], fraction=0.04)
    zc = G[:, 2].mean(); b = np.abs(Pb[:, 2] - zc) < hgt / 6; ax[row, 2].plot(Pb[b, 0] * 1000, -Pb[b, 1] * 1000, '.', c='0.6', ms=4); b2 = np.abs(G[:, 2] - zc) < hgt / 6; ax[row, 2].plot(G[b2, 0] * 1000, -G[b2, 1] * 1000, 'k.', ms=6); ax[row, 2].plot((c[0] + r * np.cos(t)) * 1000, (-c[1] + r * np.sin(t)) * 1000, 'r-', lw=1)
    ax[row, 2].set_xlim((G[:, 0].mean() - wid) * 1000, (G[:, 0].mean() + wid) * 1000); ax[row, 2].set_ylim((-G[:, 1].mean() - 1.4 * wid) * 1000, (-G[:, 1].mean() + 0.6 * wid) * 1000); ax[row, 2].set_aspect('equal'); ax[row, 2].set_title('cut across, from above (front is up); ball r %.1f mm' % (1000 * r))
    xc = G[:, 0].mean(); b = np.abs(Pb[:, 0] - xc) < wid / 8; ax[row, 3].plot(-Pb[b, 1] * 1000, Pb[b, 2] * 1000, '.', c='0.6', ms=4); b2 = np.abs(G[:, 0] - xc) < wid / 8; ax[row, 3].plot(-G[b2, 1] * 1000, G[b2, 2] * 1000, 'k.', ms=6); ax[row, 3].plot((-c[1] + r * np.cos(t)) * 1000, (c[2] + r * np.sin(t)) * 1000, 'r-', lw=1)
    ax[row, 3].set_ylim((zc - wid) * 1000, (zc + wid) * 1000); ax[row, 3].set_xlim((-G[:, 1].mean() - 1.4 * wid) * 1000, (-G[:, 1].mean() + 0.6 * wid) * 1000); ax[row, 3].set_aspect('equal'); ax[row, 3].set_title('cut down, from the side (front is right)')
xm = 0.5 * (np.array(rep['right']['centre'])[0] + np.array(rep['left']['centre'])[0]); cr, cl = np.array(rep['right']['centre']), np.array(rep['left']['centre']); cl_m = np.r_[2 * xm - cl[0], cl[1:]]
print('left against right: radius %.1f vs %.1f mm; centres %.1f mm apart; depth differs %.1f mm, height differs %.1f mm' % (rep['left']['radius_mm'], rep['right']['radius_mm'], 1000 * abs(cl[0] - cr[0]), 1000 * (cl[1] - cr[1]), 1000 * (cl[2] - cr[2])))
plt.tight_layout(); plt.savefig(PZ + '/eye_geo_check.jpg', dpi=70); json.dump(rep, open(PZ + '/eye_geo.json', 'w'), indent=1)
