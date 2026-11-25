"""Smooth tissue surfaces from a generated interior, without retraining: decode the codes to class probabilities (fixed,
crop-sized tiles), add them up per tissue group, and take each group's 50% surface with marching cubes. Probabilities
vary smoothly between voxels, so the surfaces fall between voxel centres instead of on the voxel staircase.
Also writes the same groups from the hard labels (argmax) for comparison, and a picture.
args: <interior_data root> <generated npz (codes, idx)> <shell labels npz> <autoencoder .pt> <out dir>"""
import os, sys, json, re, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import transformer_smoke as ts
from face_compress import make_ae
from transformer_smoke import DEV, UNK, encode
from skimage import measure
root, genf, shellf, aef, out = sys.argv[1:6]; os.makedirs(out, exist_ok=True)
ids = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json')))['ids']; NC = max(ids.values()) + 1
GROUPS = [('bone', r'^(skull|mandible|vertebrae_|teeth_|hard_palate)$|^5$', (0.93, 0.89, 0.78)),
          ('air', r'^(sinus_|nasal_cavity|nasopharynx|oropharynx|hypopharynx|trachea|auditory_canal)|^4$', (0.30, 0.60, 0.90)),
          ('eye', r'^eye_', (0.97, 0.97, 1.0)), ('brain', r'^(brain|Brainstem)$', (0.85, 0.60, 0.70)),
          ('muscle', r'^(masseter|temporalis|lateral_pterygoid|medial_pterygoid|digastric|tongue)', (0.78, 0.25, 0.22)),
          ('fat', r'^[12]$', (0.98, 0.84, 0.45))]
names = {v: k for k, v in ids.items()}; names.update({k: str(k) for k in range(10) if k not in names})
gidx = {g: [k for k in range(NC) if re.search(pat, names.get(k, ''))] for g, pat, _ in GROUPS}
c = torch.load(aef, map_location=DEV); ae = make_ae(NC, c['F'], c['C'], norm=c.get('norm', 'group')).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval(); F = c['F']
g = np.load(genf); codes = g['codes'][0]; idx = g['idx'].astype(np.int64)
L = np.load(shellf)['labels']; G = [n // F for n in L.shape]; L = L[:G[0] * F, :G[1] * F, :G[2] * F]; body = (L > 0) & (L < UNK)
base = encode(ae, np.where(body, 3, L).astype(np.uint8), NC, F); grid = base.copy(); grid[(slice(None),) + tuple(idx.T)] = codes.T
Pg = {gn: np.zeros(L.shape, np.float16) for gn, _, _ in GROUPS}; hard = np.zeros(L.shape, np.uint8)
with torch.no_grad():
    for a, b, m in ts.tiles(G, *ts.dec_tile(F)):
        pr = torch.softmax(ts.decode(ae, torch.tensor(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]], device=DEV)).float(), 1)[0]
        sl = tuple(slice((a[i] + m[i]) * F, b[i] * F) for i in range(3)); ss = tuple(slice(m[i] * F, None) for i in range(3))
        hard[sl] = pr.argmax(0).cpu().numpy()[ss]
        for gn, _, _ in GROUPS: Pg[gn][sl] = pr[gidx[gn]].sum(0).cpu().numpy()[ss]
for gn in Pg: Pg[gn][~body] = 0
hard[~body] = 0
res = {}; meshes = {}
for gn, _, col in GROUPS:
    for kind, vol in (('smooth', Pg[gn].astype(np.float32)), ('voxel', np.isin(hard, gidx[gn]).astype(np.float32))):
        if vol.max() < 0.5: continue
        v, f, _, _ = measure.marching_cubes(np.pad(vol, 1), 0.5, step_size=1); v -= 1
        meshes[(gn, kind)] = (v, f); res.setdefault(gn, {})[kind] = dict(vertices=len(v), ml=round(float((vol >= 0.5).sum()) / 1000, 1))
import trimesh
for (gn, kind), (v, f) in meshes.items(): trimesh.Trimesh(v, f, process=False).export(os.path.join(out, '%s_%s.ply' % (gn, kind)))
json.dump(res, open(os.path.join(out, 'meshes.json'), 'w'), indent=1); print(json.dumps(res))
# picture: close-up of the bone around the eye socket and cheek, voxel vs smooth surfaces
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
eye = np.argwhere(np.isin(hard, gidx['eye'])); ctr = eye.mean(0) if len(eye) else np.argwhere(body).mean(0)
box_lo, box_hi = ctr - [45, 40, 35], ctr + [45, 60, 25]
fig = plt.figure(figsize=(14, 7))
for i, kind in enumerate(('voxel', 'smooth')):
    ax = fig.add_subplot(1, 2, i + 1, projection='3d')
    for gn, _, col in GROUPS:
        if gn not in ('bone', 'eye') or (gn, kind) not in meshes: continue
        v, f = meshes[(gn, kind)]; keep = ((v[f] > box_lo) & (v[f] < box_hi)).all((1, 2)); tri = v[f[keep]]
        pc = Poly3DCollection(tri, facecolor=col, edgecolor='none', alpha=1.0); ax.add_collection3d(pc)
        nrm = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-9
        shade = 0.35 + 0.65 * np.clip(nrm @ np.array([0.3, 0.8, 0.5]) / np.linalg.norm([0.3, 0.8, 0.5]), 0, 1)
        pc.set_facecolor(np.clip(np.array(col)[None] * shade[:, None], 0, 1))
    ax.set_xlim(box_lo[0], box_hi[0]); ax.set_ylim(box_lo[1], box_hi[1]); ax.set_zlim(box_lo[2], box_hi[2]); ax.view_init(elev=10, azim=75); ax.set_box_aspect(box_hi - box_lo); ax.axis('off')
    ax.set_title({'voxel': 'surface from hard labels (voxel steps)', 'smooth': 'surface from decoded probabilities (smooth)'}[kind], fontsize=11)
fig.suptitle('Elf, generated skull and eyeball around the eye socket: same model output, two ways of making the surface', fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(out, 'voxel_vs_smooth.png'), dpi=85); print('picture saved')

# web export: lighter meshes (marching cubes on a coarser step for the big groups), binary float32 positions + uint32 faces
if os.environ.get('WEB'):
    web = os.environ['WEB']; os.makedirs(web, exist_ok=True); man = []
    STEP = dict(bone=2, air=1, eye=1, brain=3, muscle=2, fat=3)
    ctr = np.argwhere(body).mean(0)
    for gn, _, col in GROUPS:
        for kind, vol in (('smooth', Pg[gn].astype(np.float32)), ('voxel', np.isin(hard, gidx[gn]).astype(np.float32))):
            if vol.max() < 0.5: continue
            v, f, _, _ = measure.marching_cubes(np.pad(vol, 1), 0.5, step_size=STEP[gn]); v = (v - 1 - ctr).astype(np.float32)
            fn = '%s_%s.bin' % (gn, kind); open(os.path.join(web, fn), 'wb').write(v.tobytes() + f.astype(np.uint32).tobytes())
            man.append(dict(group=gn, kind=kind, file=fn, nv=int(len(v)), nf=int(len(f)), colour=col, ml=res[gn][kind]['ml']))
    json.dump(dict(voxel_mm=1.0, meshes=man), open(os.path.join(web, 'meshes.json'), 'w'))
    print('web', sum(os.path.getsize(os.path.join(web, m['file'])) for m in man) / 1e6, 'MB')
