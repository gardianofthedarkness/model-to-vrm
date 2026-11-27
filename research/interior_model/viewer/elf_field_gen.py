"""The elf's head interior from a field-experiment generator (field_flow.py) and its autoencoder (conv or transformer VAE):
shell tokens as in field_prep.py (8 mm cells holding body; signed distance to the skin pooled 4^3, share of the cell
known), flow sampling (16 Euler steps), decoding to tissue probabilities in crop-sized overlapping tiles, then each
tissue's smooth surface where it overtakes the strongest other tissue (the label boundary at any confidence, as scored).
Writes a picture and, with WEB, binary meshes + meshes.json for the 3D page (our own character: may be published).
args: <elf shell npz (labels 0 outside / tissue / 255 unknown)> <flow.pt> <autoencoder .pt> <out dir>  env: WEB, SEED"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from field_exp import make_ae, NC, NAMES, AMP
from flow_f1 import Flow
from transformer_smoke import sdf, tiles
from skimage import measure
import time
T0 = time.time(); TS = [T0]
def stage(name):
    """progress line: stage time and total elapsed"""
    now = time.time(); print('STAGE %-28s %7.1fs  (total %7.1fs)' % (name, now - TS[-1], now - T0), flush=True); TS.append(now)
def prog(tag, i, n, t_start, every):
    if i % every == 0 or i == n - 1:
        el = time.time() - t_start; eta = el / (i + 1) * (n - i - 1); print('  %s %d/%d  elapsed %.1fs  left ~%.1fs' % (tag, i + 1, n, el, eta), flush=True)
shellf, flowf, aef, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True); dev = 'cuda'; F = 8; C = 8
L = np.load(shellf)['labels']; G = [n // F for n in L.shape]; L = L[:G[0] * F, :G[1] * F, :G[2] * F]
outside = L == 0; unk = L == 255; body = ~outside & ~unk
ck = torch.load(aef, map_location=dev); ae = make_ae(soft=(ck['kind'] == 'soft'), arch=ck.get('arch', 'conv')).to(dev); ae.load_state_dict(ck['state_dict']); ae.eval()
fl = torch.load(flowf, map_location=dev, weights_only=False);   # our own checkpoint (mu, sd are numpy)
net = Flow(65, C, NC, 384, 8).to(dev); net.load_state_dict(fl['state_dict']); net.eval(); mu, sd = fl['mu'], fl['sd']
stage('load models')
# shell tokens (field_prep.py): unknown below the head counts as body (the body continues there)
S = np.where(outside, 0, 3).astype(np.uint8); D = sdf(S, 1.0); stage('distance to skin (CPU)'); known = (~unk).astype(np.float32)
blk = lambda A: A.reshape(G[0], F, G[1], F, G[2], F).transpose(0, 2, 4, 1, 3, 5)
tok = np.argwhere(blk(body).reshape(*G, -1).any(-1)); p = F // 4
Dp = blk(D).reshape(*G, 4, p, 4, p, 4, p).mean((4, 6, 8))
feat = np.concatenate([Dp[tuple(tok.T)].reshape(len(tok), -1), blk(known).reshape(*G, -1).mean(-1)[tuple(tok.T)][:, None]], 1).astype(np.float32)
pos = ((tok + 0.5) * F / 100.0).astype(np.float32); print('tokens', len(tok), flush=True); stage('shell tokens (CPU)')
# context codes for cells without tokens: the autoencoder's code of a plain shell (outside / lean)
x = torch.zeros((1, NC, *L.shape), device=dev); x[0, 0] = torch.tensor(outside, device=dev).float(); x[0, 3] = 1 - x[0, 0]
grid = torch.zeros((C, *G), device=dev)
with torch.no_grad():
    TL = list(tiles(G, 8, 2)); t1 = time.time()
    for j, (a, b, m) in enumerate(TL):
        prog('encode tile', j, len(TL), t1, 25)
        with torch.autocast('cuda', dtype=AMP): z = ae.encode(x[(..., slice(a[0] * F, b[0] * F), slice(a[1] * F, b[1] * F), slice(a[2] * F, b[2] * F))]).float()[0]
        grid[:, a[0] + m[0]:b[0], a[1] + m[1]:b[1], a[2] + m[2]:b[2]] = z[:, m[0]:, m[1]:, m[2]:]
del x
stage('context codes (GPU)')
T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), device=dev, dtype=dt)
occ = np.zeros(G, bool); occ[tuple(tok.T)] = True
from scipy import ndimage as ndi
shell = ~ndi.minimum_filter(occ, size=3, mode='constant', cval=0)[tuple(tok.T)]
cd = dict(pos=T(pos)[None], part=T(np.ones(len(tok)), torch.long)[None], sfeat=T(feat[shell])[None], spos=T(pos[shell])[None], spart=T(np.ones(int(shell.sum())), torch.long)[None],
          age=T([[0.0]]), agek=T([[0.0]]), sex=T([0], torch.long), cls=T(np.ones((1, NC), np.float32)), cpart=T([1], torch.long))
g = torch.Generator(device=dev); g.manual_seed(int(os.environ.get('SEED', '0')))
with torch.no_grad():
    z = torch.randn((1, len(tok), C), device=dev, generator=g); rev = torch.zeros((1, len(tok)), dtype=torch.bool, device=dev)
    t1 = time.time()
    for i in range(16):
        prog('generation step', i, 16, t1, 1)
        with torch.autocast('cuda', dtype=AMP): z = z + net(z, torch.tensor(i / 16, device=dev), rev, **cd).float() / 16
    codes = z[0].cpu().numpy() * sd + mu
stage('generation, 16 steps (GPU)')
gc = grid.cpu().numpy(); gc[(slice(None),) + tuple(tok.T)] = codes.T
P = np.zeros((NC, *L.shape), np.float16)
with torch.no_grad():
    t1 = time.time()
    for j, (a, b, m) in enumerate(TL):
        prog('decode tile', j, len(TL), t1, 25)
        with torch.autocast('cuda', dtype=AMP): pr = torch.softmax(ae.decode(T(gc[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]])).float(), 1)[0].cpu().numpy()
        P[:, (a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = pr[:, m[0] * F:, m[1] * F:, m[2] * F:]
P[:, ~body] = 0; P[0, ~body] = 1
stage('decode probabilities (GPU)')
np.savez_compressed(os.path.join(out, 'elf_probs.npz'), P=P, body=body); stage('save probabilities')
lab = P.argmax(0); vols = {NAMES[k]: round(float((lab == k).sum()) / 1000, 1) for k in range(1, NC)}; print('volumes ml', vols, flush=True)
COL = {'fat_under_skin': (0.98, 0.84, 0.45), 'fat_other': (1.0, 0.65, 0.2), 'lean': (0.78, 0.25, 0.22), 'air': (0.30, 0.60, 0.90), 'bone': (0.93, 0.89, 0.78), 'eye': (0.97, 0.97, 1.0)}
STEP = {'fat_under_skin': 2, 'fat_other': 1, 'lean': 2, 'air': 1, 'bone': 1, 'eye': 1}
ctr = np.argwhere(body).mean(0); man = []; web = os.environ.get('WEB')
if web: os.makedirs(web, exist_ok=True)
for k in range(1, NC):
    nm = NAMES[k]; vol = P[k].astype(np.float32) - np.delete(P, k, axis=0).max(0).astype(np.float32)
    if vol.max() <= 0: continue
    v, f, _, _ = measure.marching_cubes(np.pad(vol, 1, constant_values=-1), 0.0, step_size=STEP[nm]); v = (v - 1 - ctr).astype(np.float32)
    print(nm, 'vertices', len(v), flush=True); stage('surface ' + nm + ' (CPU)')
    if web:
        fn = '%s.bin' % nm; open(os.path.join(web, fn), 'wb').write(v.tobytes() + f.astype(np.uint32).tobytes())
        man.append(dict(group=nm, kind='smooth', file=fn, nv=int(len(v)), nf=int(len(f)), colour=COL[nm], ml=vols[nm]))
if web: json.dump(dict(voxel_mm=1.0, model='N (new input, transformer VAE), 10,000 steps', meshes=man), open(os.path.join(web, 'meshes.json'), 'w'))
# picture: three orthogonal slices through the eyes' centre
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
pal = np.array([[0, 0, 0], COL['fat_under_skin'], COL['fat_other'], COL['lean'], COL['air'], COL['bone'], COL['eye']])
ec = np.argwhere(lab == 6).mean(0).astype(int) if (lab == 6).any() else (np.array(L.shape) // 2)
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
for ax, sl, t in zip(axs, [(ec[0], slice(None), slice(None)), (slice(None), ec[1], slice(None)), (slice(None), slice(None), ec[2])], ['sagittal', 'coronal', 'axial']):
    ax.imshow(pal[lab[sl]].transpose(1, 0, 2), origin='lower'); ax.set_title('elf, ' + t); ax.axis('off')
plt.tight_layout(); plt.savefig(os.path.join(out, 'elf_slices.png'), dpi=70); stage('picture'); print('done', flush=True)
