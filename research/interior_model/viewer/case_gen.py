"""One real test head, three interiors for a side-by-side look (user, 2026-11-27): the truth from the new labels (tissue
shares from the continuous CT field), the autoencoder's rebuild of the true codes (what the decoder alone can do), and a
generator's interior from the shell (field_flow.py sampling). Each as smooth surfaces where a tissue overtakes the
others, lean split into brain (inside the skull) and muscle. PRIVATE: a real person's scan; output stays in private storage
and the private gallery, never published.
args: <fields dir> <prep dir> <flow.pt> <autoencoder .pt> <case> <out dir>   env: SEED"""
import os, sys, json, time, base64, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from field_exp import make_ae, load_head, shares, NC, NAMES, AMP
from flow_f1 import Flow
from transformer_smoke import tiles
from scipy import ndimage as ndi
from skimage import measure
T0 = time.time()
def log(*a): print('TS %6.1fs' % (time.time() - T0), *a, flush=True)
root, prep, flowf, aef, case, out = sys.argv[1:7]; os.makedirs(out, exist_ok=True); dev = 'cuda'; F = 8; C = 8
ck = torch.load(aef, map_location=dev, weights_only=False); ae = make_ae(soft=True, arch=ck.get('arch', 'conv')).to(dev); ae.load_state_dict(ck['state_dict']); ae.eval()
fl = torch.load(flowf, map_location=dev, weights_only=False); net = Flow(65, C, NC, 384, 8).to(dev); net.load_state_dict(fl['state_dict']); net.eval(); mu, sd = fl['mu'], fl['sd']
z = np.load(os.path.join(prep, case + '.npz')); idx = z['idx'].astype(np.int64); feat = z['feat']; pos = z['pos']; code = z['code'].astype(np.float32); G = code.shape[1:]
h = load_head(os.path.join(root, case + '.npz')); sl = tuple(slice(0, g * F) for g in G); known = ~h['unk'][sl]; outside = (h['hard'] == 0)[sl]
T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), device=dev, dtype=dt)
occ = np.zeros(G, bool); occ[tuple(idx.T)] = True; shell = ~ndi.minimum_filter(occ, size=3, mode='constant', cval=0)[tuple(idx.T)]
cd = dict(pos=T(pos)[None], part=T(np.ones(len(idx)), torch.long)[None], sfeat=T(feat[shell])[None], spos=T(pos[shell])[None], spart=T(np.ones(int(shell.sum())), torch.long)[None],
          age=T([[0.0]]), agek=T([[0.0]]), sex=T([0], torch.long), cls=T(np.ones((1, NC), np.float32)), cpart=T([1], torch.long))
g = torch.Generator(device=dev); g.manual_seed(int(os.environ.get('SEED', '0')))
with torch.no_grad():
    zz = torch.randn((1, len(idx), C), device=dev, generator=g); rev = torch.zeros((1, len(idx)), dtype=torch.bool, device=dev)
    for i in range(16):
        with torch.autocast('cuda', dtype=AMP): zz = zz + net(zz, torch.tensor(i / 16, device=dev), rev, **cd).float() / 16
    gen = zz[0].cpu().numpy() * sd + mu
log('generated', len(idx), 'tokens')
def decode(grid):
    P = np.zeros((NC, *[x * F for x in G]), np.float16)
    with torch.no_grad():
        for a, b, m in tiles(G, 8, 2):
            with torch.autocast('cuda', dtype=AMP): pr = torch.softmax(ae.decode(T(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]])).float(), 1)[0].cpu().numpy()
            P[:, (a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = pr[:, m[0] * F:, m[1] * F:, m[2] * F:]
    return P
gg = code.copy(); gg[(slice(None),) + tuple(idx.T)] = gen.T
P = {'generated': decode(gg), 'rebuild': decode(code)}
with torch.no_grad():
    P['truth'] = torch.cat([shares(T(h['field'][sl][:, :, k:k + 64])[None], T(h['eye'][sl][:, :, k:k + 64])[None], T(h['subcut'][sl][:, :, k:k + 64], torch.bool)[None],
                                   T(outside[:, :, k:k + 64], torch.bool)[None]).cpu() for k in range(0, G[2] * F, 64)], 4)[0].numpy().astype(np.float16)
log('decoded')
COL = {'bone': (0.93, 0.89, 0.78), 'eye': (0.97, 0.97, 1.0), 'air': (0.30, 0.60, 0.90), 'fat_other': (1.0, 0.65, 0.2), 'fat_under_skin': (0.98, 0.84, 0.45), 'brain': (0.85, 0.60, 0.70), 'muscle': (0.78, 0.25, 0.22)}
STEP = {'bone': 2, 'eye': 1, 'air': 1, 'fat_other': 1, 'fat_under_skin': 2, 'brain': 2, 'muscle': 2}
body = ~outside; ctr = np.argwhere(body).mean(0); man = []; stats = {}
for kind, Q in P.items():
    Q = Q.astype(np.float32); Q[:, ~known] = 0; Q[0, ~known] = 1; lab = Q.argmax(0)
    bc = ndi.binary_closing(lab == 5, np.ones((5, 5, 5), bool)); cav = ndi.binary_fill_holes(bc) & ~bc
    for z_ in range(bc.shape[2]): cav[:, :, z_] |= ndi.binary_fill_holes(bc[:, :, z_]) & ~bc[:, :, z_]
    lc, n = ndi.label(cav); brain = lc == (np.bincount(lc.ravel())[1:].argmax() + 1) if n else np.zeros(lab.shape, bool)
    marg = lambda k: Q[k] - np.delete(Q, k, 0).max(0)
    groups = {'bone': marg(5), 'eye': marg(6), 'air': marg(4), 'fat_other': marg(2), 'fat_under_skin': marg(1), 'brain': np.where(brain, marg(3), -1.0), 'muscle': np.where(brain, -1.0, marg(3))}
    for gname, vol in groups.items():
        vol = np.pad(vol.astype(np.float32), 1, constant_values=-1)
        if vol.max() <= 0: continue
        for step in range(STEP[gname], 6):
            v, f, _, _ = measure.marching_cubes(vol, 0.0, step_size=step)
            if len(v) <= 250000: break
        v = (v - 1 - ctr).astype(np.float32); raw = v.tobytes() + f.astype(np.uint32).tobytes(); fn = '%s_%s.txt' % (kind, gname)
        open(os.path.join(out, fn), 'w').write(base64.b64encode(raw).decode())
        ml = round(float((vol[1:-1, 1:-1, 1:-1] > 0).sum()) / 1000, 1)
        man.append(dict(group=gname, kind=kind, file=fn, nv=int(len(v)), nf=int(len(f)), colour=COL[gname], ml=ml, step_mm=step)); stats.setdefault(kind, {})[gname] = ml
    log(kind, stats[kind])
json.dump(dict(case=case, voxel_mm=1.0, meshes=man), open(os.path.join(out, 'meshes.json'), 'w'))
log('SUMMARY done')
