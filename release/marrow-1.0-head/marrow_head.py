"""MARROW-1.0-head: game physics compartments inside a head, generated from the head's outer shell alone.

Self-contained inference (torch, numpy, scipy, scikit-image, safetensors; trimesh only for --glb). Same computation as
research/interior_model/viewer/elf_field_gen.py, which produced the published elf example.

Input: a 1 mm label volume of the head in MARROW's head frame (see README): 0 = outside the body, 255 = unknown (no
data, e.g. below the neck window), anything else = inside the body.
Output: 14 compartment probabilities per voxel, the label volume (argmax) and, optionally, one smooth mesh per
compartment as a coloured GLB for game engines.

usage: python marrow_head.py <shell.npz> <out dir> [--weights DIR] [--seed 0] [--steps 16] [--glb [--max-faces N]] [--cpu]"""
import os, sys, json, math, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_

NAMES = ['outside', 'fat_layer', 'skull', 'mandible', 'teeth', 'eyes', 'brain', 'chew_left', 'chew_right', 'tongue', 'airway', 'spine', 'other_bone', 'other_soft']
COLOURS = {'fat_layer': (0.98, 0.84, 0.45), 'skull': (0.93, 0.89, 0.78), 'mandible': (0.85, 0.75, 0.55), 'teeth': (1.0, 1.0, 0.92), 'eyes': (0.97, 0.97, 1.0),
           'brain': (0.92, 0.62, 0.70), 'chew_left': (0.80, 0.22, 0.20), 'chew_right': (0.62, 0.15, 0.30), 'tongue': (0.95, 0.40, 0.45), 'airway': (0.30, 0.60, 0.90),
           'spine': (0.70, 0.70, 0.62), 'other_bone': (0.60, 0.55, 0.45), 'other_soft': (0.72, 0.45, 0.40)}
UNK = 255

# ---------------- autoencoder: patch transformer VAE (research/interior_model/tvae.py) ----------------
def rope_3d(q, k, g):
    D = q.shape[-1]; per = (D // 3) // 2 * 2; dev = q.device
    ii = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32) for n in g], indexing='ij'), -1).reshape(-1, 3)
    out = []
    for t in (q, k):
        parts = []
        for a in range(3):
            x = t[..., a * per:(a + 1) * per]; f = 1.0 / (100.0 ** (torch.arange(0, per, 2, device=dev, dtype=torch.float32) / per))
            ang = ii[:, a:a + 1] * f[None]; c, s = ang.cos()[None, None].to(t.dtype), ang.sin()[None, None].to(t.dtype)
            x1, x2 = x[..., 0::2], x[..., 1::2]; parts.append(torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2))
        parts.append(t[..., 3 * per:]); out.append(torch.cat(parts, -1))
    return out

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.n1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.qn = nn.RMSNorm(d // h); s.kn = nn.RMSNorm(d // h)
        s.proj = nn.Linear(d, d); s.n2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x, g):
        B, N, D = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        q, k = rope_3d(s.qn(q), s.kn(k), g); a = F_.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + s.proj(a); return x + s.mlp(s.n2(x))

class TVAE(nn.Module):
    def __init__(s, NC, C=8, P=16, d=256, depth=8, heads=8):
        super().__init__(); s.NC, s.C, s.P = NC, C, P
        s.patch = nn.Conv3d(NC, d, P, stride=P)
        s.enc = nn.ModuleList([Block(d, heads) for _ in range(depth)]); s.en = nn.LayerNorm(d); s.to_lat = nn.Linear(d, 2 * C)
        s.from_lat = nn.Linear(C, d); s.dec = nn.ModuleList([Block(d, heads) for _ in range(depth)]); s.dn = nn.LayerNorm(d)
        s.shuffle = nn.Linear(d, NC * P ** 3); s.refine = nn.Conv3d(NC, NC, 3, padding=1)
    def encode(s, x):
        h = s.patch(x); g = tuple(h.shape[2:]); h = h.flatten(2).transpose(1, 2)
        for b in s.enc: h = b(h, g)
        mu = s.to_lat(s.en(h)).chunk(2, -1)[0]; return mu.transpose(1, 2).reshape(mu.shape[0], s.C, *g)
    def decode(s, z):
        B = z.shape[0]; g = tuple(z.shape[2:]); h = s.from_lat(z.flatten(2).transpose(1, 2))
        for b in s.dec: h = b(h, g)
        P = s.P; y = s.shuffle(s.dn(h)).view(B, *g, s.NC, P, P, P).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, s.NC, g[0] * P, g[1] * P, g[2] * P)
        return y + s.refine(y)

# ---------------- generator: shell-conditioned flow transformer (research/interior_model/flow_f1.py) ----------------
class Flow(nn.Module):
    def __init__(s, fdim, C, ncmax, d=384, layers=8):
        super().__init__(); s.d = d; s.register_buffer('freq', torch.zeros(3, 16))
        s.pin = nn.Linear(35, d); s.part = nn.Embedding(2, d); s.kind = nn.Embedding(3, d)
        s.fin = nn.Sequential(nn.Linear(fdim, d), nn.SiLU(), nn.Linear(d, d))
        s.zin = nn.Linear(C, d); s.rin = nn.Linear(C, d); s.tin = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))
        s.age = nn.Linear(1, d); s.sex = nn.Embedding(3, d); s.cls = nn.Linear(ncmax, d); s.fid = nn.Embedding(3, d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, d // 64, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation='gelu'), layers, enable_nested_tensor=False)
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, C))
    def pe(s, pos): a = pos @ s.freq; return s.pin(torch.cat([a.sin(), a.cos(), pos], -1))
    def forward(s, zt, t, rev, pos, part, sfeat, spos, spart, age, agek, sex, cls, cpart):
        f = torch.exp(-math.log(1e4) * torch.arange(32, device=zt.device) / 32) * t * 1e3; te = s.tin(torch.cat([f.sin(), f.cos()])[None])
        hi = s.pe(pos) + s.part(part) + s.kind.weight[0] + torch.where(rev[..., None], s.rin(zt), s.zin(zt) + te)
        hs = s.fin(sfeat) + s.pe(spos) + s.part(spart) + s.kind.weight[1]
        hf = torch.cat([s.age(age) * agek + s.fid.weight[0], s.sex(sex) + s.fid.weight[1], s.cls(cls) + s.part(cpart) + s.fid.weight[2]], 0)[None] + s.kind.weight[2]
        return s.out(s.enc(torch.cat([hf, hs, hi], 1))[:, -zt.shape[1]:])

# ---------------- helpers ----------------
def sdf(body):
    from scipy.ndimage import distance_transform_edt as edt
    return ((edt(body) - edt(~body)) / 10.0).astype(np.float32)          # cm, positive inside

def tiles(shape, tile, ov):
    S = [max(1, int(math.ceil((n - ov) / (tile - ov)))) for n in shape]
    for i in range(S[0]):
        for j in range(S[1]):
            for k in range(S[2]):
                a = [min(max(0, q * (tile - ov)), max(0, n - tile)) for q, n in zip((i, j, k), shape)]; b = [min(n, aa + tile) for aa, n in zip(a, shape)]
                yield a, b, [ov // 2 if q > 0 else 0 for q in (i, j, k)]

def load(weights, dev):
    from safetensors.torch import load_file
    cfg = json.load(open(os.path.join(weights, 'config.json')))
    ae = TVAE(len(NAMES), C=cfg['latent_channels'], P=cfg['token_mm']); ae.load_state_dict(load_file(os.path.join(weights, 'vae.safetensors'))); ae.to(dev).eval()
    net = Flow(cfg['shell_feature_dim'], cfg['latent_channels'], len(NAMES), cfg['generator_width'], cfg['generator_layers'])
    net.load_state_dict(load_file(os.path.join(weights, 'generator.safetensors'))); net.to(dev).eval()
    return cfg, ae, net

@torch.no_grad()
def generate(labels, cfg, ae, net, dev, seed=0, steps=16, log=print):
    """labels: (X, Y, Z) uint8 shell volume at 1 mm in the head frame -> (P float16 (14, X', Y', Z'), body mask); X' etc. cut to whole tokens"""
    F, C = cfg['token_mm'], cfg['latent_channels']; mu, sd = np.array(cfg['latent_mean'], np.float32), np.array(cfg['latent_std'], np.float32)
    amp = torch.float32
    if dev.type == 'cuda':
        cc = torch.cuda.get_device_capability(dev); amp = torch.bfloat16 if cc[0] >= 8 else torch.float16 if cc[0] == 7 else torch.float32
    ac = lambda: torch.autocast(dev.type, dtype=amp, enabled=amp != torch.float32)
    G = [n // F for n in labels.shape]; L = labels[:G[0] * F, :G[1] * F, :G[2] * F]
    outside = L == 0; unk = L == UNK; body = ~outside & ~unk
    D = sdf(body); known = (~unk).astype(np.float32)
    blk = lambda A: A.reshape(G[0], F, G[1], F, G[2], F).transpose(0, 2, 4, 1, 3, 5)
    tok = np.argwhere(blk(body).reshape(*G, -1).any(-1)); p = F // 4
    Dp = blk(D).reshape(*G, 4, p, 4, p, 4, p).mean((4, 6, 8))
    feat = np.concatenate([Dp[tuple(tok.T)].reshape(len(tok), -1), blk(known).reshape(*G, -1).mean(-1)[tuple(tok.T)][:, None]], 1).astype(np.float32)
    pos = ((tok + 0.5) * F / 100.0).astype(np.float32); log('tokens', len(tok))
    # codes for token cells outside the body: the autoencoder's code of a plain shell (outside / other soft tissue)
    x = torch.zeros((1, len(NAMES), *L.shape), device=dev); x[0, 0] = torch.tensor(outside, device=dev).float(); x[0, NAMES.index('other_soft')] = 1 - x[0, 0]
    grid = torch.zeros((C, *G), device=dev); TL = list(tiles(G, 8, 2))
    for a, b, m in TL:
        with ac(): z = ae.encode(x[(..., slice(a[0] * F, b[0] * F), slice(a[1] * F, b[1] * F), slice(a[2] * F, b[2] * F))]).float()[0]
        grid[:, a[0] + m[0]:b[0], a[1] + m[1]:b[1], a[2] + m[2]:b[2]] = z[:, m[0]:, m[1]:, m[2]:]
    del x
    T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), device=dev, dtype=dt)
    from scipy import ndimage as ndi
    occ = np.zeros(G, bool); occ[tuple(tok.T)] = True
    shell = ~ndi.minimum_filter(occ, size=3, mode='constant', cval=0)[tuple(tok.T)]
    cd = dict(pos=T(pos)[None], part=T(np.ones(len(tok)), torch.long)[None], sfeat=T(feat[shell])[None], spos=T(pos[shell])[None], spart=T(np.ones(int(shell.sum())), torch.long)[None],
              age=T([[0.0]]), agek=T([[0.0]]), sex=T([0], torch.long), cls=T(np.ones((1, len(NAMES)), np.float32)), cpart=T([1], torch.long))
    g = torch.Generator(device=dev); g.manual_seed(seed)
    z = torch.randn((1, len(tok), C), device=dev, generator=g); rev = torch.zeros((1, len(tok)), dtype=torch.bool, device=dev)
    for i in range(steps):                                                   # Euler steps of the flow from noise to codes
        with ac(): z = z + net(z, torch.tensor(i / steps, device=dev), rev, **cd).float() / steps
    log('generated')
    gc = grid.cpu().numpy(); gc[(slice(None),) + tuple(tok.T)] = (z[0].cpu().numpy() * sd + mu).T
    P = np.zeros((len(NAMES), *L.shape), np.float16)
    for a, b, m in TL:
        with ac(): pr = torch.softmax(ae.decode(T(gc[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]])).float(), 1)[0].cpu().numpy()
        P[:, (a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = pr[:, m[0] * F:, m[1] * F:, m[2] * F:]
    P[:, ~body] = 0; P[0, ~body] = 1; log('decoded')
    return P, body

def meshes(P, body, mm=1.0):
    """one smooth surface per compartment, where its probability overtakes the strongest other one; vertices in mm,
    centred on the body's centre of the volume (same as the published example)"""
    from skimage import measure
    ctr = np.argwhere(body).mean(0); out = {}
    for k in range(1, len(NAMES)):
        nm = NAMES[k]; vol = P[k].astype(np.float32) - np.delete(P, k, axis=0).max(0).astype(np.float32)
        if vol.max() <= 0: continue
        v, f, _, _ = measure.marching_cubes(np.pad(vol, 1, constant_values=-1), 0.0, step_size=2 if nm in ('fat_layer', 'other_soft', 'brain') else 1)
        out[nm] = (((v - 1 - ctr) * mm).astype(np.float32), f.astype(np.uint32))
    return out

def skin_mesh(labels, F=16):
    """the input shell's outer surface (where the body ends), in the same mm frame as meshes(); for showing input and output together"""
    from skimage import measure; from scipy import ndimage as ndi
    G = [n // F for n in labels.shape]; L = labels[:G[0] * F, :G[1] * F, :G[2] * F]; body = (L != 0) & (L != UNK)
    v, f, _, _ = measure.marching_cubes(np.pad(ndi.gaussian_filter(body.astype(np.float32), 1.0), 1), 0.5)
    return ((v - 1 - np.argwhere(body).mean(0)).astype(np.float32), f.astype(np.uint32))

HEAD_TO_GLTF = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0]], np.float64)   # head frame (x to the right ear, y forward, z up) -> glTF (+y up, facing +z)

def write_glb(ms, path, max_faces=0, skin=None):
    """coloured GLB, one named node per compartment, metres, glTF axes; faces oriented outward; max_faces > 0 decimates
    each compartment to at most that many faces (needs fast-simplification); skin (v, f) adds the input shell as a
    half-transparent node 'skin_input'"""
    import trimesh
    sc = trimesh.Scene(); ms = dict(ms)
    if skin is not None: ms['skin_input'] = skin
    for nm, (v, f) in ms.items():
        if max_faces and len(f) > max_faces:
            import fast_simplification
            v, f = fast_simplification.simplify(v, f.astype(np.int64), target_reduction=1 - max_faces / len(f)); v = v.astype(np.float32); f = f.astype(np.uint32)
        vv = (v.astype(np.float64) / 1000.0) @ HEAD_TO_GLTF.T
        a, b, c = vv[f[:, 0]], vv[f[:, 1]], vv[f[:, 2]]
        if np.einsum('ij,ij->i', a, np.cross(b, c)).sum() < 0: f = f[:, ::-1]             # negative signed volume: faces point inward
        m = trimesh.Trimesh(vv, f, process=False)
        c = (np.array(COLOURS.get(nm, (0.85, 0.80, 0.78))) * 255).astype(np.uint8); sk = nm == 'skin_input'
        m.visual = trimesh.visual.TextureVisuals(material=trimesh.visual.material.PBRMaterial(name=nm, baseColorFactor=list(c) + [90 if sk else 255], roughnessFactor=0.7,
                                                                                          alphaMode='BLEND' if sk else 'OPAQUE', doubleSided=sk))
        sc.add_geometry(m, node_name=nm, geom_name=nm)
    sc.export(path)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('shell'); ap.add_argument('out'); ap.add_argument('--weights', default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument('--seed', type=int, default=0); ap.add_argument('--steps', type=int, default=16); ap.add_argument('--glb', action='store_true'); ap.add_argument('--max-faces', type=int, default=0); ap.add_argument('--cpu', action='store_true')
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    dev = torch.device('cpu' if a.cpu or not torch.cuda.is_available() else 'cuda')
    cfg, ae, net = load(a.weights, dev); sh = np.load(a.shell)
    P, body = generate(sh['labels'], cfg, ae, net, dev, a.seed, a.steps)
    lab = P.argmax(0).astype(np.uint8); lab[~body] = 0
    vols = {NAMES[k]: round(float((lab == k).sum()) / 1000, 1) for k in range(1, len(NAMES))}; print('volumes ml', vols)
    np.savez_compressed(os.path.join(a.out, 'compartments.npz'), labels=lab, probs=P, names=np.array(NAMES))
    json.dump(dict(volumes_ml=vols, seed=a.seed, steps=a.steps), open(os.path.join(a.out, 'volumes.json'), 'w'), indent=1)
    if a.glb: write_glb(meshes(P, body), os.path.join(a.out, 'compartments.glb'), a.max_faces, skin_mesh(sh['labels'], cfg['token_mm'])); print('wrote', os.path.join(a.out, 'compartments.glb'))
