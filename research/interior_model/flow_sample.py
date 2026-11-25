"""Generate an interior with the flow-matching model (flow_f1.py) for one shell-only label volume in a part frame.
The volume only needs body (any label 1..254), outside (0) and unknown (255); any anatomy in it is ignored.
Runs on the GPU (PyTorch's memory-efficient attention holds 12,000 tokens in well under 4 GB); env DEVICE=cpu moves the transformer to the CPU.
Output npz: draws (n, X, Y, Z) uint8 labels, majority, codes (n, tokens, C) as the decoder reads them, idx (tokens, 3).
args: <interior_data root> <checkpoint flow_f1.pt> <labels npz> <face|pelvis> <out npz> [age] [sex m|f] [draws=3] [steps=4]"""
import os, sys, json, math, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import ndimage as ndi

def shell_features(L, mm, F):
    """tokens with body inside, their surface features and positions, exactly as transformer_smoke.Part builds them."""
    from transformer_smoke import sdf, UNK
    G = [n // F for n in L.shape]; L = np.ascontiguousarray(L[:G[0] * F, :G[1] * F, :G[2] * F]); D = sdf(L, mm); known = (L != UNK).astype(np.float32); body = (L > 0) & (L < UNK)
    blk = lambda A: A.reshape(G[0], F, G[1], F, G[2], F).transpose(0, 2, 4, 1, 3, 5); idx = np.argwhere(blk(body).reshape(*G, -1).any(-1)); p = F // 4 if F >= 4 else 1
    Dp = blk(D).reshape(*G, 4, p, 4, p, 4, p).mean((4, 6, 8)) if p > 1 else blk(D)
    feat = np.concatenate([Dp[tuple(idx.T)].reshape(len(idx), -1), blk(known).reshape(*G, -1).mean(-1)[tuple(idx.T)][:, None]], 1).astype(np.float32)
    occ = np.zeros(G, bool); occ[tuple(idx.T)] = True; shell = ~ndi.minimum_filter(occ, size=3, mode='constant', cval=0)[tuple(idx.T)]
    return L, G, idx, feat, ((idx + 0.5) * F * mm / 100.0).astype(np.float32), shell

def load(root, ckf, part):
    from face_compress import make_ae
    from transformer_smoke import DEV
    from flow_f1 import Flow
    ck = torch.load(ckf, map_location='cpu'); net = Flow(ck['fdim'], ck['C'], ck['ncmax'], ck['d'], ck['layers']); net.load_state_dict(ck['state_dict']); net.eval().to(os.environ.get('DEVICE', DEV))
    fidx = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pidx = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
    NC = max(fidx['ids'].values()) + 1 if part == 'face' else len(pidx['classes']); names = {v: k for k, v in fidx['ids'].items()} if part == 'face' else dict(enumerate(pidx['classes']))
    c = torch.load(os.path.join(root, part + '_compress', 'autoencoder_t%d.pt' % (8 if part == 'face' else 12)), map_location=DEV); ae = make_ae(NC, c['F'], c['C'], norm=c.get('norm', 'group')).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval()
    return net, ck, ae, c['F'], NC, names

@torch.no_grad()
def sample(net, ck, part, idx, feat, pos, shell, NC, age, sex, steps, seed):
    dev = next(net.parameters()).device; pid = {'pelvis': 0, 'face': 1}[part]; T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), dtype=dt, device=dev); N = len(idx); cls = np.zeros((1, ck['ncmax']), np.float32); cls[0, :NC] = 1
    cd = dict(pos=T(pos)[None], part=T(np.full(N, pid), torch.long)[None], sfeat=T(feat[shell])[None], spos=T(pos[shell])[None], spart=T(np.full(int(shell.sum()), pid), torch.long)[None],
              age=T([[(age or 0) / 100.0]]), agek=T([[float(age is not None)]]), sex=T([{'m': 1, 'f': 2}.get(sex, 0)], torch.long), cls=T(cls), cpart=T([pid], torch.long))
    g = torch.Generator(); g.manual_seed(seed); z = torch.randn((1, N, ck['C']), generator=g).to(dev); rev = torch.zeros((1, N), dtype=torch.bool, device=dev)
    for i in range(steps): z = z + net(z, torch.tensor(i / steps, device=dev), rev, **cd) / steps
    mu, sd = [np.asarray(x, np.float32) for x in ck['norm'][part]]; return z[0].cpu().numpy() * sd + mu

@torch.no_grad()
def to_labels(ae, F, G, idx, codes, base):
    """base: the autoencoder's codes of the shell-only volume, used in tokens without body (zeros there make the decoder paint filler classes)."""
    from transformer_smoke import tiles, dec_tile, decode, DEV
    grid = base.copy(); grid[(slice(None),) + tuple(idx.T)] = codes.T; P = np.zeros([g * F for g in G], np.uint8)
    for a, b, m in tiles(G, *dec_tile(F)):
        lg = decode(ae, torch.tensor(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]], device=DEV)).float().argmax(1)[0].cpu().numpy()
        P[(a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = lg[m[0] * F:, m[1] * F:, m[2] * F:]
    return P

def vote(stack):
    best = stack[0].copy(); bc = np.zeros(best.shape, np.uint8)
    for i in range(len(stack)):
        c = np.zeros(best.shape, np.uint8)
        for j in range(len(stack)): c += stack[i] == stack[j]
        up = c > bc; best[up] = stack[i][up]; bc[up] = c[up]
    return best

if __name__ == '__main__':
    root, ckf, lf, part, outf = sys.argv[1:6]; a = sys.argv[6:]; age = float(a[0]) if len(a) > 0 and a[0] != '-' else None; sex = a[1] if len(a) > 1 else ''; draws = int(a[2]) if len(a) > 2 else 3; steps = int(a[3]) if len(a) > 3 else 4
    net, ck, ae, F, NC, names = load(root, ckf, part); mm = 1.0 if part == 'face' else 3.0
    L, G, idx, feat, pos, shell = shell_features(np.load(lf)['labels'], mm, F); body = (L > 0) & (L < 255); print(part, 'tokens', len(idx), 'shell tokens', int(shell.sum()), flush=True)
    from transformer_smoke import encode
    base = encode(ae, np.where(body, 3 if part == 'face' else 5, L).astype(np.uint8), NC, F); codes = []; outs = []
    for j in range(draws):
        z = sample(net, ck, part, idx, feat, pos, shell, NC, age, sex, steps, 100 + j); P = to_labels(ae, F, G, idx, z, base); P[~body] = 0; codes.append(z); outs.append(P); print('draw', j, 'done', flush=True)
    np.savez_compressed(outf, draws=np.stack(outs), majority=vote(outs), codes=np.stack(codes).astype(np.float32), idx=idx)
