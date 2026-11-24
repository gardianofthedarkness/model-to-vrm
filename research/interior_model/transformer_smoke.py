"""Milestone 16 (Transformer), smoke and profile: masked bidirectional transformer per person over part-relative tokens
(pelvis 12 mm tokens on 3 mm labels, face TF mm tokens on 1 mm labels), surface only at test time.

Per part a frozen label autoencoder (milestone 13/15 design) turns labels into a code grid: one code (C channels) per
token. Per token the transformer sees: the surface (signed distance to the skin on that token's voxels, pooled to 4^3,
plus the share of voxels inside the scan), its position in the part frame (Fourier features of mm), the part, and, for
a random share of tokens per person (from none to most), the true code ("revealed"). It predicts every hidden token's
code. Loss: frozen decoder on random crops of the predicted code grid -> class logits -> cross-entropy against the true
labels (hidden tokens only, unknown voxels ignored, classes weighted (share)^-1/2), plus code MSE on hidden tokens.
Test: all tokens hidden (surface only); Dice per class at label resolution against a nearest-person baseline (the
training person whose body in the part frame overlaps most, labels copied inside the test body).
Profile: peak GPU memory, seconds per step, tokens per person; written to the result for the cost table.
args: <data dir> <steps> [TF=4|8] [limit people per split]"""
import os, sys, json, time, math, numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_compress import make_ae
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
UNK = 255; DEV = 'cuda'
AMP = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float32   # GTX 1650: fp16 is slower than fp32

def sdf(L, mm):
    from scipy.ndimage import distance_transform_edt
    B = (L > 0) & (L < UNK); return ((distance_transform_edt(B, sampling=mm) - distance_transform_edt(~B, sampling=mm)) / 10.0).astype(np.float32)

def tiles(shape, tile, ov):
    S = [max(1, int(math.ceil((n - ov) / (tile - ov)))) for n in shape]
    for i in range(S[0]):
        for j in range(S[1]):
            for k in range(S[2]):
                a = [min(max(0, q * (tile - ov)), max(0, n - tile)) for q, n in zip((i, j, k), shape)]; b = [min(n, aa + tile) for aa, n in zip(a, shape)]
                yield a, b, [ov // 2 if q > 0 else 0 for q in (i, j, k)]

@torch.no_grad()
def encode(ae, L, NC, F):
    """code grid of a whole window, tiled (tiles multiple of F)."""
    x = np.where(L == UNK, NC, L).astype(np.int64); G = [n // F for n in L.shape]; out = np.zeros((ae.to_code.out_channels, *G), np.float32)
    for a, b, m in tiles(L.shape, 96, 16):
        a = [aa // F * F for aa in a]; blk = x[a[0]:b[0], a[1]:b[1], a[2]:b[2]]; pad = [(0, (-d) % F) for d in blk.shape]; blk = np.pad(blk, pad, constant_values=NC)
        with torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32):
            h = ae.inp(ae.emb(torch.tensor(blk[None], device=DEV)).permute(0, 4, 1, 2, 3))
            for d in ae.down: h = d(h)
            z = ae.to_code(h).float()[0].cpu().numpy()
        ga = [aa // F for aa in a]; mm = [q // F for q in m]; gb = [min(g, ga_ + z.shape[i + 1]) for i, (g, ga_) in enumerate(zip(G, ga))]
        out[:, ga[0] + mm[0]:gb[0], ga[1] + mm[1]:gb[1], ga[2] + mm[2]:gb[2]] = z[:, mm[0]:gb[0] - ga[0], mm[1]:gb[1] - ga[1], mm[2]:gb[2] - ga[2]]
    return out

def decode(ae, z):
    h = ae.from_code(z)
    for u in ae.up: h = u(h)
    return ae.out(h)

class Part:
    """one person's part: tokens inside the body, their features, codes and labels."""
    def __init__(s, name, L, mm, F, ae, NC, pid):
        s.name, s.mm, s.F, s.pid, s.NC = name, mm, F, pid, NC
        G = [n // F for n in L.shape]; L = L[:G[0] * F, :G[1] * F, :G[2] * F]; s.L = L; s.G = G
        D = sdf(L, mm); known = (L != UNK).astype(np.float32); body = (L > 0) & (L < UNK)
        blk = lambda A: A.reshape(G[0], F, G[1], F, G[2], F).transpose(0, 2, 4, 1, 3, 5)
        tok_body = blk(body).reshape(*G, -1).any(-1); s.idx = np.argwhere(tok_body)                       # tokens with body inside
        p = F // 4 if F >= 4 else 1
        Dp = blk(D).reshape(*G, 4, p, 4, p, 4, p).mean((4, 6, 8)) if p > 1 else blk(D)                   # pooled to 4^3 per token
        s.feat = np.concatenate([Dp[tuple(s.idx.T)].reshape(len(s.idx), -1), blk(known).reshape(*G, -1).mean(-1)[tuple(s.idx.T)][:, None]], 1).astype(np.float32)
        s.pos = ((s.idx + 0.5) * F * mm / 100.0).astype(np.float32)                                     # token centre, 100 mm units, frame-relative
        s.code = encode(ae, L, NC, F)                                                                     # (C, *G)
        s.tcode = s.code[(slice(None),) + tuple(s.idx.T)].T.copy()                                        # (T, C)

class Net(nn.Module):
    def __init__(s, fdim, C, nparts, d=384, layers=8, heads=6):
        super().__init__(); s.fin = nn.Sequential(nn.Linear(fdim, d), nn.SiLU(), nn.Linear(d, d)); s.freq = nn.Parameter(torch.randn(3, 16) * 2.0, requires_grad=False)
        s.pin = nn.Linear(35, d); s.part = nn.Embedding(nparts, d); s.cin = nn.Linear(C, d); s.mask = nn.Parameter(torch.zeros(d))
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation='gelu'), layers)
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, C))
    def forward(s, feat, pos, part, code, revealed):
        a = pos @ s.freq; pe = torch.cat([a.sin(), a.cos(), pos], -1)
        h = s.fin(feat) + s.pin(pe) + s.part(part) + torch.where(revealed[..., None], s.cin(code), s.mask.expand_as(s.cin(code)))
        return s.out(s.enc(h))

def dice_table(P, T, names):
    out = {}
    for k in np.unique(T):
        if k == 0 or k == UNK: continue
        a, b = P == k, T == k
        if b.sum() < 10: continue
        out[names.get(int(k), str(k))] = float(2 * (a & b).sum() / (a.sum() + b.sum()))
    return out

if __name__ == '__main__':
    root = sys.argv[1]; steps = int(sys.argv[2]); TF = int(sys.argv[3]) if len(sys.argv) > 3 else 4; lim = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    res = dict(token_mm=dict(pelvis=12, face=TF), amp=str(AMP), gpu=torch.cuda.get_device_name(0))
    # parts: data, label mm, F, autoencoder
    fidx = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pidx = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
    fnames = {v: k for k, v in fidx['ids'].items()}; pnames = dict(enumerate(pidx['classes']))
    NCf = max(fidx['ids'].values()) + 1; NCp = len(pidx['classes'])
    aes = {}
    for part, NC, ck in (('face', NCf, os.path.join(root, 'face_compress', 'autoencoder_t%d.pt' % TF)), ('pelvis', NCp, os.path.join(root, 'pelvis_compress', 'autoencoder_t12.pt'))):
        c = torch.load(ck, map_location=DEV); ae = make_ae(NC, c['F'], c['C']).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval()
        for p in ae.parameters(): p.requires_grad_(False)
        aes[part] = (ae, c['F'], c['C'], NC)
    C = aes['face'][2]; assert aes['pelvis'][2] == C
    hs = sorted(c for c, v in fidx['heads'].items() if v['source'] == 'hanseg')
    split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
    split.update({c: v['split'] for c, v in fidx['heads'].items() if v['source'] == 'ts'}); split.update({c: v['split'] for c, v in pidx['people'].items()})
    people = sorted(set(fidx['heads']) | set(pidx['people'])); sp = {s: [c for c in people if split[c] == s] for s in ('train', 'val', 'test')}
    if lim:                                                                                                  # balanced: half people with a face, half with a pelvis
        rs = np.random.default_rng(1)
        def pick(v):
            f = [c for c in v if c in fidx['heads']]; p = [c for c in v if c in pidx['people'] and c not in fidx['heads']]
            return sorted(set(rs.permutation(f)[:max(1, lim // 2)].tolist()) | set(rs.permutation(p)[:max(1, lim - lim // 2)].tolist()))
        sp = {s: pick(v) for s, v in sp.items()}
    log('people', {s: len(v) for s, v in sp.items()})
    t1 = time.time(); data = {}
    for s in sp:
        for c in sp[s]:
            parts = []
            if c in pidx['people']: ae, F, _, NC = aes['pelvis']; parts.append(Part('pelvis', np.load(os.path.join(root, 'pelvis_v1', c + '.npz'))['labels'], 3.0, F, ae, NC, 0))
            if c in fidx['heads']: ae, F, _, NC = aes['face']; parts.append(Part('face', np.load(os.path.join(root, 'face_v1_headframe', c + '.npz'))['labels'], 1.0, F, ae, NC, 1))
            data[c] = parts
    ntok = [sum(len(p.idx) for p in v) for v in data.values()]
    res['prep_seconds_per_person'] = round((time.time() - t1) / len(data), 2); res['tokens_per_person'] = dict(median=int(np.median(ntok)), max=int(max(ntok)), min=int(min(ntok)))
    log('prepared', res['prep_seconds_per_person'], 's/person; tokens', res['tokens_per_person'])
    fdim = data[sp['train'][0]][0].feat.shape[1]
    assert all(p.feat.shape[1] == fdim for v in data.values() for p in v), 'feature size differs between parts'
    # class weights per part
    wts = {}
    for part, (ae, F, _, NC) in aes.items():
        sh = np.zeros(NC)
        for c in sp['train']:
            for p in data[c]:
                if p.name == part: v = p.L[p.L != UNK]; sh += np.bincount(v, minlength=NC)[:NC]
        if not sh.any(): wts[part] = torch.ones(NC, device=DEV); continue
        sh /= sh.sum(); w = np.where(sh > 0, sh, sh[sh > 0].min()) ** -0.5; wts[part] = torch.tensor(w / w[sh > 0].mean(), dtype=torch.float32, device=DEV)
    net = Net(fdim, C, 2).to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=AMP == torch.float16); rng = np.random.default_rng(0); res['params_m'] = round(sum(p.numel() for p in net.parameters()) / 1e6, 1)
    CROP = 16                                                                                              # tokens per side decoded for the loss
    def batch(c, share):
        parts = data[c]; feat = np.concatenate([p.feat for p in parts]); pos = np.concatenate([p.pos for p in parts]); part = np.concatenate([np.full(len(p.idx), p.pid) for p in parts])
        code = np.concatenate([p.tcode for p in parts]); rev = rng.random(len(feat)) < share
        T = lambda a, dt=torch.float32: torch.tensor(a[None], device=DEV, dtype=dt)
        return parts, T(feat), T(pos), T(part, torch.long), T(code), T(rev, torch.bool)
    def step_loss(c, share):
        parts, feat, pos, part, code, rev = batch(c, share)
        with torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32):
            pred = net(feat, pos, part, code, rev).float()[0]
        hid = ~rev[0]; loss = Fn.mse_loss(pred[hid], code[0][hid]) if hid.any() else pred.sum() * 0; o = 0
        for p in parts:
            n = len(p.idx); pr = pred[o:o + n]; o += n; ae, F, _, NC = aes[p.name]
            grid = torch.tensor(p.code, device=DEV).clone(); hmask = torch.zeros(p.G, dtype=torch.bool, device=DEV)
            ii = torch.tensor(p.idx, device=DEV); grid[:, ii[:, 0], ii[:, 1], ii[:, 2]] = torch.where(hid[o - n:o, None], pr, grid[:, ii[:, 0], ii[:, 1], ii[:, 2]].T).T
            hmask[ii[:, 0], ii[:, 1], ii[:, 2]] = hid[o - n:o]
            cen = p.idx[rng.integers(len(p.idx))]; a = [int(np.clip(cc - CROP // 2, 0, max(0, g - CROP))) for cc, g in zip(cen, p.G)]; b = [min(g, aa + CROP) for aa, g in zip(a, p.G)]
            z = grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]]
            with torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32): lg = decode(ae, z).float()
            y = torch.tensor(p.L[a[0] * F:b[0] * F, a[1] * F:b[1] * F, a[2] * F:b[2] * F].astype(np.int64), device=DEV)
            hm = hmask[a[0]:b[0], a[1]:b[1], a[2]:b[2]].repeat_interleave(F, 0).repeat_interleave(F, 1).repeat_interleave(F, 2)
            y = torch.where(hm & (y != UNK), y, torch.full_like(y, -100))
            if (y >= 0).any(): loss = loss + Fn.cross_entropy(lg, y[None], weight=wts[p.name], ignore_index=-100)
        return loss
    torch.cuda.reset_peak_memory_stats(); t2 = time.time(); best = (np.inf, None, 0); hist = []
    for it in range(steps):
        net.train(); c = sp['train'][rng.integers(len(sp['train']))]; share = 0.0 if rng.random() < 0.3 else rng.random() * 0.9
        loss = step_loss(c, share); opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        if it % max(1, steps // 10) == 0 or it == steps - 1:
            net.eval()
            with torch.no_grad(): vl = float(np.mean([step_loss(c, 0.0).item() for c in sp['val'][:8]]))
            hist.append((it, round(loss.item(), 4), round(vl, 4)))
            if vl < best[0]: best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, it)
            log('step', it, 'train %.4f val(surface only) %.4f' % (loss.item(), vl), 'peak GB %.2f' % (torch.cuda.max_memory_allocated() / 1e9), '%.2f s/step' % ((time.time() - t2) / (it + 1)))
    res.update(seconds_per_step=round((time.time() - t2) / steps, 3), peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), curve=hist, best_step=best[2], steps=steps)
    net.load_state_dict(best[1]); net.eval(); out = os.path.join(root, 'transformer_smoke'); os.makedirs(out, exist_ok=True)
    torch.save(dict(state_dict=best[1], fdim=fdim, C=C, token_mm=res['token_mm']), os.path.join(out, 'net_TF%d.pt' % TF))
    # test: surface only, against the nearest-person baseline
    scores = {'model': {'pelvis': [], 'face': []}, 'nearest': {'pelvis': [], 'face': []}}
    for c in sp['test'][:int(os.environ.get('TEST_LIMIT', '0')) or None]:
        parts, feat, pos, part, code, rev = batch(c, 0.0)
        with torch.no_grad(), torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32): pred = net(feat, pos, part, code, rev).float()[0].cpu().numpy()
        o = 0
        for p in parts:
            n = len(p.idx); grid = np.zeros_like(p.code); grid[(slice(None),) + tuple(p.idx.T)] = pred[o:o + n].T; o += n; ae, F, _, NC = aes[p.name]
            P = np.zeros(p.L.shape, np.uint8)
            for a, b, m in tiles(p.G, 24, 4):
                with torch.no_grad(), torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32): lg = decode(ae, torch.tensor(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]], device=DEV)).float().argmax(1)[0].cpu().numpy()
                P[(a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = lg[m[0] * F:, m[1] * F:, m[2] * F:]
            body = (p.L > 0) & (p.L < UNK); P[~body] = 0; Lt = np.where(p.L == UNK, 0, p.L)
            np.savez_compressed(os.path.join(out, 'pred_%s_%s.npz' % (p.name, c)), labels=P)
            names = pnames if p.name == 'pelvis' else fnames; scores['model'][p.name].append(dice_table(P, Lt, names))
            # nearest person: most overlapping body in the part frame
            cand = [(q, d) for d in sp['train'] for q in data[d] if q.name == p.name]
            def iou(q):
                s = tuple(slice(0, min(x, y)) for x, y in zip(q.L.shape, p.L.shape)); bq = (q.L[s] > 0) & (q.L[s] < UNK); bp = body[s]; return (bq & bp).sum() / max(1, (bq | bp).sum())
            q = max(cand, key=lambda qd: iou(qd[0]))[0]; s = tuple(slice(0, min(x, y)) for x, y in zip(q.L.shape, p.L.shape))
            N = np.zeros_like(P); N[s] = np.where(q.L[s] == UNK, 0, q.L[s]); N[~body] = 0; N[body & (N == 0)] = 3 if p.name == 'face' else 5
            scores['nearest'][p.name].append(dice_table(N, Lt, names))
    summ = {}
    for who, d in scores.items():
        for part, rows in d.items():
            if not rows: continue
            keys = sorted({k for r in rows for k in r}); summ.setdefault(part, {})[who] = {k: round(float(np.median([r[k] for r in rows if k in r])), 3) for k in keys}
            summ[part][who + '_median_over_classes'] = round(float(np.median(list(summ[part][who].values()))), 3)
    res['test'] = summ; res['people'] = {s: len(v) for s, v in sp.items()}
    json.dump(res, open(os.path.join(out, 'transformer_smoke_TF%d.json' % TF), 'w'), indent=1)
    log('SUMMARY', json.dumps({k: v for k, v in res.items() if k != 'test'}), json.dumps({p: {w: v for w, v in d.items() if w.endswith('median_over_classes')} for p, d in summ.items()}))
