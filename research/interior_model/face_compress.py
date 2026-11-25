"""Milestone 15 (Tokens), face part: does a label autoencoder keep the thin face structures at part-relative tokens?

Data: face_v1 heads in their head frame, 1 mm labels (face_frames.py). Split: the skull scans' own split; HaN-Seg heads
(no split of their own) every fifth by sorted name go to test, the next to validation.
Variants (same autoencoder as Compress, milestone 13; class embedding, conv encoder with F-fold downsampling, C code
channels, conv decoder, cross-entropy weighted by (voxel share)^-1/2, unknown voxels ignored):
  t4    1 mm labels, F=4  -> 4 mm tokens   (face grid of milestone 15)
  t8    1 mm labels, F=8  -> 8 mm tokens
  t12   3 mm labels, F=4  -> 12 mm tokens  (fixed whole-body grid), decoded labels upsampled to 1 mm for scoring
All scored at 1 mm on the test heads: Dice per class against its thickness (twice the mean inside distance, mm),
key thin structures listed. Profile per variant: peak GPU memory, seconds per step. Also: tokens inside the body per
head for each token size. Runs on the local GTX 1650 (fp16). args: <data dir> <steps> [variants]
env: MARGINAL=1 marginal loss for classes a head's source never draws (face label repair, 2026-11-25); OUTDIR output folder."""
import os, sys, json, time, numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F_
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
UNK = 255
_cc, _nm = torch.cuda.get_device_capability(), torch.cuda.get_device_name()
AMP = torch.bfloat16 if _cc[0] >= 8 else torch.float16 if _cc[0] == 7 and 'GTX' not in _nm else None   # GTX 16xx: no tensor cores, fp32 faster

class ChanNorm(nn.Module):
    """normalise each voxel over its channels (layer norm on the channel axis). Unlike GroupNorm it uses no statistics over
    space, so the output does not depend on the tile size (GroupNorm did: plan v8 section 13)."""
    def __init__(s, c): super().__init__(); s.g = nn.Parameter(torch.ones(1, c, 1, 1, 1)); s.b = nn.Parameter(torch.zeros(1, c, 1, 1, 1))
    def forward(s, x): m = x.mean(1, keepdim=True); v = x.var(1, keepdim=True, unbiased=False); return (x - m) / torch.sqrt(v + 1e-5) * s.g + s.b

def make_ae(NC, F, C, emb=16, w=32, norm='group', active=None):
    nm = (lambda b: nn.GroupNorm(8, b)) if norm == 'group' else ChanNorm
    def blk(a, b, s=1): return nn.Sequential(nn.Conv3d(a, b, 3, stride=s, padding=1), nm(b), nn.SiLU(), nn.Conv3d(b, b, 3, padding=1), nm(b), nn.SiLU())
    n = int(np.log2(F)); ch = [w * 2 ** i for i in range(n + 1)]
    class AE(nn.Module):
        def __init__(s):
            super().__init__(); s.emb = nn.Embedding(NC + 1, emb); s.inp = blk(emb, ch[0])
            s.down = nn.ModuleList([blk(ch[i], ch[i + 1], 2) for i in range(n)]); s.to_code = nn.Conv3d(ch[n], C, 1); s.from_code = blk(C, ch[n])
            s.up = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), blk(ch[i + 1], ch[i])) for i in reversed(range(n))]); s.out = nn.Conv3d(ch[0], NC, 1)
            if norm != 'group':                                                                # classes that never occur are left out of the vote (a fixed part of the model)
                cm = torch.zeros(NC)
                if active is not None: cm[[k for k in range(NC) if k not in active]] = -1e4
                s.register_buffer('cmask', cm.view(1, NC, 1, 1, 1))
        def forward(s, x):
            h = s.inp(s.emb(x).permute(0, 4, 1, 2, 3))
            for d in s.down: h = d(h)
            h = s.from_code(s.to_code(h))
            for u in s.up: h = u(h)
            o = s.out(h)
            return o + s.cmask.to(o.dtype) if hasattr(s, 'cmask') else o
    return AE()

def surface_mm(a, b, mm):
    """median distance between the surfaces of two masks (boundary voxels, both ways), mm"""
    from scipy.ndimage import binary_erosion
    from scipy.spatial import cKDTree
    sa = np.argwhere(a & ~binary_erosion(a)); sb = np.argwhere(b & ~binary_erosion(b))
    if len(sa) < 3 or len(sb) < 3: return float('nan')
    sa = sa[:: max(1, len(sa) // 40000)]; sb = sb[:: max(1, len(sb) // 40000)]
    return float(np.median(np.concatenate([cKDTree(sb).query(sa, workers=-1)[0], cKDTree(sa).query(sb, workers=-1)[0]])) * mm)

def thickness(mask):
    from scipy.ndimage import distance_transform_edt
    if not mask.any(): return None
    ii = np.argwhere(mask); lo = np.maximum(ii.min(0) - 2, 0); hi = ii.max(0) + 3; m = mask[tuple(slice(a, b) for a, b in zip(lo, hi))]
    return float(2 * distance_transform_edt(m)[m].mean())

def predict(net, L, NC, step, dev, tile=96, ov=16):
    """tiled inference over a whole head; L at the variant's own label size; unknown voxels fed as class NC."""
    x = np.where(L == UNK, NC, L).astype(np.int64); out = np.zeros(L.shape, np.uint8)
    S = [max(1, int(np.ceil((n - ov) / (tile - ov)))) for n in L.shape]
    for i in range(S[0]):
        for j in range(S[1]):
            for k in range(S[2]):
                a = [min(max(0, q * (tile - ov)), max(0, n - tile)) for q, n in zip((i, j, k), L.shape)]
                b = [min(n, aa + tile) for aa, n in zip(a, L.shape)]
                blk = x[a[0]:b[0], a[1]:b[1], a[2]:b[2]]; pad = [(0, (-d) % step) for d in blk.shape]; blk = np.pad(blk, pad, constant_values=NC)
                with torch.no_grad(), torch.autocast('cuda', dtype=AMP or torch.float32, enabled=AMP is not None):
                    P = net(torch.tensor(blk[None], device=dev)).float().argmax(1)[0].cpu().numpy().astype(np.uint8)
                P = P[:b[0] - a[0], :b[1] - a[1], :b[2] - a[2]]
                m = [ov // 2 if q > 0 else 0 for q in (i, j, k)]
                out[a[0] + m[0]:b[0], a[1] + m[1]:b[1], a[2] + m[2]:b[2]] = P[m[0]:, m[1]:, m[2]:]
    return out

if __name__ == '__main__':
    root = sys.argv[1]; steps = int(sys.argv[2]); variants = sys.argv[3].split(',') if len(sys.argv) > 3 else ['t4', 't8', 't12']
    PART = os.environ.get('PART', 'face')
    if PART == 'face':
        D = os.path.join(root, 'face_v1_headframe'); idx = json.load(open(os.path.join(D, 'index.json'))); ids = idx['ids']; heads = idx['heads']
        hs = sorted(c for c, v in heads.items() if v['source'] == 'hanseg')
        split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
        split.update({c: v['split'] for c, v in heads.items() if v['source'] == 'ts'})
    else:                                                                       # pelvis part: 3 mm labels, 12 mm tokens (F=4)
        D = os.path.join(root, 'pelvis_v1'); idx = json.load(open(os.path.join(D, 'index.json'))); ids = {n: i for i, n in enumerate(idx['classes'])}; heads = idx['people']
        split = {c: v['split'] for c, v in heads.items()}
        if os.environ.get('LIMIT'): keep = {s: [c for c in sorted(heads) if split[c] == s][:int(os.environ['LIMIT'])] for s in ('train', 'val', 'test')}; heads = {c: heads[c] for v in keep.values() for c in v}
        variants = ['t12']
    names = {v: k for k, v in ids.items()}; NC = max(ids.values()) + 1; OUT = os.path.join(root, os.environ.get('OUTDIR', '%s_compress' % PART))
    MARGINAL = os.environ.get('MARGINAL') == '1'                                                 # face label repair: a source that never draws a class is not charged for it
    MISS = {c: sorted(set(range(7, NC)) - set(heads[c].get('known', range(NC)))) for c in heads} if PART == 'face' else {c: [] for c in heads}
    vols = {c: np.load(os.path.join(D, c + '.npz'))['labels'] for c in heads}
    if PART == 'face' and os.environ.get('TISSUE') == '1':                                       # tissue types (tissue_map.py); a head that never had brain or eyes drawn is not charged for them
        from tissue_map import remap, table; tab = table(ids); vols = {c: remap(v, ids) for c, v in vols.items()}
        MISS = {c: sorted({int(tab[k]) for k in MISS[c]} & {ids['brain'], ids['eye_left']}) for c in heads}
    sp = {s: [c for c in heads if split[c] == s] for s in ('train', 'val', 'test')}; log({s: len(v) for s, v in sp.items()}, 'classes', NC)
    body = {c: (v > 0) & (v < UNK) for c, v in vols.items()}
    tokens = {('%d mm' % t): int(np.median([int(np.ceil(m[::t, ::t, ::t].sum())) for m in body.values()])) for t in (4, 6, 8, 12)}   # token cells whose corner voxel is body
    log('median tokens inside the body per head', tokens)
    dev = 'cuda'; res = dict(tokens_inside_body_median=tokens, split={s: len(v) for s, v in sp.items()}, variants={})
    share = np.zeros(NC)
    for c in sp['train']: v = vols[c]; share += np.bincount(v[v < UNK].ravel(), minlength=NC)[:NC]
    share /= share.sum(); wts = torch.tensor(np.where(share > 0, share, share[share > 0].min()) ** -0.5, dtype=torch.float32); wts = (wts / wts[share > 0].mean()).to(dev)
    for var in variants:
        F, ds = {'t4': (4, 1), 't8': (8, 1), 't12': (4, 3 if PART == 'face' else 1)}[var]; C = 8; S = 64
        V = {c: v[::ds, ::ds, ::ds] for c, v in vols.items()}
        rng = np.random.default_rng(0); BI = {c: np.argwhere((v[::4, ::4, ::4] > 0) & (v[::4, ::4, ::4] < UNK)) * 4 for c, v in V.items()}
        def crop(cs, n, who=None):
            out = np.full((n, S, S, S), NC, np.int64)
            for i in range(n):
                c0 = cs[rng.integers(len(cs))]; L = V[c0]
                if who is not None: who.append(c0); ctr = BI[c0][rng.integers(len(BI[c0]))]                    # crop centred on a body voxel
                o = [int(np.clip(q - S // 2, 0, max(0, d - S))) for q, d in zip(ctr, L.shape)]; c_ = L[o[0]:o[0] + S, o[1]:o[1] + S, o[2]:o[2] + S]
                out[i, :c_.shape[0], :c_.shape[1], :c_.shape[2]] = np.where(c_ == UNK, NC, c_)
            return torch.tensor(out, device=dev)
        SHAPE = os.environ.get('SHAPE') == '1'                                                 # shape loss: per-class truncated signed distance (plan v8 section 14)
        T = 8 // ds; active = [int(k) for k in np.flatnonzero(share > 0)]
        def dil(x, cube): return F_.max_pool3d(x, 3, 1, 1) if cube else torch.maximum(torch.maximum(F_.max_pool3d(x, (3, 1, 1), 1, (1, 0, 0)), F_.max_pool3d(x, (1, 3, 1), 1, (0, 1, 0))), F_.max_pool3d(x, (1, 1, 3), 1, (0, 0, 1)))
        def depth(m):                                                                          # erosion steps survived, alternating cube and cross: near-euclidean, up to T
            cur = m; acc = torch.zeros_like(m)
            for t in range(T): cur = 1 - dil(1 - cur, t % 2 == 0); acc = acc + cur
            return acc
        act = torch.tensor(active, device=dev); inact = torch.tensor([k for k in range(NC) if k not in active], device=dev)
        def shape_targets(y):
            """u: depth inside each active class (positive) or distance outside it (negative), over T voxels, in [-1, 1];
            all active classes at once (channels). wgt: emphasis near surfaces, zero at or near unknown voxels."""
            m = (y[:, None] == act[None, :, None, None, None]).float(); unk = (y == NC).float()[:, None]
            sd = torch.where(m > 0, depth(m) + 0.5, -(depth(1 - m) + 0.5)); u = (sd / T).clamp(-1, 1)
            near = u.abs().amin(1, keepdim=True) * T
            wgt = (1 + 3 * torch.exp(-near / 1.5)) * (1 - unk) * (1 - dil(unk, True))
            return u, wgt
        def shape_loss(lg, y):
            u, wgt = shape_targets(y)                                                          # classes never present are held at 'outside' together, as one class's share
            err = (torch.tanh(lg[:, act]) - u).abs().mean(1, keepdim=True)
            return (err * wgt).sum() / wgt.sum().clamp(min=1) + ce(lg, y)                     # class term at full weight: the split is learned as fast as before; the shape term places the surfaces
        net = make_ae(NC, F, C, norm='chan' if SHAPE else 'group', active=active if SHAPE else None).to(dev); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        scaler = torch.amp.GradScaler(enabled=AMP == torch.float16); ce = nn.CrossEntropyLoss(weight=wts, ignore_index=NC)
        def lossf(lg, y, who=None):
            """weighted cross-entropy; with MARGINAL, a plain-tissue voxel (classes 1-6) of a head whose source never draws classes M
            is scored by log(p_y + sum over M of p_k) (Shi et al., marginal loss), so the code is free to hold the structure it cannot see."""
            if SHAPE: return shape_loss(lg, y)
            if not (MARGINAL and who) or not any(MISS[c] for c in who): return ce(lg, y)
            lp = torch.log_softmax(lg, 1); valid = y != NC; yc = y.clamp(max=NC - 1); lt = lp.gather(1, yc[:, None])[:, 0]
            rows = []
            for i, c in enumerate(who):
                if MISS[c]: gen = (y[i] >= 1) & (y[i] <= 6); rows.append(torch.where(gen, torch.logaddexp(lt[i], torch.logsumexp(lp[i, MISS[c]], 0)), lt[i]))
                else: rows.append(lt[i])
            lt = torch.stack(rows)
            w = wts[yc] * valid; return -(lt * w).sum() / w.sum()
        valw = [[] for _ in range(6)]; valb = [crop(sp['val'], 2, valw[i]) for i in range(6)]; best = (np.inf, None, 0)
        torch.cuda.reset_peak_memory_stats(); t1 = time.time(); bs = int(os.environ.get('BS', '2'))
        for it in range(steps):
            net.train(); who = []; x = crop(sp['train'], bs, who); y = x.clone()
            with torch.autocast('cuda', dtype=AMP or torch.float32, enabled=AMP is not None): loss = lossf(net(x).float(), y, who)
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            if it % max(1, steps // 20) == 0 or it == steps - 1:
                net.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=AMP or torch.float32, enabled=AMP is not None): vl = float(np.mean([lossf(net(b).float(), b, w).item() for b, w in zip(valb, valw)]))
                if np.isfinite(vl) and vl < best[0]: best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, it)
                log(var, 'step', it, 'train %.4f val %.4f' % (loss.item(), vl), 'peak GB %.2f' % (torch.cuda.max_memory_allocated() / 1e9))
        sec_step = (time.time() - t1) / steps; peak = torch.cuda.max_memory_allocated() / 1e9
        net.load_state_dict(best[1]); net.eval(); os.makedirs(OUT, exist_ok=True)
        torch.save(dict(state_dict=best[1], F=F, C=C, label_mm=ds, classes=NC, norm='chan' if SHAPE else 'group', shape=SHAPE), os.path.join(OUT, 'autoencoder_%s.pt' % var))
        per = {}
        for c in sp['test'][:int(os.environ.get('TEST_LIMIT', '0')) or None]:
            P = predict(net, V[c], NC, F, dev)
            if ds > 1: P = np.repeat(np.repeat(np.repeat(P, ds, 0), ds, 1), ds, 2)[:vols[c].shape[0], :vols[c].shape[1], :vols[c].shape[2]]
            T = vols[c]; ok = T != UNK; Pp = np.zeros_like(T); Pp[:P.shape[0], :P.shape[1], :P.shape[2]] = P
            np.savez_compressed(os.path.join(OUT, 'pred_%s_%s.npz' % (var, c)), labels=Pp)
            for k in np.unique(T[ok]):
                if k == 0: continue
                a, b = (Pp == k) & ok, (T == k)
                if b.sum() < 10: continue
                per.setdefault(int(k), []).append((float(2 * (a & b).sum() / (a.sum() + b.sum())), thickness(b), surface_mm(a, b, ds * (3 if PART == 'pelvis' else 1))))
        rows = {names.get(k, str(k)): dict(dice=round(float(np.median([q[0] for q in v])), 3), thickness_mm=round(float(np.median([q[1] for q in v])), 1), surface_mm=round(float(np.median([q[2] for q in v])), 2), heads=len(v)) for k, v in per.items()}
        d = np.array([r['dice'] for r in rows.values()]); th = np.array([r['thickness_mm'] for r in rows.values()])
        bins = {'%g-%g mm' % (a, b): (round(float(np.median(d[(th >= a) & (th < b)])), 3) if ((th >= a) & (th < b)).any() else None, int(((th >= a) & (th < b)).sum())) for a, b in ((0, 3), (3, 6), (6, 12), (12, 1e9))}
        res['variants'][var] = dict(token_mm=F * ds * (3 if PART == 'pelvis' else 1), label_mm=ds * (3 if PART == 'pelvis' else 1), best_step=best[2], seconds_per_step=round(sec_step, 3), peak_gpu_gb=round(peak, 2), batch=bs, crop=S,
                                    median_dice=round(float(np.median(d)), 3), by_thickness=bins, per_class=rows)
        log(var, json.dumps({k: v for k, v in res['variants'][var].items() if k != 'per_class'}))
        del net, opt; torch.cuda.empty_cache()
    json.dump(res, open(os.path.join(OUT, 'face_compress_%d.json' % steps), 'w'), indent=1)
    log('SUMMARY', json.dumps({k: (v if k != 'variants' else {kk: {x: y for x, y in vv.items() if x != 'per_class'} for kk, vv in v.items()}) for k, v in res.items()}))
