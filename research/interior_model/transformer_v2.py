"""Milestone 16 transformer, version 2: same tokens and network size as transformer_smoke.py, changed where version 1
went wrong (it painted small structures at several times their volume and lost to the nearest-person copy).

1. Correction on a real person. Every person gets a "nearest person": the training person (never themselves or their
   own mirror image) whose body tokens overlap theirs most. That person's codes at the same token places are an input,
   and the network predicts a correction added to them. The output layer starts at zero, so training starts exactly at
   the nearest-person baseline, and what it draws starts from human anatomy.
2. Loss. Cross-entropy on the decoded voxels with NO rare-class boost; each voxel instead weighs
   1 + exp(-distance to skin / token size) + exp(-distance to bone / token size), so mistakes near the surface and
   around bones cost up to three times more. Plus a Dice term over all classes with a one-voxel floor: a class painted
   where it does not exist, or at several times its size, is charged directly. Plus the code error as before.
3. Unlabelled classes. A head whose source never drew some classes (index 'known' list; HaN-Seg has no brain) is not
   charged for predicting them where its truth only says a generic tissue, in the loss and in the "fair" test score.
4. Mirror images, face only: each training head is also used mirrored about the head frame's midplane with left/right
   label names swapped. This only says a mirrored head is as plausible a head as the original; it does not make any
   prediction symmetric. The pelvis and torso are not mirrored (organs are one-sided).
5. Housekeeping: per-person preparation is cached on disk (and in the data repo when REPO is set) and can be made
   beforehand on any machine (PREP_ONLY=1, SHARD=i/n for parallel processes); labels and loss weights are read from
   disk as needed so the run fits a 15 GB machine; validation on all
   validation people 40 times per run; last and best checkpoints are saved (and uploaded) at every validation, and a
   run restarts from the last one.
Test as before: surface only, per-class Dice; reported raw (comparable with version 1) and fair (point 3).
args: <data dir> <steps> [TF=8] [limit people per split]   env: OUT (output folder name), REPO (optional, upload)"""
import os, sys, re, json, time, numpy as np
import torch, torch.nn as nn, torch.nn.functional as Fn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_compress import make_ae
from transformer_smoke import tiles, decode, Part, dice_table, AMP, UNK, DEV, log
from scipy.ndimage import distance_transform_edt as edt
BONE = re.compile(r'^(mandible|skull|teeth_|hard_palate|vertebrae_|hip|sacrum|femur_|lumbar|dense)')     # bone classes by name (anatomy, not a tuned value)
SIDE = re.compile(r'^(.*)_(left|right|L|R)$')

def upload(folder, path):
    repo = os.environ.get('REPO')
    if not repo: return
    from huggingface_hub import HfApi
    for attempt in range(4):
        try: HfApi().upload_folder(folder_path=folder, path_in_repo=path, repo_id=repo, repo_type='dataset', commit_message='m16 v2 ' + path); return
        except Exception as ex: log('upload retry', attempt, type(ex).__name__); time.sleep(60)

def weight_map(L, mm, tau, bone_ids):
    """100 * (exp(-skin distance / tau) + exp(-bone distance / tau)) as uint8; the loss weight is 1 + this / 100."""
    body = (L > 0) & (L < UNK); ds = edt(body, sampling=mm); bone = np.isin(L, bone_ids)
    w = np.exp(-ds / tau) + (np.exp(-edt(~bone, sampling=mm) / tau) if bone.any() else 0)
    return np.where(body, np.round(100 * w), 0).astype(np.uint8)

def mirror(L, lo_mm, mm, swap):
    """mirror about the frame plane x = 0 (voxel i has its centre at lo + (i + 0.5) mm) and swap left/right labels."""
    n = L.shape[0]; src = int(round(-2 * lo_mm / mm)) - 1 - np.arange(n); ok = (src >= 0) & (src < n)
    M = np.full_like(L, UNK); M[ok] = L[src[ok]]; return swap[M]

class Net(nn.Module):
    def __init__(s, fdim, C, nparts, d=384, layers=8, heads=6):
        super().__init__(); s.fin = nn.Sequential(nn.Linear(fdim, d), nn.SiLU(), nn.Linear(d, d)); s.freq = nn.Parameter(torch.randn(3, 16) * 2.0, requires_grad=False)
        s.pin = nn.Linear(35, d); s.part = nn.Embedding(nparts, d); s.cin = nn.Linear(C, d); s.mask = nn.Parameter(torch.zeros(d)); s.nin = nn.Linear(C + 1, d)
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation='gelu'), layers)
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, C)); nn.init.zeros_(s.out[1].weight); nn.init.zeros_(s.out[1].bias)
    def forward(s, feat, pos, part, code, revealed, ncode, nhas):
        a = pos @ s.freq; pe = torch.cat([a.sin(), a.cos(), pos], -1)
        h = s.fin(feat) + s.pin(pe) + s.part(part) + s.nin(torch.cat([ncode, nhas], -1)) + torch.where(revealed[..., None], s.cin(code), s.mask.expand_as(s.cin(code)))
        return ncode + s.out(s.enc(h))

if __name__ == '__main__':
    root = sys.argv[1]; steps = int(sys.argv[2]); TF = int(sys.argv[3]) if len(sys.argv) > 3 else 8; lim = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    OUT = os.environ.get('OUT', 'transformer_v2'); out = os.path.join(root, OUT); os.makedirs(out, exist_ok=True)
    cache = os.path.join(root, 'prep_v2_TF%d' % TF); os.makedirs(cache, exist_ok=True)
    res = dict(version=2, token_mm=dict(pelvis=12, face=TF), amp=str(AMP), gpu=torch.cuda.get_device_name(0))
    fidx = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pidx = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
    fnames = {v: k for k, v in fidx['ids'].items()}; pnames = dict(enumerate(pidx['classes']))
    NCf = max(fidx['ids'].values()) + 1; NCp = len(pidx['classes'])
    aes = {}
    for part, NC, ck in (('face', NCf, os.path.join(root, 'face_compress', 'autoencoder_t%d.pt' % TF)), ('pelvis', NCp, os.path.join(root, 'pelvis_compress', 'autoencoder_t12.pt'))):
        c = torch.load(ck, map_location=DEV); ae = make_ae(NC, c['F'], c['C']).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval()
        for p in ae.parameters(): p.requires_grad_(False)
        aes[part] = (ae, c['F'], c['C'], NC)
    C = aes['face'][2]; assert aes['pelvis'][2] == C
    bone = {'face': [k for k, n in fnames.items() if BONE.match(n)], 'pelvis': [k for k, n in pnames.items() if BONE.match(n)]}
    swap = np.arange(256, dtype=np.uint8); ids = fidx['ids']                                                  # left/right partner of every face class
    for n, k in ids.items():
        m = SIDE.match(n)
        if m:
            o = {'left': 'right', 'right': 'left', 'L': 'R', 'R': 'L'}[m.group(2)]; partner = '%s_%s' % (m.group(1), o)
            if partner in ids: swap[k] = ids[partner]
    log('bone classes', {p: len(v) for p, v in bone.items()}, 'left/right pairs', int((swap != np.arange(256)).sum()) // 2)
    hs = sorted(c for c, v in fidx['heads'].items() if v['source'] == 'hanseg')                              # split exactly as version 1
    split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
    split.update({c: v['split'] for c, v in fidx['heads'].items() if v['source'] == 'ts'}); split.update({c: v['split'] for c, v in pidx['people'].items()})
    people = sorted(set(fidx['heads']) | set(pidx['people'])); sp = {s: [c for c in people if split[c] == s] for s in ('train', 'val', 'test')}
    if lim:
        rs = np.random.default_rng(1)
        def pick(v):
            f = [c for c in v if c in fidx['heads']]; p = [c for c in v if c in pidx['people'] and c not in fidx['heads']]
            return sorted(set(rs.permutation(f)[:max(1, lim // 2)].tolist()) | set(rs.permutation(p)[:max(1, lim - lim // 2)].tolist()))
        sp = {s: pick(v) for s, v in sp.items()}
    log('people', {s: len(v) for s, v in sp.items()})

    # ---- preparation (cached): tokens, features, codes, loss weights; mirrored copies of training heads
    def prep(key, name, L, mm, F, ae, NC, pid, known):
        f = os.path.join(cache, '%s_%s.npz' % (key, name)); G = [n // F for n in L.shape]; L = np.ascontiguousarray(L[:G[0] * F, :G[1] * F, :G[2] * F])
        if os.path.exists(f):
            z = np.load(f); p = Part.__new__(Part); p.name, p.mm, p.F, p.pid, p.NC, p.L, p.G = name, mm, F, pid, NC, L, G
            p.idx, p.feat, p.pos, p.code, p.W = z['idx'], z['feat'], z['pos'], z['code'].astype(np.float32), z['W']; p.tcode = p.code[(slice(None),) + tuple(p.idx.T)].T.copy()
        else:
            p = Part(name, L, mm, F, ae, NC, pid); p.W = weight_map(p.L, mm, F * mm, bone[name])
            np.savez_compressed(f, idx=p.idx, feat=p.feat, pos=p.pos, code=p.code.astype(np.float16), W=p.W)
        if MM:                                                                                             # keep labels and weights on disk, not in RAM
            for k in ('L', 'W'):
                g = os.path.join(MM, '%s_%s_%s.npy' % (key, name, k))
                if not os.path.exists(g): np.save(g, getattr(p, k))
                setattr(p, k, np.load(g, mmap_mode='r'))
        p.miss = sorted(set(range(7, NC)) - set(known)) if known is not None else []                         # named classes this person's source never drew
        return p
    PREP_ONLY = os.environ.get('PREP_ONLY') == '1'; SH = [int(x) for x in os.environ.get('SHARD', '0/1').split('/')]   # PREP_ONLY: fill the cache and stop (any machine with a GPU); SHARD i/n: every n-th person
    MM = None if PREP_ONLY else os.path.join(root, 'mmap_v2'); MM and os.makedirs(MM, exist_ok=True)
    t1 = time.time(); data = {}; base = {}; made = len(os.listdir(cache)); count = 0
    for s in sp:
        for c in list(sp[s]):
            count += 1
            if PREP_ONLY and count % SH[1] != SH[0]: continue
            parts = []
            if c in pidx['people']: ae, F, _, NC = aes['pelvis']; parts.append(prep(c, 'pelvis', np.load(os.path.join(root, 'pelvis_v1', c + '.npz'))['labels'], 3.0, F, ae, NC, 0, None))
            if c in fidx['heads']:
                ae, F, _, NC = aes['face']; L = np.load(os.path.join(root, 'face_v1_headframe', c + '.npz'))['labels']; kn = fidx['heads'][c]['known']
                parts.append(prep(c, 'face', L, 1.0, F, ae, NC, 1, kn))
                if s == 'train':
                    m = c + '~m'; data[m] = [prep(m, 'face', mirror(L, fidx['window_lo_mm'][0], 1.0, swap), 1.0, F, ae, NC, 1, [int(swap[k]) for k in kn])]; base[m] = c; sp[s].append(m)
            data[c] = parts; base[c] = c
            if PREP_ONLY: data.clear(); parts.clear()                                                      # only the cache files matter; keep nothing in memory
    if PREP_ONLY: log('prepared shard', SH, 'in', round(time.time() - t1), 's'); sys.exit(0)
    ntok = [sum(len(p.idx) for p in v) for v in data.values()]
    res['prep_seconds'] = round(time.time() - t1); res['tokens_per_person'] = dict(median=int(np.median(ntok)), max=int(max(ntok)), min=int(min(ntok)))
    log('prepared', len(data), 'people incl. mirrors in', res['prep_seconds'], 's; tokens', res['tokens_per_person'])
    if len(os.listdir(cache)) > made and not lim: upload(cache, 'm16_full/prep_v2_TF%d' % TF)
    fdim = data[sp['train'][0]][0].feat.shape[1]

    # ---- nearest training person per part (token-level body overlap), never the same person or their mirror
    for name in ('pelvis', 'face'):
        ents = [(c, p) for c, v in data.items() for p in v if p.name == name]
        if not ents: continue
        Gm = np.max([p.G for _, p in ents], 0); occ = torch.zeros((len(ents), *Gm), dtype=torch.float32, device=DEV)
        for i, (c, p) in enumerate(ents): occ[i][tuple(torch.tensor(p.idx.T, device=DEV))] = 1
        occ = occ.flatten(1); n = occ.sum(1); tr = torch.tensor([split[base[c]] == 'train' for c, _ in ents], device=DEV)
        for i, (c, p) in enumerate(ents):
            inter = occ @ occ[i]; iou = inter / (n + n[i] - inter); iou[~tr] = -1
            for j, (c2, _) in enumerate(ents):
                if base[c2] == base[c]: iou[j] = -1
            q = ents[int(iou.argmax())][1]; s_ = tuple(slice(0, min(a, b)) for a, b in zip(q.G, p.G))
            g = np.zeros((C, *p.G), np.float32); g[(slice(None),) + s_] = q.code[(slice(None),) + s_]; has = np.zeros(p.G, np.float32); qi = q.idx[(q.idx < np.array(p.G)).all(1)]; has[tuple(qi.T)] = 1
            p.ntcode = g[(slice(None),) + tuple(p.idx.T)].T.copy(); p.nhas = has[tuple(p.idx.T)][:, None].copy(); p.near = q
    log('nearest people assigned')

    net = Net(fdim, C, 2).to(DEV); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=AMP == torch.float16); rng = np.random.default_rng(0); res['params_m'] = round(sum(p.numel() for p in net.parameters()) / 1e6, 1)
    CROP = 16
    def batch(c, share):
        parts = data[c]; cat = lambda k: np.concatenate([getattr(p, k) for p in parts]); part = np.concatenate([np.full(len(p.idx), p.pid) for p in parts])
        rev = rng.random(len(part)) < share; T = lambda a, dt=torch.float32: torch.tensor(a[None], device=DEV, dtype=dt)
        return parts, T(cat('feat')), T(cat('pos')), T(part, torch.long), T(cat('tcode')), T(rev, torch.bool), T(cat('ntcode')), T(cat('nhas'))
    def step_loss(c, share):
        parts, feat, pos, part, code, rev, ncode, nhas = batch(c, share)
        with torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32): pred = net(feat, pos, part, code, rev, ncode, nhas).float()[0]
        hid = ~rev[0]; loss = Fn.mse_loss(pred[hid], code[0][hid]) if hid.any() else pred.sum() * 0; o = 0
        for p in parts:
            n = len(p.idx); pr = pred[o:o + n]; o += n; ae, F, _, NC = aes[p.name]
            grid = torch.tensor(p.code, device=DEV).clone(); hmask = torch.zeros(p.G, dtype=torch.bool, device=DEV)
            ii = torch.tensor(p.idx, device=DEV); grid[:, ii[:, 0], ii[:, 1], ii[:, 2]] = torch.where(hid[o - n:o, None], pr, grid[:, ii[:, 0], ii[:, 1], ii[:, 2]].T).T
            hmask[ii[:, 0], ii[:, 1], ii[:, 2]] = hid[o - n:o]
            cen = p.idx[rng.integers(len(p.idx))]; a = [int(np.clip(cc - CROP // 2, 0, max(0, g - CROP))) for cc, g in zip(cen, p.G)]; b = [min(g, aa + CROP) for aa, g in zip(a, p.G)]
            with torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32): lg = decode(ae, grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]]).float()[0]
            vs = (slice(a[0] * F, b[0] * F), slice(a[1] * F, b[1] * F), slice(a[2] * F, b[2] * F))
            y = torch.tensor(p.L[vs].astype(np.int64), device=DEV); w = 1 + torch.tensor(p.W[vs], device=DEV).float() / 100
            valid = hmask[a[0]:b[0], a[1]:b[1], a[2]:b[2]].repeat_interleave(F, 0).repeat_interleave(F, 1).repeat_interleave(F, 2) & (y != UNK)
            if not valid.any(): continue
            yc = y.clamp(max=NC - 1); lp = Fn.log_softmax(lg, 0); lt0 = lp.gather(0, yc[None])[0]; lt = lt0; prob = lp.exp() * valid
            if p.miss:
                mi = torch.tensor(p.miss, device=DEV); gen = (y >= 1) & (y <= 6)                              # generic tissue may really be a class this source never drew
                lt = torch.where(gen, torch.logaddexp(lt0, torch.logsumexp(lp[mi], 0)), lt0); prob[mi] = prob[mi] * ~gen
            ce = -(lt * w)[valid].sum() / w[valid].sum()
            tv = torch.bincount(yc[valid], minlength=NC).float(); inter = torch.zeros(NC, device=DEV).scatter_add_(0, yc[valid], lt0.exp()[valid]); pv = prob.flatten(1).sum(1)
            dice = 1 - ((2 * inter + 1) / (tv + pv + 1))[1:].mean()                                           # all classes but 'outside'; one-voxel floor
            loss = loss + ce + dice
        return loss

    # ---- training with checkpoints
    state = dict(step=0, best=np.inf, best_step=0, hist=[]); last = os.path.join(out, 'last.pt'); bestf = os.path.join(out, 'net_v2_TF%d.pt' % TF)
    if os.path.exists(last):
        ck = torch.load(last, map_location=DEV); net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); scaler.load_state_dict(ck['scaler']); state = ck['state']
        rng = np.random.default_rng(state['step']); log('resumed at step', state['step'], 'best', state['best'], 'at', state['best_step'])
    torch.cuda.reset_peak_memory_stats(); t2 = time.time(); start = state['step']; every = max(1, steps // 40)
    def evaluate(it, tl):
        net.eval()
        with torch.no_grad(): vl = float(np.mean([step_loss(c, 0.0).item() for c in sp['val']]))
        state['hist'].append((it, round(tl, 4), round(vl, 4))); state['step'] = it + 1
        better = vl < state['best']
        if better: state['best'], state['best_step'] = vl, it; torch.save(dict(state_dict=net.state_dict(), fdim=fdim, C=C, token_mm=res['token_mm'], step=it, val=vl), bestf)
        torch.save(dict(net=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), state=state), last)
        json.dump(dict(res, curve=state['hist'], best_step=state['best_step'], steps=steps), open(os.path.join(out, 'progress.json'), 'w'))
        log('step', it, 'train %.4f val(surface only) %.4f%s' % (tl, vl, ' *' if better else ''), 'peak GB %.2f' % (torch.cuda.max_memory_allocated() / 1e9), '%.2f s/step' % ((time.time() - t2) / max(1, it + 1 - start)))
        if not lim: upload(out, 'm16_full/' + OUT)
    run = []
    for it in range(start, steps):
        net.train(); c = sp['train'][rng.integers(len(sp['train']))]; share = 0.0 if rng.random() < 0.3 else rng.random() * 0.9
        loss = step_loss(c, share); opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        run.append(loss.item())
        if it % every == 0 or it == steps - 1: evaluate(it, float(np.mean(run))); run = []
    res.update(seconds_per_step=round((time.time() - t2) / max(1, steps - start), 3), peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), curve=state['hist'], best_step=state['best_step'], steps=steps)
    net.load_state_dict(torch.load(bestf, map_location=DEV)['state_dict']); net.eval()

    # ---- test: surface only; raw Dice (as version 1) and fair Dice (classes the person's source never drew are left out)
    scores = {w: {'pelvis': [], 'face': []} for w in ('model', 'nearest', 'model_fair', 'nearest_fair')}
    for c in sp['test'][:int(os.environ.get('TEST_LIMIT', '0')) or None]:
        parts, feat, pos, part, code, rev, ncode, nhas = batch(c, 0.0)
        with torch.no_grad(), torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32): pred = net(feat, pos, part, code, rev, ncode, nhas).float()[0].cpu().numpy()
        o = 0
        for p in parts:
            n = len(p.idx); grid = np.zeros_like(p.code); grid[(slice(None),) + tuple(p.idx.T)] = pred[o:o + n].T; o += n; ae, F, _, NC = aes[p.name]
            P = np.zeros(p.L.shape, np.uint8)
            for a, b, m in tiles(p.G, 24, 4):
                with torch.no_grad(), torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32): lg = decode(ae, torch.tensor(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]], device=DEV)).float().argmax(1)[0].cpu().numpy()
                P[(a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = lg[m[0] * F:, m[1] * F:, m[2] * F:]
            body = (p.L > 0) & (p.L < UNK); P[~body] = 0; Lt = np.where(p.L == UNK, 0, p.L)
            np.savez_compressed(os.path.join(out, 'pred_%s_%s.npz' % (p.name, c)), labels=P)
            q = p.near; s_ = tuple(slice(0, min(x, y)) for x, y in zip(q.L.shape, p.L.shape))
            N = np.zeros_like(P); N[s_] = np.where(q.L[s_] == UNK, 0, q.L[s_]); N[~body] = 0; N[body & (N == 0)] = 3 if p.name == 'face' else 5
            names = pnames if p.name == 'pelvis' else fnames; gen = (p.L >= 1) & (p.L <= 6)
            for who, X in (('model', P), ('nearest', N)):
                scores[who][p.name].append(dice_table(X, Lt, names))
                skip = gen & np.isin(X, p.miss) if p.miss else np.zeros(X.shape, bool)
                scores[who + '_fair'][p.name].append(dice_table(np.where(skip, 0, X), np.where(skip, UNK, np.where(p.L == UNK, UNK, Lt)), names))
    summ = {}
    for who, d in scores.items():
        for part, rows in d.items():
            if not rows: continue
            keys = sorted({k for r in rows for k in r}); summ.setdefault(part, {})[who] = {k: round(float(np.median([r[k] for r in rows if k in r])), 3) for k in keys}
            summ[part][who + '_median_over_classes'] = round(float(np.median(list(summ[part][who].values()))), 3)
    res['test'] = summ; res['people'] = {s: len(v) for s, v in sp.items()}
    json.dump(res, open(os.path.join(out, 'transformer_v2_TF%d.json' % TF), 'w'), indent=1)
    log('SUMMARY', json.dumps({k: v for k, v in res.items() if k not in ('test', 'curve')}), json.dumps({p: {w: v for w, v in d.items() if w.endswith('median_over_classes')} for p, d in summ.items()}))
