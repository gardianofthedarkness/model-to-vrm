"""Test F1 of plan v8: flow matching over the frozen autoencoder's tokens, conditioned in context on the body shell.

One sequence per person, three kinds of token, one bidirectional transformer:
  shell tokens  the reference: every body token that touches the outside (a neighbour cell has no body). Content: the
                signed distance to the skin on that cell (pooled 4^3) and the share of the cell inside the scan.
  fact tokens   age, sex (each may be unknown) and the list of classes this person's source draws.
  noise tokens  one per body token; they start as Gaussian noise and end as that token's code (8 numbers).
Each kind has its own input projection; all share one 3D position encoding in the part frame, which is the only link
between the shell (geometry) and the interior (anatomy code).
Training: z_t = (1 - t) z_0 + t z_1 with z_0 noise and z_1 the normalised true codes; the network outputs the velocity
and is charged ||v - (z_1 - z_0)||^2 on the unknown tokens, averaged within the person. For half the steps a random
share of tokens is given clean with a flag ("partial reveal"); they are not charged. Age and sex are hidden at random.
No decoder and no labels are used in training (plain velocity loss; voxel-level terms are test F3).
Sampling: Euler steps from noise; the number of steps is picked on validation people by Dice.
Test (surface only): per-class Dice of single draws and of the per-voxel majority over draws, against the nearest
training person; plausibility (per structure: volume, number of pieces, centre inside the range seen in training
people); shell fidelity (generated body outline against the input shell); agreement between draws.
Uses the preparation cache of transformer_v2.py (<root>/prep_v2_TF<TF>), which must exist.
args: <data dir> <steps> [TF=8] [limit people per split]
env: OUT, REPO (optional upload), CFG = comma list of parts=pelvis|face|pelvis+face, d=384, layers=8, draws=3, valn=8, evals=20 (validations per run)"""
import os, sys, json, time, math, numpy as np
import torch, torch.nn as nn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_compress import make_ae
from transformer_smoke import tiles, decode, encode, AMP, UNK, DEV, log
from scipy import ndimage as ndi
CFG = dict(kv.split('=') for kv in os.environ.get('CFG', '').split(',') if kv)
PARTS = CFG.get('parts', 'pelvis').split('+'); D = int(CFG.get('d', 384)); LAYERS = int(CFG.get('layers', 8)); DRAWS = int(CFG.get('draws', 3)); VALN = int(CFG.get('valn', 8))
PID = {'pelvis': 0, 'face': 1}

_up = [None]
def upload(folder, path):
    """checkpoint upload in the background: a slow or stuck upload must never stop training (the job uploads everything again at the end)."""
    repo = os.environ.get('REPO')
    if not repo: return
    if _up[0] is not None and _up[0].is_alive(): log('earlier upload still running; skipped this one'); return
    def go():
        from huggingface_hub import HfApi
        try: HfApi().upload_folder(folder_path=folder, path_in_repo=path, repo_id=repo, repo_type='dataset', commit_message='flow ' + path)
        except Exception as ex: log('upload failed', type(ex).__name__)
    import threading; _up[0] = threading.Thread(target=go, daemon=True); _up[0].start()
def save(obj, f): torch.save(obj, f + '.tmp'); os.replace(f + '.tmp', f)                                   # whole-file swap: an upload in flight keeps reading the old file

class Flow(nn.Module):
    def __init__(s, fdim, C, ncmax, d=384, layers=8):
        super().__init__(); s.d = d; s.register_buffer('freq', torch.randn(3, 16) * 2.0)
        s.pin = nn.Linear(35, d); s.part = nn.Embedding(2, d); s.kind = nn.Embedding(3, d)                     # shared position encoding; kind: 0 interior, 1 shell, 2 fact
        s.fin = nn.Sequential(nn.Linear(fdim, d), nn.SiLU(), nn.Linear(d, d))                                # shell projection
        s.zin = nn.Linear(C, d); s.rin = nn.Linear(C, d); s.tin = nn.Sequential(nn.Linear(64, d), nn.SiLU(), nn.Linear(d, d))   # noisy code, given clean code, time
        s.age = nn.Linear(1, d); s.sex = nn.Embedding(3, d); s.cls = nn.Linear(ncmax, d); s.fid = nn.Embedding(3, d)            # facts; sex 0 = unknown
        s.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, d // 64, 4 * d, dropout=0.0, batch_first=True, norm_first=True, activation='gelu'), layers)
        s.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, C)); nn.init.zeros_(s.out[1].weight); nn.init.zeros_(s.out[1].bias)
    def pe(s, pos): a = pos @ s.freq; return s.pin(torch.cat([a.sin(), a.cos(), pos], -1))
    def forward(s, zt, t, rev, pos, part, sfeat, spos, spart, age, agek, sex, cls, cpart):
        f = torch.exp(-math.log(1e4) * torch.arange(32, device=zt.device) / 32) * t * 1e3; te = s.tin(torch.cat([f.sin(), f.cos()])[None])
        hi = s.pe(pos) + s.part(part) + s.kind.weight[0] + torch.where(rev[..., None], s.rin(zt), s.zin(zt) + te)
        hs = s.fin(sfeat) + s.pe(spos) + s.part(spart) + s.kind.weight[1]
        hf = torch.cat([s.age(age) * agek + s.fid.weight[0], s.sex(sex) + s.fid.weight[1], s.cls(cls) + s.part(cpart) + s.fid.weight[2]], 0)[None] + s.kind.weight[2]
        return s.out(s.enc(torch.cat([hf, hs, hi], 1))[:, -zt.shape[1]:])

def dice_table(P, T, names):
    """per-class Dice as transformer_smoke.dice_table (classes with >= 10 true voxels; outside and unknown left out), from one joint count."""
    m = np.bincount(P.ravel().astype(np.int64) * 256 + T.ravel(), minlength=65536).reshape(256, 256); out = {}
    for k in np.nonzero(m.sum(0) >= 10)[0]:
        if k == 0 or k == UNK: continue
        out[names.get(int(k), str(k))] = float(2 * m[k, k] / (m[k].sum() + m[:, k].sum()))
    return out

def stats(L, NC, mm):
    """per class present (>= 10 voxels): volume (ml), number of connected pieces, centre (mm in the part frame)."""
    L = np.where(L == UNK, 0, L).astype(np.int32); out = {}
    for k, sl in enumerate(ndi.find_objects(L, max_label=NC - 1), 1):
        if sl is None: continue
        m = L[sl] == k; n = int(m.sum())
        if n < 10: continue
        cen = [(c + q.start + 0.5) * mm for c, q in zip(ndi.center_of_mass(m), sl)]
        out[k] = [n * mm ** 3 / 1000.0, int(ndi.label(m)[1]), *cen]
    return out

if __name__ == '__main__':
    root = sys.argv[1]; steps = int(sys.argv[2]); TF = int(sys.argv[3]) if len(sys.argv) > 3 else 8; lim = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    OUT = os.environ.get('OUT', 'flow_f1'); out = os.path.join(root, OUT); os.makedirs(out, exist_ok=True); cache = os.path.join(root, os.environ.get('PREP', 'prep_v2_TF%d' % TF))
    res = dict(test='F1', parts=PARTS, d=D, layers=LAYERS, draws=DRAWS, amp=str(AMP), gpu=torch.cuda.get_device_name(0))
    fidx = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pidx = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
    names = {'face': {v: k for k, v in fidx['ids'].items()}, 'pelvis': dict(enumerate(pidx['classes']))}
    NCS = {'face': max(fidx['ids'].values()) + 1, 'pelvis': len(pidx['classes'])}; ncmax = max(NCS.values())
    aes = {}
    for part, ck in (('face', os.path.join(root, os.environ.get('FACE_AE_DIR', 'face_compress'), 'autoencoder_t%d.pt' % TF)), ('pelvis', os.path.join(root, 'pelvis_compress', 'autoencoder_t12.pt'))):
        if part not in PARTS: continue
        c = torch.load(ck, map_location=DEV); ae = make_ae(NCS[part], c['F'], c['C']).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval()
        for p in ae.parameters(): p.requires_grad_(False)
        aes[part] = (ae, c['F'], c['C'])
    C = next(iter(aes.values()))[2]; MM = {'face': 1.0, 'pelvis': 3.0}; FOLD = {'face': 'face_v1_headframe', 'pelvis': 'pelvis_v1'}
    hs = sorted(c for c, v in fidx['heads'].items() if v['source'] == 'hanseg')                              # split exactly as versions 1 and 2
    split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
    split.update({c: v['split'] for c, v in fidx['heads'].items() if v['source'] == 'ts'}); split.update({c: v['split'] for c, v in pidx['people'].items()})
    members = {'face': fidx['heads'], 'pelvis': pidx['people']}

    # ---- people from the preparation cache (training needs no labels)
    class Ent: pass
    data = {}; t1 = time.time()
    EXCL = json.load(open(os.environ['EXCLUDE']))['people'] if os.environ.get('EXCLUDE') and os.path.exists(os.environ['EXCLUDE']) else {}   # people whose shell fit failed (far outliers)
    for f in sorted(os.listdir(cache)):
        if not f.endswith('.npz'): continue
        key, part = f[:-4].rsplit('_', 1)
        if part not in PARTS or key.split('~')[0] in EXCL.get(part, []): continue
        base = key.split('~')[0]; z = np.load(os.path.join(cache, f)); e = Ent(); e.name, e.key, e.base, e.F, e.mm, e.file = part, key, base, aes[part][1], MM[part], os.path.join(cache, f); e.has_L = 'L' in z.files
        e.idx, e.feat, e.pos = z['idx'], z['feat'], z['pos']; code = z['code'].astype(np.float32); e.G = code.shape[1:]; e.raw = code[(slice(None),) + tuple(e.idx.T)].T.copy()
        occ = np.zeros(e.G, bool); occ[tuple(e.idx.T)] = True
        e.shell = ~ndi.minimum_filter(occ, size=3, mode='constant', cval=0)[tuple(e.idx.T)]                  # body tokens with a neighbour cell outside the body
        kn = members[part][base].get('known') if part == 'face' else None; e.cls = np.zeros(ncmax, np.float32)
        e.cls[list(kn) if kn is not None else range(NCS[part])] = 1                                          # (a mirrored head draws the mirrored classes: the list is symmetric in left/right pairs)
        data.setdefault(key, []).append(e)
    sp = {s: sorted(k for k in data if split[k.split('~')[0]] == s) for s in ('train', 'val', 'test')}
    sp['val'] = [k for k in sp['val'] if '~' not in k]; sp['test'] = [k for k in sp['test'] if '~' not in k or k.endswith('~h')]   # scored: clean people, and covered ones (prep v3)
    if lim: sp = {s: v[:: max(1, len(v) // lim)][:lim] for s, v in sp.items()}
    for s in ('val', 'test'):
        if not sp[s]: sp[s] = sp['train'][:2]; log('NOTE: no', s, 'people in the cache, using training people (plumbing check only)')
    def facts(key):
        b = key.split('~')[0]; m = pidx['people'].get(b, {}); a = m.get('age') or ''; x = m.get('sex') or ''
        try: age = float(a)
        except ValueError: age = None
        return age, {'m': 1, 'f': 2}.get(x, 0)
    ents = lambda keys: [e for k in keys for e in data[k]]
    norm = {}
    for part in PARTS:
        X = np.concatenate([e.raw for e in ents(sp['train']) if e.name == part]); norm[part] = (X.mean(0), X.std(0))
    for v in data.values():
        for e in v: e.z = ((e.raw - norm[e.name][0]) / norm[e.name][1]).astype(np.float32)
    ntok = [sum(len(e.idx) + int(e.shell.sum()) for e in data[k]) for s in sp for k in sp[s]]; shs = [float(e.shell.mean()) for e in ents(sp['train'])]
    res.update(people={s: len(v) for s, v in sp.items()}, sequence_tokens=dict(median=int(np.median(ntok)), max=int(max(ntok))), shell_share=round(float(np.mean(shs)), 3),
               code_mean={p: [round(float(x), 3) for x in v[0]] for p, v in norm.items()}, code_std={p: [round(float(x), 3) for x in v[1]] for p, v in norm.items()})
    log('people', res['people'], 'sequence', res['sequence_tokens'], 'shell share of body tokens', res['shell_share'], 'loaded in %.0f s' % (time.time() - t1))
    fdim = ents(sp['train'])[0].feat.shape[1]

    net = Flow(fdim, C, ncmax, D, LAYERS).to(DEV); ema = Flow(fdim, C, ncmax, D, LAYERS).to(DEV); ema.load_state_dict(net.state_dict()); ema.eval()
    for p in ema.parameters(): p.requires_grad_(False)
    opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    scaler = torch.amp.GradScaler(enabled=AMP == torch.float16); rng = np.random.default_rng(0); res['params_m'] = round(sum(p.numel() for p in net.parameters()) / 1e6, 1)
    T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), device=DEV, dtype=dt)
    def cond(key, drop):
        """everything but the noisy codes: positions, shell block, facts. drop: hide age and sex at random (training)."""
        v = data[key]; cat = lambda g: np.concatenate([g(e) for e in v]); age, sex = facts(key)
        if drop and rng.random() < 0.1: age = None
        if drop and rng.random() < 0.1: sex = 0
        return dict(pos=T(cat(lambda e: e.pos))[None], part=T(cat(lambda e: np.full(len(e.idx), PID[e.name])), torch.long)[None],
                    sfeat=T(cat(lambda e: e.feat[e.shell]))[None], spos=T(cat(lambda e: e.pos[e.shell]))[None], spart=T(cat(lambda e: np.full(int(e.shell.sum()), PID[e.name])), torch.long)[None],
                    age=T([[(age or 0) / 100.0]]), agek=T([[float(age is not None)]]), sex=T([sex], torch.long), cls=T(np.stack([e.cls for e in v])), cpart=T([PID[e.name] for e in v], torch.long))
    def run(model, zt, t, rev, cd):
        with torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32): return model(zt, torch.tensor(float(t), device=DEV), rev, **cd).float()
    def fm_loss(model, key, t, share, gen, drop):
        z1 = T(np.concatenate([e.z for e in data[key]]))[None]; z0 = torch.randn(z1.shape, device=DEV, generator=gen)
        rev = torch.rand(z1.shape[:2], device=DEV, generator=gen) < share; zt = torch.where(rev[..., None], z1, (1 - t) * z0 + t * z1)
        v = run(model, zt, t, rev, cond(key, drop)); return ((v - (z1 - z0)) ** 2)[~rev].mean()

    # ---- training with checkpoints (validation: fixed noise, four fixed times, nothing revealed)
    state = dict(step=0, best=np.inf, best_step=0, hist=[]); last = os.path.join(out, 'last.pt'); bestf = os.path.join(out, 'flow_f1.pt')
    if os.path.exists(last):
        ck = torch.load(last, map_location=DEV); net.load_state_dict(ck['net']); ema.load_state_dict(ck['ema']); opt.load_state_dict(ck['opt']); sched.load_state_dict(ck['sched']); state = ck['state']
        rng = np.random.default_rng(state['step']); log('resumed at step', state['step'])
    torch.cuda.reset_peak_memory_stats(); t2 = time.time(); start = state['step']; every = max(1, steps // int(CFG.get('evals', 20))); gt = torch.Generator(device=DEV); gt.manual_seed(start)
    def evaluate(it, tl):
        gv = torch.Generator(device=DEV); gv.manual_seed(12345)
        with torch.no_grad(): vl = float(np.mean([fm_loss(ema, k, (j + 0.5) / 4, 0.0, gv, False).item() for k in sp['val'] for j in range(4)]))
        state['hist'].append((it, round(tl, 4), round(vl, 4))); state['step'] = it + 1; better = vl < state['best']
        if better: state['best'], state['best_step'] = vl, it; save(dict(state_dict=ema.state_dict(), fdim=fdim, C=C, ncmax=ncmax, d=D, layers=LAYERS, norm={p: [x.tolist() for x in v] for p, v in norm.items()}, step=it, val=vl), bestf)
        save(dict(net=net.state_dict(), ema=ema.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), state=state), last)
        json.dump(dict(res, curve=state['hist'], best_step=state['best_step'], steps=steps), open(os.path.join(out, 'progress.json'), 'w'))
        log('step', it, 'train %.4f val %.4f%s' % (tl, vl, ' *' if better else ''), 'peak GB %.2f' % (torch.cuda.max_memory_allocated() / 1e9), '%.2f s/step' % ((time.time() - t2) / max(1, it + 1 - start)))
        if not lim: upload(out, 'm16_full/' + OUT)
    acc = []
    for it in range(start, steps):
        net.train(); key = sp['train'][rng.integers(len(sp['train']))]; share = 0.0 if rng.random() < 0.5 else rng.random() * 0.9
        t = float(torch.sigmoid(torch.randn((), device=DEV, generator=gt)))                                   # logit-normal times (Esser et al. 2024)
        loss = fm_loss(net, key, t, share, gt, True); opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        with torch.no_grad():
            a = min(0.999, (it + 1) / (it + 10))
            for pe_, p_ in zip(ema.parameters(), net.parameters()): pe_.lerp_(p_, 1 - a)
        acc.append(loss.item())
        if it % every == 0 or it == steps - 1: evaluate(it, float(np.mean(acc))); acc = []
    res.update(seconds_per_step=round((time.time() - t2) / max(1, steps - start), 3), peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), curve=state['hist'], best_step=state['best_step'], steps=steps)
    ema.load_state_dict(torch.load(bestf, map_location=DEV)['state_dict'])

    # ---- sampling and decoding
    def labels(e):                                                                           # true labels: stored with the copy (prep v3, shell frame) or the original set
        L = np.load(e.file)['L'] if e.has_L else np.load(os.path.join(root, FOLD[e.name], e.base + '.npz'))['labels']; return np.ascontiguousarray(L[:e.G[0] * e.F, :e.G[1] * e.F, :e.G[2] * e.F])
    def cover(e):
        if not e.has_L: return None
        z = np.load(e.file); n = int(np.prod([g * e.F for g in e.G])); return np.unpackbits(z['cover'])[:n].reshape([g * e.F for g in e.G]).astype(bool)
    @torch.no_grad()
    def sample(key, n, seed):
        cd = cond(key, False); g = torch.Generator(device=DEV); g.manual_seed(seed); N = cd['pos'].shape[1]
        z = torch.randn((1, N, C), device=DEV, generator=g); rev = torch.zeros((1, N), dtype=torch.bool, device=DEV)
        for i in range(n): z = z + run(ema, z, i / n, rev, cd) / n
        return z[0].cpu().numpy()
    bases = {}
    def base(e):
        """codes for the tokens the model does not generate: the autoencoder's own codes of the shell-only volume (outside, unknown, one
        generic tissue inside the body). The decoder reads its neighbours; zeros there are not a code it knows and made it paint a few
        classes (right parotid, right eye, brainstem) over plain tissue next to the body's edge. Found 2026-11-24."""
        if (e.key, e.name) not in bases:
            L = labels(e); cv = cover(e); sh = (L > 0) & (L < UNK) | (cv if cv is not None else False); bases.clear()
            bases[(e.key, e.name)] = encode(aes[e.name][0], np.where(sh, 3 if e.name == 'face' else 5, np.where(L == UNK, UNK, 0)).astype(np.uint8), NCS[e.name], e.F)
        return bases[(e.key, e.name)]
    @torch.no_grad()
    def to_labels(e, z):
        ae, F, _ = aes[e.name]; grid = base(e).copy(); grid[(slice(None),) + tuple(e.idx.T)] = (z * norm[e.name][1] + norm[e.name][0]).T
        P = np.zeros([g * F for g in e.G], np.uint8)
        for a, b, m in tiles(e.G, 24, 4):
            with torch.autocast('cuda', dtype=AMP, enabled=AMP != torch.float32): lg = decode(ae, torch.tensor(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]], device=DEV)).float().argmax(1)[0].cpu().numpy()
            P[(a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = lg[m[0] * F:, m[1] * F:, m[2] * F:]
        return P
    def draw(key, n, seed):
        z = sample(key, n, seed); o = 0; outp = []
        for e in data[key]: outp.append(to_labels(e, z[o:o + len(e.idx)])); o += len(e.idx)
        return outp
    med = lambda d: float(np.median(list(d.values()))) if d else 0.0
    truth = {}
    def tr(e):
        if (e.key, e.name) not in truth:                                                     # body = the shell the model read (body plus any cover); truth outside under a cover
            L = labels(e); cv = cover(e); truth[(e.key, e.name)] = (L, (L > 0) & (L < UNK) | (cv if cv is not None else False), np.where(L == UNK, 0, L))
        return truth[(e.key, e.name)]
    pick = {}
    for n in (4, 8, 16, 32):                                                                                 # number of Euler steps, picked on validation people by Dice of one draw
        sc = []
        for k in sp['val'][:VALN]:
            for e, P in zip(data[k], draw(k, n, 1)): L, body, Lt = tr(e); P[~body] = 0; sc.append(med(dice_table(P, Lt, names[e.name])))
        pick[n] = round(float(np.mean(sc)), 4); log('sampling steps', n, 'validation Dice', pick[n])
    NS = max(pick, key=pick.get); res['sampling_steps'] = dict(tried=pick, chosen=NS); truth.clear()

    # ---- reference ranges from training people (volume, pieces, centre) and nearest training person
    ref = {p: {} for p in PARTS}; trn = [e for e in ents(sp['train']) if '~' not in e.key]
    for e in trn:
        for k, v in stats(labels(e), NCS[e.name], e.mm).items(): ref[e.name].setdefault(k, []).append(v)
    rngs = {p: {k: (np.min(v, 0), np.max(v, 0)) for k, v in d.items()} for p, d in ref.items()}
    def inside(P, Lt, e):
        """share of this person's true structures whose predicted volume / piece count / centre lies inside the training range."""
        st = stats(P, NCS[e.name], e.mm); ok = np.zeros(3); n = 0
        for k in stats(Lt, NCS[e.name], e.mm):
            if k not in rngs[e.name]: continue
            n += 1; lo, hi = rngs[e.name][k]
            if k in st: v = np.array(st[k]); g = (v >= lo) & (v <= hi); ok += [g[0], g[1], g[2:].all()]
        return (ok / max(1, n)).tolist()
    def nearest(e):
        cand = [q for q in trn if q.name == e.name and q.base != e.base]; a = np.zeros(e.G, bool); a[tuple(e.idx.T)] = True; best, bi = None, -1
        for q in cand:
            s_ = tuple(slice(0, min(x, y)) for x, y in zip(q.G, e.G)); b = np.zeros(e.G, bool); qi = q.idx[(q.idx < np.array(e.G)).all(1)]; b[tuple(qi.T)] = True
            i = (a & b).sum() / max(1, (a | b).sum())
            if i > bi: best, bi = q, i
        return best
    def vote(stack):
        best = stack[0].copy(); bc = np.zeros(best.shape, np.uint8)
        for i in range(len(stack)):
            c = np.zeros(best.shape, np.uint8)
            for j in range(len(stack)): c += stack[i] == stack[j]
            up = c > bc; best[up] = stack[i][up]; bc[up] = c[up]
        return best
    from collections import defaultdict
    rows = defaultdict(lambda: {w: [] for w in ('single', 'majority', 'nearest')}); plaus = defaultdict(lambda: {w: [] for w in ('single', 'majority', 'nearest', 'truth')})
    shellfid = defaultdict(list); agree = defaultdict(list); cleared = defaultdict(list); t3 = time.time()
    for k in sp['test'][:int(os.environ.get('TEST_LIMIT', '0')) or None]:
        draws = [draw(k, NS, 100 + j) for j in range(DRAWS)]
        for pi, e in enumerate(data[k]):
            L, body, Lt = tr(e); raw = [d[pi] for d in draws]; nm = names[e.name]; pn = e.name + ('_covered' if e.key.endswith('~h') else '')
            cell = np.zeros(e.G, bool); cell[tuple(e.idx.T)] = True; cell = cell.repeat(e.F, 0).repeat(e.F, 1).repeat(e.F, 2)   # voxels of the body tokens: the only place the model generates
            shellfid[pn].append(float(np.mean([((P > 0) & body & cell).sum() / max(1, (((P > 0) | body) & cell).sum()) for P in raw])))
            for P in raw: P[~body] = 0
            cv = cover(e)
            if cv is not None and cv.any(): cleared[pn].append(float(np.mean([(P[cv] == 0).mean() for P in raw])))   # share of the cover the model emptied
            M = vote(raw); q = nearest(e); Lq = labels(q); s_ = tuple(slice(0, min(x, y)) for x, y in zip(Lq.shape, L.shape))
            N = np.zeros_like(M); N[s_] = np.where(Lq[s_] == UNK, 0, Lq[s_]); N[~body] = 0; N[body & (N == 0)] = 3 if e.name == 'face' else 5
            tabs = [dice_table(P, Lt, nm) for P in raw]; keys = sorted({x for t_ in tabs for x in t_})
            rows[pn]['single'].append({x: float(np.mean([t_.get(x, 0.0) for t_ in tabs])) for x in keys}); rows[pn]['majority'].append(dice_table(M, Lt, nm)); rows[pn]['nearest'].append(dice_table(N, Lt, nm))
            for w, X in (('single', raw[0]), ('majority', M), ('nearest', N), ('truth', Lt)): plaus[pn][w].append(inside(X, Lt, e))
            agree[pn].append(float(np.mean([med(dice_table(raw[i], raw[j], nm)) for i in range(DRAWS) for j in range(i + 1, DRAWS)])) if DRAWS > 1 else 1.0)
            np.savez_compressed(os.path.join(out, 'pred_%s_%s.npz' % (e.name, k)), labels=M, draw0=raw[0])
        truth.clear()
    summ = {}
    for p in list(rows):
        if not rows[p]['single']: continue
        s = summ[p] = {}
        for w, rr in rows[p].items():
            keys = sorted({x for r in rr for x in r}); s[w] = {x: round(float(np.median([r[x] for r in rr if x in r])), 3) for x in keys}; s[w + '_median_over_classes'] = round(med(s[w]), 3)
        s['inside_training_range'] = {w: dict(zip(('volume', 'pieces', 'centre'), [round(float(x), 3) for x in np.mean(v, 0)])) for w, v in plaus[p].items()}
        s['shell_fidelity_iou'] = round(float(np.mean(shellfid[p])), 4); s['cover_emptied'] = round(float(np.mean(cleared[p])), 3) if cleared[p] else None; s['agreement_between_draws'] = round(float(np.mean(agree[p])), 3)
    res['test_results'] = summ; res['test_seconds_per_person'] = round((time.time() - t3) / max(1, len(sp['test'])), 1)
    res['decode'] = 'shell-volume codes in tokens without body'; json.dump(res, open(os.path.join(out, 'flow_f1_shellfill.json'), 'w'), indent=1)
    log('SUMMARY', json.dumps({k: v for k, v in res.items() if k not in ('test_results', 'curve', 'code_mean', 'code_std')}),
        json.dumps({p: {w: v for w, v in d.items() if not isinstance(v, dict) or w == 'inside_training_range'} for p, d in summ.items()}))
