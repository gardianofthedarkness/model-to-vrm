"""Flow matching for the field experiment (field_exp.py), face only, with an optional shell/topology loss.
Model: the F1 transformer (flow_f1.Flow) over the autoencoder's codes, conditioned on the shell tokens (field_prep.py).
Loss: flow matching on the codes; with SHAPE > 0 also, every step, on a random block of 8^3 tokens (64 mm): the one-step
estimate z1_hat = z_t + (1 - t) v is decoded by the frozen decoder (cells outside the body keep their true codes) and
charged per tissue against the true shares from the CT field (known voxels only):
  shape     1 - soft Dice                         size   |soft volume - true volume| / true volume
  position  distance of soft centroids / block     topology  1 - soft clDice (Shit et al. 2021: skeletons by soft erosion)
Test: Euler sampling (16 steps), decoded probabilities; against the true shares on known voxels: Dice of the argmax per
tissue, surface distance at the 50% level (dense even sampling), and pieces per tissue (argmax, >= 20 voxels) against truth.
env: OUT, REPO, SHAPE (weight, 0 = off, auto = shell part equal to the flow loss each step), SEED, STEPS, TRAIN_SOURCES, TEST_SOURCES (field_exp.splits). args: <fields dir> <prep dir> <autoencoder .pt>"""
import os, sys, shutil, json, time, math, numpy as np, torch, torch.nn.functional as F_
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_exp import AMP, scaler, cached_head, load_head, splits, shares, make_ae, NC, NAMES, log, profile_line
from flow_f1 import Flow
from transformer_smoke import tiles
root, prep, aef = sys.argv[1:4]; dev = 'cuda'; OUT = os.environ.get('OUT', 'field_flow'); os.makedirs(OUT, exist_ok=True)
STEPS = int(os.environ.get('STEPS', '10000')); AUTO = os.environ.get('SHAPE', '0') == 'auto'; SHAPE = 1.0 if AUTO else float(os.environ.get('SHAPE', '0')); SEED = int(os.environ.get('SEED', '0')); torch.manual_seed(SEED); rng = np.random.default_rng(SEED)
idx = json.load(open(os.path.join(root, 'index.json'))); sp = splits(idx, lambda c: os.path.exists(os.path.join(prep, c + '.npz')))
if os.environ.get('TEST_LIMIT'): sp['test'] = sp['test'][:int(os.environ['TEST_LIMIT'])]
ck = torch.load(aef, map_location=dev); ae = make_ae(soft=(ck['kind'] == 'soft'), arch=ck.get('arch', 'conv')).to(dev); ae.load_state_dict(ck['state_dict']); ae.eval()
for p in ae.parameters(): p.requires_grad_(False)
F = 8; C = 8
class Ent: pass
data = {}
for s, cs in sp.items():
    for c in cs:
        z = np.load(os.path.join(prep, c + '.npz')); e = Ent(); e.idx = z['idx'].astype(np.int64); e.feat = z['feat']; e.pos = z['pos']; e.code = z['code'].astype(np.float32); e.G = e.code.shape[1:]
        occ = np.zeros(e.G, bool); occ[tuple(e.idx.T)] = True; e.shell = ~ndi.minimum_filter(occ, size=3, mode='constant', cval=0)[tuple(e.idx.T)]; e.raw = e.code[(slice(None),) + tuple(e.idx.T)].T.copy()
        data[c] = e
X = np.concatenate([data[c].raw for c in sp['train']]); mu, sd = X.mean(0), X.std(0)
for e in data.values(): e.z = ((e.raw - mu) / sd).astype(np.float32)
log({s: len(v) for s, v in sp.items()}, 'shape weight', SHAPE, 'seed', SEED, 'autoencoder', ck['kind'])
heads = {}
def head(c):
    if c not in heads: heads[c] = cached_head(root, c, idx['eye_id'])
    return heads[c]
net = Flow(66 - 1, C, NC, 384, 8).to(dev); ema = Flow(66 - 1, C, NC, 384, 8).to(dev); ema.load_state_dict(net.state_dict()); ema.eval()
for p in ema.parameters(): p.requires_grad_(False)
opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=0.05); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=STEPS, pct_start=max(0.05, 2.5 / STEPS))
T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), device=dev, dtype=dt)
CLS = T(np.ones((1, NC), np.float32))
def cond(e):
    n = len(e.idx); return dict(pos=T(e.pos)[None], part=T(np.ones(n), torch.long)[None], sfeat=T(e.feat[e.shell])[None], spos=T(e.pos[e.shell])[None], spart=T(np.ones(int(e.shell.sum())), torch.long)[None],
                                age=T([[0.0]]), agek=T([[0.0]]), sex=T([0], torch.long), cls=CLS, cpart=T([1], torch.long))
def soft_skel(x, it=6):
    def erode(y): return -F_.max_pool3d(-y, 3, 1, 1)
    def dilate(y): return F_.max_pool3d(y, 3, 1, 1)
    skel = F_.relu(x - dilate(erode(x)))
    for _ in range(it):
        x = erode(x); d = F_.relu(x - dilate(erode(x))); skel = skel + F_.relu(d - skel * d)
    return skel
def shape_loss(P, Tt, K):
    """P, Tt: (1, NC, S, S, S) probabilities / true shares; K: (1, 1, S, S, S) known. Terms averaged over tissues present."""
    P = P * K; Tt = Tt * K; terms = []; S = P.shape[-1]; g = torch.stack(torch.meshgrid(*[torch.arange(S, device=dev, dtype=torch.float32)] * 3, indexing='ij'), 0)
    for k in range(1, NC):
        p, t = P[0, k], Tt[0, k]; vt = t.sum()
        if vt < 50: continue
        vp = p.sum(); dice = 1 - (2 * (p * t).sum() + 1) / (vp + vt + 1); size = (vp - vt).abs() / vt
        cp = (g * p).sum((1, 2, 3)) / (vp + 1e-3); ct = (g * t).sum((1, 2, 3)) / vt; pos = torch.linalg.norm(cp - ct) / S
        sp_, st = soft_skel(p[None, None]), soft_skel(t[None, None]); tprec = ((sp_ * t).sum() + 1) / (sp_.sum() + 1); tsens = ((st * p).sum() + 1) / (st.sum() + 1)
        terms.append(dice + size + pos + (1 - 2 * tprec * tsens / (tprec + tsens)))
    return torch.stack(terms).mean() if terms else None
def step_loss(c, t, gen):
    e = data[c]; z1 = T(e.z)[None]; z0 = torch.randn(z1.shape, device=dev, generator=gen); rev = torch.rand(z1.shape[:2], device=dev, generator=gen) < (0.3 if rng.random() < 0.5 else 0.0)
    zt = torch.where(rev[..., None], z1, (1 - t) * z0 + t * z1)
    with torch.autocast('cuda', dtype=AMP): v = net(zt, torch.tensor(float(t), device=dev), rev, **cond(e)).float()
    loss = ((v - (z1 - z0)) ** 2)[~rev].mean(); sl = None
    if SHAPE > 0:
        h = head(c); occ = np.zeros(e.G, bool); occ[tuple(e.idx.T)] = True; cand = np.argwhere(occ[:max(1, e.G[0] - 7), :max(1, e.G[1] - 7), :max(1, e.G[2] - 7)])
        a = cand[rng.integers(len(cand))]; b = np.minimum(a + 8, e.G); bx = tuple(slice(x, y) for x, y in zip(a, b))
        z1h = (zt + (1 - t) * v)[0] * T(sd) + T(mu); grid = T(e.code[(slice(None),) + bx]).clone()
        inb = np.all((e.idx >= a) & (e.idx < b), 1); loc = e.idx[inb] - a
        g2 = grid.clone(); g2[:, loc[:, 0], loc[:, 1], loc[:, 2]] = z1h[T(np.flatnonzero(inb), torch.long)].T
        with torch.autocast('cuda', dtype=AMP): P = torch.softmax(ae.decode(g2[None]).float(), 1)
        vs = tuple(slice(x * F, y * F) for x, y in zip(a, b))
        Tt = shares(T(h['field'][vs])[None], (T(h['eye'][vs]) / 255)[None], T(h['subcut'][vs], torch.bool)[None], T(h['outside'][vs], torch.bool)[None]); K = T(~h['unk'][vs])[None, None]
        sl = shape_loss(P, Tt, K)
        if sl is not None:
            w = (loss.detach() / sl.detach().clamp(min=1e-6)) if AUTO else SHAPE   # auto (user, 2026-11-26): the shell part equals the flow loss at every step, no set weight
            loss = loss + w * sl
    return loss, (sl.item() if sl is not None else float('nan'))
sc = scaler(); gen = torch.Generator(device=dev); gen.manual_seed(SEED); hist = []; torch.cuda.reset_peak_memory_stats(); t_tr = time.time()
for it in range(STEPS):
    net.train(); c = sp['train'][rng.integers(len(sp['train']))]; t = float(rng.random())
    loss, sl = step_loss(c, t, gen); opt.zero_grad(); sc.scale(loss).backward(); sc.unscale_(opt); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); sc.step(opt); sc.update(); sched.step()
    with torch.no_grad():
        for pe_, pn in zip(ema.parameters(), net.parameters()): pe_.mul_(0.999).add_(pn, alpha=0.001)
    if it % max(1, STEPS // 20) == 0 or it == STEPS - 1: log('step', it, 'loss %.4f shape %.4f' % (loss.item(), sl)); hist.append((it, loss.item(), sl))
log(profile_line('flow_shape%s' % ('auto' if AUTO else '%g' % SHAPE), t_tr, STEPS)); torch.cuda.reset_peak_memory_stats(); t_te = time.time()
torch.save(dict(state_dict=ema.state_dict(), mu=mu, sd=sd, shape='auto' if AUTO else SHAPE, seed=SEED), os.path.join(OUT, 'flow.pt'))
# ---- test
@torch.no_grad()
def sample(e, steps=16, seed=0):
    g = torch.Generator(device=dev); g.manual_seed(seed); z = torch.randn((1, len(e.idx), C), device=dev, generator=g); rev = torch.zeros((1, len(e.idx)), dtype=torch.bool, device=dev); cd = cond(e)
    for i in range(steps):
        with torch.autocast('cuda', dtype=AMP): z = z + ema(z, torch.tensor(i / steps, device=dev), rev, **cd).float() / steps
    return z[0].cpu().numpy() * sd + mu
@torch.no_grad()
def decode_probs(e, codes):
    grid = e.code.copy(); grid[(slice(None),) + tuple(e.idx.T)] = codes.T; P = np.zeros((NC, *[g * F for g in e.G]), np.float16)
    for a, b, m in tiles(e.G, 8, 2):
        with torch.autocast('cuda', dtype=AMP): pr = torch.softmax(ae.decode(T(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]])).float(), 1)[0].cpu().numpy()
        P[:, (a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = pr[:, m[0] * F:, m[1] * F:, m[2] * F:]
    return P
def margin(Q, k):
    """tissue k minus the strongest other tissue: its zero level is the boundary of the label map, at any confidence"""
    o = np.delete(Q, k, axis=0).max(0); return Q[k].astype(np.float32) - o.astype(np.float32)
def surf(vol):
    from skimage import measure
    if vol.max() < 0 or vol.min() > 0: return np.zeros((0, 3))
    v, f, _, _ = measure.marching_cubes(vol.astype(np.float32), 0.0); t_ = v[f]; ar = 0.5 * np.linalg.norm(np.cross(t_[:, 1] - t_[:, 0], t_[:, 2] - t_[:, 0]), axis=1)
    r = np.random.default_rng(0); k = r.choice(len(f), 200000, p=ar / ar.sum()); u = r.random((200000, 2)); mm = u.sum(1) > 1; u[mm] = 1 - u[mm]
    return t_[k, 0] + u[:, :1] * (t_[k, 1] - t_[k, 0]) + u[:, 1:] * (t_[k, 2] - t_[k, 0])
res = {}
for c in sp['test']:
    e = data[c]; h = head(c); sl = tuple(slice(0, g * F) for g in e.G); known = ~h['unk'][sl]
    with torch.no_grad(): Tt = torch.cat([shares(T(h['field'][sl][:, :, k0:k0 + 64])[None], (T(h['eye'][sl][:, :, k0:k0 + 64]) / 255)[None], T(h['subcut'][sl][:, :, k0:k0 + 64], torch.bool)[None], T(h['outside'][sl][:, :, k0:k0 + 64], torch.bool)[None]).cpu() for k0 in range(0, e.G[2] * F, 64)], 4)[0].numpy()
    P = decode_probs(e, sample(e)); ap, at = P.argmax(0), Tt.argmax(0); r = {}
    for k in range(1, NC):
        a_, b_ = (ap == k) & known, (at == k) & known
        if b_.sum() < 50: continue
        sp_, st = surf(np.where(known, margin(P, k), -1.0)), surf(np.where(known, margin(Tt, k), -1.0))
        d = np.concatenate([cKDTree(st).query(sp_, workers=-1)[0], cKDTree(sp_).query(st, workers=-1)[0]]) if len(sp_) and len(st) else np.array([np.nan])
        npc = lambda m: int(sum(np.bincount(ndi.label(m)[0].ravel())[1:] >= 20))
        r[NAMES[k]] = dict(dice=round(float(2 * (a_ & b_).sum() / (a_.sum() + b_.sum())), 3), surface_median_mm=round(float(np.nanmedian(d)), 2), surface_p90_mm=round(float(np.nanpercentile(d, 90)), 2),
                          pieces_model=npc(a_), pieces_true=npc(b_))
    res[c] = r; log(c, json.dumps(r))
    np.savez_compressed(os.path.join(OUT, 'pred_%s.npz' % c), labels=ap.astype(np.uint8))
summ = {}
for k in NAMES[1:]:
    rows = [r[k] for r in res.values() if k in r]
    if rows: summ[k] = {m: round(float(np.median([x[m] for x in rows])), 3) for m in ('dice', 'surface_median_mm', 'surface_p90_mm')} | dict(pieces_error=round(float(np.median([abs(x['pieces_model'] - x['pieces_true']) for x in rows])), 1), heads=len(rows))
json.dump(dict(summary=summ, per_head=res, shape='auto' if AUTO else SHAPE, seed=SEED, steps=STEPS, kind=ck['kind'], hist=hist), open(os.path.join(OUT, 'field_flow.json'), 'w'), indent=1)
log(profile_line('test', t_te, max(1, len(sp['test'])))); log('SUMMARY', json.dumps(summ))
