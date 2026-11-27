"""Check 0 before S1.1a (user agreed 2026-11-27): is the gap between the autoencoder's rebuild and the generated heads
caused by codes the decoder reads badly (off the code set it was trained on: a decoder fine-tune can fix it), or by the
shell not deciding the interior (the generator draws another plausible interior: no decoder can fix it)?

Per head (validation heads, plus some training heads for the seen/unseen gap), with the frozen autoencoder and generator:
  ceiling      Dice(decode(true codes), truth)                      what the autoencoder can do
  generated    Dice(decode(generated codes), truth), seeds 0 and 1  what the model does
  draw spread  Dice(decode(gen seed 0), decode(gen seed 1))         two draws against each other
  re-encoded   Dice(decode(gen), decode(encode(decode(gen))))       does projecting onto the decoder's own output change it
and per token, in normalised code units:
  knn true     distance of the head's TRUE codes to the nearest training-head code (tokens of other heads)
  knn gen      the same for the GENERATED codes
  round trip   |encode(decode(z)) - z| for true and for generated codes
Reading: knn gen and round trip gen far above their true-code values = off-target codes (S1.1a can help).
Draw spread close to generated = the shell leaves the interior open (S1.1a cannot help much).
The prep dir may hold only some training heads (the job prepares N_REF of them): they form the reference code set.
args: <fields dir> <prep dir> <autoencoder .pt> <flow.pt> <out json>  env: N_TRAIN (training heads scored, default 8)"""
import os, sys, json, time, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_exp import AMP, splits, make_ae, NC, NAMES, CLASSES, log, cached_head
from flow_f1 import Flow
from transformer_smoke import tiles
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
assert CLASSES == 'game', 'set CLASSES=game'
root, prep, aef, flowf, outf = sys.argv[1:6]; dev = 'cuda'; C = 8
idx = json.load(open(os.path.join(root, 'index.json'))); sp = splits(idx, lambda c: os.path.exists(os.path.join(prep, c + '.npz')))
ck = torch.load(aef, map_location=dev, weights_only=False); F = int(ck['F'])
ae = make_ae(F=F, soft=True, arch=ck.get('arch', 'tvae')).to(dev); ae.load_state_dict(ck['state_dict']); ae.eval()
fl = torch.load(flowf, map_location=dev, weights_only=False); mu, sd = fl['mu'].astype(np.float32), fl['sd'].astype(np.float32)
net = Flow(65, C, NC, 384, 8).to(dev); net.load_state_dict(fl['state_dict']); net.eval()
T = lambda a, dt=torch.float32: torch.tensor(np.asarray(a), device=dev, dtype=dt)
class Ent: pass
def load(c):
    z = np.load(os.path.join(prep, c + '.npz')); e = Ent(); e.c = c; e.idx = z['idx'].astype(np.int64); e.feat = z['feat']; e.pos = z['pos']; e.code = z['code'].astype(np.float32); e.G = e.code.shape[1:]
    occ = np.zeros(e.G, bool); occ[tuple(e.idx.T)] = True; e.shell = ~ndi.minimum_filter(occ, size=3, mode='constant', cval=0)[tuple(e.idx.T)]
    e.raw = e.code[(slice(None),) + tuple(e.idx.T)].T.copy(); return e
def cond(e):
    n = len(e.idx); return dict(pos=T(e.pos)[None], part=T(np.ones(n), torch.long)[None], sfeat=T(e.feat[e.shell])[None], spos=T(e.pos[e.shell])[None], spart=T(np.ones(int(e.shell.sum())), torch.long)[None],
                                age=T([[0.0]]), agek=T([[0.0]]), sex=T([0], torch.long), cls=T(np.ones((1, NC), np.float32)), cpart=T([1], torch.long))
@torch.no_grad()
def sample(e, seed, steps=16):
    g = torch.Generator(device=dev); g.manual_seed(seed); z = torch.randn((1, len(e.idx), C), device=dev, generator=g); rev = torch.zeros((1, len(e.idx)), dtype=torch.bool, device=dev); cd = cond(e)
    for i in range(steps):
        with torch.autocast('cuda', dtype=AMP): z = z + net(z, torch.tensor(i / steps, device=dev), rev, **cd).float() / steps
    return z[0].cpu().numpy() * sd + mu
@torch.no_grad()
def decode(e, codes):
    grid = e.code.copy(); grid[(slice(None),) + tuple(e.idx.T)] = codes.T; P = np.zeros((NC, *[g * F for g in e.G]), np.float16)
    for a, b, m in tiles(e.G, 8, 2):
        with torch.autocast('cuda', dtype=AMP): pr = torch.softmax(ae.decode(T(grid[None, :, a[0]:b[0], a[1]:b[1], a[2]:b[2]])).float(), 1)[0].cpu().numpy()
        P[:, (a[0] + m[0]) * F:b[0] * F, (a[1] + m[1]) * F:b[1] * F, (a[2] + m[2]) * F:b[2] * F] = pr[:, m[0] * F:, m[1] * F:, m[2] * F:]
    return P
@torch.no_grad()
def encode(e, P):
    code = np.zeros((C, *e.G), np.float32)
    for a, b, m in tiles(e.G, 8, 2):
        with torch.autocast('cuda', dtype=AMP): z = ae.encode(T(P[None, :, a[0] * F:b[0] * F, a[1] * F:b[1] * F, a[2] * F:b[2] * F].astype(np.float32))).float()[0].cpu().numpy()
        code[:, a[0] + m[0]:b[0], a[1] + m[1]:b[1], a[2] + m[2]:b[2]] = z[:, m[0]:, m[1]:, m[2]:]
    return code[(slice(None),) + tuple(e.idx.T)].T
def dice(a, b, known):
    r = {}
    for k in range(1, NC):
        x, y = (a == k) & known, (b == k) & known
        if y.sum() >= 50 or x.sum() >= 50: r[NAMES[k]] = float(2 * (x & y).sum() / max(1, x.sum() + y.sum()))
    return r
n_train = int(os.environ.get('N_TRAIN', '8')); rng = np.random.default_rng(0)
tr_all = [c for c in sp['train'] if '~' not in c]; tr_score = [tr_all[i] for i in sorted(rng.choice(len(tr_all), min(n_train, len(tr_all)), replace=False))]
log('heads: train', len(tr_all), 'val', len(sp['val']), 'train scored', len(tr_score))
# reference code set: true codes of all training heads (normalised)
raws = [load(c).raw for c in tr_all]; ref = np.concatenate([(r_ - mu) / sd for r_ in raws]); owner = np.concatenate([np.full(len(r_), i) for i, r_ in enumerate(raws)]); del raws
tree = cKDTree(ref); log('reference tokens', len(ref))
def knn(Z, own=None):
    """distance to the nearest training token, excluding the head's own tokens (k neighbours, first foreign one)"""
    d, j = tree.query((Z - mu) / sd, k=64, workers=-1)
    if own is None: return d[:, 0]
    foreign = owner[j] != own; first = foreign.argmax(1); return np.where(foreign.any(1), d[np.arange(len(d)), first], np.nan)
res = {}; t0 = time.time()
for group, cases in (('val', sp['val']), ('train', tr_score)):
    for c in cases:
        e = load(c); h = cached_head(root, c, None); sl = tuple(slice(0, g * F) for g in e.G)
        truth = np.asarray(h['cls'][sl]); known = ~np.asarray(h['unk'][sl]); body = known & (truth != 0)
        own = tr_all.index(c) if c in tr_all else None
        Pt = decode(e, e.raw); g0, g1 = sample(e, 0), sample(e, 1); P0, P1 = decode(e, g0), decode(e, g1)
        z_rt_t = encode(e, Pt); z_rt_g = encode(e, P0); Pr = decode(e, z_rt_g)
        A = lambda P: np.where(body, P.argmax(0), 0)
        at, a0, a1, ar = A(Pt), A(P0), A(P1), A(Pr)
        nrm = lambda x: np.linalg.norm((x) / sd, axis=1)
        r = dict(group=group, ceiling=dice(at, truth, known), generated=dice(a0, truth, known), generated_s1=dice(a1, truth, known),
                 spread=dice(a0, a1, known), reencoded=dice(ar, a0, known),
                 knn_true=float(np.nanmedian(knn(e.raw, own))), knn_gen=float(np.nanmedian(knn(g0, own))),
                 roundtrip_true=float(np.median(nrm(z_rt_t - e.raw))), roundtrip_gen=float(np.median(nrm(z_rt_g - g0))),
                 gen_to_true_code=float(np.median(nrm(g0 - e.raw))), tokens=len(e.idx))
        res[c] = r; log(group, c, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, dict)}),
                       'skull ceil/gen/spread/reenc', *[round(r[m].get('skull', float('nan')), 3) for m in ('ceiling', 'generated', 'spread', 'reencoded')])
summ = {}
for group in ('val', 'train'):
    rows = [r for r in res.values() if r['group'] == group]
    if not rows: continue
    s = {m: float(np.median([r[m] for r in rows])) for m in ('knn_true', 'knn_gen', 'roundtrip_true', 'roundtrip_gen', 'gen_to_true_code')}
    for m in ('ceiling', 'generated', 'generated_s1', 'spread', 'reencoded'):
        s[m] = {k: round(float(np.median([r[m][k] for r in rows if k in r[m]])), 3) for k in NAMES[1:] if any(k in r[m] for r in rows)}
    s['heads'] = len(rows); summ[group] = s
json.dump(dict(summary=summ, per_head=res, autoencoder=aef, flow=flowf), open(outf, 'w'), indent=1)
log('SUMMARY', json.dumps(summ)); log('done in %.1f min' % ((time.time() - t0) / 60))
