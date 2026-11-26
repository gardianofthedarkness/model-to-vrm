"""Experiment (user, 2026-11-26): effect of the clean head input (continuous CT field, tissue + eye only, unknowns) and of
the shell/topology loss. Shared pieces for the three arms:
  B0  old-style hard labels (head_fields_v1 'hard') -> hard autoencoder -> flow
  N   new soft targets (tissue shares per voxel from the field)  -> soft autoencoder -> flow
  NS  as N, flow with shell/topology terms on the decoded estimate
Seven classes: 0 outside, 1 fat under the skin, 2 other fat, 3 lean, 4 air, 5 bone, 6 eye. Unknown voxels: ignored.
This module: loading a head, tissue shares from the field (sub-voxel sampling), the autoencoder (tile-independent channel
norm, soft or class input), and its training. CLI: train an autoencoder.  args: <fields dir> <steps> <hard|soft> <out dir>"""
import os, sys, json, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_compress import ChanNorm
NC = 7; NAMES = ['outside', 'fat_under_skin', 'fat_other', 'lean', 'air', 'bone', 'eye']; UNK = 255
# autocast type: bfloat16 where the card has it (Ampere and newer), else float16 with a gradient scaler (T4, user 2026-11-26)
AMP = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8) else torch.float16
def scaler(): return torch.amp.GradScaler('cuda', enabled=(AMP == torch.float16))
def log(*a): print('TS %6.0fs' % (time.time() - T0), *a, flush=True)
T0 = time.time()

def profile_line(tag, t_start, steps):
    """seconds per step, peak VRAM and peak RAM of this process (for packing runs onto cards)"""
    import resource
    try: rss = int([l for l in open('/proc/self/status') if l.startswith('VmHWM')][0].split()[1]) / 1e6
    except Exception: rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    return 'PROFILE %s s_per_step %.3f peak_vram_gb %.2f peak_ram_gb %.2f' % (tag, (time.time() - t_start) / max(1, steps), torch.cuda.max_memory_allocated() / 1e9, rss)

def splits(idx, have=lambda c: True):
    """train / val / test lists. TEST_SOURCES (default hanseg,pddca: full faces) give val and test by their split; a source in
    TRAIN_SOURCES but not a test source (e.g. synthrad, whose faces are masked) trains with all its heads."""
    TS = os.environ.get('TRAIN_SOURCES', 'hanseg,pddca').split(','); ES = os.environ.get('TEST_SOURCES', 'hanseg,pddca').split(',')
    sp = {'train': [], 'val': [], 'test': []}
    for c, v in sorted(idx['heads'].items()):
        if not have(c): continue
        if v['source'] in ES:
            if v['split'] != 'train' or v['source'] in TS: sp[v['split']].append(c)
        elif v['source'] in TS: sp['train'].append(c)
    return sp

def load_head(path):
    z = np.load(path); shape = tuple(z['shape']); n = int(np.prod(shape))
    unpack = lambda k: np.unpackbits(z[k])[:n].reshape(shape).astype(bool)
    return dict(field=z['field'].astype(np.float32), eye=z['eye'].astype(np.float32) / 255.0, subcut=unpack('subcut'), unk=unpack('unk'), hard=z['hard'], shape=shape)

def hard_classes(hard, eye_id):
    """old labels -> the seven classes (unknown 255 kept)"""
    out = np.full(hard.shape, UNK, np.uint8); m = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, eye_id: 6}
    for k, v in m.items(): out[hard == k] = v
    return out

CACHE_KEYS = ('field', 'eye', 'subcut', 'unk', 'outside', 'cls')
def cached_head(root, c, eye_id):
    """a head's arrays memory-mapped from a decompressed cache on disk (HEADCACHE, default <fields>/../hc), written once and
    shared by every process on the card, so RAM stays small however many runs share it. field float16, eye 0-255, cls the
    old classes (hard_classes)."""
    import shutil
    d = os.path.join(os.environ.get('HEADCACHE') or os.path.join(os.path.dirname(os.path.abspath(root)), 'hc'), c)
    if not os.path.exists(os.path.join(d, 'done')):
        h = load_head(os.path.join(root, c + '.npz')); tmp = '%s.tmp%d' % (d, os.getpid()); os.makedirs(tmp, exist_ok=True)
        arrs = dict(field=h['field'].astype(np.float16), eye=np.round(h['eye'] * 255).astype(np.uint8), subcut=h['subcut'], unk=h['unk'], outside=h['hard'] == 0, cls=hard_classes(h['hard'], eye_id))
        for k in CACHE_KEYS: np.save(os.path.join(tmp, k + '.npy'), arrs[k])
        open(os.path.join(tmp, 'done'), 'w').close(); del h, arrs
        try: os.rename(tmp, d)
        except OSError: shutil.rmtree(tmp, ignore_errors=True)   # another process wrote it first
    return {k: np.load(os.path.join(d, k + '.npy'), mmap_mode='r') for k in CACHE_KEYS}

def shares(field, eye, subcut, outside, up=2):
    """tissue shares per 1 mm voxel (torch, (B, X, Y, Z) inputs -> (B, 7, X, Y, Z)): the field is sampled `up` times finer
    by trilinear interpolation and each sample is classified at the HU levels; eye share and outside override"""
    B = field.shape[0]; f = F_.interpolate(field[:, None], scale_factor=up, mode='trilinear', align_corners=False)[:, 0]
    cls = torch.stack([f < -500, (f >= -190) & (f <= -30), (f > -30) & (f <= 150), f > 150], 1).float()   # air, fat, lean, bone
    cls = torch.cat([cls, ((f >= -500) & (f < -190)).float()[:, None]], 1)                                # -500..-190: between air and fat, counted as fat
    cls = F_.avg_pool3d(cls, up)                                                                          # back to 1 mm: shares
    air, fat, lean, bone, low = cls[:, 0], cls[:, 1] + cls[:, 4], cls[:, 2], cls[:, 3], None
    sc = subcut[:, None].float(); e = eye[:, None]; inside = 1.0 - outside[:, None].float()
    out = torch.cat([outside[:, None].float(), fat[:, None] * sc, fat[:, None] * (1 - sc), lean[:, None], air[:, None], bone[:, None], torch.zeros_like(lean[:, None])], 1)
    out[:, 1:6] = out[:, 1:6] * inside * (1 - e); out[:, 6:7] = e * inside
    return out / out.sum(1, keepdim=True).clamp(min=1e-6)

def make_ae(F=8, C=8, w=32, soft=True):
    """the label autoencoder of face_compress.py with the channel norm; input: shares (soft) or class ids (hard)"""
    blk = lambda a, b, s=1: nn.Sequential(nn.Conv3d(a, b, 3, stride=s, padding=1), ChanNorm(b), nn.SiLU(), nn.Conv3d(b, b, 3, padding=1), ChanNorm(b), nn.SiLU())
    n = int(np.log2(F)); ch = [w * 2 ** i for i in range(n + 1)]
    class AE(nn.Module):
        def __init__(s):
            super().__init__(); s.soft = soft; s.emb = None if soft else nn.Embedding(NC + 1, 16)
            s.mix = nn.Linear(NC, 16, bias=False) if (soft and os.environ.get('SOFT_IN') == 'emb') else None   # soft shares through a learned 16-d tissue embedding (as the hard input)
            if s.mix is not None: nn.init.normal_(s.mix.weight)
            s.inp = blk(16 if (s.mix is not None or not soft) else NC, ch[0])
            s.down = nn.ModuleList([blk(ch[i], ch[i + 1], 2) for i in range(n)]); s.to_code = nn.Conv3d(ch[n], C, 1); s.from_code = blk(C, ch[n])
            s.up = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), blk(ch[i + 1], ch[i])) for i in reversed(range(n))]); s.out = nn.Conv3d(ch[0], NC, 1)
        def encode(s, x):
            h = s.inp((s.mix(x.movedim(1, -1)).movedim(-1, 1) if s.mix is not None else x) if s.soft else s.emb(x).permute(0, 4, 1, 2, 3))
            for d in s.down: h = d(h)
            return s.to_code(h)
        def decode(s, z):
            h = s.from_code(z)
            for u in s.up: h = u(h)
            return s.out(h)
        def forward(s, x): return s.decode(s.encode(x))
    return AE()

def soft_ce(logits, target, known, w):
    lp = F_.log_softmax(logits.float(), 1); l = -(target * lp * w[None, :, None, None, None]).sum(1)
    return (l * known).sum() / known.sum().clamp(min=1)

if __name__ == '__main__':
    root, steps, kind, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]; os.makedirs(out, exist_ok=True); dev = 'cuda'
    idx = json.load(open(os.path.join(root, 'index.json'))); eye_id = idx['eye_id']
    sp = splits(idx)
    log({s: len(v) for s, v in sp.items()}, kind)
    H = {c: cached_head(root, c, eye_id) for c in sp['train'] + sp['val']}                # memory-mapped, shared across processes
    KS = ('field', 'eye', 'subcut', 'outside', 'unk') if kind == 'soft' else ('outside', 'unk', 'cls')
    S = 64; rng = np.random.default_rng(0); BI = {c: np.argwhere(~h['outside'][::4, ::4, ::4] & ~h['unk'][::4, ::4, ::4]) * 4 for c, h in H.items()}
    def crop(cs, n):
        out = {k: [] for k in KS}
        for _ in range(n):
            c = cs[rng.integers(len(cs))]; h = H[c]; q = BI[c][rng.integers(len(BI[c]))]; o = [int(np.clip(a - S // 2, 0, max(0, d - S))) for a, d in zip(q, h['field'].shape)]
            sl = tuple(slice(a, a + S) for a in o)
            for k in out: out[k].append(np.pad(h[k][sl], [(0, S - x) for x in h[k][sl].shape], constant_values={'field': -1000, 'eye': 0, 'subcut': 0, 'outside': 1, 'unk': 1, 'cls': UNK}[k]))
        return {k: torch.tensor(np.stack(v), device=dev) for k, v in out.items()}
    # class weights from training shares (rarer tissue, larger weight)
    cnt = np.bincount(np.concatenate([np.asarray(H[c]['cls'][::4, ::4, ::4]).ravel() for c in sp['train']]), minlength=256)[:NC].astype(float); share = cnt / cnt.sum()
    w = torch.tensor(np.where(share > 0, share, share[share > 0].min()) ** -0.5, dtype=torch.float32, device=dev); w = w / w.mean(); log('class shares', np.round(share, 4).tolist())
    net = make_ae(soft=(kind == 'soft')).to(dev); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    def batch_targets(b):
        known = (~b['unk']).float()
        if kind == 'soft': tgt = shares(b['field'].float(), b['eye'].float() / 255, b['subcut'], b['outside']); x = tgt
        else: c = b['cls'].long(); tgt = F_.one_hot(c.clamp(max=NC - 1), NC).permute(0, 4, 1, 2, 3).float(); known = known * (c != UNK).float(); x = c.clamp(max=NC)
        return x, tgt, known
    val = [crop(sp['val'], 2) for _ in range(6)]; best = (1e9, None, 0); torch.cuda.reset_peak_memory_stats(); t_tr = time.time()
    sc = scaler()
    for it in range(steps):
        net.train(); b = crop(sp['train'], 2); x, tgt, known = batch_targets(b)
        with torch.autocast('cuda', dtype=AMP): lg = net(x)
        loss = soft_ce(lg, tgt, known, w); opt.zero_grad(); sc.scale(loss).backward(); sc.step(opt); sc.update(); sched.step()
        if it % max(1, steps // 20) == 0 or it == steps - 1:
            net.eval()
            with torch.no_grad(), torch.autocast('cuda', dtype=AMP): vl = float(np.mean([soft_ce(net(batch_targets(v)[0]), *batch_targets(v)[1:], w).item() for v in val]))
            if vl < best[0]: best = (vl, {k: v.detach().clone() for k, v in net.state_dict().items()}, it)
            log('step', it, 'train %.4f val %.4f' % (loss.item(), vl))
    log(profile_line('ae_' + kind, t_tr, steps))
    torch.save(dict(state_dict=best[1], kind=kind, soft_in=os.environ.get('SOFT_IN'), F=8, C=8, classes=NC, names=NAMES, best_step=best[2]), os.path.join(out, 'ae_%s.pt' % kind)); log('SUMMARY saved', kind, 'best step', best[2])
