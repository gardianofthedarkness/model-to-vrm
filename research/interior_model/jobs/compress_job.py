# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch"]
# ///
"""Milestone 13 (Compress): does a label autoencoder keep thin structures at coarse tokens?

Data: label volumes v1 of all adult CT scans (127 classes: 6 basic + 117 structures... ids up to 126), read at 3 mm
(every second voxel of the 1.5 mm grid), official split. Model: class embedding -> 3D conv encoder with F-fold
downsampling -> C code channels -> conv decoder -> class logits; cross-entropy with weights (training voxel share)^-1/2.
Variants: F = 4 (12 mm tokens) and F = 8 (24 mm tokens), C = 8. Trained on random 64^3 crops of training scans;
checkpoint by validation loss. Scored on whole test scans (padded to a multiple of 16): Dice per class, summarised by
group and by the structure's own thickness (twice its mean inside distance, mm), and subcutaneous-fat thickness error
at the skin. Tokens for a 75-litre body are reported for each F. args: <repo> [limit scans] [steps]"""
import os, sys, csv, json, time, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
NC = 127
def load(path):
    L = np.load(path)['labels'][::2, ::2, ::2]; return np.ascontiguousarray(L)

def make_ae(F, C, emb=16, w=48):
    import torch, torch.nn as nn
    def blk(a, b, s=1): return nn.Sequential(nn.Conv3d(a, b, 3, stride=s, padding=1), nn.GroupNorm(8, b), nn.SiLU(), nn.Conv3d(b, b, 3, padding=1), nn.GroupNorm(8, b), nn.SiLU())
    n = int(np.log2(F)); ch = [w * 2 ** i for i in range(n + 1)]
    class AE(nn.Module):
        def __init__(s):
            super().__init__(); s.emb = nn.Embedding(NC, emb); s.inp = blk(emb, ch[0])
            s.down = nn.ModuleList([blk(ch[i], ch[i + 1], 2) for i in range(n)]); s.to_code = nn.Conv3d(ch[n], C, 1); s.from_code = blk(C, ch[n])
            s.up = nn.ModuleList([nn.Sequential(nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False), blk(ch[i + 1], ch[i])) for i in reversed(range(n))]); s.out = nn.Conv3d(ch[0], NC, 1)
        def encode(s, x):
            h = s.inp(s.emb(x).permute(0, 4, 1, 2, 3))
            for d in s.down: h = d(h)
            return s.to_code(h)
        def decode(s, z):
            h = s.from_code(z)
            for u in s.up: h = u(h)
            return s.out(h)
        def forward(s, x): return s.decode(s.encode(x))
    return AE()

def thickness_mm(mask, sp=3.0):
    from scipy.ndimage import distance_transform_edt
    return float(2 * distance_transform_edt(mask, sampling=sp)[mask].mean()) if mask.any() else None

def evaluate(net, vols, dev, names, sp=3.0):
    import torch
    from scipy.ndimage import distance_transform_edt, binary_erosion
    per = {}; fat_err = []
    for L in vols:
        pad = [(0, (-n) % 16) for n in L.shape]; Lp = np.pad(L, pad)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'):
            P = net(torch.tensor(Lp[None].astype(np.int64), device=dev)).float().argmax(1)[0].cpu().numpy().astype(np.uint8)
        P = P[:L.shape[0], :L.shape[1], :L.shape[2]]
        for k in np.unique(L):
            if k == 0: continue
            a, b = P == k, L == k
            if b.sum() < 30: continue
            per.setdefault(int(k), []).append((float(2 * (a & b).sum() / (a.sum() + b.sum())), thickness_mm(b)))
        M = L > 0; skin = M & ~binary_erosion(M); deep = binary_erosion(M, iterations=2)
        tp, tt = M & (P != 1) & deep, M & (L != 1) & deep
        if tp.any() and tt.any() and (L == 1).sum() > 100: fat_err.append(float(np.abs(distance_transform_edt(~tp, sampling=sp)[skin] - distance_transform_edt(~tt, sampling=sp)[skin]).mean()))
    rows = [dict(id=k, name=names.get(k, str(k)), dice=float(np.median([d for d, _ in v])), thickness_mm=float(np.median([t for _, t in v])), scans=len(v)) for k, v in per.items()]
    return rows, (float(np.median(fat_err)) if fat_err else None)

def summarise(rows, fat_err):
    d = np.array([r['dice'] for r in rows]); th = np.array([r['thickness_mm'] for r in rows]); nm = [r['name'] for r in rows]
    grp = lambda f: round(float(np.median([r['dice'] for r in rows if f(r['name'])])), 3) if any(f(r['name']) for r in rows) else None
    bins = [(0, 6), (6, 12), (12, 24), (24, 1e9)]
    return dict(classes=len(rows), median_dice=round(float(np.median(d)), 3), p10_dice=round(float(np.percentile(d, 10)), 3),
                by_thickness_mm={'%g-%s' % (a, ('%g' % b) if b < 1e8 else 'up'): dict(n=int(((th >= a) & (th < b)).sum()), median_dice=round(float(np.median(d[(th >= a) & (th < b)])), 3) if ((th >= a) & (th < b)).any() else None) for a, b in bins},
                groups=dict(ribs=grp(lambda n: n.startswith('rib_')), vertebrae=grp(lambda n: n.startswith('vertebrae_')), hips_femurs_sacrum=grp(lambda n: n.startswith(('hip_', 'femur_', 'sacrum'))), gluteal=grp(lambda n: n.startswith('gluteus_')),
                            subcut_fat=grp(lambda n: n == 'subcutaneous fat'), other_fat=grp(lambda n: n == 'other fat'), lean=grp(lambda n: n == 'other lean'), vessels=grp(lambda n: any(k in n for k in ('artery', 'vein', 'vena', 'aorta')))),
                worst=[dict(name=r['name'], dice=round(r['dice'], 3), thickness_mm=round(r['thickness_mm'], 1)) for r in sorted(rows, key=lambda r: r['dice'])[:8]], subcut_fat_thickness_err_mm=round(fat_err, 2) if fat_err is not None else None)

def run(train, val, test, names, dev, steps, log=print, variants=((4, 8), (8, 8))):
    import torch, copy
    share = np.zeros(NC)
    for L in train[:200]: share += np.bincount(L.ravel(), minlength=NC)
    share /= share.sum(); wts = torch.tensor(np.where(share > 0, share, share[share > 0].min()) ** -0.5, dtype=torch.float32); wts = (wts / wts[share > 0].mean()).to(dev)
    rng = np.random.default_rng(0); S = 64
    def crop(vs, n):
        out = np.zeros((n, S, S, S), np.int64)
        for i in range(n):
            L = vs[rng.integers(len(vs))]; o = [rng.integers(0, max(1, d - S + 1)) for d in L.shape]; c = L[o[0]:o[0] + S, o[1]:o[1] + S, o[2]:o[2] + S]; out[i, :c.shape[0], :c.shape[1], :c.shape[2]] = c
        return torch.tensor(out, device=dev)
    valb = [crop(val, 4) for _ in range(8)]; res = {}
    for F, C in variants:
        net = make_ae(F, C).to(dev); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); lossf = torch.nn.CrossEntropyLoss(weight=wts); best = (np.inf, None, 0); t1 = time.time()
        for it in range(steps):
            net.train(); x = crop(train, 4)
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'): loss = lossf(net(x).float(), x)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            if it % max(1, steps // 20) == 0 or it == steps - 1:
                net.eval()
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16, enabled=dev == 'cuda'): vl = float(np.mean([lossf(net(b).float(), b).item() for b in valb]))
                if vl < best[0]: best = (vl, copy.deepcopy(net.state_dict()), it)
                log('F=%d C=%d step %d train %.4f val %.4f' % (F, C, it, loss.item(), vl))
        net.load_state_dict(best[1]); net.eval(); rows, fe = evaluate(net, test, dev, names)
        tok = 75e6 / (3.0 * F) ** 3
        res['F%d_C%d' % (F, C)] = dict(token_mm=3 * F, code_channels=C, tokens_for_75_litre_body=int(tok), best_step=best[2], train_minutes=round((time.time() - t1) / 60, 1), params_m=round(sum(p.numel() for p in net.parameters()) / 1e6, 1), **summarise(rows, fe), per_class=rows)
        log('F=%d' % F, json.dumps({k: v for k, v in res['F%d_C%d' % (F, C)].items() if k != 'per_class'}))
    return res

if __name__ == '__main__':
    import torch
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0; steps = int(sys.argv[3]) if len(sys.argv) > 3 else 12000
    meta = {m['image_id']: m for m in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    snap = snapshot_download(repo, repo_type='dataset', allow_patterns=['labels_v1/*.npz', 'labels_v1/label_ids.json'] if not limit else ['labels_v1/s00*.npz', 'labels_v1/label_ids.json'], max_workers=16)
    info = json.load(open(os.path.join(snap, 'labels_v1', 'label_ids.json'))); names = {int(k): v for k, v in info['classes'].items()}; names.update({v: k for k, v in info['ids'].items()})
    files = sorted(f for f in os.listdir(os.path.join(snap, 'labels_v1')) if f.endswith('.npz')); cases = [f[:-4] for f in files if f[:-4] in meta]
    if limit: cases = cases[:limit]
    with ThreadPoolExecutor(8) as ex: vols = dict(zip(cases, ex.map(lambda c: load(os.path.join(snap, 'labels_v1', c + '.npz')), cases)))
    sp = {s: [vols[c] for c in cases if meta[c]['split'] == s] for s in ('train', 'val', 'test')}
    if limit and (not sp['val'] or not sp['test']): sp['val'] = sp['train'][-2:]; sp['test'] = sp['train'][-4:-2]; sp['train'] = sp['train'][:-4]
    log('scans', {s: len(v) for s, v in sp.items()}, 'voxels', int(sum(v.size for v in vols.values())))
    res = run(sp['train'], sp['val'], sp['test'], names, 'cuda', steps, log); res['gpu'] = torch.cuda.get_device_name(0); res['scans'] = {s: len(v) for s, v in sp.items()}
    os.makedirs('out', exist_ok=True); tag = 'smoke' if limit else 'full'; json.dump(res, open('out/compress_%s.json' % tag, 'w'), indent=1)
    api.upload_folder(folder_path='out', path_in_repo='compress/%s' % tag, repo_id=repo, repo_type='dataset', commit_message='Compress %s' % tag)
    log('SUMMARY ' + json.dumps({k: ({kk: vv for kk, vv in v.items() if kk != 'per_class'} if isinstance(v, dict) and 'per_class' in v else v) for k, v in res.items()}))
