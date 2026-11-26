"""Stage gate (user, 2026-11-26: check the encoding before training a generator on it). Reconstruction of whole validation
heads by each autoencoder: encode and decode in 64-voxel blocks, then per tissue the overlap (Dice) and the median and
90th-percentile surface distance of the decoded label map against the truth, on known voxels. Rare tissues (eye, air)
are what fails first. Writes <out>.json and prints one line per autoencoder.
args: <fields dir> <out .json> <autoencoder .pt> [more .pt ...]   env: SPLIT (default val), GATE_HEADS (first n heads), SOFT_IN per file via name *emb*"""
import os, sys, json, numpy as np, torch
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_exp import AMP, scaler, load_head, hard_classes, shares, make_ae, splits, NAMES, NC, log

def surf_pts(m):
    from skimage import measure
    if m.sum() < 20 or m.all(): return np.zeros((0, 3))
    v, f, _, _ = measure.marching_cubes(m.astype(np.float32), 0.5); return v

@torch.no_grad()
def recon(ae, kind, h, eye_id, dev='cuda', S=64):
    out = h['hard'] == 0; G = [n // S * S for n in h['shape']]; P = np.zeros((NC, *G), np.float16)
    for a in range(0, G[0], S):
        for b in range(0, G[1], S):
            for c in range(0, G[2], S):
                sl = (slice(a, a + S), slice(b, b + S), slice(c, c + S)); T = lambda x, dt=torch.float32: torch.tensor(np.ascontiguousarray(x[sl]), device=dev, dtype=dt)[None]
                if kind == 'soft': x = shares(T(h['field']), T(h['eye']), T(h['subcut'], torch.bool), T(out, torch.bool))
                else: hc = hard_classes(h['hard'][sl], eye_id); x = torch.tensor(np.where(hc == 255, NC, hc).astype(np.int64), device=dev)[None]
                with torch.autocast('cuda', dtype=AMP): P[(slice(None),) + sl] = torch.softmax(ae(x).float(), 1)[0].cpu().numpy()
    sl = tuple(slice(0, g) for g in G)
    with torch.no_grad():
        tgt = torch.cat([shares(torch.tensor(h['field'][sl][:, :, k:k + S], device=dev)[None], torch.tensor(h['eye'][sl][:, :, k:k + S], device=dev)[None],
                                torch.tensor(h['subcut'][sl][:, :, k:k + S], device=dev)[None], torch.tensor(out[sl][:, :, k:k + S], device=dev)[None]).cpu() for k in range(0, G[2], S)], 4)[0].numpy()
    return P.argmax(0), tgt.argmax(0), ~h['unk'][sl]

if __name__ == '__main__':
    root, outj, aes = sys.argv[1], sys.argv[2], sys.argv[3:]
    idx = json.load(open(os.path.join(root, 'index.json'))); eye_id = idx['eye_id']; cases = splits(idx)[os.environ.get('SPLIT', 'val')]
    if os.environ.get('GATE_HEADS'): cases = cases[:int(os.environ['GATE_HEADS'])]
    log('gate on', len(cases), 'heads')
    res = {}
    for p in aes:
        ck = torch.load(p, map_location='cuda'); kind = ck['kind']
        if 'emb' in os.path.basename(os.path.dirname(p)) or ck.get('soft_in') == 'emb': os.environ['SOFT_IN'] = 'emb'
        else: os.environ.pop('SOFT_IN', None)
        ae = make_ae(soft=(kind == 'soft'), arch=ck.get('arch', 'conv')).cuda(); ae.load_state_dict(ck['state_dict']); ae.eval(); rows = {}
        for c in cases:
            h = load_head(os.path.join(root, c + '.npz')); a, b, kn = recon(ae, kind, h, eye_id)
            for k in range(1, NC):
                pa, pb = (a == k) & kn, (b == k) & kn
                if pb.sum() < 50: continue
                sa, sb = surf_pts(pa), surf_pts(pb)
                d = np.concatenate([cKDTree(sb).query(sa, workers=-1)[0], cKDTree(sa).query(sb, workers=-1)[0]]) if len(sa) and len(sb) else np.array([np.nan])
                rows.setdefault(NAMES[k], []).append((2 * (pa & pb).sum() / (pa.sum() + pb.sum()), np.nanmedian(d), np.nanpercentile(d, 90) if len(d) > 1 else np.nan))
        res[os.path.basename(os.path.dirname(p)) or p] = {k: dict(dice=round(float(np.median([r[0] for r in v])), 3), surface_median_mm=round(float(np.nanmedian([r[1] for r in v])), 2),
                                                             surface_p90_mm=round(float(np.nanmedian([r[2] for r in v])), 2), heads=len(v)) for k, v in rows.items()}
        log('GATE', os.path.dirname(p), json.dumps(res[os.path.basename(os.path.dirname(p)) or p]))
    json.dump(res, open(outj, 'w'), indent=1); log('SUMMARY gate written', outj)
