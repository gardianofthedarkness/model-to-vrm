"""Preparation for the field experiment (field_exp.py): per head, the shell tokens and the autoencoder's codes.
Tokens: F mm cells (F from the autoencoder: 8 or 16) holding any known body voxel. Shell features as before: signed distance to the skin pooled 4^3
inside the cell, and the share of the cell inside the scan (64 + 1 numbers). Codes: the autoencoder's encoder on the head's
input (soft: tissue shares from the field; hard: the old class labels), in crop-sized tiles (the channel norm makes the
result tile-independent apart from the receptive field, so tiles overlap and keep their centres).
Output <out>/<case>.npz: idx (int16), feat, pos, code (C, G) float16. args: <fields dir> <autoencoder .pt> <out dir>"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_exp import AMP, scaler, load_head, hard_classes, make_targets, make_ae, splits, NC, CLASSES, log, profile_line
import time; t_p = time.time(); torch.cuda.reset_peak_memory_stats(); nh = 0
from transformer_smoke import sdf, tiles
root, aef, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); dev = 'cuda'
idx = json.load(open(os.path.join(root, 'index.json'))); eye_id = idx.get('eye_id')
c = torch.load(aef, map_location=dev, weights_only=False); F = int(c.get('F', 8))
ae = make_ae(F=F, soft=(c['kind'] == 'soft'), arch=c.get('arch', 'conv')).to(dev); ae.load_state_dict(c['state_dict']); ae.eval(); kind = c['kind']
SH = [int(x) for x in os.environ.get('SHARD', '0/1').split('/')]
HAIR = int(os.environ.get('HAIR', '0')); train = set(splits(idx)['train'])              # HAIR: augmented shells per training head (game.hair_shell)
def tokens_feats(S, body, known, G):
    """shell tokens of one shell: cells holding body, signed distance pooled 4^3 per cell, share of the cell known"""
    D = sdf(S, 1.0); blk = lambda A: A.reshape(G[0], F, G[1], F, G[2], F).transpose(0, 2, 4, 1, 3, 5)
    tok = np.argwhere(blk(body).reshape(*G, -1).any(-1)); p = F // 4
    Dp = blk(D).reshape(*G, 4, p, 4, p, 4, p).mean((4, 6, 8))
    feat = np.concatenate([Dp[tuple(tok.T)].reshape(len(tok), -1), blk(known).reshape(*G, -1).mean(-1)[tuple(tok.T)][:, None]], 1).astype(np.float32)
    return tok, feat
for case in sorted(idx['heads'])[SH[0]::SH[1]]:
    h = load_head(os.path.join(root, case + '.npz')); outside = h['hard'] == 0; G = [n // F for n in h['shape']]; sl = tuple(slice(0, g * F) for g in G)
    T = lambda a: torch.tensor(np.ascontiguousarray(a[sl]), device=dev)[None]
    with torch.no_grad():
        if kind == 'soft':
            CH = 8 * F; get = lambda k, k0: T(h[k] if k != 'outside' else outside)[..., k0:k0 + CH]
            x = torch.cat([make_targets(lambda k, k0=k0: get(k, k0)) for k0 in range(0, G[2] * F, CH)], 4)
        else: x = torch.tensor(np.where(np.ascontiguousarray(hard_classes(h['hard'], eye_id)[sl]) == 255, NC, hard_classes(h['hard'], eye_id)[sl]).astype(np.int64), device=dev)[None]
        code = torch.zeros((8, *G), device=dev)
        for a, b, m in tiles(G, 8, 2):
            xs = x[(..., slice(a[0] * F, b[0] * F), slice(a[1] * F, b[1] * F), slice(a[2] * F, b[2] * F))]
            with torch.autocast('cuda', dtype=AMP): zz = ae.encode(xs).float()[0]
            code[:, a[0] + m[0]:b[0], a[1] + m[1]:b[1], a[2] + m[2]:b[2]] = zz[:, m[0]:, m[1]:, m[2]:]
    unk = h['unk'][sl]; out_ = outside[sl]; body = ~out_ & ~unk; known = (~unk).astype(np.float32)
    if CLASSES == 'game': S = np.where(out_ | unk, 0, 3).astype(np.uint8)                # as F1: unknown is not body; the known share tells the model
    else: S = np.where(out_ | (unk & (h['field'][sl] <= -500)), 0, 3).astype(np.uint8)   # unknown air-level voxels (SynthRAD's masked face) count as outside; unknown tissue (metal streaks) as body
    tok, feat = tokens_feats(S, body, known, G); codef = code.cpu().numpy().astype(np.float16)
    np.savez_compressed(os.path.join(out, case + '.npz'), idx=tok.astype(np.int16), feat=feat, pos=((tok + 0.5) * F / 100.0).astype(np.float32), code=codef)
    log(case, kind, 'tokens', len(tok)); nh += 1
    if HAIR and case in train:                                                          # hair / cloth: extra shell, interior unchanged (those cells must come out as outside)
        from game import hair_shell
        import zlib; rng = np.random.default_rng(zlib.crc32(case.encode()))
        for k in range(1, HAIR + 1):
            extra = hair_shell(out_, unk, rng); S2 = S.copy(); S2[extra] = 3
            tok2, feat2 = tokens_feats(S2, body | extra, known, G)
            np.savez_compressed(os.path.join(out, '%s~hair%d.npz' % (case, k)), idx=tok2.astype(np.int16), feat=feat2, pos=((tok2 + 0.5) * F / 100.0).astype(np.float32), code=codef)
            log(case, 'hair', k, 'extra ml', round(extra.sum() / 1000, 1), 'tokens', len(tok2))
log(profile_line('prep_per_head', t_p, nh))
json.dump(dict(autoencoder=os.path.basename(aef), kind=kind, F=F, hair=HAIR), open(os.path.join(out, 'prep_shard_%d.json' % SH[0]), 'w'))
