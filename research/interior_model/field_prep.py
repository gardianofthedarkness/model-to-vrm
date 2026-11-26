"""Preparation for the field experiment (field_exp.py): per head, the shell tokens and the autoencoder's codes.
Tokens: 8 mm cells (F = 8) holding any known body voxel. Shell features as before: signed distance to the skin pooled 4^3
inside the cell, and the share of the cell inside the scan (64 + 1 numbers). Codes: the autoencoder's encoder on the head's
input (soft: tissue shares from the field; hard: the old class labels), in crop-sized tiles (the channel norm makes the
result tile-independent apart from the receptive field, so tiles overlap and keep their centres).
Output <out>/<case>.npz: idx (int16), feat, pos, code (C, G) float16. args: <fields dir> <autoencoder .pt> <out dir>"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from field_exp import load_head, hard_classes, shares, make_ae, NC, log
from transformer_smoke import sdf, tiles
root, aef, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True); dev = 'cuda'; F = 8
idx = json.load(open(os.path.join(root, 'index.json'))); eye_id = idx['eye_id']
c = torch.load(aef, map_location=dev); ae = make_ae(soft=(c['kind'] == 'soft')).to(dev); ae.load_state_dict(c['state_dict']); ae.eval(); kind = c['kind']
SH = [int(x) for x in os.environ.get('SHARD', '0/1').split('/')]
for case in sorted(idx['heads'])[SH[0]::SH[1]]:
    h = load_head(os.path.join(root, case + '.npz')); outside = h['hard'] == 0; G = [n // F for n in h['shape']]; sl = tuple(slice(0, g * F) for g in G)
    fld, eye, sub, unk, out_ = [torch.tensor(np.ascontiguousarray(a[sl]), device=dev) for a in (h['field'], h['eye'], h['subcut'], h['unk'], outside)]
    with torch.no_grad():
        if kind == 'soft': x = torch.cat([shares(fld[None, :, :, k0:k0 + 64], eye[None, :, :, k0:k0 + 64], sub[None, :, :, k0:k0 + 64], out_[None, :, :, k0:k0 + 64]) for k0 in range(0, fld.shape[2], 64)], 4)
        else: x = torch.tensor(np.where(np.ascontiguousarray(hard_classes(h['hard'], eye_id)[sl]) == 255, NC, hard_classes(h['hard'], eye_id)[sl]).astype(np.int64), device=dev)[None]
        code = torch.zeros((8, *G), device=dev)
        for a, b, m in tiles(G, 8, 2):
            xs = x[(..., slice(a[0] * F, b[0] * F), slice(a[1] * F, b[1] * F), slice(a[2] * F, b[2] * F))]
            with torch.autocast('cuda', dtype=torch.bfloat16): z = ae.encode(xs).float()[0]
            code[:, a[0] + m[0]:b[0], a[1] + m[1]:b[1], a[2] + m[2]:b[2]] = z[:, m[0]:, m[1]:, m[2]:]
    body = (~outside[sl]) & (~h['unk'][sl]); S = np.where(outside[sl], 0, 3).astype(np.uint8); S[h['unk'][sl] & ~outside[sl]] = 3
    D = sdf(S, 1.0); known = (~h['unk'][sl]).astype(np.float32)
    blk = lambda A: A.reshape(G[0], F, G[1], F, G[2], F).transpose(0, 2, 4, 1, 3, 5)
    tok = np.argwhere(blk(body).reshape(*G, -1).any(-1)); p = F // 4
    Dp = blk(D).reshape(*G, 4, p, 4, p, 4, p).mean((4, 6, 8))
    feat = np.concatenate([Dp[tuple(tok.T)].reshape(len(tok), -1), blk(known).reshape(*G, -1).mean(-1)[tuple(tok.T)][:, None]], 1).astype(np.float32)
    np.savez_compressed(os.path.join(out, case + '.npz'), idx=tok.astype(np.int16), feat=feat, pos=((tok + 0.5) * F / 100.0).astype(np.float32), code=code.cpu().numpy().astype(np.float16))
    log(case, kind, 'tokens', len(tok))
json.dump(dict(autoencoder=os.path.basename(aef), kind=kind), open(os.path.join(out, 'prep_shard_%d.json' % SH[0]), 'w'))
