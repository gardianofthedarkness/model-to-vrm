"""Preparation v3 for the flow model (plan v8, tests P2 to P4): every person in the shell frame, with jittered and covered copies.

Per person and part (face: head frame window at 1 mm, 8 mm tokens; pelvis: 3 mm, 12 mm tokens):
1. Shell frame (P2). Skin points (body next to outside) are fitted to the population's mean shell by a rigid transform
   (shell_align.fit, scale fixed: CT is in mm); the labels are resampled into that frame. A character is placed the same
   way at inference, with its size also fitted.
2. Copies.  <c>      the person in the shell frame
            <c>~j    the same, moved by a small random rigid transform (P3); its spread comes from the measured recovery
                     error of the shell fit (align/<part>.json, case 'all three', 90th percentile), not a set number
            <c>~h    covered (P4): a character's hair or loose cloth (cover_<part>.npz, in the same frame) laid over the
                     shell. The shell the model reads includes the cover; the target labels say 'outside' there.
            <c>~m    face only, training only: mirrored about the frame's midplane with left/right labels swapped
3. Per copy: body tokens (cells holding shell-body voxels), shell features (signed distance pooled to 4^3, share of the
   cell inside the scan, share of the cell's outer surface that is hair or cloth), positions, the autoencoder's codes of the TRUE labels,
   the true labels (for scoring) and the cover mask.
Output: <root>/<OUT>/<key>_<part>.npz.  env: OUT (default prep_v3_TF8), FACE_AE (default face_compress_v3/autoencoder_t8.pt),
SHARD i/n, PARTS (comma list, default all).  args: <interior_data root> <align dir> <covers dir>"""
import os, sys, json, time, numpy as np, torch
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_compress import make_ae
from transformer_smoke import encode, sdf, UNK, DEV
from transformer_v2 import mirror, SIDE
from shell_align import MeanShell, fit, skin_points
from tissue_map import remap
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)

def reframe(L, mm, lo, Q, b, cval=UNK):
    """labels in the new frame y = Q p + b (p: old frame mm): L'(y) = L(Q^T (y - b)), nearest neighbour."""
    sh = L.shape; ax = [lo[i] + mm * (np.arange(sh[i]) + 0.5) for i in range(3)]; Y = np.stack(np.meshgrid(*ax, indexing='ij'), -1).reshape(-1, 3)
    p = (Y - b) @ Q; q = (p - lo) / mm - 0.5
    return ndi.map_coordinates(L, q.T, order=0, mode='constant', cval=cval).reshape(sh).astype(np.uint8)

def tokens(L, cover, mm, F, ae, NC, fill):
    """shell = body plus cover; target codes from the true labels."""
    G = [n // F for n in L.shape]; L = np.ascontiguousarray(L[:G[0] * F, :G[1] * F, :G[2] * F]); cover = cover[:G[0] * F, :G[1] * F, :G[2] * F]
    body = (L > 0) & (L < UNK); shell = body | cover; S = np.where(shell, fill, np.where(L == UNK, UNK, 0)).astype(np.uint8)
    D = sdf(S, mm); known = (S != UNK).astype(np.float32)
    blk = lambda A: A.reshape(G[0], F, G[1], F, G[2], F).transpose(0, 2, 4, 1, 3, 5)
    idx = np.argwhere(blk(shell).reshape(*G, -1).any(-1)); p = F // 4 if F >= 4 else 1
    Dp = blk(D).reshape(*G, 4, p, 4, p, 4, p).mean((4, 6, 8)) if p > 1 else blk(D)
    surf = shell & ndi.binary_dilation(~shell & (S != UNK))                                  # outer surface of the shell (not the scan's cut faces)
    cs = blk(surf & cover).reshape(*G, -1).sum(-1) / np.maximum(blk(surf).reshape(*G, -1).sum(-1), 1)   # share of the cell's outer surface that is hair or cloth: what a
                                                                                               # character gives (surface vertices marked loose); under its hair there is no skin surface
    feat = np.concatenate([Dp[tuple(idx.T)].reshape(len(idx), -1), blk(known).reshape(*G, -1).mean(-1)[tuple(idx.T)][:, None], cs[tuple(idx.T)][:, None]], 1).astype(np.float32)
    pos = ((idx + 0.5) * F * mm / 100.0).astype(np.float32)
    code = encode(ae, L, NC, F)
    return idx, feat, pos, code, L, cover

if __name__ == '__main__':
    root, alignd, coverd = sys.argv[1:4]; OUT = os.path.join(root, os.environ.get('OUT', 'prep_v3_TF8')); os.makedirs(OUT, exist_ok=True)
    SH = [int(x) for x in os.environ.get('SHARD', '0/1').split('/')]; rng = np.random.default_rng(1000 + SH[0])
    fi = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pi = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
    hs = sorted(c for c, v in fi['heads'].items() if v['source'] == 'hanseg'); split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
    split.update({c: v['split'] for c, v in fi['heads'].items() if v['source'] == 'ts'}); split.update({c: v['split'] for c, v in pi['people'].items()})
    ids = fi['ids']; swap = np.arange(256, dtype=np.uint8)
    for n, k in ids.items():
        m = SIDE.match(n)
        if m:
            o = {'left': 'right', 'right': 'left', 'L': 'R', 'R': 'L'}[m.group(2)]; q = '%s_%s' % (m.group(1), o)
            if q in ids: swap[k] = ids[q]
    PARTS = {}
    for part, folder, members, mm, lo, NC, ck, fill in (('face', 'face_v1_headframe', fi['heads'], 1.0, np.array(fi['window_lo_mm']), max(ids.values()) + 1, os.environ.get('FACE_AE', 'face_compress_v3/autoencoder_t8.pt'), 3),
                                                       ('pelvis', 'pelvis_v1', pi['people'], 3.0, np.array(pi['window_lo_mm']), len(pi['classes']), 'pelvis_compress/autoencoder_t12.pt', 5)):
        c = torch.load(os.path.join(root, ck), map_location=DEV); ae = make_ae(NC, c['F'], c['C']).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval()
        al = json.load(open(os.path.join(alignd, '%s.json' % part)))['recovery']['all three']; cv = np.load(os.path.join(coverd, 'cover_%s.npz' % part))
        PARTS[part] = dict(folder=folder, members=members, mm=mm, lo=lo, NC=NC, ae=ae, F=c['F'], fill=fill, mean=MeanShell.load(os.path.join(alignd, '%s_mean.npz' % part)),
                           jit_mm=float(al['p90_mm']), cover=cv['cover'])
        log(part, 'jitter scale (mm, from the shell-fit recovery error)', PARTS[part]['jit_mm'], 'cover voxels', int(cv['cover'].sum()))
    only = [x for x in os.environ.get('PARTS', '').split(',') if x]                          # PARTS=face: prepare one part only
    work = [(part, c_) for part, P in PARTS.items() for c_ in sorted(P['members']) if (not only or part in only) and os.path.exists(os.path.join(root, P['folder'], c_ + '.npz'))]
    work = work[SH[0]::SH[1]]; log('shard', SH, len(work), 'part-people'); meta = {}
    for part, c_ in work:
        P = PARTS[part]; mm, lo, F = P['mm'], P['lo'], P['F']; t1 = time.time()
        try:
            L0 = np.load(os.path.join(root, P['folder'], c_ + '.npz'))['labels']
            pts = skin_points(L0, mm, lo, step=4 if part == 'face' else 1); x, ap = fit(pts, P['mean'], scale=False)
            Q = Rotation.from_rotvec(x[1:4]).as_matrix(); b = x[4:7]; L = reframe(L0, mm, lo, Q, b)
            copies = {c_: (L, np.zeros(L.shape, bool))}
            # jitter: rigid move with a spread equal to the measured fit error (shift) and that error over the part's radius (turn)
            body = (L > 0) & (L < UNK); rad = float(np.sqrt(((np.argwhere(body[::4, ::4, ::4]) * 4 * mm + lo - (np.argwhere(body[::4, ::4, ::4]) * 4 * mm + lo).mean(0)) ** 2).sum(1).mean()))
            rv = rng.normal(size=3) * (P['jit_mm'] / rad) / np.sqrt(3); bj = rng.normal(size=3) * P['jit_mm'] / np.sqrt(3); Qj = Rotation.from_rotvec(rv).as_matrix()
            copies[c_ + '~j'] = (reframe(L, mm, lo, Qj, bj), np.zeros(L.shape, bool))
            # cover: the template laid over this shell (randomly mirrored), cut to pieces that touch the body
            T = P['cover'][::-1] if rng.random() < 0.5 else P['cover']; T = T[:L.shape[0], :L.shape[1], :L.shape[2]]
            cov = T & ~body; lab, n = ndi.label(cov); touch = np.unique(lab[ndi.binary_dilation(body) & cov]); cov = np.isin(lab, touch[touch > 0])
            copies[c_ + '~h'] = (L, cov)
            if part == 'face' and split.get(c_) == 'train': copies[c_ + '~m'] = (mirror(L, lo[0], mm, swap), np.zeros(L.shape, bool))
            for key, (Lk, ck_) in copies.items():
                if part == 'face' and os.environ.get('TISSUE') == '1': Lk = remap(Lk, ids)          # tissue types, after mirroring (both eyes one class, brain and brainstem one organ class)
                idx, feat, pos, code, Lc, cc = tokens(Lk, ck_, mm, F, P['ae'], P['NC'], P['fill'])
                np.savez_compressed(os.path.join(OUT, '%s_%s.npz' % (key, part)), idx=idx.astype(np.int16), feat=feat, pos=pos, code=code.astype(np.float16), L=Lc, cover=np.packbits(cc))
            meta['%s_%s' % (c_, part)] = dict(split=split.get(c_), fit_shift_mm=round(float(np.linalg.norm(b)), 1), fit_turn_deg=round(float(np.degrees(np.linalg.norm(x[1:4]))), 2),
                                              cover_ml=round(float(cov.sum() * mm ** 3 / 1000)), copies=list(copies), seconds=round(time.time() - t1, 1))
            log(c_, part, meta['%s_%s' % (c_, part)])
        except Exception as e: meta['%s_%s' % (c_, part)] = 'failed: %s %s' % (type(e).__name__, str(e)[:160]); log(c_, part, meta['%s_%s' % (c_, part)])
    json.dump(meta, open(os.path.join(OUT, 'meta_shard_%d_of_%d.json' % tuple(SH)), 'w'), indent=1)
    log('SUMMARY', json.dumps(dict(shard=SH, done=sum(isinstance(v, dict) for v in meta.values()), failed=sum(isinstance(v, str) for v in meta.values()))))
