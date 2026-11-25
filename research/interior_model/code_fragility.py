"""How fragile is an autoencoder's code space? Decode a head's stored codes with Gaussian noise added (in units of each
channel's spread over the head) and score against the true labels. A generator never hits codes exactly, so a code space
where small errors wipe out small structures is hard to generate in, even if it reconstructs perfectly.
args: <autoencoder .pt> <face index.json> <tissue 0|1> <prep npz> [noise levels, default 0,0.1,0.2,0.4]"""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from face_compress import make_ae
from transformer_smoke import DEV, encode
from flow_sample import to_labels
aef, idxf, tissue, pf = sys.argv[1:5]; levels = [float(x) for x in (sys.argv[5] if len(sys.argv) > 5 else '0,0.1,0.2,0.4').split(',')]
ids = json.load(open(idxf))['ids']; NC = max(ids.values()) + 1; inv = {v: k for k, v in ids.items()}
c = torch.load(aef, map_location=DEV); ae = make_ae(NC, c['F'], c['C']).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval(); F = c['F']
z = np.load(pf); idx = z['idx'].astype(np.int64); L = z['L']; G = [n // F for n in L.shape]
code = z['code'].astype(np.float32); code = code[(slice(None),) + tuple(idx.T)].T if code.ndim == 4 else code
body = (L > 0) & (L < 255); base = encode(ae, np.where(body, 3, L).astype(np.uint8), NC, F); known = L != 255
sd = code.std(0, keepdims=True); rng = np.random.default_rng(0)
WATCH = ['5', 'eye_left', '4', '3', '1'] if tissue == '1' else ['skull', 'sinus_maxillary', 'teeth_upper', 'mandible', 'eye_left', 'lateral_pterygoid_right']
for s in levels:
    P = to_labels(ae, F, G, idx, code + rng.normal(size=code.shape).astype(np.float32) * sd * s, base); P[~body] = 0; d = {}
    for k in sorted(set(np.unique(L[known])) - {0, 255}):
        a = L == k
        if a.sum() >= 50: d[inv.get(int(k), str(k))] = 2 * (a & (P == k) & known).sum() / (a.sum() + ((P == k) & known).sum())
    print('noise %.2f  median %.3f  ' % (s, np.median(list(d.values()))) + ' '.join('%s %.2f' % (w, d.get(w, float('nan'))) for w in WATCH), flush=True)
