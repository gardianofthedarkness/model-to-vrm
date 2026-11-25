"""Diagnosis: does a flow checkpoint reproduce heads it trained on as well as held-out ones? Uses the stored v3
preparation (features with the cover column) so the input is exactly what training saw. One draw per head.
args: <checkpoint flow_f1.pt> <face autoencoder .pt> <face index.json> <steps> <prep npz>..."""
import os, sys, json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy import ndimage as ndi
from face_compress import make_ae
from transformer_smoke import DEV, encode
from flow_f1 import Flow
from flow_sample import sample, to_labels
ckf, aef, idxf, steps = sys.argv[1:5]; files = sys.argv[5:]; steps = int(steps)
ck = torch.load(ckf, map_location='cpu'); net = Flow(ck['fdim'], ck['C'], ck['ncmax'], ck['d'], ck['layers']); net.load_state_dict(ck['state_dict']); net.eval().to(DEV)
ids = json.load(open(idxf))['ids']; NC = max(ids.values()) + 1; inv = {v: k for k, v in ids.items()}
c = torch.load(aef, map_location=DEV); ae = make_ae(NC, c['F'], c['C']).to(DEV); ae.load_state_dict(c['state_dict']); ae.eval(); F = c['F']
WATCH = ['skull', 'sinus_maxillary', 'teeth_upper', 'Cavity_Oral', 'mandible', 'eye_left', 'brain']
for f in files:
    z = np.load(f); idx = z['idx'].astype(np.int64); feat = z['feat']; pos = z['pos']; L = z['L']; G = [n // F for n in L.shape]
    occ = np.zeros(G, bool); occ[tuple(idx.T)] = True; shell = ~ndi.minimum_filter(occ, size=3, mode='constant', cval=0)[tuple(idx.T)]
    body = (L > 0) & (L < 255); base = encode(ae, np.where(body, 3, L).astype(np.uint8), NC, F)
    zc = sample(net, ck, 'face', idx, feat, pos, shell, NC, None, '', steps, 100); P = to_labels(ae, F, G, idx, zc, base); P[~body] = 0
    known = L != 255; d = {}
    for k in sorted(set(np.unique(L[known])) - {0, 255}):
        a = L == k; b = (P == k) & known
        if a.sum() >= 50: d[inv.get(int(k), str(k))] = 2 * (a & b).sum() / (a.sum() + b.sum())
    print(os.path.basename(f), 'model median %.3f' % np.median(list(d.values())), ' '.join('%s %.2f' % (w, d.get(w, float('nan'))) for w in WATCH), flush=True)
    # stored codes decoded: the ceiling for this head
    cs = z['code'].astype(np.float32); cs = cs[(slice(None),) + tuple(idx.T)].T if cs.ndim == 4 else cs;   # stored as (C, G0, G1, G2)
    Q = to_labels(ae, F, G, idx, cs, base); Q[~body] = 0
    dq = [2 * ((L == k) & (Q == k) & known).sum() / ((L == k).sum() + ((Q == k) & known).sum()) for k in sorted(set(np.unique(L[known])) - {0, 255}) if (L == k).sum() >= 50]
    print('   stored codes (ceiling) median %.3f' % np.median(dq), flush=True)
