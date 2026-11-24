"""Exports the loaded-vs-hanging muscle runs (stance_runs.sh) for the interactive viewer: per muscle and case, the surface
every 2nd saved frame (int16-quantised), the per-frame metrics, the MyoSuite activations, and the leg bones at the held
frame, clustered at 2 mm.
  E:\\splat\\.venv\\Scripts\\python.exe stance_web_export.py <fits dir> <leg_transforms npz> <sim dir> <myo_single_leg.json> <out json>"""
import sys, os, json, base64, numpy as np
FD, TRF, SD, MJ, OUT = sys.argv[1:6]
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode()
J = json.load(open(MJ)); HOLD = J['clip_frame']; TR = np.load(TRF)
def cluster(V, F, cell):
    key = np.floor(V / cell).astype(np.int64); _, lab = np.unique(key, axis=0, return_inverse=True); lab = lab.ravel()
    W = np.zeros((lab.max() + 1, 3)); np.add.at(W, lab, V); W /= np.bincount(lab)[:, None]
    G = lab[F]; return W, G[(G[:, 0] != G[:, 1]) & (G[:, 1] != G[:, 2]) & (G[:, 0] != G[:, 2])]
R = np.load(os.path.join(FD, 'muscle_repair.npz')); V, OB, FB = R['V'].astype(np.float64), R['obj'], R['F_bone']
names = [str(x) for x in np.load(os.path.join(FD, 'atlas_warp.npz'))['names']]; nid = {n: i for i, n in enumerate(names)}
def seg(sg):
    if sg == 0: return np.eye(3), np.zeros(3)
    return (TR['thigh_R'][HOLD], TR['thigh_t'][HOLD]) if sg == 1 else (TR['shank_R'][HOLD], TR['shank_t'][HOLD])
bones = []
for n, sg in (('Hip bone.l', 0), ('Femur.l', 1), ('Tibia.l', 2), ('Fibula.l', 2), ('Patella.l', 2)):
    fb = FB[OB[FB[:, 0]] == nid[n]]; u, inv = np.unique(fb, return_inverse=True); W, G = cluster(V[u], inv.reshape(-1, 3), 0.002)
    vol = np.einsum('ij,ij->i', W[G[:, 0]], np.cross(W[G[:, 1]], W[G[:, 2]])).sum()
    if vol < 0: G = G[:, [0, 2, 1]]                         # atlas bones are wound inward
    Rb, tb = seg(sg); bones.append((n, W @ Rb.T + tb, G))
mus = []; allP = [b[1] for b in bones]
for m in J['cases']['loaded']['activation']:
    if m.startswith('Sartorius'): continue
    tag = m.replace(' ', '_').replace('.', '_'); runs = {}
    for case in ('loaded', 'hanging'):
        a = J['cases'][case]['activation'][m]; f = os.path.join(SD, 'stance', '%s_a%.4f.npz' % (tag, a))
        D = np.load(f); Jr = json.load(open(f.replace('.npz', '.json'))); S = D['surf'].astype(np.float64); S = np.where(np.isfinite(S), S, D['rest'][None])
        idx = list(range(0, len(S), 2)); idx = idx if idx[-1] == len(S) - 1 else idx + [len(S) - 1]
        fr = [int(D['frames'][i]) for i in idx]; rows = [Jr['frames'][k] for k in fr]
        runs[case] = dict(a=a, S=S[idx], frames=fr, t=[r['t'] for r in rows], act=[r['activation'] for r in rows], belly=[r['belly_thickness_ratio'] for r in rows],
                          fib=[r['fibre_stretch_p5_50_95'][1] for r in rows], vol=[r['volume_ratio'] for r in rows], inside=[r['inside_bone_nodes'] for r in rows], deep=[r['deepest_mm'] for r in rows])
        allP.append(S.reshape(-1, 3))
    mus.append((m, runs, D['F'], D['rest']))
allP = np.concatenate(allP); lo, hi = allP.min(0) - 0.01, allP.max(0) + 0.01
q = lambda P: np.round((P - lo) / (hi - lo) * 65534 - 32767).astype(np.int16)
out = dict(lo=lo.tolist(), hi=hi.tolist(), clip_frame=HOLD, myo=J['cases'], bones=[dict(name=n, v=b64(q(W)), f=b64(G.astype(np.uint32))) for n, W, G in bones], muscles=[])
for m, runs, F, rest in mus:
    out['muscles'].append(dict(name=m, F=b64(F.astype(np.uint32)), nv=int(len(rest)), nframes=len(runs['loaded']['frames']),
                               cases={c: dict(a=r['a'], S=b64(q(r['S'])), t=r['t'], act=r['act'], belly=r['belly'], fib=r['fib'], vol=r['vol'], inside=r['inside'], deep=r['deep']) for c, r in runs.items()}))
json.dump(out, open(OUT, 'w'), separators=(',', ':')); print('wrote %s (%.1f MB), %d muscles' % (OUT, os.path.getsize(OUT) / 1e6, len(mus)))
