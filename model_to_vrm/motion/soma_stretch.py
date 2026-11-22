"""ARDY's standard SOMA body stretched onto the twin's skeleton: python soma_stretch.py <skin_standard.npz> <ab_free.npz> <out.npz>
Each SOMA bone gets the turn that lays it along the twin's matching bone, a stretch along its length to the twin bone's
length, and across its length the overall size ratio (heights, sole to crown); points move by SOMA's own skin weights.
Bones with no twin match (eyes, jaw, finger tips, toe ends, the second neck joint) ride with their parent.
Frames: SOMA is y-up facing +z (ARDY's frame); the twin is z-up facing -y; the axes are swapped first."""
import sys, numpy as np
S, Fz, out = np.load(sys.argv[1]), np.load(sys.argv[2]), sys.argv[3]
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
Vs = S['bind_vertices'] @ M.T; Js = S['bind_rig_transform'][:, :3, 3] @ M.T; names = [str(n) for n in S['rig_joint_names']]
con = S['rig_joint_connections']; par = {int(c): int(p) for p, c in con}; kids = {}
for c, p in par.items(): kids.setdefault(p, []).append(c)
tn = [str(n).split(':')[-1] for n in Fz['names']]; H0 = Fz['head0']; V0 = Fz['V0']
mp = {'Hips': 'Hips', 'Spine1': 'Spine', 'Spine2': 'Spine1', 'Chest': 'Spine2', 'Neck1': 'Neck', 'Head': 'Head'}
for s in ('Left', 'Right'):
    mp.update({s + 'Shoulder': s + 'Shoulder', s + 'Arm': s + 'Arm', s + 'ForeArm': s + 'ForeArm', s + 'Hand': s + 'Hand',
               s + 'Leg': s + 'UpLeg', s + 'Shin': s + 'Leg', s + 'Foot': s + 'Foot', s + 'ToeBase': s + 'ToeBase'})
    for f in ('Thumb', 'Index', 'Middle', 'Ring', 'Pinky'):
        for k in (1, 2, 3): mp['%sHand%s%d' % (s, f, k)] = '%sHand%s%d' % (s, f, k)
P = {names.index(a): H0[tn.index(b)] for a, b in mp.items() if a in names and b in tn}
q = (V0[:, 2].max() - V0[:, 2].min()) / (Vs[:, 1 @ np.eye(3, dtype=int)[1]] if False else (Vs[:, 2].max() - Vs[:, 2].min()))
print('SS mapped joints %d of %d; overall size ratio %.4f' % (len(P), len(names), q))
def turn(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b); v = np.cross(a, b); c = a @ b
    if np.linalg.norm(v) < 1e-9: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K / (1 + c)
nJ = len(names); A = np.zeros((nJ, 3, 3)); T = np.zeros((nJ, 3)); order = []
seen = set()
while len(order) < nJ:
    for j in range(nJ):
        if j not in seen and (j not in par or par[j] in seen): order.append(j); seen.add(j)
for j in order:
    p = par.get(j)
    # where the joint lands: its twin match, else carried by its parent's transform
    tgt = P[j] if j in P else (A[p] @ (Js[j] - Js[p]) + T[p] if p is not None else Js[j] * q)
    ch = [c for c in kids.get(j, []) if c in P]
    if j in P and ch:
        c = max(ch, key=lambda c: np.linalg.norm(Js[c] - Js[j])); s = Js[c] - Js[j]; t = P[c] - P[j]
        R = turn(s, t); sh = s / np.linalg.norm(s); k = np.linalg.norm(t) / np.linalg.norm(s)
        A[j] = R @ (q * np.eye(3) + (k - q) * np.outer(sh, sh))
    else: A[j] = A[p] if p is not None else q * np.eye(3)
    T[j] = tgt
    # store as x' = A (x - Js[j]) + T[j]
li, lw = S['lbs_indices'], S['lbs_weights']
Y = np.einsum('vkab,vkb->vka', A[li], Vs[:, None, :] - Js[li]) + T[li]; Vn = (lw[..., None] * Y).sum(1) / lw.sum(1, keepdims=True)
Jn = np.array([T[j] for j in range(nJ)])
np.savez(out, V=Vn, F=S['faces'], J=Jn, names=np.array(names), lbs_indices=li, lbs_weights=lw, A=A)
print('SS body z %.3f..%.3f (twin %.3f..%.3f)' % (Vn[:, 2].min(), Vn[:, 2].max(), V0[:, 2].min(), V0[:, 2].max()))
