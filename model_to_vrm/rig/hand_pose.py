"""Finger poses on SOMA's finger joints, from anatomy: import and call fingers(D, P, J, pose, side).
A pose gives each finger a curl fraction 0..1 of its joints' normal range of motion (AAOS normal values, degrees):
  fingers  MCP 90, PIP 100, DIP 80 (SOMA joints 1, 2, 3 of each finger)
  thumb    CMC/MCP 50, IP 80      (SOMA Thumb1 small, Thumb2, Thumb3)
Each joint flexes about the axis across its bone and the palm's normal (palm normal from the hand's own joints: index
and pinky knuckles across, middle knuckle along), toward the palm. The finger's rotation replaces ARDY's (which ARDY
leaves fixed); the hand itself keeps ARDY's turn."""
import numpy as np
ROM = {'Thumb': (0.0, 50.0, 80.0), 'Index': (90.0, 100.0, 80.0), 'Middle': (90.0, 100.0, 80.0), 'Ring': (90.0, 100.0, 80.0), 'Pinky': (90.0, 100.0, 80.0)}
POSES = {'open': dict(Thumb=0, Index=0, Middle=0, Ring=0, Pinky=0),
         'relaxed': dict(Thumb=0.2, Index=0.25, Middle=0.3, Ring=0.35, Pinky=0.4),
         'fist': dict(Thumb=1, Index=1, Middle=1, Ring=1, Pinky=1),
         'point': dict(Thumb=1, Index=0, Middle=1, Ring=1, Pinky=1)}
NAMES = [s for s in open('E:/figures/ardy/soma77_names.txt').read().split('\n')]
S = np.load('E:/code/ardy/ardy/assets/skeletons/somaskel77/skin_standard.npz'); PAR = {int(c): int(p) for p, c in S['rig_joint_connections']}
def rotv(axis, deg):
    a = axis / np.linalg.norm(axis); t = np.radians(deg); K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K
def fingers(D, P, J, pose, side):
    """D (nJ,3,3) turns from the body's rest, P (nJ,3) joint places, J rest joints; returns new D, P for one frame."""
    D, P = D.copy(), P.copy(); ix = lambda n: NAMES.index(side + n); h = ix('Hand')
    across = J[ix('HandIndex1')] - J[ix('HandPinky1')]; along = J[ix('HandMiddle1')] - J[h]
    palm = np.cross(along, across) if side == 'Left' else np.cross(across, along); palm /= np.linalg.norm(palm)
    spec = POSES[pose] if isinstance(pose, str) else pose
    for f, frac in spec.items():
        C = np.eye(3)
        for k in (1, 2, 3):
            j = ix('Hand%s%d' % (f, k)); c = ix('Hand%s%d' % (f, k + 1)) if k < 3 else ix('Hand%s%d' % (f, 4)) if (side + 'Hand%s4' % f) in NAMES else ix('Hand%sEnd' % f)
            bone = J[c] - J[j]; ax = np.cross(bone, palm)                              # turning about this moves the tip toward the palm
            C = C @ rotv(ax, frac * ROM[f][k - 1]) if True else C
            D[j] = D[h] @ C
        # descendants beyond joint 3 ride joint 3
        for n in NAMES:
            if n.startswith(side + 'Hand' + f) and n[len(side + 'Hand' + f):] in ('4', 'End'): D[ix(n[len(side):])] = D[ix('Hand%s3' % f)]
    # joint places by forward kinematics below the hand
    order = sorted([i for i, n in enumerate(NAMES) if n.startswith(side + 'Hand') and n != side + 'Hand'], key=lambda i: depth(i))
    for j in order: p = PAR[j]; P[j] = P[p] + D[p] @ (J[j] - J[p])
    return D, P
def depth(i):
    d = 0
    while i in PAR: i = PAR[i]; d += 1
    return d
