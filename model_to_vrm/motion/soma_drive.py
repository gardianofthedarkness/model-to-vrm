"""Pose the fitted SOMA body by an ARDY clip (skinning only, the kinematic target): python soma_drive.py <body_fit.npz> <soma_stretched.npz> <clip.npz> <out.npz>
ARDY gives each joint's world turn G from its neutral T-pose (y-up, facing +z). The fitted body's rest is that T-pose
after the stage-2 stretch, which turned each bone by R_j (the rotation part of the stretch A_j). So a joint's turn from
the body's rest is Q_j R_j^-1, with Q = M G M^T in the body's axes; joints follow by forward kinematics from the body's
own joint places, the hips also travel by ARDY's root path scaled by the hip-height ratio; points by SOMA's own weights."""
import sys, numpy as np
B, St, C, out = np.load(sys.argv[1]), np.load(sys.argv[2]), np.load(sys.argv[3]), sys.argv[4]
import numpy.linalg as la
M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
V, J = B['V'], B['J']; li, lw = B['lbs_indices'], B['lbs_weights']; A = St['A']
S = np.load(r'E:\code\ardy\ardy\assets\skeletons\somaskel77\skin_standard.npz'); par = {int(c): int(p) for p, c in S['rig_joint_connections']}
def polar(A):   # rotation part
    U, _, Vt = la.svd(A); R = U @ Vt
    if la.det(R) < 0: U[:, -1] *= -1; R = U @ Vt
    return R
# the body's rest turn from ARDY's T-pose, per bone: ARDY's neutral bone (into these axes) laid onto the body's rest bone;
# a joint without children (ends) takes its parent's; the body's bind (an A-pose) does not enter
N = np.load('E:/figures/ardy/soma77_neutral.npy') @ M.T; kids = {}
for c_, p_ in par.items(): kids.setdefault(p_, []).append(c_)
def turn(a, b):
    a, b = a / la.norm(a), b / la.norm(b); v = np.cross(a, b); c = a @ b
    if la.norm(v) < 1e-9: return np.eye(3)
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K / (1 + c)
R = np.zeros((len(J), 3, 3)); done = set(); todo = list(range(len(J)))
while todo:
    j = todo.pop(0)
    if j in kids:
        c_ = max(kids[j], key=lambda c: la.norm(N[c] - N[j])); R[j] = turn(N[c_] - N[j], J[c_] - J[j]); done.add(j)
    elif par.get(j) in done: R[j] = R[par[j]]; done.add(j)
    else: todo.append(j)
G = C['global_rot_mats']; root = C['root_positions']; Tn = len(G); nJ = len(J)
Q = np.einsum('ij,tnjk,lk->tnil', M, G, M); D = np.einsum('tnij,nkj->tnik', Q, R)      # Q R^T per joint
order = []; seen = set()
while len(order) < nJ:
    for j in range(nJ):
        if j not in seen and (j not in par or par[j] in seen): order.append(j); seen.add(j)
s_hip = (J[0, 2] - V[:, 2].min()) / root[0, 1]; P = np.zeros((Tn, nJ, 3))
for j in order:
    if j not in par: P[:, j] = J[j] + np.einsum('ij,tj->ti', M, root - root[0]) * s_hip
    else: p = par[j]; P[:, j] = P[:, p] + np.einsum('tij,j->ti', D[:, p], J[j] - J[p])
Y = np.einsum('tvkab,vkb->tvka', D[:, li], V[:, None, :] - J[li]) + P[:, li]
Vp = (lw[None, ..., None] * Y).sum(2) / lw.sum(1)[None, :, None]
np.savez(out, V=Vp.astype(np.float32), P=P.astype(np.float32), D=D.astype(np.float32), fps=C['fps'], s_hip=s_hip)
print('SD %d frames; hip scale %.3f; body points %d' % (Tn, s_hip, len(V)))
