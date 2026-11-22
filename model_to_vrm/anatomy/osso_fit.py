"""The figure's skeleton from OSSO (Keller et al., CVPR 2022: a skeleton shape model learned from DXA scans, non-commercial
research licence), placed inside the model's SMPL body; then the stick joints taken from the fitted bones:
  E:\\splat\\.venv\\Scripts\\python.exe osso_fit.py <model dir> [female|male]
Inputs: anatomy/smpl_fit.npz (mhr_to_smpl.py: the body as SMPL), parts/orig_colours.npz + orig_normals.npz (the model's
surface), body/skeleton.npz (the stick skeleton, for the hands, feet and face that OSSO's 21 parts do not have).
OSSO files: OSSO_DIR (default E:\\figures\\models\\osso: skeleton/*.pkl) and OSSO's code data (E:\\code\\external\\OSSO\\data\\loss).
What runs (OSSO's own inference, without STAR and without psbody/chumpy):
  landmarks   OSSO's linear landmark regressor on the SMPL body vertices (63 bone landmarks, each a weighted mean of 4-7
              nearby body vertices, so it also holds in the T-pose, not only the lying pose it was learned in)
  skeleton    OSSO's stitched puppet: 21 parts; shape = OSSO's skeleton PCA (300 modes, prior: their variances); each part
              rotated and moved on its own; one global size (the model's units are not metres)
  energy      landmarks (part point - predicted landmark)                                            / sigma_l
              stitching (neighbouring parts keep their interface offsets; collarbones free, as OSSO)  / sigma_s
              skin contacts (OSSO's skin-bone contact pairs and weights, 5 mm skin x size, as OSSO)   / sigma_c
              inside the model (each part's convex-hull points below the surface, one-sided)          / sigma_l
              PCA prior beta_i / sqrt(Var_i)
  start       one similarity on all 63 landmarks (size); each part by its 3 landmarks (Kabsch)
  solve       scipy least_squares (trust region, soft-L1), every sigma re-estimated from its own residuals (1.4826 x MAD)
              until no point moves more than sigma_l
Joints from the bones (written to body/skeleton.npz; the stick fit kept as body/skeleton_mhr.npz):
  ball joints   sphere centres on OSSO's joint surfaces: shoulder = humerus head, hip = femur head;
                elbow = middle of the humerus end and the ulna end spheres, knee = femur end and tibia end
  interfaces    the mean of two parts' interface points: spine base (pelvis-spine), the spine levels, neck base,
                head (neck-head), wrist (forearm-hand), ankle (shin-foot), collarbone's inner end (collarbone-neck)
  kept          hands, feet, eyes and jaw from the stick skeleton (measured keypoints), moved with their parent joint
Writes anatomy/osso_fit.npz (bone vertices, parts, faces, parameters, joints) and anatomy/osso_fit.json (checks)."""
import sys, os, json, time, pickle, numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation as Rot
from scipy.sparse import coo_matrix
R = sys.argv[1]; G = sys.argv[2] if len(sys.argv) > 2 else 'female'
OD = os.environ.get('OSSO_DIR', r'E:\figures\models\osso'); LD = os.environ.get('OSSO_LOSS', r'E:\code\external\OSSO\data\loss')
L_ = lambda f: pickle.load(open(f, 'rb'), encoding='latin1'); j = lambda *p: os.path.join(R, *p); t00 = time.time()
log = lambda *a: print('OS', *a, flush=True)
SK = L_(os.path.join(OD, 'skeleton', 'skeleton_model.pkl')); NP = 21; names = list(SK['names'])
n = [SK['shape'][p]['T'].size // 3 for p in range(NP)]; sh = np.r_[0, np.cumsum(n)]; part = np.repeat(np.arange(NP), n); NV = sh[-1]
PCA = L_(os.path.join(OD, 'skeleton', 'skeleton_pca_%s.pkl' % G)); M, Bm, Var = PCA['M'].astype(np.float64), PCA['B'], PCA['Var'].astype(np.float64); K = Bm.shape[2]
ldm_reg = L_(os.path.join(OD, 'skeleton', 'ldm_regressor_%s.pkl' % G)); ldm_reg = np.asarray(ldm_reg.todense() if hasattr(ldm_reg, 'todense') else ldm_reg, float)
ldm_idx = np.array(L_(os.path.join(OD, 'skeleton', 'ldm_indices.pkl')))
SM = np.load(j('anatomy', 'smpl_fit.npz')); VS = SM['V'].astype(np.float64); FS = SM['F']
Lm = ldm_reg @ VS                                                   # predicted bone landmarks, in the world
MV = np.load(j('parts', 'orig_colours.npz'))['V'].astype(np.float64); MN = np.load(j('parts', 'orig_normals.npz'))['N'].astype(np.float64); mtree = cKDTree(MV)
# part-centred PCA: local(beta) = (M + B beta) - part mean
Mc = M.copy(); Bc = np.asarray(Bm, np.float64)
for p in range(NP):
    s_, e_ = sh[p], sh[p + 1]; Mc[s_:e_] -= Mc[s_:e_].mean(0); Bc[s_:e_] -= Bc[s_:e_].mean(0)
del Bm
# interfaces between parts (OSSO's stitching), collarbone pairs free as in OSSO
pairs = SK['pairs'][:len(SK['pairs']) // 2]; Ii, Ij, Pi = [], [], []
for k, (a, b) in enumerate(pairs):
    fa, fb = SK['interfacePointsFromTo'][a][b], SK['interfacePointsFromTo'][b][a]
    if fa is None or fb is None: continue
    Ii.append(np.asarray(fa) + sh[a]); Ij.append(np.asarray(fb) + sh[b]); Pi.append(np.full(len(fa), k))
IiA, IjA, PiA = np.concatenate(Ii), np.concatenate(Ij), np.concatenate(Pi); keep = ~np.isin(PiA, [6, 7]); Ii, Ij = IiA[keep], IjA[keep]   # all interfaces (joints), stitched ones
# template topology -> part topology (OSSO's from_template_topology): the same points, another order
X0 = np.vstack([SK['shape'][p]['T'].reshape(-1, 3) for p in range(NP)]) + np.repeat(SK['template_t'], n, 0)
d_, t2s = cKDTree(X0).query(SK['template_points']); log('template -> part topology: largest mismatch %.2e' % d_.max())
c_skin, c_skel, c_w = [np.asarray(x) for x in L_(os.path.join(LD, 'skin_skel_contacts.pkl'))]; c_skel = t2s[c_skel]
BJ = L_(os.path.join(LD, 'gloss_ball_joints_ldm.pkl')); bj = {nm: np.asarray(v) for nm, v in zip(BJ['names'], BJ['verts_ids'])}
hull = np.concatenate([sh[p] + ConvexHull(Mc[sh[p]:sh[p + 1]]).vertices for p in range(NP)])
U = np.unique(np.concatenate([ldm_idx, Ii, Ij, c_skel, hull] + list(bj.values())))   # the points the energy looks at
ui = {v: i for i, v in enumerate(U)}; uix = lambda a: np.array([ui[v] for v in a])
McU, BcU, partU = Mc[U], Bc[U], part[U]
# SMPL vertex normals for the contact skin points
fn = np.cross(VS[FS[:, 1]] - VS[FS[:, 0]], VS[FS[:, 2]] - VS[FS[:, 0]]); VN = np.zeros_like(VS)
for c in range(3): np.add.at(VN, FS[:, c], fn)
VN /= np.linalg.norm(VN, axis=1, keepdims=True) + 1e-12
# --- start: global size by all landmarks, then each part by its 3 landmarks
def umeyama(X, Y):
    mx, my = X.mean(0), Y.mean(0); Xc, Yc = X - mx, Y - my; U_, S_, Vt = np.linalg.svd(Yc.T @ Xc / len(X))
    D = np.diag([1, 1, np.sign(np.linalg.det(U_ @ Vt))]); Rr = U_ @ D @ Vt; s = np.trace(np.diag(S_) @ D) / (Xc ** 2).sum(1).mean(); return Rr, s, my - s * Rr @ mx
_, s0, _ = umeyama(M[ldm_idx], Lm); R0 = np.zeros((NP, 3, 3)); t0 = np.zeros((NP, 3)); lpart = part[ldm_idx]
for p in range(NP):
    k = lpart == p; X, Y = s0 * Mc[ldm_idx[k]], Lm[k]; mx, my = X.mean(0), Y.mean(0); U_, _, Vt = np.linalg.svd((Y - my).T @ (X - mx))
    D = np.diag([1, 1, np.sign(np.linalg.det(U_ @ Vt))]); R0[p] = U_ @ D @ Vt; t0[p] = my - R0[p] @ mx
log('start: size %.3f (model units per metre), %d landmarks, %d interface pairs, %d skin contacts, %d hull points' % (s0, len(Lm), len(Ii), len(c_skel), len(hull)))
# parameters: [log s, beta (K), r (NP x 3), t (NP x 3)]
nb = 1 + K; NPAR = nb + 6 * NP
def unpack(x): return np.exp(x[0]), x[1:nb], x[nb:nb + 3 * NP].reshape(NP, 3), x[nb + 3 * NP:].reshape(NP, 3)
def world(x, Mc_, Bc_, pt):
    s, b, r, t = unpack(x); Rr = Rot.from_rotvec(r).as_matrix() @ R0; loc = Mc_ + np.einsum('nck,k->nc', Bc_, b)
    return s * np.einsum('nij,nj->ni', Rr[pt], loc) + t[pt]
li, ii, jj, ci, hi = uix(ldm_idx), uix(Ii), uix(Ij), uix(c_skel), uix(hull)
def depth(p): _, k = mtree.query(p); return -np.einsum('ij,ij->i', p - MV[k], MN[k])
x = np.zeros(NPAR); x[0] = np.log(s0); x[nb + 3 * NP:] = t0.ravel()
XU0 = world(x, McU, BcU, partU); D0 = (XU0[ii] - XU0[jj]) / s0      # interface offsets of the start (per unit size)
def resid(x, S):
    s = np.exp(x[0]); XU = world(x, McU, BcU, partU)
    rl = ((XU[li] - Lm) / S['l']).ravel()
    rs = ((XU[ii] - XU[jj] - s * D0) / S['s']).ravel()
    skin = VS[c_skin] - 0.005 * s * VN[c_skin]                      # OSSO: skin 5 mm thick (scaled to the figure's size)
    rc = (np.sqrt(c_w)[:, None] * (skin - XU[ci]) / S['c']).ravel()
    rk = np.maximum(0, -depth(XU[hi])) / S['l']
    rp = x[1:nb] / np.sqrt(Var)
    return np.r_[rl, rs, rc, rk, rp]
nl, ns, nc, nk = 3 * len(li), 3 * len(ii), 3 * len(ci), len(hi)
rows, cols = [], []
def dep(r0, pts, k):                                                # rows r0.. depend on: size, betas, and their points' part
    rr = r0 + np.arange(len(pts) * k); pp = np.repeat(pts, k)
    rows.append(np.repeat(rr, nb)); cols.append(np.tile(np.arange(nb), len(rr)))
    for q in range(3): rows.append(rr); cols.append(nb + 3 * pp + q); rows.append(rr); cols.append(nb + 3 * NP + 3 * pp + q)
dep(0, partU[li], 3); dep(nl, partU[ii], 3); dep(nl, partU[jj], 3); dep(nl + ns, partU[ci], 3); dep(nl + ns + nc, partU[hi], 1)
rows.append(nl + ns + nc + nk + np.arange(K)); cols.append(1 + np.arange(K))
NR = nl + ns + nc + nk + K; rows, cols = np.concatenate(rows), np.concatenate(cols)
Jsp = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(NR, NPAR)).tocsr(); Jsp.data[:] = 1
mad = lambda r: 1.4826 * float(np.median(np.abs(r))) + 1e-12
r_ = XU0[li] - Lm; S = {'l': mad(r_), 's': mad(XU0[ii] - XU0[jj] - s0 * D0) + mad(r_), 'c': mad(np.sqrt(c_w)[:, None] * (VS[c_skin] - XU0[ci]))}
log('start scales: landmarks %.4f, stitching %.4f, contacts %.4f (model units)' % (S['l'], S['s'], S['c'])); hist = []
for rnd in range(15):
    old = world(x, McU, BcU, partU)
    r = least_squares(resid, x, args=(S,), jac_sparsity=Jsp, method='trf', tr_solver='lsmr', loss='soft_l1', x_scale='jac', max_nfev=50)
    x = r.x; new = world(x, McU, BcU, partU); move = float(np.linalg.norm(new - old, axis=1).max()); rr = resid(x, S)
    nS = {'l': mad(rr[:nl] * S['l']), 's': mad(rr[nl:nl + ns] * S['s']), 'c': mad(rr[nl + ns:nl + ns + nc] * S['c'])}
    hist.append(dict(round=rnd, cost=float(r.cost), nfev=int(r.nfev), largest_move=move, **S))
    log('round %d: cost %.0f (%d evaluations), largest move %.4f; new scales landmarks %.4f, stitching %.4f, contacts %.4f (%.0f s)' % (rnd, r.cost, r.nfev, move, nS['l'], nS['s'], nS['c'], time.time() - t00))
    if rnd > 0 and move < S['l']: break
    S = nS
# --- the whole skeleton, checks
XA = world(x, Mc, Bc, part); s, beta, rv, tv = unpack(x)
F = np.vstack([np.asarray(SK['partFaces'][p]) + sh[p] for p in range(NP)])
el = np.linalg.norm(XA[ldm_idx] - Lm, axis=1); es = np.linalg.norm(XA[Ii] - XA[Ij] - s * D0, axis=1)
dA = depth(XA); out = dA < 0
def sphere(P):                                                      # algebraic sphere fit (as OSSO's sphere_fit)
    A = np.c_[2 * P, np.ones(len(P))]; f = (P ** 2).sum(1); C = np.linalg.lstsq(A, f, rcond=None)[0]; return C[:3], float(np.sqrt(C[3] + C[:3] @ C[:3]))
SP = {nm: sphere(XA[v]) for nm, v in bj.items()}
def iface(a, b):
    k = [q for q, (pa, pb) in enumerate(pairs) if {pa, pb} == {names.index(a), names.index(b)}]; m = PiA == (k[0] if k else -1)
    return (XA[IiA[m]] + XA[IjA[m]]).mean(0) / 2 if m.any() else None
mid = lambda a, b: (SP[a][0] + SP[b][0]) / 2
JO = {'l_uparm': SP['humerus_PL'][0], 'r_uparm': SP['humerus_PR'][0], 'l_upleg': SP['femur_PL'][0], 'r_upleg': SP['femur_PR'][0],
      'l_lowarm': mid('humerus_DL', 'ulna_PL'), 'r_lowarm': mid('humerus_DR', 'ulna_PR'), 'l_lowleg': mid('femur_DL', 'tibia_PL'), 'r_lowleg': mid('femur_DR', 'tibia_PR'),
      'l_wrist': iface('LeftArm2', 'LeftHand'), 'r_wrist': iface('RightArm2', 'RightHand'), 'l_talocrural': iface('LeftLeg2', 'LeftFoot'), 'r_talocrural': iface('RightLeg2', 'RightFoot'),
      'c_spine0': iface('Pelvis', 'Spine'), 'c_spine1': iface('Spine', 'Spine1'), 'c_spine2': iface('Spine1', 'Spine2'), 'c_neck': iface('Spine2', 'Neck'),
      'c_head': iface('Neck', 'Head'), 'l_clavicle': iface('LeftClavicle', 'Neck'), 'r_clavicle': iface('RightClavicle', 'Neck')}
JO = {k: v for k, v in JO.items() if v is not None}
if 'c_spine2' in JO and 'c_neck' in JO: JO['c_spine3'] = (JO['c_spine2'] + JO['c_neck']) / 2   # upper thorax: the middle of the Spine2 part's two interfaces
# side check: OSSO's Left = the person's left; the stick skeleton's l_ the same - compare to be sure
SKM = np.load(j('body', 'skeleton.npz'), allow_pickle=True); H, par, lab = SKM['heads'].astype(np.float64), SKM['parents'], [str(q) for q in SKM['labels']]
sw = np.mean([np.linalg.norm(JO['l_' + q] - H[lab.index('l_' + q)]) > np.linalg.norm(JO['l_' + q] - H[lab.index('r_' + q)]) for q in ('uparm', 'upleg', 'wrist')])
if sw > 0.5:
    log('OSSO left/right is mirrored against the stick skeleton: swapping'); JO = {(('r' if k[0] == 'l' else 'l') + k[1:] if k[0] in 'lr' else k): v for k, v in JO.items()}
if not os.path.exists(j('body', 'skeleton_mhr.npz')): np.savez(j('body', 'skeleton_mhr.npz'), **{k: SKM[k] for k in SKM.files})
H2 = H.copy(); moved = {}
order = list(range(len(lab)))                                       # parents come before children in MHR's order
for i in order:
    if lab[i] in JO: H2[i] = JO[lab[i]]; moved[i] = H2[i] - H[i]
    elif par[i] >= 0:
        p_ = par[i]; d = moved.get(p_, np.zeros(3)); H2[i] = H[i] + d; moved[i] = d   # kept joints ride with their parent
shift = {lab[i]: float(np.linalg.norm(H2[i] - H[i])) for i in range(len(lab)) if lab[i] in JO}
np.savez(j('body', 'skeleton.npz'), **{**{k: SKM[k] for k in SKM.files}, 'heads': H2})
js = {'size_model_units_per_metre': float(s), 'rounds': hist, 'landmark_error': {'median': float(np.median(el)), 'p90': float(np.percentile(el, 90))},
      'stitch_error': {'median': float(np.median(es)), 'p90': float(np.percentile(es, 90))}, 'bone_points_outside_model': float(out.mean()),
      'ball_joint_radii': {k: v[1] for k, v in SP.items()}, 'joint_shift_from_sticks': shift, 'betas_in_sd': (beta[:10] / np.sqrt(Var[:10])).round(2).tolist(),
      'seconds': round(time.time() - t00)}
json.dump(js, open(j('anatomy', 'osso_fit.json'), 'w'), indent=1)
np.savez(j('anatomy', 'osso_fit.npz'), V=XA.astype(np.float32), F=F.astype(np.int32), part=part, names=np.array(names), beta=beta, r=rv, t=tv, R0=R0, size=s,
         landmarks=Lm, joints_names=np.array(list(JO)), joints=np.array(list(JO.values())))
log('done: size %.3f; landmarks median %.4f (p90 %.4f), stitching median %.4f; %.1f%% bone points outside the model; first PCA modes in SD %s; joints moved (model units): %s (%.0f s)' % (
    s, np.median(el), np.percentile(el, 90), np.median(es), 100 * out.mean(), js['betas_in_sd'][:4], ', '.join('%s %.3f' % kv for kv in list(shift.items())[:8]), time.time() - t00))
