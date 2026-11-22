"""Bend the T-pose twin's rig into the original model's pose (the twin's own skinning does the bending).
python fit_pose.py <twin_rig.npz> <twin_in.npz> <orig.npz> <orig_front.png> <orig_front.json> <out.npz>
  frame    the rig job hands the twin back in its own frame (uniformly scaled, shifted): one scale + shift fitted back onto the
           twin's own points (nearest-point rounds), so rig and original share the model frame; each rig point takes the part
           number of the nearest twin point
  joints   the original's joints read on its front render with the pipeline's pose detector (rtmlib Wholebody): shoulders,
           elbows, wrists, hips, knees, ankles, big toes; only their front positions (x, z) are used, never a guessed depth
  model    per bone a turn (axis-angle, world frame at rest) and a length factor (head-to-head offset), a root shift; points move
           by the twin's skin weights (linear blend)
  stage 1  joints: the rig's joints, seen from the front, onto the detected ones (each by its detector score)
  parts    each original part paired with the twin part most of its points lie nearest to (after stage 1)
  stage 2  shape: both ways nearest-point distance between the bent twin and the original, only within paired parts, plus the
           joints, plus the regularisers
  every term is a mean distance in the model's units; the regularisers are in the same units (a turn counted as the arc its
  bone's tip sweeps, a length change as the length gained), so no term needs a weight to be on the same footing"""
import sys, os, json, numpy as np, cv2, torch
FREEZE = os.environ.get('FREEZE') == '1'; J2 = os.environ.get('J2', '1') == '1'   # A/B: one size for all bones (proportions kept); joints in stage 2 or not
from scipy.spatial import cKDTree
rig_p, tin_p, org_p, img_p, cam_p, out = sys.argv[1:7]
dev = 'cuda' if torch.cuda.is_available() else 'cpu'; tt = lambda a: torch.tensor(np.asarray(a), dtype=torch.float32, device=dev)
R_ = np.load(rig_p); TI = np.load(tin_p); OG = np.load(org_p)
V0, W, names, par, H0, T0 = R_['V'], R_['W'], [str(n) for n in R_['names']], R_['parent'], R_['head'], R_['tail']
# --- frame: one scale + shift, the rig's points onto the twin's own
A = TI['V']; kd = cKDTree(A); s = np.ptp(A[:, 2]) / np.ptp(V0[:, 2]); t = A.mean(0) - s * V0.mean(0)
for _ in range(20):                                                                  # NUMERIC: nearest-point rounds (converges in a few)
    C = s * V0 + t; _, i = kd.query(C); X, Y = V0, A[i]; xm, ym = X.mean(0), Y.mean(0)
    s = ((X - xm) * (Y - ym)).sum() / ((X - xm) ** 2).sum(); t = ym - s * xm
miss = kd.query(s * V0 + t)[0]; print('FP frame scale %.4f, miss median %.4f' % (s, np.median(miss)))
V0 = s * V0 + t; H0 = s * H0 + t; T0 = s * T0 + t; part_t = TI['part'][kd.query(V0)[1]]
nb = len(names); bi = {n: i for i, n in enumerate(names)}
order = []; seen = set()
while len(order) < nb:
    for j in range(nb):
        if j not in seen and (par[j] < 0 or par[j] in seen): order.append(j); seen.add(j)
side = np.sign(H0[bi['mixamorig:LeftArm'], 0]); print('FP the twin\'s left arm lies at x %s' % ('+' if side > 0 else '-'))
# --- the original's joints from its front render
from rtmlib import Wholebody
img = cv2.imread(img_p); cam = json.load(open(cam_p)); kp, sc = Wholebody(mode='balanced', backend='onnxruntime', device='cpu')(img)
k = int(np.argmax(sc[:, :17].mean(1))); K, S = kp[k], sc[k]
to_w = lambda u, v: (cam['cx'] + (u - cam['W'] / 2) * cam['mpp'], cam['cz'] - (v - cam['H'] / 2) * cam['mpp'])
pairs = {5: 'LeftArm', 6: 'RightArm', 7: 'LeftForeArm', 8: 'RightForeArm', 9: 'LeftHand', 10: 'RightHand', 11: 'LeftUpLeg', 12: 'RightUpLeg',
         13: 'LeftLeg', 14: 'RightLeg', 15: 'LeftFoot', 16: 'RightFoot', 17: 'LeftToeBase', 20: 'RightToeBase'}       # COCO-WholeBody index -> the bone whose head sits there
# COCO "left" is the person's left; it is matched to the rig's Left only if both lie on the same side of the picture
img_left = np.sign(to_w(K[5][0], 0)[0] - to_w(K[6][0], 0)[0])
if img_left != side: pairs = {k_: (b.replace('Left', 'TMP').replace('Right', 'Left').replace('TMP', 'Right')) for k_, b in pairs.items()}
jb = [bi['mixamorig:' + b] for b in pairs.values()]; jt = tt([to_w(*K[k_]) for k_ in pairs]); js = tt([S[k_] for k_ in pairs])
for k_, b in pairs.items(): print('FP joint %-12s score %.2f' % (b, S[k_]))
# the detector's joint is not the rig's joint (a sleeve crease, not the bone end): the same detector on the twin's own front
# render says where it lands relative to each rig joint on this character; that offset rides with its bone
if len(sys.argv) > 8:
    img2 = cv2.imread(sys.argv[7]); cam2 = json.load(open(sys.argv[8])); kp2, sc2 = Wholebody(mode='balanced', backend='onnxruntime', device='cpu')(img2)
    K2 = kp2[int(np.argmax(sc2[:, :17].mean(1)))]
    to_w2 = lambda u, v: (cam2['cx'] + (u - cam2['W'] / 2) * cam2['mpp'], cam2['cz'] - (v - cam2['H'] / 2) * cam2['mpp'])
    off = np.zeros((len(jb), 3))
    for n_, (k_, j) in enumerate(zip(pairs, jb)): x_, z_ = to_w2(*K2[k_]); off[n_] = (x_ - H0[j, 0], 0, z_ - H0[j, 2])
    for b, o in zip(pairs.values(), off): print('FP detector offset %-12s dx %+.4f dz %+.4f' % (b, o[0], o[2]))
else: off = np.zeros((len(jb), 3))
offt = tt(off)
# --- the bending model
Vt, Wt, H0t, T0t = tt(V0), tt(W), tt(H0), tt(T0); par_t = torch.tensor(par, device=dev)
blen = tt(np.linalg.norm(T0 - H0, axis=1))
rot = torch.zeros(nb, 3, device=dev, requires_grad=True); lf = torch.zeros(nb, device=dev, requires_grad=True); sh = torch.zeros(3, device=dev, requires_grad=True); gs = torch.zeros(1, device=dev, requires_grad=True)
LF = lambda j: torch.exp(gs[0]) if FREEZE else torch.exp(lf[j])
def rodr(r):
    th = r.norm(dim=-1, keepdim=True).clamp_min(1e-8); k_ = r / th; Kx = torch.zeros(r.shape[0], 3, 3, device=dev)
    Kx[:, 0, 1], Kx[:, 0, 2], Kx[:, 1, 0], Kx[:, 1, 2], Kx[:, 2, 0], Kx[:, 2, 1] = -k_[:, 2], k_[:, 1], k_[:, 2], -k_[:, 0], -k_[:, 1], k_[:, 0]
    th = th[..., None]; return torch.eye(3, device=dev) + torch.sin(th) * Kx + (1 - torch.cos(th)) * Kx @ Kx
def pose():
    Rl = rodr(rot); Rg = [None] * nb; P = [None] * nb
    for j in order:
        p = par[j]
        if p < 0: Rg[j] = Rl[j]; P[j] = H0t[j] + sh
        else: Rg[j] = Rg[p] @ Rl[j]; P[j] = P[p] + Rg[p] @ ((H0t[j] - H0t[p]) * LF(j))
    Rg = torch.stack(Rg); P = torch.stack(P); return Rg, P
def skin(Rg, P, V, Wv):
    Y = torch.einsum('bij,nbj->nbi', Rg, V[:, None, :] - H0t[None]) * (torch.exp(gs[0]) if FREEZE else 1) + P[None]; return (Wv[..., None] * Y).sum(1)
bdir = tt((T0 - H0) / np.maximum(np.linalg.norm(T0 - H0, axis=1, keepdims=True), 1e-9))
def reg():   # the same character: a length change costs the length gained; bends are what is being solved (free); a twist about
             # the bone's own axis (which front joints cannot see) costs the arc its tip sweeps - both distances in model units
    twist = (rot * bdir).sum(1).abs()
    return (twist * blen).mean() + (0 if FREEZE else (torch.exp(lf) - 1).abs().mul(blen).mean())
RG_ = [None]
def jterm(P, Rg=None):
    Rg = RG_[0] if Rg is None else Rg; Q = P[jb] + torch.einsum('bij,bj->bi', Rg[jb], offt)   # where the detector would see each joint
    return ((Q[:, [0, 2]] - jt).norm(dim=1) * js).sum() / js.sum()
_pose = pose
def pose():
    Rg, P = _pose(); RG_[0] = Rg; return Rg, P
opt = torch.optim.Adam([rot, lf, sh, gs], lr=0.01)                                     # NUMERIC: step size (radians / log-length per step)
for it in range(400):                                                               # NUMERIC: stage-1 steps
    Rg, P = pose(); L = jterm(P) + reg(); opt.zero_grad(); L.backward(); opt.step()
print('FP stage 1 joint miss %.4f (of height %.3f)' % (jterm(pose()[1]).item(), np.ptp(OG['V'][:, 2])))
# --- parts: each original part paired with the twin part both overlap most: votes both ways (original points to their
#     nearest twin point, twin points to their nearest original point), scored by the geometric mean of the two fractions,
#     so a small part (a hand in a pocket) pairs with the small twin part that lands on it, not with the big part it touches
Vo = tt(OG['V']); grp_t = tt(part_t); NS = 5000                                    # NUMERIC: points sampled per side per shape step
def pairing():
    with torch.no_grad(): Vp = skin(*pose(), Vt, Wt).cpu().numpy()
    po, pt_ = OG['part'], part_t; no, nt = po.max() + 1, pt_.max() + 1
    a = np.zeros((no, nt)); np.add.at(a, (po, pt_[cKDTree(Vp).query(OG['V'])[1]]), 1)
    b = np.zeros((no, nt)); np.add.at(b, (po[cKDTree(OG['V']).query(Vp)[1]], pt_), 1)
    fa = a / np.maximum(a.sum(1, keepdims=True), 1); fb = b / np.maximum(b.sum(0, keepdims=True), 1); sc_ = np.sqrt(fa * fb)
    pr = {int(p): int(np.argmax(sc_[p])) for p in np.unique(po)}
    return pr, tt(np.array([pr[int(p)] for p in po]))
def shape(Vp, grp_o):
    a = torch.randint(0, len(Vt), (NS,), device=dev); b = torch.randint(0, len(Vo), (NS,), device=dev)
    D = torch.cdist(Vp[a], Vo[b]); D = D + 1e3 * (grp_t[a][:, None] != grp_o[b][None]).float()   # other parts are out of reach
    da = D.min(1).values; db = D.min(0).values; ok_a = da < 1e2; ok_b = db < 1e2
    return da[ok_a].mean() + db[ok_b].mean()
# --- stage 2: shape within paired parts, plus joints, plus regularisers; the pairing redone after each round
for rnd in range(3):                                                                # NUMERIC: pairing rounds
    pair, grp_o = pairing()
    print('FP round %d pairs: ' % rnd + ' '.join('%d>%d' % kv for kv in sorted(pair.items())))
    opt = torch.optim.Adam([rot, lf, sh, gs], lr=0.005)                                # NUMERIC: a smaller step for the finer stage
    for it in range(600):                                                           # NUMERIC: shape steps per round
        Rg, P = pose(); Vp = skin(Rg, P, Vt, Wt); L = shape(Vp, grp_o) + (jterm(P) if J2 else 0) + reg(); opt.zero_grad(); L.backward(); opt.step()
    print('FP round %d: shape %.4f joints %.4f' % (rnd, shape(Vp, grp_o).item(), jterm(P).item()), flush=True)
with torch.no_grad():
    Rg, P = pose(); Vp = skin(Rg, P, Vt, Wt); Tl = P + torch.einsum('bij,bj->bi', Rg, T0t - H0t) * torch.exp(lf)[:, None]
    d_o = []
    for g in torch.unique(grp_o):
        mo, mt = grp_o == g, grp_t == g
        if mt.any(): d_o.append(torch.cdist(Vo[mo], Vp[mt]).min(1).values)
    d_o = torch.cat(d_o).cpu().numpy()
    dfull = np.full(len(OG['V']), np.nan); gnp = grp_o.cpu().numpy()
    for g in np.unique(gnp):
        mo = gnp == g; mt = (grp_t == int(g)).cpu().numpy()
        if mt.any(): dfull[mo] = cKDTree(Vp.cpu().numpy()[mt]).query(OG['V'][mo])[0]
print('FP global size %.3f' % float(torch.exp(gs[0])))
print('FP final: original points to the bent twin (same parts): median %.4f, 95%% %.4f; joints %.4f' % (np.median(d_o), np.percentile(d_o, 95), jterm(P).item()))
print('FP length factors: ' + ', '.join('%s %.2f' % (n.split(':')[-1], f) for n, f in zip(names, torch.exp(lf).detach().cpu().numpy()) if abs(f - 1) > 0.05))
np.savez(out, Vbent=Vp.cpu().numpy(), V0=V0, W=W, part_t=part_t, pair=np.array([[k_, v] for k_, v in pair.items()]), names=np.array(names), parent=par,
         head=P.cpu().numpy(), tail=Tl.cpu().numpy(), Rg=Rg.cpu().numpy(), head0=H0, tail0=T0, joints2d=jt.cpu().numpy(), jbones=np.array(jb), dist=dfull, gscale=float(torch.exp(gs[0])))
