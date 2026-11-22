"""Whole-body assembly by the same strategy as the parts: named detector points, measured joints, strict inside rule.
The skull (skull_stage2.py), the hands and the feet (part_bone_fit.py) are taken as fitted and held; every other bone -
arms, legs, spine, ribs, shoulder and hip girdles - is a rigid piece of its own size that may also be longer or shorter
along its own main axis, joined to its neighbours and to the held parts.
  E:\\splat\\.venv\\Scripts\\python.exe body_assembly.py <fits dir> <model dir>
Per free bone b, 9 numbers on top of one size and pose S for the whole body (Umeyama 1991, atlas places of the named
joints -> detected joints): turn r_b, shift t_b, size s_b, and two stretches e_b, f_b along the bone's own main axis a_b
and second axis a2_b (a pelvis may be wide but shallow; never sheared)
  T_b(x) = S( c_b + exp(s_b) Rot(r_b) (I + (exp(e_b) - 1) a_b a_b^T + (exp(f_b) - 1) a2_b a2_b^T) (x - c_b) + t_b )
Start: each bone where the body skeleton fit put it (closest size-and-pose to its vertices there).
Least squares (Levenberg-Marquardt with a direct sparse solve; Jacobian by finite differences), every scale from data:
  joints   every two touching bones that form a joint (bone_gaps.npz, less the pairs agents/joint_names.py found to be
           mere neighbours) share the middle of their contact: T_a(m) = T_b(m); for a held bone
           the place of its own contact points.                                                              / sigma_c
  points   the named joints (shoulder, elbow, wrist, hip, knee, ankle), carried by the bones its line runs along
           (the knee by thigh and shin bone, not by the kneecap, which only rides on the joint)                 / sigma_j
  size     s_a - s_b for touching free bones; turn r_a - r_b likewise                                        / sigma_s, sigma_r
  stretch  e_b, f_b                                                                                          / sigma_e
  knees    (when SAM 3 found them on the bare front view) the kneecap bone under the kneecap bump on the skin, the
           thigh bone's end centred in the leg there, its side-to-side axis straight across      / the left-right agreement
  crotch   the atlas's skin point under the sit bones, carried by each hip bone, at the height of the figure's crotch
           (where its skin crosses its own mirror plane under the hips)                          / the knees' scale
  shared   a bone of a named joint with no bone-to-bone contact to a held part (the ulna at the wrist) carries the
           joint's place to where the held bones put it                                         / their atlas distance
  kneecap  the patellar ligament keeps its length as a share of the kneecap's size (attachment areas from the atlas) / sigma_c
  mirror   a left bone and its right partner share one size and one pair of stretches (not a penalty: the same numbers);
           the named joints are pooled left with right as far as the figure is its own mirror image (symmetry.py)
  inside   every free bone hull point's depth outside its solid - the whole body's, or, for a bone that lies in one
           separate piece of flesh (a leg, an arm: flesh/islands.npz), that piece's own - as a penalty whose scale is
           halved until the worst point is inside to within one grid cell.
Scales: sigma_c = the skeleton fit's joint gap; sigma_j = the detector's spread, from the figure's left-right symmetry
(left joints mirrored against right); sigma_s and sigma_e the
skeleton fit's spreads of bone size and shape; sigma_r as measured on the hands (two named points on one bone).
Writes <fits dir>/body_assembly.npz (V: every atlas vertex; bones placed, the rest as the skeleton fit left them) and
body_assembly.json (the checks, before = the body skeleton fit)."""
import sys, os, re, json, glob, time, numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree, ConvexHull
from scipy.spatial.transform import Rotation as Rot
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import symmetry
FD, MD = sys.argv[1:3]; t00 = time.time()
def log(*a): print('BA', *a, flush=True)
W = np.load(os.path.join(FD, 'atlas_warp.npz')); V0 = W['V0'].astype(np.float64); OB = W['obj']; names = [str(x) for x in W['names']]; FB = W['F_bone']; sysn = [str(x) for x in W['system']]; nid = {n: i for i, n in enumerate(names)}
SK = np.load(os.path.join(FD, 'skeleton_fit.npz')); Vsk = SK['V_rigid'].astype(np.float64); SKJ = json.load(open(os.path.join(FD, 'skeleton_fit.json')))
NM = json.load(open(os.path.join(MD, 'lines', 'points_named.json'), encoding='utf-8')); G = np.load(os.path.join(FD, 'bone_gaps.npz'))
allb = [int(b) for b in np.unique(OB) if sysn[b] == 'bone']; idx = {b: np.flatnonzero(OB == b) for b in allb}
# --- held parts: their fitted places
Vh = Vsk.copy(); held = {}
S2 = np.load(os.path.join(FD, 'skull_stage2.npz')); moved = np.linalg.norm(S2['V'] - S2['V_fit'], axis=1) > 0
for b in allb:
    if moved[idx[b]].any(): held[b] = 'skull'; Vh[idx[b]] = S2['V'][idx[b]]
for f in sorted(glob.glob(os.path.join(FD, '*_bones.npz'))):
    Z = np.load(f); reg = os.path.basename(f)[:-10]
    for b in Z['bones']: held[int(b)] = reg; Vh[idx[int(b)]] = Z['V'][idx[int(b)]]
free = [b for b in allb if b not in held]; B = len(free); bi = {b: k for k, b in enumerate(free)}; C = np.array([V0[idx[b]].mean(0) for b in free])
EV = [np.linalg.eigh(np.cov((V0[idx[b]] - V0[idx[b]].mean(0)).T))[1] for b in free]; AX = np.array([e_[:, -1] for e_ in EV]); AX2 = np.array([e_[:, -2] for e_ in EV])   # each bone's own main and second axis
# --- left and right bones are the same bone: the atlas's left bones are the exact mirror images of its right ones, and a
# body's two sides are built alike whatever its pose. A right bone therefore has no size and stretches of its own: it
# takes its left partner's (shared numbers, not a penalty - there is no scale to choose). Its turn and shift stay its own.
TL, TR = [], []
for k, b in enumerate(free):
    r_ = nid.get(names[b][:-2] + '.r') if names[b].endswith('.l') else None
    if r_ in bi and len(idx[b]) == len(idx[r_]) and np.allclose(V0[idx[b]] * np.array([-1, 1, 1]), V0[idx[r_]], atol=np.finfo(np.float32).eps * np.abs(V0).max() * 8): TL.append(k); TR.append(bi[r_])
TL, TR = np.array(TL, int), np.array(TR, int)
def QQ(x): q = x.reshape(B, NPB).copy(); q[TR, 6:9] = q[TL, 6:9]; return q
log('%d bones: %d held (%s), %d free' % (len(allb), len(held), ', '.join('%s %d' % (k, list(held.values()).count(k)) for k in sorted(set(held.values()))), B))
# --- named joints of the body (COCO-WholeBody order: left, right)
Zraw = np.load(os.path.join(MD, 'body', 'joints3d.npz'))['X'].astype(np.float64); SYMF = symmetry.figure_mirror(MD); Zb = symmetry.body_joints(MD, SYMF, log=lambda *a: print('BA', *a, flush=True)); CO = {'shoulder': 5, 'elbow': 7, 'wrist': 9, 'hip': 11, 'knee': 13, 'ankle': 15}; P = []
for k, i0 in CO.items():
    for o, sd in ((0, '.l'), (1, '.r')):
        e = NM['points'][k + sd]
        on = {r_['bones'][j] for r_ in NM['lines'] if r_['side'] == sd and k in r_['points'] for i_ in [r_['points'].index(k)] for j in (i_ - 1, i_) if 0 <= j < len(r_['bones'])} & set(e['bones'])   # the bones the line runs along here
        if e['atlas'] is not None and np.isfinite(Zb[i0 + o]).all(): P.append((k + sd, np.array(e['atlas']), Zb[i0 + o], [nid[n] for n in (sorted(on) or e['bones'])]))
A0 = np.array([p[1] for p in P]); X = np.array([p[2] for p in P])
def umeyama(A, Bx):
    ma, mb = A.mean(0), Bx.mean(0); U, D_, Vt = np.linalg.svd((Bx - mb).T @ (A - ma) / len(A)); Sg = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]); R_ = U @ Sg @ Vt
    return np.trace(np.diag(D_) @ Sg) / ((A - ma) ** 2).sum(1).mean(), R_, mb - np.trace(np.diag(D_) @ Sg) / ((A - ma) ** 2).sum(1).mean() * R_ @ ma
s1, R1, t1 = umeyama(A0, X); W2A = lambda Y: (Y - t1) @ R1 / s1; A2W = lambda Y: s1 * Y @ R1.T + t1; Xa = W2A(X)
log('whole body: size %.3f from %d named joints; left after one size and pose: median %.1f mm, max %.1f mm' % (s1, len(P), 1000 * np.median(np.linalg.norm(A2W(A0) - X, axis=1)), 1000 * np.linalg.norm(A2W(A0) - X, axis=1).max()))
# --- start: each free bone where the skeleton fit put it
NPB = 9; x0 = np.zeros((B, NPB))
for k, b in enumerate(free):
    Y = W2A(Vsk[idx[b]]); s_, R_, t_ = umeyama(V0[idx[b]], Y); x0[k, :3] = Rot.from_matrix(R_).as_rotvec(); x0[k, 6] = np.log(s_); x0[k, 3:6] = s_ * R_ @ C[k] + t_ - C[k]
x0[TL, 6] = x0[TR, 6] = (x0[TL, 6] + x0[TR, 6]) / 2                # a pair starts at the mean of its two sizes
def TF(x, pts, own):                                               # atlas frame (S applied after)
    q = QQ(x); d = pts - C[own]; a = AX[own]; a2 = AX2[own]; d = d + (np.exp(q[own, 7]) - 1)[:, None] * (d * a).sum(1)[:, None] * a + (np.exp(q[own, 8]) - 1)[:, None] * (d * a2).sum(1)[:, None] * a2
    return C[own] + np.exp(q[own, 6])[:, None] * np.einsum('nij,nj->ni', Rot.from_rotvec(q[own, :3]).as_matrix(), d) + q[own, 3:6]
# --- joints: measured touching pairs. free-free: shared middle; free-held: the held bone's own contact points
jn = os.path.join(FD, 'joints_named.json'); NOJ = {tuple(sorted((nid[e_['a']], nid[e_['b']]))) for e_ in json.load(open(jn, encoding='utf-8')) if not e_['joint']} if os.path.exists(jn) else set()   # touching, but no joint between them (agents/joint_names.py)
thr = float(G['threshold']); n_double = 0; fh_h = []; ff_a, ff_b, ff_m, fh_b, fh_m, fh_t, fh_n = [], [], [], [], [], [], []
for (a, b), g in zip(G['pairs'], G['rel_gap']):
    a, b = int(a), int(b)
    if g >= thr or a not in idx or b not in idx or (a in held and b in held) or tuple(sorted((a, b))) in NOJ: continue
    Pa, Pb = V0[idx[a]], V0[idx[b]]; d, i = cKDTree(Pb).query(Pa); k = np.argsort(d)[:max(3, len(Pa) // 100)]; groups = [np.arange(len(k))]
    ax_ = np.linalg.eigh(np.cov((Pa - Pa.mean(0)).T))[1][:, -1]; u_ = Pa[k] @ ax_; o_ = np.argsort(u_); j_ = int(np.argmax(np.diff(u_[o_]))) if len(k) > 3 else 0   # two separate contacts (shin and calf bone, radius and ulna
    if len(k) > 3 and u_[o_][j_ + 1] - u_[o_][j_] > max(u_[o_][j_] - u_[o_][0], u_[o_][-1] - u_[o_][j_ + 1]): groups = [o_[:j_ + 1], o_[j_ + 1:]]; n_double += 1      # meet at both ends): the gap between the groups exceeds either group's own length
    for g_ in groups:
        ia, ib = idx[a][k[g_]], idx[b][i[k[g_]]]; m = (V0[ia].mean(0) + V0[ib].mean(0)) / 2
        if a in bi and b in bi: ff_a.append(bi[a]); ff_b.append(bi[b]); ff_m.append(m)
        else:
            fr, hd, ih = (a, b, ib) if a in bi else (b, a, ia); fh_b.append(bi[fr]); fh_m.append(m); fh_t.append(W2A(Vh[ih].mean(0)[None])[0]); fh_n.append(held[hd]); fh_h.append((bi[fr], hd))
ff_a, ff_b, ff_m, fh_b, fh_m, fh_t = map(np.array, (ff_a, ff_b, ff_m, fh_b, fh_m, fh_t))
log('joints: %d between free bones, %d to held parts (%s); %d bone pairs meet in two places' % (len(ff_a), len(fh_b), ', '.join('%s %d' % (k, fh_n.count(k)) for k in sorted(set(fh_n))), n_double))
# --- solid model of the whole body (grid two surface edges fine: the whole figure at one edge does not fit in memory)
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']; hstep = 2 * float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1)))
glo = MV.min(0) - 4 * hstep; gn = np.ceil((MV.max(0) + 4 * hstep - glo) / hstep).astype(int) + 1; T = MV[MF]; nsub = np.clip(np.ceil(np.maximum(np.linalg.norm(T[:, 1] - T[:, 0], axis=1), np.linalg.norm(T[:, 2] - T[:, 0], axis=1)) / (hstep / 2)).astype(int), 1, 64); occ = np.zeros(gn, bool)
for n_ in np.unique(nsub):
    Tn = T[nsub == n_]; u, v = np.meshgrid(np.arange(n_ + 1) / n_, np.arange(n_ + 1) / n_); m_ = (u + v) <= 1 + 1e-9; u, v = u[m_], v[m_]
    Pn = (Tn[:, None, 0] * (1 - u - v)[None, :, None] + Tn[:, None, 1] * u[None, :, None] + Tn[:, None, 2] * v[None, :, None]).reshape(-1, 3); ii = np.rint((Pn - glo) / hstep).astype(int); occ[ii[:, 0], ii[:, 1], ii[:, 2]] = True
for seal in range(1, 17):
    thick = ndimage.binary_dilation(occ, iterations=seal); fill = ndimage.binary_fill_holes(thick)
    if fill.sum() - thick.sum() > thick.sum(): break
else: print('WARNING body_assembly.py: the solid did not close within 17 sealing steps; the widest sealing is used and the result may be wrong', flush=True)
solid = ndimage.binary_erosion(fill, iterations=seal) | occ; sdf = ((ndimage.distance_transform_edt(solid) - ndimage.distance_transform_edt(~solid)) * hstep).astype(np.float32); del thick, fill
def depth(Pw): return ndimage.map_coordinates(sdf, ((Pw - glo) / hstep).T, order=1, mode='nearest')
log('solid: grid %s at %.2f mm; gaps sealed by %d cells; volume %.0f cm3 (%.0f s)' % (tuple(int(g) for g in gn), 1000 * hstep, seal, 1e6 * solid.sum() * hstep ** 3, time.time() - t00))
hull, hown = [], []
for b in free:
    ii = idx[b]
    try: ii = ii[np.unique(ConvexHull(V0[ii]).vertices)] if len(ii) >= 8 else ii
    except Exception: pass
    hull.append(ii); hown.append(np.full(len(ii), bi[b]))
hull = np.concatenate(hull); hown = np.concatenate(hown); H0 = V0[hull]
# --- a limb's bones stay inside that limb. The whole-body solid is one piece where two limbs press together (her legs),
# so nothing there keeps a bone in its own leg. The flesh step (parts/flesh_islands.py) already cut the skin into separate
# pieces at such contacts: each closed piece gets its own solid (its open ends - the cut at the hip, at the shoe, along the
# contact - closed by a fan of triangles to the middle of each open edge loop), and a bone that starts inside one piece
# may not enter any other piece.
hsol = np.zeros(len(hull), int); SOL = [(sdf, glo, hstep)]; CAPS = [[]]; isf = os.path.join(MD, 'flesh', 'islands.npz')
if os.path.exists(isf):
    ISL = np.load(isf)['island']; _, wid = np.unique(np.round(MV, 6), axis=0, return_inverse=True); wid = wid.ravel(); mt = cKDTree(MV); est = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1)))
    near = ISL[mt.query(A2W(TF(x0.ravel(), H0, hown)))[1]]; own_isl = {}
    for b in range(B):
        u_, c_ = np.unique(near[hown == b], return_counts=True); k_ = int(u_[c_.argmax()])
        if k_ >= 0 and c_.max() > (hown == b).sum() / 2: own_isl[b] = k_
    for k_ in sorted(set(own_isl.values())):
        Fk = MF[(ISL[MF] == k_).all(1)]; Wk = wid[Fk]; E_ = np.sort(np.r_[Wk[:, [0, 1]], Wk[:, [1, 2]], Wk[:, [2, 0]]], axis=1); ue, ce = np.unique(E_, axis=0, return_counts=True); be = ue[ce == 1]      # open edges (after welding the UV seams)
        rv = np.zeros(wid.max() + 1, int); rv[wid] = np.arange(len(MV)); bv = np.unique(be); loc = {v: i for i, v in enumerate(bv)}
        from scipy.sparse import coo_matrix as _coo
        from scipy.sparse.csgraph import connected_components as _cc
        nl, lab_ = _cc(_coo((np.ones(len(be)), ([loc[a] for a in be[:, 0]], [loc[b_] for b_ in be[:, 1]])), shape=(len(bv), len(bv))), directed=False)
        Tk = [MV[Fk]]; caps_ = []; pc_ = MV[np.unique(Fk)].mean(0)
        for l_ in range(nl):
            vv = bv[lab_ == l_]; cen_ = MV[rv[vv]].mean(0); ee = be[lab_[[loc[a] for a in be[:, 0]]] == l_]; Tk.append(np.stack([MV[rv[ee[:, 0]]], MV[rv[ee[:, 1]]], np.repeat(cen_[None], len(ee), 0)], axis=1)); nn_ = np.linalg.svd(MV[rv[vv]] - cen_)[2][-1]; nn_ = nn_ if nn_ @ (cen_ - pc_) > 0 else -nn_; caps_.append((cen_, nn_, float(np.linalg.norm(MV[rv[ee[:, 0]]] - MV[rv[ee[:, 1]]], axis=1).sum())))
        Tk = np.vstack(Tk); g0 = Tk.reshape(-1, 3).min(0) - 4 * est; gk = np.ceil((Tk.reshape(-1, 3).max(0) + 4 * est - g0) / est).astype(int) + 1; oc = np.zeros(gk, bool)
        ns = np.clip(np.ceil(np.maximum(np.linalg.norm(Tk[:, 1] - Tk[:, 0], axis=1), np.linalg.norm(Tk[:, 2] - Tk[:, 0], axis=1)) / (est / 2)).astype(int), 1, 256)
        for n_ in np.unique(ns):
            Tn = Tk[ns == n_]; u, v = np.meshgrid(np.arange(n_ + 1) / n_, np.arange(n_ + 1) / n_); m_ = (u + v) <= 1 + 1e-9; u, v = u[m_], v[m_]
            for c0 in range(0, len(Tn), 2000):
                Tc = Tn[c0:c0 + 2000]; Pn = (Tc[:, None, 0] * (1 - u - v)[None, :, None] + Tc[:, None, 1] * u[None, :, None] + Tc[:, None, 2] * v[None, :, None]).reshape(-1, 3); ii = np.rint((Pn - g0) / est).astype(int); oc[ii[:, 0], ii[:, 1], ii[:, 2]] = True
        closed_ = False
        for seal_ in range(1, 5):
            th_ = ndimage.binary_dilation(oc, iterations=seal_); fl_ = ndimage.binary_fill_holes(th_)
            if fl_.sum() - th_.sum() > th_.sum(): closed_ = True; break
        if not closed_: log('flesh piece %d does not close into a solid (an open patch of skin): not used' % k_); continue
        so_ = ndimage.binary_erosion(fl_, iterations=seal_) | oc; sd_ = ((ndimage.distance_transform_edt(so_) - ndimage.distance_transform_edt(~so_)) * est).astype(np.float32); SOL.append((sd_, g0, est)); big_ = max(c_[2] for c_ in caps_); CAPS.append([c_ for c_ in caps_ if c_[2] >= big_ / 2])   # the piece's real openings: loops at least half as long as its longest
        P0_ = A2W(TF(x0.ravel(), H0, hown)); in_ = ndimage.map_coordinates(sd_, ((P0_ - g0) / est).T, order=1, mode='nearest') > 0                       # a bone belongs to the piece if most of it starts inside the piece's solid
        bs_ = [b for b, kk in own_isl.items() if kk == k_ and in_[hown == b].mean() > 0.5]
        if not bs_: SOL.pop(); CAPS.pop(); continue
        hsol[np.isin(hown, bs_)] = len(SOL) - 1
        log('own solid for flesh piece %d: %d open edge loops closed, grid %s at %.2f mm, sealed by %d cells, volume %.0f cm3; bones kept inside it: %s' % (k_, nl, tuple(int(g) for g in gk), 1000 * est, seal_, 1e6 * so_.sum() * est ** 3, ', '.join(names[free[b]] for b in bs_)))
def depth_h(Pw):
    """a trunk bone: inside the body. A limb bone: inside its own piece, or - past one of the piece's open ends (the hip,
    the shoulder) - inside the body but in no other limb's piece. The gap between two limbs is not inside."""
    samp = lambda i_, P_: ndimage.map_coordinates(SOL[i_][0], ((P_ - SOL[i_][1]) / SOL[i_][2]).T, order=1, mode='nearest'); out_ = samp(0, Pw)
    for i_ in range(1, len(SOL)):
        m_ = hsol == i_
        if not m_.any(): continue
        P_ = Pw[m_]; other = np.max([samp(j_, P_) for j_ in range(1, len(SOL)) if j_ != i_], axis=0) if len(SOL) > 2 else np.full(len(P_), -np.inf)
        beyond = np.max([(P_ - c_) @ n_ for c_, n_, _ in CAPS[i_]], axis=0); out_[m_] = np.maximum(samp(i_, P_), np.minimum(np.minimum(out_[m_], -other), beyond))
    return out_
# --- the fit
drow = [(k, bi[b]) for k, p in enumerate(P) for b in p[3] if b in bi]; dk = np.array([k for k, _ in drow]); db = np.array([b for _, b in drow])
sig_c = SKJ['joint_gap_mm']['median'] / 1000 / s1; sig_s = float(SKJ['scales_final']['s']); sig_e = float(SKJ['scales_final']['e'])
tr = [json.load(open(f))['scales'].get('turn_deg') for f in glob.glob(os.path.join(FD, '*_bones.json')) if json.load(open(f)).get('stretches')]; sig_r = float(np.radians(np.median(tr)))
# kneecap: the atlas has the patellar ligament's two attachment areas (origin on the kneecap, end on the shin bone). The
# distance between them, as a share of the kneecap's own size, is kept (the ligament does not stretch; the ratio of its
# length to the kneecap's is about 1 in people - Insall and Salvati 1971). Hard rule: each area must lie on its bone.
LIG = []; AT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(os.path.dirname(os.path.dirname(MD.rstrip('/\\'))), 'figures', 'models', 'zanatomy', 'export', 'atlas.npz')
for cand in (AT, r'E:\figures\models\zanatomy\export\atlas.npz'):
    if os.path.exists(cand): AT = cand; break
if os.path.exists(AT):
    ZA = np.load(AT, allow_pickle=True); zn = [str(n_) for n_ in ZA['names']]
    for sd in ('l', 'r'):
        o_, e2_ = 'Patellar ligament.o' + sd, 'Patellar ligament.e' + sd; pat, tib = nid.get('Patella.' + sd), nid.get('Tibia.' + sd)
        if o_ not in zn or e2_ not in zn or pat not in bi or tib not in bi: continue
        Po, Pe = ZA['V'][ZA['obj'] == zn.index(o_)].astype(float), ZA['V'][ZA['obj'] == zn.index(e2_)].astype(float); do = cKDTree(V0[idx[pat]]).query(Po)[0].min(); de = cKDTree(V0[idx[tib]]).query(Pe)[0].min()
        gapmax = float(np.median(np.linalg.norm(V0[FB[:, 0]] - V0[FB[:, 1]], axis=1)))                                                  # "on the bone": within one atlas mesh edge
        if do > gapmax or de > gapmax: log('patellar ligament %s: its areas are not on the kneecap and shin bone (%.1f, %.1f mm): not used' % (sd, 1000 * do, 1000 * de)); continue
        LIG.append((bi[pat], bi[tib], Po.mean(0), Pe.mean(0), float(np.linalg.norm(Po.mean(0) - Pe.mean(0)))))
    log('patellar ligament: %d knees; atlas length %s mm between its attachment areas' % (len(LIG), ', '.join('%.1f' % (1000 * l_[4]) for l_ in LIG)))
def CORD(x):
    if not LIG: return np.zeros(0)
    q = QQ(x); return np.array([np.linalg.norm(TF(x, po[None], np.array([a]))[0] - TF(x, pe[None], np.array([b]))[0]) - L0 * np.exp(q[a, 6]) for a, b, po, pe, L0 in LIG])
# --- knees from the surface. A detector's knee point is a guess at a joint centre inside the leg; what can be seen is the
# knee on the skin. SAM 3 "knee" on the bare front view (parts/part_concept_masks.py on body/clay_a000) gives each knee's
# region; in it the kneecap is the part that stands out in front of the straight line from the region's top to its bottom
# (side profile). Targets, all at the scale of how well left and right agree after mirroring:
#   kneecap   the front of the kneecap bone lies under that bump, one surface edge deep (bone under thin skin)
#   middle    the two sides of the thigh bone's end (at the kneecap's level) have their middle at the middle of the
#             leg's cross-section there
#   axis      the line between those two sides - the knee's bending axis - runs straight across the leg   / sigma_r
KNEE = []; kf = os.path.join(MD, 'body', 'masks', 'clay_a000.npz')
if os.path.exists(kf) and 's_knee' in np.load(kf).files and os.path.exists(isf):
    from skimage.filters import threshold_otsu
    km = np.load(kf); kres = int(km['res']); kS = km['s_knee']; kM = np.unpackbits(km['m_knee'], axis=1)[:, :kres * kres].reshape(-1, kres, kres).astype(bool); kc = json.load(open(os.path.join(MD, 'body', 'cameras.json')))['clay_a000']
    Mi_ = np.array(kc['matrix_inv']); cc_ = MV @ Mi_[:3, :3].T + Mi_[:3, 3]; pp_ = np.clip(np.rint(np.c_[(cc_[:, 0] / kc['scale'] + 0.5) * kres, (0.5 - cc_[:, 1] / kc['scale']) * kres]).astype(int), 0, kres - 1)
    up_w = np.array([0, 0, 1.0]); fw_ = (Zb[17] - Zb[19]) + (Zb[20] - Zb[22]); fw_[2] = 0; fwd_w = fw_ / np.linalg.norm(fw_); lat_w = np.cross(up_w, fwd_w); xm_ = float(np.mean([(Zb[i_] + Zb[i_ + 1])[0] / 2 for i_ in CO.values()]))   # forward: heels to toes
    fwd_a, up_a, lat_a = fwd_w @ R1, up_w @ R1, lat_w @ R1
    for k_ in np.flatnonzero(kS > threshold_otsu(kS)):
        ii = np.flatnonzero(kM[k_][pp_[:, 1], pp_[:, 0]] & (ISL >= 0))
        if not len(ii): continue
        isl_ = int(np.bincount(ISL[ii]).argmax()); ii = ii[ISL[ii] == isl_]; Pk = MV[ii]; f_ = Pk @ fwd_w; Pk = Pk[f_ >= np.median((MV[ISL == isl_] @ fwd_w)[np.abs(MV[ISL == isl_][:, 2] - Pk[:, 2].mean()) < np.ptp(Pk[:, 2]) / 2])]   # the front half of the leg there
        sd = 'l' if Pk[:, 0].mean() > xm_ else 'r'; pat, fem = nid.get('Patella.' + sd), nid.get('Femur.' + sd)
        if pat not in bi or fem not in bi: continue
        zb = np.arange(Pk[:, 2].min(), Pk[:, 2].max() + est, est); prof = np.array([(Pk[np.abs(Pk[:, 2] - z_) < est] @ fwd_w).max() if (np.abs(Pk[:, 2] - z_) < est).any() else np.nan for z_ in zb]); ok_ = np.isfinite(prof); zb, prof = zb[ok_], prof[ok_]
        chord = prof[0] + (prof[-1] - prof[0]) * (zb - zb[0]) / (zb[-1] - zb[0]); jb_ = int(np.argmax(prof - chord)); sl_ = Pk[np.abs(Pk[:, 2] - zb[jb_]) < est]; Bp = sl_[np.argmax(sl_ @ fwd_w)]                                          # the kneecap bump on the skin
        cs_ = MV[(ISL == isl_) & (np.abs(MV[:, 2] - zb[jb_]) < est)] @ lat_w; cmid = float((cs_.min() + cs_.max()) / 2)
        Pp = V0[idx[pat]]; cpa = Pp.mean(0); pf = cpa + fwd_a * ((Pp - cpa) @ fwd_a).max(); hh = np.abs((Pp - cpa) @ up_a).max(); Pf = V0[idx[fem]]; lv = Pf[np.abs((Pf - cpa) @ up_a) <= hh]; u_ = lv @ lat_a; o_ = np.argsort(u_); n1 = max(3, len(lv) // 100)
        KNEE.append(dict(side=sd, pat=bi[pat], fem=bi[fem], pf=pf, pm=lv[o_[:n1]].mean(0), pl=lv[o_[-n1:]].mean(0), B=Bp, cmid=cmid, bump_mm=1000 * float((prof - chord)[jb_]), region_z=(float(zb[0]), float(zb[-1]))))
    if len(KNEE) == 2: Bm = KNEE[0]['B'] * np.array([-1, 1, 1]) + np.array([2 * xm_, 0, 0]); sig_k = max(float(np.linalg.norm(Bm - KNEE[1]['B']) / np.sqrt(2)), est)
    else: sig_k = est
    for K_ in KNEE: log('knee %s on the surface: region %.0f..%.0f mm high; kneecap bump %.1f mm proud at height %.0f mm, %.0f mm from the midline; leg middle there %.0f mm' % (K_['side'], 1000 * K_['region_z'][0], 1000 * K_['region_z'][1], K_['bump_mm'], 1000 * K_['B'][2], 1000 * abs(K_['B'][0] - xm_), 1000 * abs(K_['cmid'] - xm_ * lat_w[0])))
    log('knee landmark scale (left against right, mirrored): %.1f mm' % (1000 * sig_k))
def KN(x):
    r_ = []
    for K_ in KNEE:
        Wp = A2W(TF(x, K_['pf'][None], np.array([K_['pat']])))[0]; r_ += list((Wp - (K_['B'] - fwd_w * est)) / sig_k); Wm, Wl = A2W(TF(x, np.array([K_['pm'], K_['pl']]), np.array([K_['fem'], K_['fem']])))
        ax_ = (Wl - Wm) / np.linalg.norm(Wl - Wm); r_ += [(((Wm + Wl) / 2) @ lat_w - K_['cmid']) / sig_k, (ax_ @ fwd_w) / sig_r, (ax_ @ up_w) / sig_r]
    return np.array(r_)
# --- the crotch. The detector's hip points are guesses at joint centres; nothing above says where the pelvis ends below.
# What can be seen is the crotch: on the figure's own mirror plane (symmetry.py) the skin crosses the plane along the
# trunk's outline, and its underside between the legs is the first crossing met going straight down from the height of
# the hips. In the atlas the same: straight down from the middle between the two sit bones (the lowest points of the hip
# bones) to the atlas's skin. That atlas skin point, carried by each hip bone, lies at the height of the figure's crotch
# below it (soft tissue under the sit bones grows with the bone, since the point is carried by the bone's own size).
# Scale: that of the other surface landmarks (the knees' left-right agreement). Clothing over the crotch is not removed.
def vline(T_, x_, y_):
    """heights at which the vertical line through (x_, y_) crosses the triangles T_ (n, 3, 3)."""
    a_, b_, c_ = T_[:, 0, :2] - (x_, y_), T_[:, 1, :2] - (x_, y_), T_[:, 2, :2] - (x_, y_); w0 = b_[:, 0] * c_[:, 1] - b_[:, 1] * c_[:, 0]; w1 = c_[:, 0] * a_[:, 1] - c_[:, 1] * a_[:, 0]; w2 = a_[:, 0] * b_[:, 1] - a_[:, 1] * b_[:, 0]
    k_ = ((w0 >= 0) & (w1 >= 0) & (w2 >= 0)) | ((w0 <= 0) & (w1 <= 0) & (w2 <= 0)); sm = (w0 + w1 + w2)[k_]; k_[np.flatnonzero(k_)[sm == 0]] = False; sm = sm[sm != 0]
    return (w0[k_] * T_[k_, 0, 2] + w1[k_] * T_[k_, 1, 2] + w2[k_] * T_[k_, 2, 2]) / sm
CRO = None; hl_, hr_ = nid.get('Hip bone.l'), nid.get('Hip bone.r')
if hl_ in bi and hr_ in bi and 'F_skin' in W.files:
    sit = (V0[idx[hl_]][V0[idx[hl_]][:, 2].argmin()] + V0[idx[hr_]][V0[idx[hr_]][:, 2].argmin()]) / 2; za_ = vline(V0[W['F_skin']], sit[0], sit[1]); za_ = za_[za_ < sit[2]]
    zh_ = float((Zb[11, 2] + Zb[12, 2]) / 2); yh_ = np.r_[Vh[idx[hl_]][:, 1], Vh[idx[hr_]][:, 1]]; est_c = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1))); ys_ = np.arange(yh_.min(), yh_.max() + est_c, est_c); Tm_ = MV[MF]
    xs_ = (SYMF['d'] - SYMF['n'][1] * ys_ - SYMF['n'][2] * zh_) / SYMF['n'][0]; near_ = (Tm_[:, :, 0].min(1) <= xs_.max()) & (Tm_[:, :, 0].max(1) >= xs_.min()) & (Tm_[:, :, 2].min(1) < zh_); Tm_ = Tm_[near_]; zc_ = np.full(len(ys_), np.nan)
    for i_, (x_, y_) in enumerate(zip(xs_, ys_)):
        z_ = vline(Tm_, x_, y_); z_ = z_[z_ < zh_]
        if len(z_) and (vline(MV[MF][(MV[MF][:, :, 2].max(1) >= zh_)], x_, y_) > zh_).any(): zc_[i_] = z_.max()                 # only where the line starts inside the body (skin above it too)
    if len(za_) and np.isfinite(zc_).any():
        ca_ = np.array([sit[0], sit[1], za_.max()]); CRO = dict(ca=ca_, ys=ys_[np.isfinite(zc_)], zc=zc_[np.isfinite(zc_)], b=np.array([bi[hl_], bi[hr_]]), gap_atlas=float(sit[2] - za_.max()))
        log('crotch: atlas skin %.1f mm under the sit bones (%.2f%% of the atlas height); the figure\'s skin crosses its mirror plane under the hips at %.0f..%.0f mm high over %d front-to-back steps' % (1000 * CRO['gap_atlas'], 100 * CRO['gap_atlas'] / np.ptp(V0[np.unique(W['F_skin'])][:, 2]), 1000 * CRO['zc'].min(), 1000 * CRO['zc'].max(), len(CRO['ys'])))
if 'sig_k' not in globals(): sig_k = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1)))   # no knees found: one surface edge
def CR(x):
    if CRO is None: return np.zeros(0)
    Pw = A2W(TF(x, np.repeat(CRO['ca'][None], 2, 0), CRO['b'])); return (Pw[:, 2] - np.interp(Pw[:, 1], CRO['ys'], CRO['zc'])) / sig_k
# --- a named joint is one place shared by ALL its bones. The contact joints above tie bones that touch bone to bone; some
# bones of a joint do not (the ulna reaches the wrist bones through a disc of cartilage, not in the skeleton), so the
# forearm could hang on the hand by the radius alone. The naming step (agents/point_names.py) lists every bone of each
# joint and its place in the atlas: a free bone of the joint that has no contact joint with the held part carries that
# place to where the held bones of the joint put it. (Tying every bone this way was tried: with the contact joint it
# fixes two points of a bone and locks the joint - the shoulders could no longer lift.) Scale: how far these places lie from their
# own bones in the atlas (a joint's middle is in the gap, not on a bone).
NJ = []; njd = []
for k_, e_ in NM['points'].items():
    if e_.get('kind') != 'joint' or e_.get('atlas') is None: continue
    ids_ = [nid[n_] for n_ in e_['bones'] if nid.get(n_) in idx]; a_ = np.array(e_['atlas']); fr_ = [bi[b] for b in ids_ if b in bi]; hd_ = [b for b in ids_ if b in held]
    fr_ = [b for b in fr_ if not any((b, h_) in fh_h for h_ in hd_)]   # only a bone with no bone-to-bone joint to the held part needs this tie (the ulna at the wrist); a second tie would lock a bone that already has one
    if not fr_ or not hd_: continue
    njd += [float(cKDTree(V0[idx[b]]).query(a_)[0]) for b in ids_]; tgt_ = None
    if hd_:                                                         # where the held bones carry the place: each one's closest size-and-pose from its atlas shape to its fitted shape
        im_ = []
        for b in hd_: s_, R_, t_ = umeyama(V0[idx[b]], W2A(Vh[idx[b]])); im_.append(s_ * R_ @ a_ + t_)
        tgt_ = np.mean(im_, axis=0)
    if tgt_ is None and len(fr_) < 2: continue
    NJ.append((k_, a_, fr_, tgt_))
sig_n = float(np.median(njd)) if njd else 1.0; NJ_rows = [[b] if t_ is not None else [b, fr_[0]] for _, _, fr_, t_ in NJ for b in (fr_ if t_ is not None else fr_[1:])]
log('named joints as shared places: %d joints (%d tied to held parts: %s); scale %.1f mm' % (len(NJ), sum(t_ is not None for *_, t_ in NJ), ', '.join(k_ for k_, _, _, t_ in NJ if t_ is not None), 1000 * sig_n * s1))
def NJR(x):
    r_ = []
    for _, a_, fr_, t_ in NJ:
        Pb = TF(x, np.repeat(a_[None], len(fr_), 0), np.array(fr_))
        r_ += [((Pb[i_] - t_) / sig_n) for i_ in range(len(fr_))] if t_ is not None else [((Pb[i_] - Pb[0]) / sig_n) for i_ in range(1, len(fr_))]
    return np.concatenate(r_) if r_ else np.zeros(0)
# --- check only: how far the left bones are from the mirror images of the right ones (six points per bone: its centre
# moved one standard deviation each way along its three own axes), in the figure's own mirror plane (symmetry.py).
SYMP = []; isf_ = os.path.join(MD, 'flesh', 'islands.npz'); est_s = float(np.median(np.linalg.norm(SYMF['MV'][np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F'][:, 0]] - SYMF['MV'][np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F'][:, 1]], axis=1)))
bv_ = np.flatnonzero(np.isin(OB, allb)); own_ = OB[bv_][cKDTree(Vh[bv_]).query(SYMF['MV'][SYMF['flesh']])[1]]; rs_ = SYMF['resid'][SYMF['flesh']]; rmed = float(np.median(rs_)); loc_ = {int(b): float(np.median(rs_[own_ == b])) for b in np.unique(own_)}
for k, b in enumerate(free):
    n_ = names[b]
    if not n_.endswith('.l') or nid.get(n_[:-2] + '.r') not in bi: continue
    r_ = bi[nid[n_[:-2] + '.r']]; lam_, ev_ = np.linalg.eigh(np.cov((V0[idx[b]] - C[k]).T)); pts = np.concatenate([C[k] + sg * np.sqrt(max(l_, 0)) * ev_[:, i_] for i_, l_ in enumerate(lam_) for sg in (1, -1)]).reshape(-1, 3)
    if np.abs(V0[idx[b]] * np.array([-1, 1, 1]) - V0[idx[free[r_]]]).max() > est_s if len(idx[b]) == len(idx[free[r_]]) else True: continue      # the atlas pair must itself be a mirror pair
    SYMP.append((k, r_, pts, pts * np.array([-1, 1, 1]), max((loc_.get(b, rmed) + loc_.get(free[r_], rmed)) / 2, est_s)))
sy_l = np.concatenate([np.full(len(p_[2]), p_[0]) for p_ in SYMP]); sy_r = np.concatenate([np.full(len(p_[2]), p_[1]) for p_ in SYMP]); sy_pl = np.vstack([p_[2] for p_ in SYMP]); sy_pr = np.vstack([p_[3] for p_ in SYMP]); sy_sig = np.concatenate([np.full(len(p_[2]), p_[4]) for p_ in SYMP])
log('left-right pairs of free bones sharing size and stretches: %d; mirror plane normal %s, %.1f mm from x = 0' % (len(TL), np.round(SYMF['n'], 4).tolist(), 1000 * SYMF['d']))
def SYMD(x): return symmetry.reflect(SYMF, A2W(TF(x, sy_pl, sy_l))) - A2W(TF(x, sy_pr, sy_r))        # metres, per point
def parts(x):
    q = QQ(x); return (TF(x, A0[dk], db) - Xa[dk], TF(x, ff_m, ff_a) - TF(x, ff_m, ff_b), TF(x, fh_m, fh_b) - fh_t, q[ff_a, 6] - q[ff_b, 6], q[ff_a, :3] - q[ff_b, :3], q[:, 7:9].ravel(), CORD(x), np.r_[KN(x), NJR(x), CR(x)], depth_h(A2W(TF(x, H0, hown))))
def checks(x):
    d, jf, jh, _, tn, e_, cd_, kn_, dp = parts(x); gaps = np.r_[np.linalg.norm(jf, axis=1), np.linalg.norm(jh, axis=1)] * s1 * 1000
    return dict(named_joints_mm=dict(median=1000 * s1 * float(np.median(np.linalg.norm(d, axis=1))), max=1000 * s1 * float(np.linalg.norm(d, axis=1).max())), joint_gap_mm=dict(median=float(np.median(gaps)), p90=float(np.percentile(gaps, 90)), max=float(gaps.max())),
                gap_to_held_mm={k: float(np.median(np.linalg.norm(jh[[n == k for n in fh_n]], axis=1)) * s1 * 1000) for k in sorted(set(fh_n))}, outside_share=float((dp < 0).mean()), outside_worst_mm=1000 * float(max(0, -dp.min())))
x = x0.ravel().copy(); rep = {'before': checks(x)}; c = rep['before']; wf = os.path.join(FD, 'body_assembly.npz')
if os.environ.get('BA_WARM') and os.path.exists(wf) and list(np.load(wf)['free']) == free and np.load(wf)['params'].shape == (B, NPB): x = np.load(wf)['params'].ravel().astype(float); log('warm start from the last result')
log('before (body skeleton fit): named joints %.1f mm median (max %.1f); joint gaps %.1f mm median, %.1f p90, %.1f max; to held parts %s; outside %.1f%% of hull points, worst %.1f mm' % (c['named_joints_mm']['median'], c['named_joints_mm']['max'], c['joint_gap_mm']['median'], c['joint_gap_mm']['p90'],
    c['joint_gap_mm']['max'], {k: round(v, 1) for k, v in c['gap_to_held_mm'].items()}, 100 * c['outside_share'], c['outside_worst_mm']))
Lp = np.array([Zraw[i0] for i0 in CO.values()]); Rp = np.array([Zraw[i0 + 1] for i0 in CO.values()]); xm = float((Lp[:, 0] + Rp[:, 0]).mean() / 2); Lm = Lp * np.array([-1, 1, 1]) + np.array([2 * xm, 0, 0])   # the detector's own spread: the joints as detected, before pooling
spread = float(np.median(np.linalg.norm(Lm - Rp, axis=1)) / np.sqrt(2)); sig_j = spread / s1; sig_in = spread; hist = []
log('detector spread from left-right symmetry (left joints mirrored against right): %.1f mm -> scale of the named joints' % (1000 * spread))
def resid(x_, sig_in_):
    d, jf, jh, ds, tn, e_, cd_, kn_, dp = parts(x_); return np.r_[(d / sig_j).ravel(), (jf / sig_c).ravel(), (jh / sig_c).ravel(), ds / sig_s, (tn / sig_r).ravel(), e_ / sig_e, cd_ / sig_c, kn_, np.maximum(0, -dp) / sig_in_]
rows_b = [[b] for b in np.repeat(db, 3)] + [[a, b] for a, b in zip(np.repeat(ff_a, 3), np.repeat(ff_b, 3))] + [[b] for b in np.repeat(fh_b, 3)] + [[a, b] for a, b in zip(ff_a, ff_b)] + [[a, b] for a, b in zip(np.repeat(ff_a, 3), np.repeat(ff_b, 3))] + [[b] for b in np.repeat(np.arange(B), 2)] + [[a, b] for a, b, _, _, _ in LIG] + [[b] for K_ in KNEE for b in [K_['pat']] * 3 + [K_['fem']] * 3] + [r_ for r_ in NJ_rows for _ in range(3)] + ([[b] for b in CRO['b']] if CRO is not None else []) + [[b] for b in hown]
SP = lil_matrix((len(rows_b), NPB * B), dtype=np.int8)
for r_, bb in enumerate(rows_b):
    for b in bb:
        SP[r_, NPB * b:NPB * b + NPB] = 1
        if b in TR: l_ = int(TL[list(TR).index(b)]); SP[r_, NPB * l_ + 6:NPB * l_ + NPB] = 1          # a right bone's size and stretches are its left partner's
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.sparse import csc_matrix, diags
from scipy.sparse.linalg import spsolve
SPc = csc_matrix(SP); GR = group_columns(SPc)
def lm(x, sig_in_):                                                # Marquardt 1963; Jacobian by grouped finite differences on the known sparsity
    f = lambda z: resid(z, sig_in_); r = f(x); cost = float(r @ r); lam = 1e-3; it = 0
    for it in range(100):
        J = csc_matrix(approx_derivative(f, x, f0=r, sparsity=(SPc, GR))); H = (J.T @ J).tocsc(); g = J.T @ r; dg = H.diagonal() + 1e-12; ok = False
        for _ in range(12):
            xn = x - spsolve(H + diags(lam * dg), g); rn = f(xn); cn = float(rn @ rn)
            if np.isfinite(cn) and cn < cost: ok = True; break
            lam *= 4
        if not ok: break
        gain = (cost - cn) / cost; x, r, cost = xn, rn, cn; lam = max(lam / 4, 1e-9)
        if gain < 1e-5: break
    return x, it + 1, cost
if os.environ.get('BA_PROBE'):                                      # debugging aid: can the named bones alone do better from here, the rest held?
    pb = [bi[nid[n]] for n in os.environ['BA_PROBE'].split(',')]; cols_ = np.concatenate([NPB * b + np.arange(NPB) for b in pb]); f_ = lambda y: resid(np.r_[x][None][0] if False else _put(x, cols_, y), spread / 8)
    def _put(x_, c_, y_): z_ = x_.copy(); z_[c_] = y_; return z_
    r0 = f_(x[cols_]); sol = least_squares(f_, x[cols_], method='trf', x_scale='jac', max_nfev=400); x2 = _put(x, cols_, sol.x); q0, q2 = x.reshape(B, NPB), x2.reshape(B, NPB)
    log('probe %s: cost %.1f -> %.1f (%s, %d evaluations)' % (os.environ['BA_PROBE'], float(r0 @ r0), float(sol.fun @ sol.fun), sol.message, sol.nfev))
    for b in pb: log('   %-10s stretch x%.2f -> x%.2f, size %.3f -> %.3f, shift %.1f mm, turn %.1f deg' % (names[free[b]], np.exp(q0[b, 7]), np.exp(q2[b, 7]), s1 * np.exp(q0[b, 6]), s1 * np.exp(q2[b, 6]), 1000 * s1 * np.linalg.norm(q2[b, 3:6] - q0[b, 3:6]), np.degrees(np.linalg.norm(q2[b, :3] - q0[b, :3]))))
    c0, c2 = checks(x), checks(x2); log('   named joints max %.1f -> %.1f mm; joint gaps max %.2f -> %.2f mm' % (c0['named_joints_mm']['max'], c2['named_joints_mm']['max'], c0['joint_gap_mm']['max'], c2['joint_gap_mm']['max']))
    for (ja_, jb_, m_) in [(a_, b_, m__) for a_, b_, m__ in zip(ff_a, ff_b, ff_m) if a_ in pb or b_ in pb]: log('   joint %s / %s at atlas z %.3f' % (names[free[ja_]], names[free[jb_]], m_[2]))
    for b_, m_, n_ in zip(fh_b, fh_m, fh_n):
        if b_ in pb: log('   joint %s / held %s at atlas z %.3f' % (names[free[b_]], n_, m_[2]))
    sys.exit(0)
xbest = None
for outer in range(16):
    xprev = x.copy(); x, nit, cost = lm(x, sig_in); log('      solve: %d steps, cost %.1f' % (nit, cost)); c = checks(x); hist.append(dict(round=outer, penalty_mm=1000 * sig_in, **c))
    log('   round %d: inside penalty %.3f mm, worst outside %.2f mm (%.1f%%), named joints %.1f mm, joint gaps %.2f mm median / %.2f max (%.0f s)' % (outer, 1000 * sig_in, c['outside_worst_mm'], 100 * c['outside_share'], c['named_joints_mm']['median'], c['joint_gap_mm']['median'], c['joint_gap_mm']['max'], time.time() - t00))
    dw_ = depth_h(A2W(TF(x, H0, hown))); log('      furthest out: %s' % names[free[int(hown[dw_.argmin()])]])
    if c['outside_worst_mm'] / 1000 < hstep and outer > 0: break
    if xbest is not None and c['outside_worst_mm'] >= xbest[1]: x = xbest[0]; hist.pop(); c = checks(x); log('   a harder penalty no longer brings the worst point in: the round before is kept (worst %.2f mm)' % c['outside_worst_mm']); break   # pressing harder only bends the bones
    xbest = (x.copy(), c['outside_worst_mm'])
    sig_in /= 2
# which term holds each badly placed joint: cost (sum of squared scaled residuals) per term, over the rows that involve that joint's free bones
d_, jf_, jh_, ds_, tn_, e_, cd_, kn2_, dp_ = parts(x); e_ = e_.reshape(B, 2); dn = np.linalg.norm(d_, axis=1) * s1 * 1000; blame = {}
for k in np.argsort(-np.array([dn[dk == k_].max() if (dk == k_).any() else 0 for k_ in range(len(P))]))[:3]:
    bs = sorted(set(db[dk == k])); row = {}
    for b in bs:
        row[names[free[b]]] = dict(named=float(((d_[db == b] / sig_j) ** 2).sum()), joints=float(((jf_[(ff_a == b) | (ff_b == b)] / sig_c) ** 2).sum() + ((jh_[fh_b == b] / sig_c) ** 2).sum()), size=float(((ds_[(ff_a == b) | (ff_b == b)] / sig_s) ** 2).sum()),
                                   turn=float(((tn_[(ff_a == b) | (ff_b == b)] / sig_r) ** 2).sum()), stretch=float(((e_[b] / sig_e) ** 2).sum()), inside=float((np.maximum(0, -dp_[hown == b]) ** 2).sum() / sig_in ** 2), outside_points=int((dp_[hown == b] < 0).sum()), nearest_surface_mm=float(1000 * dp_[hown == b].min()))
    blame[P[k][0]] = row; log('held back at %s: %s' % (P[k][0], '; '.join('%s: %s' % (n, ', '.join('%s %.1f' % (t, v) for t, v in r.items())) for n, r in row.items())))
for K_ in KNEE:
    Wp = A2W(TF(x, K_['pf'][None], np.array([K_['pat']])))[0]; Pt = A2W(TF(x, V0[idx[free[K_['pat']]]], np.full(len(idx[free[K_['pat']]]), K_['pat']))); Wm, Wl = A2W(TF(x, np.array([K_['pm'], K_['pl']]), np.array([K_['fem'], K_['fem']])))
    K_['report'] = dict(kneecap_front_to_bump_mm=1000 * float(np.linalg.norm(Wp - K_['B'])), kneecap_z_mm=[1000 * float(Pt[:, 2].min()), 1000 * float(Pt[:, 2].max())], sides_middle_off_mm=1000 * float(((Wm + Wl) / 2) @ lat_w - K_['cmid']), axis_off_deg=float(np.degrees(np.arccos(np.clip(abs(((Wl - Wm) / np.linalg.norm(Wl - Wm)) @ lat_w), 0, 1)))))
    log('knee %s after: kneecap front %.1f mm from the bump; kneecap spans %.0f..%.0f mm high (surface region %.0f..%.0f); thigh-bone sides\' middle %.1f mm off the leg\'s middle; bending axis %.1f deg off straight across' % (K_['side'], K_['report']['kneecap_front_to_bump_mm'], *K_['report']['kneecap_z_mm'], 1000 * K_['region_z'][0], 1000 * K_['region_z'][1], K_['report']['sides_middle_off_mm'], K_['report']['axis_off_deg']))
sd_ = 1000 * np.linalg.norm(SYMD(x), axis=1); sd0_ = 1000 * np.linalg.norm(SYMD(x0.ravel()), axis=1); rep['left_right_mm'] = dict(before=dict(median=float(np.median(sd0_)), max=float(sd0_.max())), after=dict(median=float(np.median(sd_)), max=float(sd_.max())), worst=names[free[int(sy_l[sd_.argmax()])]]); log('left bones against mirrored right bones: %.1f mm median (max %.1f) before, %.1f mm (max %.1f, %s) after' % (np.median(sd0_), sd0_.max(), np.median(sd_), sd_.max(), rep['left_right_mm']['worst']))
if CRO is not None: cr0_, cr1_ = 1000 * sig_k * CR(x0.ravel()), 1000 * sig_k * CR(x); rep['crotch_mm'] = dict(before=cr0_.tolist(), after=cr1_.tolist(), atlas_gap_mm=1000 * CRO['gap_atlas']); log('crotch: the atlas crotch point carried by the hip bones is %s mm above the crotch of the figure before, %s mm after' % (np.round(cr0_, 1).tolist(), np.round(cr1_, 1).tolist()))
rep['after'] = c; rep['blame'] = blame; rep['knees'] = [dict(side=K_['side'], bump=K_['B'].tolist(), bump_mm=K_['bump_mm'], **K_['report']) for K_ in KNEE]; q = QQ(x); V = Vh.copy(); m = np.isin(OB, free); own_all = np.array([bi[int(b)] for b in OB[m]]); V[m] = A2W(TF(x, V0[m], own_all)); allm = np.isin(OB, allb)
size = s1 * np.exp(q[:, 6]); o = np.argsort(-np.abs(q[:, 7])); bend = np.degrees(np.linalg.norm(q[ff_a, :3] - q[ff_b, :3], axis=1)); ob_ = np.argsort(-bend)
rep.update(size=float(s1), bones=dict(free=B, held=len(held)), bone_size=dict(median=float(np.median(size)), min=[names[free[int(size.argmin())]], float(size.min())], max=[names[free[int(size.argmax())]], float(size.max())]),
           stretch=dict(median_abs=float(np.median(np.abs(q[:, 7]))), largest=[[names[free[i]], float(np.exp(q[i, 7]))] for i in o[:6]]), bend_deg=dict(median=float(np.median(bend)), largest=[['%s / %s' % (names[free[ff_a[i]]], names[free[ff_b[i]]]), float(bend[i])] for i in ob_[:6]]),
           outside_all_bone=dict(before=float((depth(Vsk[allm]) < 0).mean()), after=float((depth(V[allm]) < 0).mean())), rounds=hist, scales=dict(joint_gap_mm=1000 * sig_c * s1, named_joints_mm=1000 * sig_j * s1, size=sig_s, stretch=sig_e, turn_deg=float(np.degrees(sig_r)), grid_mm=1000 * hstep), seconds=time.time() - t00)
log('after: named joints %.1f mm median (max %.1f); joint gaps %.2f mm median, %.2f p90, %.2f max; to held parts %s; free bone hull outside %.2f%%, worst %.1f mm' % (c['named_joints_mm']['median'], c['named_joints_mm']['max'], c['joint_gap_mm']['median'], c['joint_gap_mm']['p90'], c['joint_gap_mm']['max'],
    {k: round(v, 2) for k, v in c['gap_to_held_mm'].items()}, 100 * c['outside_share'], c['outside_worst_mm']))
log('all bone vertices outside the body: before %.1f%%, after %.1f%%; bone size median %.3f (%s %.3f to %s %.3f); stretched most: %s; widened most: %s; bent most: %s' % (100 * rep['outside_all_bone']['before'], 100 * rep['outside_all_bone']['after'], rep['bone_size']['median'], *rep['bone_size']['min'], *rep['bone_size']['max'],
    '; '.join('%s x%.2f' % (n, v) for n, v in rep['stretch']['largest'][:4]), '; '.join('%s x%.2f' % (names[free[i]], np.exp(q[i, 8])) for i in np.argsort(-np.abs(q[:, 8]))[:4]), '; '.join('%s %.0f deg' % (n, v) for n, v in rep['bend_deg']['largest'][:3])))
np.savez(os.path.join(FD, 'body_assembly.npz'), V=V, V_before=Vsk, F=FB, obj=OB, params=q, free=np.array(free), held=np.array(sorted(held)), J_det=X, s1=s1, R1=R1, t1=t1)
json.dump(rep, open(os.path.join(FD, 'body_assembly.json'), 'w'), indent=1); log('done in %.0f s' % (time.time() - t00))
