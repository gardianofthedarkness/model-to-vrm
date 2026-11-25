"""R0 (runtime-plan.md): plain skinning (A) against the small flesh runtime (B) on the elf's left leg and hip, for one ARDY clip.
The only live inputs are the bone motion and the muscle effort computed from it (r0_clip_activations.py); every setting
comes from the elf's own anatomy or from the cited table in r0_params.json.

  A  plain skinning: bone-heat weights (Baran & Popovic 2007, Blender's automatic weights) for the groups pelvis / thigh /
     shank / foot, blended as dual quaternions; the foot turns about the ankle axis through the malleoli tips (Inman 1976) by
     ARDY's ankle angle.  The skin is first joined into one surface.  The pelvis moves in
     the world with ARDY's hips (rotation as given, displacement scaled by the ratio of leg lengths); thigh and shank ride
     on it with the calibrated leg transforms (ardy_to_elf_leg_calibrated.py).
  B  A plus two physical layers:
     bulge   the muscle under a skin vertex (nearest muscle rest tube point, if nearer than the nearest bone) thickens as it
             shortens at constant volume: its local radius r (tube point to the tube's centre line) grows by
             r (sqrt(l_rest / l) - 1), l = MuJoCo's normalised fibre length; plus the cited active thickening at full
             activation, scaled by activation.  Pushed along the skin normal, spread over the skin across one local fat
             depth (the fat layer smooths the muscle's shape, as it blends the compartment borders below).
     jiggle  one damped 3-D oscillator per muscle compartment (quadriceps, hamstrings, adductors, gluteals, calf, front and
             side of the shank), in its carrying bone's frame, driven by the inertial and gravity load at the compartment's
             centre:  x'' + 2 z w x' + w^2 x = R^T (g - a) - R0^T g   (the rest shape already holds the sag at rest).
             w^2 is interpolated between the cited relaxed and tensed values by the compartment's activation (stiffnesses
             add), z likewise.  Each skin vertex moves by its compartment's x times its soft depth (skin to bone) over the
             compartment's largest; compartment borders are blended over the skin by diffusion across one local fat depth
             (the skin's transfer length, skin_fat.py).
Bone motion is resampled to 240 Hz (Slerp, cubic splines) for the oscillators; output at 60 fps.  The time per frame of B's
own work (oscillators + offsets + normals, numpy on the CPU) is measured.
  E:\\splat\\.venv-sim\\Scripts\\python.exe r0_flesh.py <fits dir> <model dir> <ARDY clip> <leg_transforms npz> <activation npz> <params json> <out dir>"""
import sys, os, re, json, time, math, numpy as np, scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot, Slerp
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'muscle_engine'))
from epack_members import build
FD, MD, CLIP, TRF, ACT, PRM, OUT = sys.argv[1:8]; os.makedirs(OUT, exist_ok=True)
def say(*a): print('R0', *a, flush=True)
B = build(FD, MD); V, OB, names, system, nid, S = B['V'], B['OB'], B['names'], B['system'], B['nid'], B['S']
R = np.load(os.path.join(FD, 'muscle_repair.npz')); Wt = csr_matrix((R['W_data'], R['W_indices'], R['W_indptr'])); wb = [int(b) for b in R['bones']]
P0, F = B['SP'], B['SG']
# the skin piece is stored as 60 texture patches whose border points coincide; joined into one surface (exact same position),
# so weights, jiggle and bulge spread across the seams instead of tearing them open
P0, wi = np.unique(P0.round(7), axis=0, return_inverse=True); F = wi.ravel()[F]; NV = len(P0)
PRMS = json.load(open(PRM))
# ---------- A: bone shares per skin vertex
FOOT = re.compile(r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform|phalan|sesamoid', re.I)
def grp(b):
    n = names[b]
    if n in ('Femur.l',): return 1
    if n in ('Tibia.l', 'Fibula.l', 'Patella.l'): return 2
    if n.endswith('.l') and FOOT.search(n): return 3
    return 0
NG = 4                                                                             # pelvis, thigh, shank, foot
bone_ids = [i for i, s_ in enumerate(system) if s_ == 'bone']; bpt = np.flatnonzero(np.isin(OB, bone_ids))
# ---------- world motion of the three groups
C = np.load(CLIP); M = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float); fps = float(C['fps']); J = C['posed_joints'] @ M.T
sn = open(os.path.join(os.path.dirname(os.path.dirname(CLIP)), 'soma77_names.txt')).read().split('\n'); si = {n: i for i, n in enumerate(sn)}
J0 = np.load(os.path.join(os.path.dirname(os.path.dirname(CLIP)), 'soma77_neutral.npy')).reshape(-1, 3) @ M.T
Rh = np.array([M @ g @ M.T for g in C['global_rot_mats'][:, si['Hips']]])
fem, tib, tal = (V[OB == nid[n]] for n in ('Femur.l', 'Tibia.l', 'Talus.l'))
leg_elf = fem[:, 2].max() - tal[:, 2].mean()                                     # hip to ankle height, as ARDY's LeftLeg -> LeftFoot
leg_ardy = J0[si['LeftLeg'], 2] - J0[si['LeftFoot'], 2]; sc = leg_elf / leg_ardy
c0 = 0.5 * (V[OB == nid['Hip bone.l']].mean(0) + V[OB == nid['Hip bone.r']].mean(0)); Hd = sc * (J[:, si['Hips']] - J[0, si['Hips']])
T = np.load(TRF); NT = len(Rh)
# foot: turns about the ankle axis through the tips of the two malleoli (Inman 1976) by ARDY's ankle angle (r0_clip_activations)
fib = V[OB == nid['Fibula.l']]; med, lat = tib[np.argmin(tib[:, 2])], fib[np.argmin(fib[:, 2])]; ax_a = (lat - med) / np.linalg.norm(lat - med); piv = 0.5 * (med + lat)
A_ = np.load(ACT); ank = np.interp(np.arange(NT) / fps, np.arange(len(A_['ankle_l'])) / float(A_['fps']), A_['ankle_l'])
toe = V[np.isin(OB, [i for i, n_ in enumerate(names) if n_.endswith('.l') and FOOT.search(n_)])]; toe = toe[np.argmin(toe[:, 1])]     # most forward foot point
sgn = 1.0 if (Rot.from_rotvec(ax_a * 0.2).apply(toe - piv)[2] > (toe - piv)[2]) else -1.0                 # + = dorsiflexion: toes up
Rg = np.zeros((NT, NG, 3, 3)); tg = np.zeros((NT, NG, 3))
for t in range(NT):
    Rp, tp = Rh[t], c0 - Rh[t] @ c0 + Hd[t]; Ra = Rot.from_rotvec(ax_a * sgn * ank[t]).as_matrix()
    Rs_, ts_ = T['shank_R'][t], T['shank_t'][t]
    for k, (Rr, tr) in enumerate(((np.eye(3), np.zeros(3)), (T['thigh_R'][t], T['thigh_t'][t]), (Rs_, ts_), (Rs_ @ Ra, Rs_ @ (piv - Ra @ piv) + ts_))):
        Rg[t, k] = Rp @ Rr; tg[t, k] = Rp @ tr + tp
say('clip %d frames %.0f fps; elf/ARDY leg scale %.3f; %d skin vertices' % (NT, fps, sc, NV))
SUB = 240; OUTF = int(os.environ.get("OUTF", 60)); tt = np.arange(NT) / fps; ts = np.arange(0, tt[-1], 1 / SUB)
slerps = [Slerp(tt, Rot.from_matrix(Rg[:, k])) for k in range(NG)]; spl = [CubicSpline(tt, tg[:, k]) for k in range(NG)]
RS = np.stack([s(ts).as_matrix() for s in slerps], 1); TS = np.stack([s(ts) for s in spl], 1)    # (Ns, 3, 3, 3), (Ns, 3, 3)
# ---------- muscles under the skin, compartments
MYO = {'Rectus femoris': ['recfem'], 'Vastus lateralis': ['vaslat'], 'Vastus medialis': ['vasmed'], 'Vastus intermedius': ['vasint'],
       'Long head of biceps': ['bflh'], 'Short head of biceps': ['bfsh'], 'Semitendinosus': ['semiten'], 'Semimembranosus': ['semimem'],
       'Gracilis': ['grac'], 'Sartorius': ['sart'], 'Adductor longus': ['addlong'], 'Adductor brevis': ['addbrev'],
       'Adductor magnus': ['addmagDist', 'addmagIsch', 'addmagMid', 'addmagProx'], 'Gluteus maximus': ['glmax1', 'glmax2', 'glmax3'],
       'Gluteus medius': ['glmed1', 'glmed2', 'glmed3'], 'Gluteus minimus': ['glmin1', 'glmin2', 'glmin3'], 'Tensor fasciae': ['tfl'],
       'Lateral head of gastrocnemius': ['gaslat'], 'Medial head of gastrocnemius': ['gasmed'], 'Soleus': ['soleus'], 'Tibialis anterior': ['tibant'],
       'Tibialis posterior': ['tibpost'], 'Fibularis longus': ['perlong'], 'Fibularis brevis': ['perbrev'], 'Extensor digitorum longus': ['edl'],
       'Extensor hallucis longus': ['ehl'], 'Flexor digitorum longus': ['fdl'], 'Flexor hallucis longus': ['fhl'], 'Iliacus': ['iliacus'], 'Psoas': ['psoas'],
       'Piriformis': ['piri'], 'Pectineus': [], 'Popliteus': [], 'Plantaris': []}
COMP = {'quadriceps': ['Rectus femoris', 'Vastus'], 'hamstrings': ['biceps femoris', 'Semitendinosus', 'Semimembranosus'],
        'adductors': ['Adductor', 'Gracilis', 'Pectineus', 'Sartorius'], 'gluteals': ['Gluteus', 'Tensor fasciae', 'Piriformis', 'gemellus', 'Obturator', 'Quadratus femoris', 'Iliacus', 'Psoas'],
        'calf': ['gastrocnemius', 'Soleus', 'Plantaris', 'Popliteus', 'Tibialis posterior', 'Flexor'], 'shank_front': ['Tibialis anterior', 'Extensor'], 'shank_side': ['Fibularis']}
CARRY = {'quadriceps': 1, 'hamstrings': 1, 'adductors': 1, 'gluteals': 0, 'calf': 2, 'shank_front': 2, 'shank_side': 2}
comps = list(COMP)
def comp_of(n): return next((c for c, keys in COMP.items() if any(k.lower() in n.lower() for k in keys)), None)
mus = [i for i in B['mem'] if system[i] != 'bone' and comp_of(names[i])]
mpts = np.concatenate([S[i][0] for i in mus]); mown = np.concatenate([np.full(len(S[i][0]), k) for k, i in enumerate(mus)])
rad = np.zeros(len(mpts))                                       # local radius: tube point to the tube's centre line (slices along the long axis)
for k, i in enumerate(mus):
    P = S[i][0]; c = P.mean(0); ax = np.linalg.svd(P - c, full_matrices=False)[2][0]; s = (P - c) @ ax
    nb = max(2, int(round((s.max() - s.min()) / (2 * B['thick'][names[i]])))); e = np.linspace(s.min(), s.max() + 1e-9, nb + 1); b = np.clip(np.digitize(s, e) - 1, 0, nb - 1)
    cen = np.array([P[b == j].mean(0) if (b == j).any() else c for j in range(nb)]); r = P - cen[b]; r -= np.outer(r @ ax, ax); rad[mown == k] = np.linalg.norm(r, axis=1)
bones_pts = V[np.isin(OB, [i for i, s_ in enumerate(system) if s_ == 'bone'])]                 # every bone, foot and trunk included
d_bone = cKDTree(bones_pts).query(P0)[0]; d_mus, j_mus = cKDTree(mpts).query(P0)
under = np.where(d_mus < d_bone, mown[j_mus], -1); r_under = np.where(under >= 0, rad[j_mus], 0.0)
vc = np.array([comps.index(comp_of(names[mus[k]])) for k in mown[j_mus]])          # every vertex: compartment of the nearest muscle
fat = np.where(under >= 0, d_mus, d_bone)                                          # local fat depth (skin to the first deeper structure)
# blend compartment borders over the skin: implicit diffusion with length = local fat depth
E = np.unique(np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1), axis=0); el = np.linalg.norm(P0[E[:, 0]] - P0[E[:, 1]], axis=1)
Wg = sp.coo_matrix((1 / el ** 2, (E[:, 0], E[:, 1])), shape=(NV, NV)); Wg = Wg + Wg.T; Lap = sp.diags(np.asarray(Wg.sum(1)).ravel()) - Wg
solve = factorized((sp.eye(NV) + sp.diags(fat ** 2) @ Lap).tocsc())
share = np.stack([solve((vc == c).astype(float)) for c in range(len(comps))], 1); share /= share.sum(1, keepdims=True)
# A's weights: bone heat (Baran & Popovic 2007, Blender's automatic weights): (L + H) w_k = H p_k, H = 1 / d^2 to the nearest
# bone, p_k = 1 where group k's bone is the nearest; the blend across a joint is as wide as the joint's own geometry
dg = np.stack([cKDTree(V[bpt[np.array([grp(b) for b in OB[bpt]]) == k]]).query(P0)[0] for k in range(NG)], 1)
Hh = sp.diags(1 / dg.min(1) ** 2); heat = factorized((Lap + Hh).tocsc())
W3 = np.stack([heat(Hh @ (dg.argmin(1) == k).astype(float)) for k in range(NG)], 1); W3 = np.clip(W3, 0, None); W3 /= W3.sum(1, keepdims=True)
wsoft = np.zeros(NV)
for c in range(len(comps)):
    m = vc == c
    if m.any(): wsoft[m] = d_bone[m] / d_bone[m].max()
# ---------- muscle effort per compartment, fibre length per muscle under the skin
an = [str(x) for x in A_['names']]; fpsA = float(A_['fps']); act = A_['activation']; fn = A_['fiber_norm']; fn0 = A_['fiber_norm_standing']
def myo_cols(n):
    keys = next((v for k, v in MYO.items() if k.lower() in n.lower()), [])
    return [an.index(k + '_l') for k in keys if k + '_l' in an]
mcols = [myo_cols(names[i]) for i in mus]
vol = np.array([abs(B['vol_area'](*S[i])[0]) for i in mus])
ta = np.arange(len(act)) / fpsA
def at(x, t): return np.interp(t, ta, x)
comp_act = np.zeros((len(ts), len(comps)))
for c in range(len(comps)):
    ks = [k for k, i in enumerate(mus) if comps.index(comp_of(names[i])) == c and mcols[k]]
    if ks: comp_act[:, c] = sum(vol[k] * at(act[:, mcols[k]].mean(1), ts) for k in ks) / sum(vol[k] for k in ks)
bulge_ratio = np.zeros((len(ts), len(mus)))
for k in range(len(mus)):
    if mcols[k]:
        l = at(fn[:, mcols[k]].mean(1), ts); l0 = fn0[mcols[k]].mean(); a = at(act[:, mcols[k]].mean(1), ts)
        bulge_ratio[:, k] = np.sqrt(l0 / l) - 1 + PRMS['bulge_active_full'] * a
# ---------- oscillators
g = np.array([0, 0, -9.81]); cw = np.array([(share[:, c] * wsoft) for c in range(len(comps))])
cen0 = np.array([(cw[c] @ P0) / max(cw[c].sum(), 1e-12) for c in range(len(comps))])
def omz(c, a):
    p = PRMS['compartments'].get(comps[c], PRMS['compartments']['default'])
    w2 = (2 * math.pi * p['f_relaxed_hz']) ** 2 + a * ((2 * math.pi * p['f_tensed_hz']) ** 2 - (2 * math.pi * p['f_relaxed_hz']) ** 2)
    return math.sqrt(w2), p['zeta_relaxed'] + a * (p['zeta_tensed'] - p['zeta_relaxed'])
def skin_A(Rk, tk):
    """dual-quaternion blend of the three group transforms"""
    q = Rot.from_matrix(Rk).as_quat()[:, [3, 0, 1, 2]]; q[q[:, 0] < 0] *= -1
    for k in range(1, NG):
        if q[k] @ q[0] < 0: q[k] *= -1
    tq = np.stack([np.r_[0, tk[k]] for k in range(NG)])
    def qm(a, b):
        return np.stack([a[..., 0] * b[..., 0] - (a[..., 1:] * b[..., 1:]).sum(-1),
                         *(a[..., 0, None] * b[..., 1:] + b[..., 0, None] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])).T], -1)
    qd = 0.5 * qm(tq, q); b0 = W3 @ q; bd = W3 @ qd; n = np.linalg.norm(b0, axis=1, keepdims=True); b0 /= n; bd /= n
    w, v = b0[:, :1], b0[:, 1:]; t = 2 * (bd[:, :1] * 0 + (w * bd[:, 1:] - bd[:, :1] * v + np.cross(v, bd[:, 1:])))
    rot = Rot.from_quat(np.c_[v, w]); return rot.apply(P0) + t, rot.apply(N0)        # skinned positions and normals (what A already pays)
def normals(X):
    fn_ = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]]); idx = F.ravel(); fr = np.repeat(fn_, 3, 0)
    n = np.stack([np.bincount(idx, fr[:, k], minlength=len(X)) for k in range(3)], 1)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
N0 = normals(P0)
# B's own per-frame data, prepared once: jiggle weights (vertex x compartment), muscle index and radius per vertex
WJ = (share * wsoft[:, None]).astype(np.float32)
# bulge: each muscle's local radius under the skin, spread over the skin by the fat layer (same diffusion, one fat depth)
WB = np.stack([solve(np.where(under == k, r_under, 0.0)) for k in range(len(mus))], 1).astype(np.float32)
CAR = np.array([CARRY[c] for c in comps])
NC = len(comps); x = np.zeros((NC, 3)); v = np.zeros((NC, 3)); dt = 1 / SUB
R0c = np.array([RS[0, CARRY[comps[c]]] for c in range(NC)])
cpos = lambda i, c: RS[i, CARRY[comps[c]]] @ cen0[c] + TS[i, CARRY[comps[c]]]
cp = np.array([[cpos(i, c) for c in range(NC)] for i in range(len(ts))]); acc = np.gradient(np.gradient(cp, dt, axis=0), dt, axis=0)
step_out = SUB // OUTF; framesA, framesB, Xc, Bt, At, rec = [], [], [], [], [], []
for i in range(len(ts)):
    t0 = time.perf_counter()
    for c in range(NC):
        Rc = RS[i, CARRY[comps[c]]]; w, z = omz(c, comp_act[i, c]); f = Rc.T @ (g - acc[i, c]) - R0c[c].T @ g
        v[c] += dt * (f - 2 * z * w * v[c] - w * w * x[c]); x[c] += dt * v[c]          # semi-implicit Euler
    rec.append(time.perf_counter() - t0)
    if i % step_out: continue
    ta0 = time.perf_counter(); XA, NA = skin_A(RS[i], TS[i]); At.append(time.perf_counter() - ta0); t1 = time.perf_counter()
    Y = np.einsum('cij,cj->ci', RS[i, CAR], x).astype(np.float32)                 # each compartment's offset in the world
    XB = XA + WJ @ Y + (WB @ bulge_ratio[i].astype(np.float32))[:, None] * NA
    Bt.append(time.perf_counter() - t1 + sum(rec[-step_out:]))
    framesA.append(XA.astype(np.float32)); framesB.append(XB.astype(np.float32)); Xc.append(x.copy())
FA, FB = np.array(framesA), np.array(framesB); dmag = np.linalg.norm(FB - FA, axis=2)
res = dict(frames_out=len(FA), fps_out=OUTF, substep_hz=SUB, vertices=NV, compartments=comps, leg_scale=sc,
           B_extra_ms_per_frame=dict(median=1e3 * float(np.median(Bt)), p95=1e3 * float(np.percentile(Bt, 95)), note='oscillators + jiggle + bulge on top of skinning, numpy CPU'),
           A_skinning_ms_per_frame=dict(median=1e3 * float(np.median(At)), note='dual-quaternion skinning in numpy (a game does this on the GPU)'),
           offset_mm=dict(p50=1e3 * float(np.median(dmag)), p99=1e3 * float(np.percentile(dmag, 99)), max=1e3 * float(dmag.max())),
           peak_compartment_activation={c: round(float(comp_act[:, k].max()), 3) for k, c in enumerate(comps)},
           peak_oscillator_mm={c: round(1e3 * float(np.linalg.norm(np.array(Xc)[:, k], axis=1).max()), 2) for k, c in enumerate(comps)}, params=PRM)
np.savez_compressed(os.path.join(OUT, 'r0_skin.npz'), A=FA, B=FB, F=F, rest=P0, under=under, comp=vc, wsoft=wsoft, comp_act=comp_act[::step_out], comps=np.array(comps))
json.dump(res, open(os.path.join(OUT, 'r0_flesh.json'), 'w'), indent=1); say(json.dumps(res, indent=1))
