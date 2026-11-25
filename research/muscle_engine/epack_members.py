"""E-PACK members for the left thigh, shared by epack_thigh.py (wildmeshing) and epack_grid.py (GPU label grid): the leg's
skin piece, the crop box, the bones and the muscles' rest tubes, and the element size of each (rules in epack_thigh.py)."""
import os, re, hashlib, numpy as np
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
def say(*a): print('EP', *a, flush=True)
def build(FD, MD):
    R = np.load(os.path.join(FD, 'muscle_repair.npz')); V = R['V'].astype(np.float64); OB = R['obj']
    A = np.load(os.path.join(FD, 'atlas_warp.npz'), allow_pickle=True); names = [str(x) for x in A['names']]; system = [str(x) for x in A['system']]
    nid = {n: i for i, n in enumerate(names)}
    NOT_MUSCLE = re.compile(r'fascia|septum|bursa|aponeurosis|retinacul|sheath|ligament|tendon|membrane', re.I)
    def surf(i):
        F = R['F_bone'] if system[i] == 'bone' else R['F_muscle']
        f = F[OB[F[:, 0]] == i]; u, inv = np.unique(f, return_inverse=True); P = V[u]; G = inv.reshape(-1, 3)
        if np.einsum('ij,ij->i', P[G[:, 0]], np.cross(P[G[:, 1]], P[G[:, 2]])).sum() < 0: G = G[:, [0, 2, 1]]   # atlas is wound inward
        return P, G
    def vol_area(P, G):
        c = np.cross(P[G[:, 1]] - P[G[:, 0]], P[G[:, 2]] - P[G[:, 0]])
        return np.einsum('ij,ij->i', P[G[:, 0]], np.cross(P[G[:, 1]], P[G[:, 2]])).sum() / 6, np.linalg.norm(c, axis=1).sum() / 2
    # skin: the leg's own flesh piece (as elf_leg_flesh.py)
    MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']
    ISL = np.load(os.path.join(MD, 'flesh', 'islands.npz'))['island']
    near = ISL[cKDTree(MV).query(V[OB == nid['Femur.l']])[1]]; leg = int(np.bincount(near[near >= 0]).argmax())
    sk = np.flatnonzero(ISL == leg); SF = MF[(ISL[MF] == leg).all(1)]; loc = np.full(len(MV), -1); loc[sk] = np.arange(len(sk)); SP, SG = MV[sk], loc[SF]
    if vol_area(SP, SG)[0] < 0: SG = SG[:, [0, 2, 1]]
    # crop box
    hipL, hipR, tib = (V[OB == nid[n]] for n in ('Hip bone.l', 'Hip bone.r', 'Tibia.l'))
    midx = 0.5 * (hipL[:, 0].mean() + hipR[:, 0].mean()); side = np.sign(hipL[:, 0].mean() - midx)
    ztop, zbot = hipL[:, 2].max(), 0.5 * (tib[:, 2].min() + tib[:, 2].max())
    # members.  Bones: the fitted left leg bones reaching into the box.  Muscles: every left muscle attached to at least two leg
    # segments (pelvis, thigh, shank, foot), i.e. every muscle that moves the leg; its surface is the rest tube of
    # muscle_rest.py (the warp crushed several fitted muscles, e.g. rectus femoris 64 -> 5.7 cm3): the unwarped atlas muscle
    # (V0) moved by the one similarity that puts its attachment centroids on the fitted ones.
    inbox = lambda P: (P[:, 2] <= ztop) & (P[:, 2] >= zbot) & (side * (P[:, 0] - midx) >= 0)
    V0 = A['V0'].astype(np.float64); contact = R['contact'].astype(bool); wb = [int(b) for b in R['bones']]
    Wt = csr_matrix((R['W_data'], R['W_indices'], R['W_indptr']))
    FOOT = re.compile(r'of foot|metatarsal|Calcaneus|Talus|Cuboid|Navicular|cuneiform|phalan')
    SEG = {nid['Hip bone.l']: 0, nid['Femur.l']: 1, nid['Tibia.l']: 2, nid['Fibula.l']: 2, nid['Patella.l']: 2,
           **{i: 3 for i, n in enumerate(names) if n.endswith('.l') and FOOT.search(n)}}
    def turn(a, b):
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b); v = np.cross(a, b); c = a @ b
        K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]]); return np.eye(3) + K + K @ K / (1 + c)
    mem, S, ends = [], {}, {}
    for i, n in enumerate(names):
        if not n.endswith('.l'): continue
        if system[i] == 'bone':
            if i in SEG and SEG[i] <= 2 and inbox(V[OB == i]).any(): mem.append(i); S[i] = surf(i)
            continue
        if system[i] != 'muscle' or NOT_MUSCLE.search(n) or re.search(r'tract|arch', n, re.I): continue
        f = R['F_muscle'][OB[R['F_muscle'][:, 0]] == i]
        if not len(f): continue
        u, inv = np.unique(f, return_inverse=True); G = inv.reshape(-1, 3); c = contact[u]
        if not c.any(): continue
        seg = np.full(len(u), -1); seg[np.flatnonzero(c)] = [SEG.get(wb[k], -1) for k in np.asarray(Wt[u[c]].argmax(1)).ravel()]
        present = sorted(set(seg[seg >= 0]))
        if len(present) < 2: continue
        oi, ii = np.flatnonzero(seg == present[0]), np.flatnonzero(seg == present[-1])
        o0, i0, o1, i1 = V0[u[oi]].mean(0), V0[u[ii]].mean(0), V[u[oi]].mean(0), V[u[ii]].mean(0)
        s_ = np.linalg.norm(i1 - o1) / np.linalg.norm(i0 - o0); P = (V0[u] - o0) @ turn(i0 - o0, i1 - o1).T * s_ + o1
        if vol_area(P, G)[0] < 0: G = G[:, [0, 2, 1]]
        if not inbox(P).any(): continue
        mem.append(i); S[i] = (P, G); ends[n] = dict(segments=[int(x) for x in present], scale=round(float(s_), 4), origin=oi.tolist(), insertion=ii.tolist())
    # element size: volume / area (a quarter of the diameter for a cylinder, so about four elements across)
    thick = {names[i]: abs(vol_area(*S[i])[0]) / vol_area(*S[i])[1] for i in mem}
    struct = np.concatenate([S[i][0] for i in mem]); fat_depth = cKDTree(struct).query(SP)[0]
    h_fat = 0.5 * float(np.median(fat_depth[inbox(SP)]))
    pad = 2 * h_fat; lo = np.array([min(midx, SP[:, 0].min()) - pad, SP[:, 1].min() - pad, zbot]); hi = np.array([max(midx, SP[:, 0].max()) + pad, SP[:, 1].max() + pad, ztop])
    if side > 0: lo[0] = midx
    else: hi[0] = midx
    BX = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    BG = np.array([[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]])
    if vol_area(BX, BG)[0] < 0: BG = BG[:, [0, 2, 1]]
    say('leg skin island %d: %d verts; box z %.3f..%.3f, midline x %.4f (side %+d); %d members; fat element %.1f mm' % (leg, len(SP), zbot, ztop, midx, side, len(mem), 1e3 * h_fat))
    # inputs
    return dict(V=V, OB=OB, names=names, system=system, nid=nid, mem=mem, S=S, ends=ends, thick=thick, h_fat=h_fat, SP=SP, SG=SG,
                lo=lo, hi=hi, midx=midx, side=side, zbot=zbot, ztop=ztop, leg=leg, inbox=inbox, vol_area=vol_area)
