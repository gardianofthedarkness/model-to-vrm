"""Solid models on a voxel grid, shared by the muscle steps (the same construction as in body_assembly.py)."""
import os, numpy as np
from scipy import ndimage
def solid_of(P, F, step):
    """the surface's cells: every triangle sampled finer than the grid."""
    lo = P.min(0) - 4 * step; gn = np.ceil((P.max(0) + 4 * step - lo) / step).astype(int) + 1; T = P[F]; occ = np.zeros(gn, bool)
    ns = np.clip(np.ceil(np.maximum(np.linalg.norm(T[:, 1] - T[:, 0], axis=1), np.linalg.norm(T[:, 2] - T[:, 0], axis=1)) / (step / 2)).astype(int), 1, 64)
    for n_ in np.unique(ns):
        Tn = T[ns == n_]; u, v = np.meshgrid(np.arange(n_ + 1) / n_, np.arange(n_ + 1) / n_); k = (u + v) <= 1 + 1e-9; u, v = u[k], v[k]
        for c0 in range(0, len(Tn), 20000):
            Tc = Tn[c0:c0 + 20000]; Pn = (Tc[:, None, 0] * (1 - u - v)[None, :, None] + Tc[:, None, 1] * u[None, :, None] + Tc[:, None, 2] * v[None, :, None]).reshape(-1, 3); ii = np.rint((Pn - lo) / step).astype(int); occ[ii[:, 0], ii[:, 1], ii[:, 2]] = True
    return occ, lo
def sdf_of(sol, step): return ((ndimage.distance_transform_edt(sol) - ndimage.distance_transform_edt(~sol)) * step).astype(np.float32)
def sample(sd, lo, step, P): return ndimage.map_coordinates(sd, ((P - lo) / step).T, order=1, mode='nearest')
def body_solid(MD):
    """the figure's solid: grid step two mesh edges; open gaps sealed by the fewest cells that make an inside appear.
    Returns depth grid (positive inside, metres), grid origin, step, the solid, the mesh."""
    MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); MF = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['F']; step = 2 * float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1)))
    occ, lo = solid_of(MV, MF, step)
    for seal in range(1, 17):
        thick = ndimage.binary_dilation(occ, iterations=seal); fill = ndimage.binary_fill_holes(thick)
        if fill.sum() - thick.sum() > thick.sum(): break
    else: print('WARNING solids.py: the solid did not close within 17 sealing steps; the widest sealing is used and the result may be wrong', flush=True)
    body = ndimage.binary_erosion(fill, iterations=seal) | occ
    return sdf_of(body, step), lo, step, body, MV, MF
def bone_solid(V, FB, step):
    o, lo = solid_of(V, FB, step); f = ndimage.binary_fill_holes(o); return sdf_of(f, step), lo, f
def closed_volumes(Vref, F, obj_of_face, items, *Vs):
    """volume per item for each vertex set in Vs, counting only meshes that are closed (every welded edge shared by two
    faces): an open sheet (a fascia) has no volume, and the signed sum over its faces would be a meaningless number."""
    out = {}
    for m in items:
        Fm = F[obj_of_face == m]; vi = np.unique(Fm); _, wid = np.unique(np.round(Vref[vi], 6), axis=0, return_inverse=True); loc = np.zeros(len(Vref), np.int64); loc[vi] = wid.ravel(); Fw = loc[Fm]
        E = np.sort(np.r_[Fw[:, [0, 1]], Fw[:, [1, 2]], Fw[:, [2, 0]]], axis=1); E = E[E[:, 0] != E[:, 1]]
        if not len(E) or (np.unique(E, axis=0, return_counts=True)[1] != 2).any(): continue
        out[int(m)] = [abs(float((P[Fm[:, 0]] * np.cross(P[Fm[:, 1]], P[Fm[:, 2]])).sum() / 6)) for P in Vs]
    return out
def piece_solids(MD, MV, MF):
    """the closed solids of the separate pieces of flesh (flesh/islands.npz: a leg, an arm), as in body_assembly.py: each
    piece's open ends (the cut at the hip, at the shoe) closed by a fan of triangles to the middle of each open edge
    loop. Returns per piece that closes: (depth grid, origin, step, caps), caps = the piece's real openings (loops at
    least half as long as its longest) as (centre, outward normal)."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    isf = os.path.join(MD, 'flesh', 'islands.npz'); out = {}
    if not os.path.exists(isf): return out
    ISL = np.load(isf)['island']; _, wid = np.unique(np.round(MV, 6), axis=0, return_inverse=True); wid = wid.ravel(); est = float(np.median(np.linalg.norm(MV[MF[:, 0]] - MV[MF[:, 1]], axis=1))); rv = np.zeros(wid.max() + 1, int); rv[wid] = np.arange(len(MV))
    for k in [int(k) for k in np.unique(ISL) if k >= 0]:
        Fk = MF[(ISL[MF] == k).all(1)]
        if not len(Fk): continue
        Wk = wid[Fk]; E = np.sort(np.r_[Wk[:, [0, 1]], Wk[:, [1, 2]], Wk[:, [2, 0]]], axis=1); ue, ce = np.unique(E, axis=0, return_counts=True); be = ue[ce == 1]
        if not len(be): continue
        bv = np.unique(be); loc = np.zeros(wid.max() + 1, int); loc[bv] = np.arange(len(bv)); nl, lab = connected_components(coo_matrix((np.ones(len(be)), (loc[be[:, 0]], loc[be[:, 1]])), shape=(len(bv), len(bv))), directed=False)
        Tk = [MV[Fk]]; caps = []; pc = MV[np.unique(Fk)].mean(0)
        for l in range(nl):
            vv = bv[lab == l]; cen = MV[rv[vv]].mean(0); ee = be[lab[loc[be[:, 0]]] == l]; Tk.append(np.stack([MV[rv[ee[:, 0]]], MV[rv[ee[:, 1]]], np.repeat(cen[None], len(ee), 0)], axis=1))
            nn = np.linalg.svd(MV[rv[vv]] - cen)[2][-1]; nn = nn if nn @ (cen - pc) > 0 else -nn; caps.append((cen, nn, float(np.linalg.norm(MV[rv[ee[:, 0]]] - MV[rv[ee[:, 1]]], axis=1).sum())))
        Tk = np.vstack(Tk); V_ = Tk.reshape(-1, 3); F_ = np.arange(len(V_)).reshape(-1, 3); oc, g0 = solid_of(V_, F_, est); closed = False
        for seal in range(1, 5):
            th = ndimage.binary_dilation(oc, iterations=seal); fl = ndimage.binary_fill_holes(th)
            if fl.sum() - th.sum() > th.sum(): closed = True; break
        if not closed: continue
        big = max(c[2] for c in caps); out[k] = (sdf_of(ndimage.binary_erosion(fl, iterations=seal) | oc, est), g0, est, [(c, n) for c, n, L in caps if L >= big / 2])
    return out
def carried_skin_solid(FD, V, step):
    """the atlas's own skin placed in the figure by the fitted bones: every atlas skin vertex moves as its nearest bone
    moved (that bone's closest affine map, atlas -> figure). A stand-in for her body where her own surface cannot be
    seen (under hair, under loose cloth). Returns (depth grid, origin)."""
    from scipy.spatial import cKDTree
    W = np.load(os.path.join(FD, 'atlas_warp.npz')); V0 = W['V0'].astype(np.float64); OB = W['obj']; FS = W['F_skin']; FB = W['F_bone']; bv = np.unique(FB); sv = np.unique(FS); own = OB[bv][cKDTree(V0[bv]).query(V0[sv])[1]]; P = V0.copy()
    for b in np.unique(own):
        A = V0[OB == b]; B_ = V[OB == b]; M, *_ = np.linalg.lstsq(np.c_[A, np.ones(len(A))], B_, rcond=None) if len(A) >= 4 else (np.r_[np.eye(3), (B_.mean(0) - A.mean(0))[None]],); k = sv[own == b]; P[k] = np.c_[V0[k], np.ones(len(k))] @ M
    occ, lo = solid_of(P, FS, step)
    for seal in range(1, 17):
        thick = ndimage.binary_dilation(occ, iterations=seal); fill = ndimage.binary_fill_holes(thick)
        if fill.sum() - thick.sum() > thick.sum(): break
    else: print('WARNING solids.py: the solid did not close within 17 sealing steps; the widest sealing is used and the result may be wrong', flush=True)
    return sdf_of(ndimage.binary_erosion(fill, iterations=seal) | occ, step), lo
def inner_solid(MD, bone_pts, bone_obj, FD=None, V=None):
    """the figure's solid without what is not her body. The whole-figure solid of body_solid() is one piece where
    clothing and hair are fused to the body. Two rules take out what hangs on her:
      limbs   a limb has its own closed skin (piece_solids), and it holds that limb's bones (a bone belongs to a piece
              when most of its vertices are inside it). Every cell of the solid is flesh of its nearest bone; a cell whose
              nearest bone is a limb's bone must be inside that limb's skin, or past one of the limb's open ends (the
              trunk past the hip, the hand past the sleeve).
      loose   where the nearest surface is loose - hair, and the cloth Jev's boxes mark as hanging (parts/loose_vote.py:
              <model dir>/loose/loose.npz) - her own surface is not seen. Her other side stands in: a body is its own
              mirror image (symmetry.py: the plane measured on her skin), so such a cell must also be inside the solid
              at its mirror place, to within one grid cell (a lobe of hair behind one shoulder has no partner on the
              other side; hair on both sides of the head has). Where the nearest surface is hair, the atlas's skin
              placed by the fitted bones (carried_skin_solid) must hold the cell too: on the head both sides are
              hidden. (Needs FD and V.)
    Returns the same as body_solid(), the volume taken out (m3) and the limb pieces used."""
    from scipy.spatial import cKDTree
    sdf, lo, step, body, MV, MF = body_solid(MD); PS = piece_solids(MD, MV, MF); idx = np.argwhere(body); P = lo + step * idx; near = bone_obj[cKDTree(bone_pts).query(P)[1]]; bad = np.zeros(len(P), bool); used = []
    for k, (sd, g0, est, caps) in PS.items():
        inb = sample(sd, g0, est, bone_pts) > 0; own = [b for b in np.unique(bone_obj[inb]) if inb[bone_obj == b].mean() > 0.5]
        if not own: continue
        used.append(k); m = np.isin(near, own); Q = P[m]; bad[np.flatnonzero(m)[(sample(sd, g0, est, Q) <= 0) & (np.max([(Q - c) @ n for c, n in caps], axis=0) < 0)]] = True
    lf = os.path.join(MD, 'loose', 'loose.npz')
    if FD is not None and V is not None and os.path.exists(lf):
        import symmetry
        loose = np.load(lf)['loose']; MT = np.load(os.path.join(MD, 'contacts', 'material.npz'), allow_pickle=True); hair = np.array([str(x) for x in MT['materials']])[MT['material']] == 'hair'; jn = cKDTree(MV).query(P)[1]
        S = symmetry.figure_mirror(MD); sk, slo = carried_skin_solid(FD, V, step); bad |= loose[jn] & (sample(sdf, lo, step, symmetry.reflect(S, P)) < -step); bad |= hair[jn] & (sample(sk, slo, step, P) <= 0)
    inner = body.copy(); inner[tuple(idx[bad].T)] = False
    return sdf_of(inner, step), lo, step, inner, MV, MF, float(bad.sum() * step ** 3), used
