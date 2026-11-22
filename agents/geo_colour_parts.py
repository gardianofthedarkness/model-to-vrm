"""Segments of a one-piece model for the labeller, by shape first and true colour second (no shading splits):
  python geo_colour_parts.py <model dir>        -> sam_body/geo_parts.npz (stage1 and stage2 label per vertex, -1 none)
  stage 1  shape: faces joined across edges whose dihedral angle is under the Otsu split of all edge angles (a smooth,
           continuous surface stays one part; creases, carved edges and separate pieces split it); crumbs below the
           face-weighted Otsu split of the log part sizes go into the neighbour they share the most border with
  stage 2  true colour inside each stage-1 part: chroma only (Lab a, b; brightness L left out, so shading of one colour
           never splits), clusters by a Gaussian mixture with the count by BIC, joined along the mesh, crumbs merged the
           same way inside the part
  The UV-seam twins of a vertex are welded first (one surface)."""
import sys, os, numpy as np, cv2
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.mixture import GaussianMixture
R = sys.argv[1]; j = lambda *p: os.path.join(R, *p)
O = np.load(j('parts', 'orig_colours.npz')); V, C = O['V'], O['C'].astype(np.float32); F = np.load(j('parts', 'orig_normals.npz'))['F']
_, rep, inv = np.unique(V, axis=0, return_index=True, return_inverse=True); Wd = rep[inv.ravel()]; FW = Wd[F]
fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]); fn /= np.linalg.norm(fn, axis=1, keepdims=True) + 1e-30
e = np.r_[FW[:, [0, 1]], FW[:, [1, 2]], FW[:, [2, 0]]]; fid = np.tile(np.arange(len(F)), 3); e.sort(1)
key = e[:, 0].astype(np.int64) * len(V) + e[:, 1]; o = np.argsort(key); ks = key[o]; same = ks[1:] == ks[:-1]; FA, FB = fid[o][:-1][same], fid[o][1:][same]
ang = np.arccos(np.clip((fn[FA] * fn[FB]).sum(1), -1, 1))
def otsu(v):
    v = np.sort(v); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
def otsu_w(v, w):
    o = np.argsort(v); v, w = v[o], w[o]; c = np.cumsum(w); cv = np.cumsum(w * v); Wt = c[-1]; w0 = c[:-1]; w1 = Wt - w0
    m0 = cv[:-1] / np.maximum(w0, 1e-30); m1 = (cv[-1] - cv[:-1]) / np.maximum(w1, 1e-30); return v[int(np.argmax(np.where((w0 > 0) & (w1 > 0), w0 * w1 * (m0 - m1) ** 2, -1)))]
def comps(ok, faces):                                             # face components over the allowed edges, among the given faces
    n, c = connected_components(coo_matrix((np.ones(ok.sum()), (FA[ok], FB[ok])), shape=(len(F), len(F))), directed=False); return c
def merge(comp, faces):
    """crumbs (below the face-weighted Otsu split of the log sizes, among these faces) into the neighbour with the most border"""
    inp = np.zeros(len(F), bool); inp[faces] = True; A, B = FA[inp[FA] & inp[FB]], FB[inp[FA] & inp[FB]]; N = comp.max() + 1
    sz = np.bincount(comp[faces], minlength=N); u = np.unique(comp[faces])
    if len(u) < 2: return comp
    th = np.exp(otsu_w(np.log(sz[u]), sz[u].astype(float)))
    for _ in range(200):
        sz = np.bincount(comp[faces], minlength=N); ca, cb = comp[A], comp[B]; m = ca != cb; a_, b_ = np.r_[ca[m], cb[m]], np.r_[cb[m], ca[m]]
        s = sz[a_] < th; a_, b_ = a_[s], b_[s]
        if not len(a_): break
        up, cnt = np.unique(a_.astype(np.int64) * N + b_, return_counts=True); a, t = up // N, up % N
        o = np.lexsort((-cnt, a)); f = o[np.r_[True, a[o][1:] != a[o][:-1]]]; a, t = a[f], t[f]; ok = (sz[t] > sz[a]) | ((sz[t] == sz[a]) & (t > a))
        if not ok.any(): break
        rm = np.arange(N); rm[a[ok]] = t[ok]
        for _ in range(64):
            nx = rm[rm]
            if (nx == rm).all(): break
            rm = nx
        comp = comp.copy(); comp[faces] = rm[comp[faces]]
    return comp
# --- stage 1: relief. Local altitude h = height of each (welded) vertex above a smoothed copy of the surface; its first
# derivative (change along each edge / edge length) and second derivative (Laplacian of h); a vertex is a boundary when
# either is above the upper Tukey fence (Q3 + 1.5 IQR) of all vertices. The smoothing scale is not chosen: the spread
# of h is measured at doubling smoothing steps and the knee of that curve (largest distance to its chord) is taken.
from scipy.sparse import csr_matrix, diags
UW, uinv = np.unique(Wd, return_inverse=True); X = V[UW].astype(np.float64); FU = uinv[F]; nU = len(UW)
EU = np.unique(np.sort(np.r_[FU[:, [0, 1]], FU[:, [1, 2]], FU[:, [2, 0]]], 1), axis=0)
A = csr_matrix((np.ones(2 * len(EU)), (np.r_[EU[:, 0], EU[:, 1]], np.r_[EU[:, 1], EU[:, 0]])), shape=(nU, nU)); P = diags(1 / np.maximum(np.asarray(A.sum(1)).ravel(), 1)) @ A
NU = np.zeros((nU, 3)); fnn = np.cross(X[FU[:, 1]] - X[FU[:, 0]], X[FU[:, 2]] - X[FU[:, 0]])
for c in range(3): np.add.at(NU, FU[:, c], fnn)
NU /= np.linalg.norm(NU, axis=1, keepdims=True) + 1e-30
S = X.copy(); steps, sd, hs = [], [], []; n_ = 0
for k in range(11):                                               # 1, 2, 4, ... 1024 smoothing steps
    while n_ < 2 ** k: S = P @ S; n_ += 1
    h_ = ((X - S) * NU).sum(1); steps.append(k); sd.append(float(np.std(h_))); hs.append(h_)
x_ = np.array(steps, float); y_ = np.array(sd); xn, yn = (x_ - x_[0]) / (x_[-1] - x_[0]), (y_ - y_[0]) / (y_[-1] - y_[0] + 1e-30)
kk = int(np.argmax(np.abs(yn - xn))); h = hs[kk]
el = np.linalg.norm(X[EU[:, 0]] - X[EU[:, 1]], axis=1) + 1e-30; g_e = np.abs(h[EU[:, 0]] - h[EU[:, 1]]) / el
g = np.zeros(nU); np.maximum.at(g, EU[:, 0], g_e); np.maximum.at(g, EU[:, 1], g_e)
lap = np.abs(h - P @ h) / np.median(el) ** 2
def fence(v): q1, q3 = np.percentile(v, [25, 75]); return q3 + 1.5 * (q3 - q1)
bnd = (g > fence(g)) | (lap > fence(lap))
print('GP stage 1 (relief): smoothing scale %d steps (knee of the altitude spread); boundary vertices %.1f%% (slope %.1f%%, curvature %.1f%%)' % (
    2 ** kk, 100 * bnd.mean(), 100 * (g > fence(g)).mean(), 100 * (lap > fence(lap)).mean()), flush=True)
bnd = bnd | (A @ bnd.astype(float) > 0)                            # closing: one ring around each boundary vertex, so broken lines still separate
ok = ~bnd[EU[:, 0]] & ~bnd[EU[:, 1]]
nc, vc = connected_components(csr_matrix((np.ones(ok.sum()), (EU[ok, 0], EU[ok, 1])), shape=(nU, nU)), directed=False)
vc[bnd] = -1
for _ in range(200):                                              # boundary vertices join the part most of their labelled neighbours are in
    un = np.flatnonzero(vc < 0)
    if not len(un): break
    a_ = np.r_[EU[:, 0], EU[:, 1]]; b_ = np.r_[EU[:, 1], EU[:, 0]]; m = (vc[a_] < 0) & (vc[b_] >= 0)
    if not m.any(): break
    up, cnt = np.unique(a_[m].astype(np.int64) * (nc + 1) + vc[b_[m]], return_counts=True); va, lb = up // (nc + 1), up % (nc + 1)
    o = np.lexsort((-cnt, va)); f = o[np.r_[True, va[o][1:] != va[o][:-1]]]; vc[va[f]] = lb[f]
fl = vc[FU]; f1 = np.where(fl[:, 0] == fl[:, 1], fl[:, 0], np.where(fl[:, 1] == fl[:, 2], fl[:, 1], fl[:, 0]))   # face: its vertices' majority
# big parts: neighbouring relief regions stay apart only where the true colour jumps across their border (skin -> blue),
# not where only brightness changes. Per region its mean chroma (Lab a, b); per neighbouring pair the chroma distance,
# weighted by the length of their shared border; the Otsu split of the log distances separates "same material" from
# "material change"; pairs below it are joined. Crumbs are merged after that, so a crumb never swallows a colour change.
LABc = cv2.cvtColor(C.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float64); fab0 = LABc[F][:, :, 1:].mean(1)
_, r = np.unique(np.maximum(f1, 0), return_inverse=True); r = r.ravel(); nr = r.max() + 1   # compact ids: regions that own faces
cnt_r = np.bincount(r, minlength=nr).astype(float)
mab = np.c_[np.bincount(r, fab0[:, 0], nr), np.bincount(r, fab0[:, 1], nr)] / np.maximum(cnt_r, 1)[:, None]
m = r[FA] != r[FB]; pa, pb = np.minimum(r[FA][m], r[FB][m]), np.maximum(r[FA][m], r[FB][m])
up, blen = np.unique(pa.astype(np.int64) * nr + pb, return_counts=True); pa, pb = up // nr, up % nr
dist = np.linalg.norm(mab[pa] - mab[pb], axis=1)
split = np.exp(otsu_w(np.log(dist + 1e-6), blen.astype(float)))   # all borders, weighted by their length
print('GP big parts: %d relief regions; colour-change split %.1f (chroma distance, Lab a b; borders: median %.1f, p90 %.1f)' % (nr, split, np.median(dist), np.percentile(dist, 90)), flush=True)
# colour ranges inside regions: a relief line that never closed leaves skin, cloth and hair in one region. Face chroma
# clustered over the whole model (Gaussian mixture, count by BIC); clusters closer than the colour-change split above
# (same material, different shading) form one colour range; each region is split where its faces change colour range
sub0 = fab0[np.random.default_rng(0).choice(len(F), min(30000, len(F)), replace=False)]
fits0 = [GaussianMixture(k, random_state=0).fit(sub0) for k in range(1, 13)]; g0 = fits0[int(np.argmin([f.bic(sub0) for f in fits0]))]
cd_ = np.linalg.norm(g0.means_[:, None] - g0.means_[None], axis=2); ncr, crange = connected_components(csr_matrix(cd_ < split), directed=False)
fcr = crange[g0.predict(fab0)]
key2 = r.astype(np.int64) * ncr + fcr; okf = key2[FA] == key2[FB]
_, r2 = connected_components(csr_matrix((np.ones(okf.sum()), (FA[okf], FB[okf])), shape=(len(F), len(F))), directed=False)
_, r = np.unique(r2, return_inverse=True); r = r.ravel(); nr = r.max() + 1
print('GP colour ranges: %d mixture colours -> %d colour ranges (centres closer than %.1f joined); regions split by range -> %d' % (g0.n_components, ncr, split, nr), flush=True)
cnt_r = np.bincount(r, minlength=nr).astype(float); mab = np.c_[np.bincount(r, fab0[:, 0], nr), np.bincount(r, fab0[:, 1], nr)] / np.maximum(cnt_r, 1)[:, None]
m = r[FA] != r[FB]; pa, pb = np.minimum(r[FA][m], r[FB][m]), np.maximum(r[FA][m], r[FB][m])
up, blen = np.unique(pa.astype(np.int64) * nr + pb, return_counts=True); pa, pb = up // nr, up % nr; dist = np.linalg.norm(mab[pa] - mab[pb], axis=1)
# SAM 3 as the filter: a relief boundary between two regions is kept only where SAM 3's label also changes across it
# (each region's dominant SAM 3 concept, from sam_body/top_concept.npz); neighbours with the same concept are joined
TC = np.load(j('sam_body', 'top_concept.npz'), allow_pickle=True); tcl = np.asarray(TC['label']).astype(int); tcn = [str(x) for x in TC['names']]
fcon = tcl[F]; fcon = np.where(fcon[:, 0] >= 0, fcon[:, 0], np.where(fcon[:, 1] >= 0, fcon[:, 1], fcon[:, 2]))
nk = len(tcn) + 1; dcon = np.zeros(nr, int); cm_ = np.zeros((nr, nk))
np.add.at(cm_, (r, fcon + 1), 1); cm_[:, 0] = 0; dcon = cm_.argmax(1) - 1          # -1: no SAM 3 label on the region
same = (dcon[pa] == dcon[pb]) & (dcon[pa] >= 0) & (dist < split)   # joined only when SAM 3 agrees AND the true colour does not jump
nb_, rc = connected_components(csr_matrix((np.ones(same.sum()), (pa[same], pb[same])), shape=(nr, nr)), directed=False)
print('GP SAM 3 + colour filter: %d relief regions, %d borders, %d kept (SAM 3 changes %d, colour jumps %d) -> %d groups' % (nr, len(pa), (~same).sum(), (dcon[pa] != dcon[pb]).sum(), (dist >= split).sum(), nb_), flush=True)
_, g1 = np.unique(rc[r], return_inverse=True); ng = g1.max() + 1; allf = np.arange(len(F))
# crumbs (below the Otsu split of the log group sizes, unweighted) go to the neighbouring group closest in colour
cg0 = np.bincount(g1, minlength=ng).astype(float); th = np.exp(otsu(np.log(cg0[cg0 > 0]))) if (cg0 > 0).sum() > 2 else 0   # unweighted, fixed once: tiny crumbs vs real groups
for _ in range(50):
    cg = np.bincount(g1, minlength=ng).astype(float); live = cg > 0
    mg = np.c_[np.bincount(g1, fab0[:, 0], ng), np.bincount(g1, fab0[:, 1], ng)] / np.maximum(cg, 1)[:, None]
    m = g1[FA] != g1[FB]; a_, b_ = np.r_[g1[FA][m], g1[FB][m]], np.r_[g1[FB][m], g1[FA][m]]; k = cg[a_] < th; a_, b_ = a_[k], b_[k]
    if not len(a_): break
    d_ = np.linalg.norm(mg[a_] - mg[b_], axis=1); o = np.lexsort((d_, a_)); f = o[np.r_[True, a_[o][1:] != a_[o][:-1]]]
    rm = np.arange(ng); src, dst = a_[f], b_[f]; ok = (cg[dst] > cg[src]) | ((cg[dst] == cg[src]) & (dst > src)); rm[src[ok]] = dst[ok]
    if not ok.any(): break
    for _ in range(64):
        nx = rm[rm]
        if (nx == rm).all(): break
        rm = nx
    g1 = rm[g1]
u1, s1 = np.unique(g1, return_inverse=True)
print('GP big parts after crumb merge: %d; sizes %s' % (len(u1), sorted(np.bincount(s1).tolist())[::-1][:30]), flush=True)
np.savez(j('sam_body', 'relief.npz'), h=h, slope=g, lap=lap, boundary=bnd, welded=UW, scale_steps=2 ** kk, sd=np.array(sd))
thr = otsu(ang)
LAB = cv2.cvtColor(C.reshape(-1, 1, 3), cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32); fab = LAB[F][:, :, 1:].mean(1)   # face chroma (a, b)
s2 = np.full(len(F), -1); nxt = 0; rng = np.random.default_rng(0)
for p in range(len(u1)):
    fi = np.flatnonzero(s1 == p); X = fab[fi]
    if len(fi) < 20: s2[fi] = nxt; nxt += 1; continue
    sub = X[rng.choice(len(fi), min(20000, len(fi)), replace=False)]
    fits = [GaussianMixture(k, random_state=0).fit(sub) for k in range(1, 9) if k * 20 <= len(sub)]; g = fits[int(np.argmin([f.bic(sub) for f in fits]))]
    lab = np.full(len(F), -1); lab[fi] = g.predict(X); ok = (lab[FA] >= 0) & (lab[FA] == lab[FB])
    c = merge(comps(ok, fi), fi); uu, ci = np.unique(c[fi], return_inverse=True); s2[fi] = nxt + ci; nxt += len(uu)
    print('GP   part %d (%d faces): %d colours by BIC -> %d pieces' % (p, len(fi), g.n_components, len(uu)), flush=True)
print('GP stage 2 (true colour): %d pieces' % nxt, flush=True)
vlab = lambda fl: (lambda out: (np.maximum.at(out, F.ravel(), np.repeat(fl, 3)), out)[1])(np.full(len(V), -1))
np.savez(j('sam_body', 'geo_parts.npz'), stage1=vlab(s1), stage2=vlab(s2), face_stage1=s1, face_stage2=s2, crease_deg=np.degrees(thr))
for k, lab in (('stage1', vlab(s1)), ('stage2', vlab(s2))): np.savez(j('sam_body', 'geo_%s_label.npz' % k), label=lab, names=np.array(['p%d' % i for i in range(lab.max() + 1)]))
print('GP done')
