"""Stage 2 of the skull fit: after the one-size placement (skull_line_fit.py) the skull bends, as a skull can - an
anatomical deformation graph (page "Stage 2 Bone Deformation"):
  E:\\splat\\.venv\\Scripts\\python.exe skull_stage2.py <fits dir> <model dir> <atlas head dir> <ct data dir>
  nodes     a few control nodes per head bone (farthest-point samples of its vertices; how many: the bone's extent over
            the median extent of the non-tooth bones, squared), placed mirror-symmetric left / right. A node j carries a
            small transform (A_j, t_j); a bone point x moves with its own bone's nearest nodes:
              D(x) = x + sum_j w_j(x) [ (A_j - I)(x - g_j) + t_j ]          (atlas space; then stage 1's size and pose)
  edges     stiff  inside a bone: each node predicts its neighbours' places, scale = the strain real skulls show
            seam   bones that touch predict the same place for their contact points, scale = the joint gap the
                   skeleton fit reached
            hinge  the jaw (mandible + lower teeth) hangs on the skull only by its two joints: the middle of each joint's
                   contact stays in its socket
  data      lines  under each named skin line (jaw outline, eyebrows) the atlas has a bone curve at a known depth; her
                   bone curve must lie under her line at that depth (scaled), free to slide along it; a line counts only
                   where it runs over its own main bone (the jaw outline's ends by the ears lie over the cheekbone)
            sites  CT tissue depth (388 people, medians and spreads, scaled with the skull) at the points the detector
                   shares with the CT study: nose root, between the brows, above the eyes, under the nose, mid upper lip,
                   upper and lower lip, chin. They outrank the lines: besides the soft pull to the CT median, the bone
                   may not come nearer the skin than the median minus one spread - a hard bound at the steadier sites
                   only (spread under the median site's: nose root, between the brows, above the eyes). At the lips,
                   under the nose and the chin a hard bound pushed the lower teeth back and took the whole lower face
                   with them (upper lip and chin 3 mm too deep), so those stay a soft pull
  shape     each node's stretch (symmetric part of A_j - I) / the spread of skull proportions between people (CT skulls),
            and never beyond three such spreads (hard)
  mirror    left and right nodes deform as mirror images (scale: the stiff scale)
  philtrum  the bone under the philtrum area follows her face even past what real skulls show: it may not lie deeper
            than the CT median plus one spread there (hard), it has nodes of its own under the area, and those nodes are
            loosened stage after stage while their depth is unmet (no stop for small gains)
  nose      the nose's cartilages are parts of the fit like the bones, with her nose bridge and nose base lines over
            them; the philtrum (Jev's box, face/philtrum_area.py) is covered with soft CT depth sites as an area
  pose      on top of the nodes the whole skull may nod, shift forward and up, and be larger or smaller - the moves that
            keep its mirror plane where stage 1 put it; these are settled first, alone, before any bone bends. It may not tilt sideways, turn or shift sideways: the skull stays symmetric
            about the head's midline.
  eyes      the atlas's eye centre, carried by the bones round each socket, lies under her eye centre (across and
            up-down; scale: the detector's noise)
  inside    hard: every bone hull point stays inside the head solid (signed distance from stage 1), linearised per step,
            with a slack that is driven to zero; under bare skin the margin is the thinnest CT site's lower end
            (5th percentile, scaled), under hair and cloth zero
Each round is one convex problem (cvxpy, Clarabel): correspondences and the inside rule are re-linearised, steps damped
at the line scale, until no hull point moves more than a tenth of a grid cell.
  loosening per part: once settled, nodes whose own targets are still unmet (line farther than the detector's noise, CT
            depth off by more than its spread) get their stiffness and stretch cap doubled, mirror partner too, and the
            fit runs again; nodes at their standard keep theirs; when a loosening gains less than the standard itself, it
            is taken back and the node is left alone (without this a cheekbone was loosened 256 times and stretched 150%). A bone held
            at the surface stays inside, and the other parts go on. Inside rule and seams never loosen.
Writes <fits dir>/skull_stage2.npz (V placed for all atlas vertices, V_stage1, faces) and skull_stage2.json (checks)."""
import sys, os, re, json, time, numpy as np, cvxpy as cp
from scipy import ndimage
from scipy.interpolate import splprep, splev
from scipy.sparse import coo_matrix, vstack as svstack
from scipy.spatial import cKDTree, ConvexHull
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segments import SEG
FD, MD, AH, CT = sys.argv[1:5]; t00 = time.time(); log = lambda *a: print('S2', *a, flush=True)
Z = np.load(os.path.join(FD, 'skull_lines.npz')); R0, s0, t0 = Z['R'], float(Z['s']), Z['t']; XE, XA = Z['points_model'], Z['points_atlas']; sdf, glo, hst = Z['sdf'].astype(np.float64), Z['sdf_origin'], float(Z['sdf_step'])
W = np.load(os.path.join(FD, 'atlas_warp.npz')); V0 = W['V0'].astype(np.float64); OB = W['obj']; names = [str(x) for x in W['names']]; sysn = np.array([str(x) for x in W['system']]); FB = W['F_bone']
K = np.load(os.path.join(FD, 'skeleton_fit.npz'), allow_pickle=True); SJ = json.load(open(os.path.join(FD, 'skeleton_fit.json'))); L1 = json.load(open(os.path.join(FD, 'skull_lines.json')))
CTS = json.load(open(os.path.join(CT, 'summary.json')))['sites']; SIG_S = float(json.load(open(os.path.join(CT, 'skull_shape_spread.json')))['proportion_spread'])
to_atlas = lambda Y: (Y - t0) @ R0 / s0; to_world = lambda X: X @ (s0 * R0).T + t0
hall = [i for i, n in enumerate(names) if sysn[i] == 'bone' and (re.search(SEG['head']['bones'], n) or re.search('cartilage|Incus|Malleus|Stapes|Sinus|concha', n)) and not re.search('rib|Thyroid|Cricoid|Costal|Arytenoid|Epiglot|trachea', n, re.I)]
hc = [i for i in hall if 'cartilage' in names[i].lower()]; hb = [i for i in hall if i not in hc]
nose = [i for i in hc if re.search('nasal|alar', names[i], re.I)]; hb += nose; hc = [i for i in hc if i not in nose]   # the nose's cartilages are fitted with the skull: her nose says how far forward the middle of the face is
vid_b = {b: np.flatnonzero(OB == b) for b in hall}
# how much a nose varies between people, against how much a skull does: the spread of the nose's own proportions (the
# distances among nose tip, alar curvatures and subnasale over face size, 388 CT faces: face_skull_stats.py) over the
# spread of skull proportions. The nose's cartilages bend that much more easily than bone.
FM_ = np.load(os.path.join(CT, 'face_skull_model.npz'), allow_pickle=True); nf_ = [i for i, f in enumerate(FM_['features']) if set(str(f).split('-')) <= {'prn', 'acL', 'acR', 'sn'}]
NOSE_X = float(np.median(FM_['feat_sd'][nf_])) / SIG_S if nf_ else 1.0
jaw = lambda b: names[b].startswith('Mandible') or names[b].startswith('Lower ')       # the jaw's group: mandible and lower teeth
# --- nodes
ext = {b: float(np.linalg.norm(np.ptp(V0[vid_b[b]], 0))) for b in hb}; d0 = float(np.median([ext[b] for b in hb if not re.search('incisor|canine|premolar|molar|cartilage', names[b])]))
def fps(P, n):
    idx = [int(np.argmin(np.linalg.norm(P - P.mean(0), axis=1)))]; d = np.linalg.norm(P - P[idx[0]], axis=1)
    while len(idx) < n: idx.append(int(np.argmax(d))); d = np.minimum(d, np.linalg.norm(P - P[idx[-1]], axis=1))
    return P[idx]
G, GB, MIR = [], [], []                                             # node positions, their bone, mirror pairs
byname = {names[b]: b for b in hb}; done = set()
for b in hb:
    if b in done: continue
    n = max(1, int(round((ext[b] / d0) ** 2))); nm = names[b]; P = V0[vid_b[b]]
    if nm.endswith('.r') and nm[:-2] + '.l' in byname:
        o = byname[nm[:-2] + '.l']; g = fps(P, n); k0 = len(G); G += list(g); GB += [b] * n; Po = V0[vid_b[o]]; gm = Po[cKDTree(Po).query(g * [-1, 1, 1])[1]]; G += list(gm); GB += [o] * n
        MIR += [(k0 + i, k0 + n + i) for i in range(n)]; done |= {b, o}
    elif nm.endswith('.l') and nm[:-2] + '.r' in byname: continue
    else:                                                           # a midline bone: its x >= 0 half sampled, then mirrored
        h = P[P[:, 0] >= 0] if (P[:, 0] >= 0).sum() > 3 else P; g = fps(h, max(1, (n + 1) // 2)); k0 = len(G); sp = ext[b] / max(np.sqrt(n), 1)
        for gi in g:
            if abs(gi[0]) < sp / 4 or n == 1: G.append(gi * [0, 1, 1] if n > 1 else gi); GB.append(b); MIR.append((len(G) - 1, len(G) - 1))
            else: gm = P[cKDTree(P).query(gi * [-1, 1, 1])[1]]; G += [gi, gm]; GB += [b, b]; MIR.append((len(G) - 2, len(G) - 1))
        done.add(b)
# a node of its own under the philtrum area (Jev's box): the standard nodes are spread by bone size, and a half upper jaw
# gets one or two - too few to bend under the lip alone (loosened 128 times it still could not: every seam of the bone
# moved with it). Each bone lying under the area gets a node there (the middle of its points nearest the area), with its
# mirror partner, unless it already has one closer than the area's own sites are apart.
pf0_ = os.path.join(MD, 'parts_zoom', 'philtrum.npz'); n_area_nodes = 0
if os.path.exists(pf0_):
    PA_ = to_atlas(np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'][np.load(pf0_)['vids']].astype(np.float64)); sp0_ = float(np.linalg.norm(XE[33] - (XE[33] + XE[51]) / 2)) / s0
    ii_ = np.concatenate([vid_b[b] for b in hb if b not in nose]); kk_ = cKDTree(V0[ii_]).query(PA_)[1]; bb_ = OB[ii_[kk_]]; seen_ = set()
    for b in np.unique(bb_):
        nm = names[b]; o = byname.get(nm[:-2] + ('.r' if nm.endswith('.l') else '.l')) if nm[-2:] in ('.l', '.r') else None
        if b in seen_: continue
        Pb = V0[vid_b[b]]; g = Pb[cKDTree(Pb).query(V0[ii_[kk_[bb_ == b]]].mean(0))[1]]; have = np.array([G[j] for j in range(len(G)) if GB[j] == b])
        if len(have) and np.linalg.norm(have - g, axis=1).min() < sp0_: continue
        if o is not None: Po = V0[vid_b[o]]; gm = Po[cKDTree(Po).query(g * [-1, 1, 1])[1]]; G += [g, gm]; GB += [b, o]; MIR.append((len(G) - 2, len(G) - 1)); seen_ |= {b, o}; n_area_nodes += 2
        else: G.append(g * [0, 1, 1]); GB.append(b); MIR.append((len(G) - 1, len(G) - 1)); n_area_nodes += 1
    log('philtrum area: %d nodes added under it (%s)' % (n_area_nodes, ', '.join(sorted({names[b] for b in seen_}) or ['none needed'])))
G = np.array(G); GB = np.array(GB); M = len(G); nodes_of = {b: np.flatnonzero(GB == b) for b in hb}; NQ = 12 * M; NU = NQ + 7   # node numbers, then 6 for the whole skull's pose (turn, shift) and 1 for its size
log('%d head bones, %d nodes (median bone extent %.0f mm; most: %s); %d mirror pairs; strain scale %.3f' % (len(hb), M, 1000 * d0, ', '.join('%s %d' % (names[b], len(nodes_of[b])) for b in sorted(hb, key=lambda b: -len(nodes_of[b]))[:5]), len(MIR), SIG_S))
CEN = np.concatenate([V0[vid_b[b]] for b in hb]).mean(0)
def Bmat(X, bone):
    """sparse (3n x 12M): D(X) - X = B z for points X moving with the nodes of 'bone' (an id per point, or one id)"""
    bone = np.full(len(X), bone) if np.isscalar(bone) else np.asarray(bone); rows, cols, vals = [], [], []
    for b in np.unique(bone):
        m = np.flatnonzero(bone == b); nd = nodes_of[b]; P = X[m]; k = min(4, len(nd)); d = np.linalg.norm(P[:, None] - G[nd][None], axis=2); o = np.argsort(d, 1)
        if len(nd) == 1: w = np.ones((len(m), 1)); sel = np.zeros((len(m), 1), int)
        else:
            sel = o[:, :k]; dk = np.take_along_axis(d, sel, 1); dn = np.take_along_axis(d, o[:, k:k + 1], 1) if len(nd) > k else 1.5 * dk[:, -1:]
            w = np.maximum(1 - dk / (dn + 1e-12), 0) ** 2 + 1e-9; w /= w.sum(1, keepdims=True)
        for c in range(sel.shape[1]):
            j = nd[sel[:, c]]; dx = P - G[j]
            for a in range(3):
                for q in range(3): rows.append(3 * m + a); cols.append(12 * j + 3 * a + q); vals.append(w[:, c] * dx[:, q])
                rows.append(3 * m + a); cols.append(12 * j + 9 + a); vals.append(w[:, c])
    n = len(X); i3 = 3 * np.arange(n); xc = X - CEN                                  # whole-skull pose: w x (x - centre) + shift
    for a, (q1, q2, sg) in enumerate(((1, 2, 1.0), (2, 0, 1.0), (0, 1, 1.0))):       # column NQ + a: turn about axis a
        rows += [i3 + q2, i3 + q1]; cols += [np.full(n, NQ + a), np.full(n, NQ + a)]; vals += [sg * xc[:, q1], -sg * xc[:, q2]]
        rows.append(i3 + a); cols.append(np.full(n, NQ + 3 + a)); vals.append(np.ones(n))
        rows.append(i3 + a); cols.append(np.full(n, NQ + 6)); vals.append(xc[:, a])                 # column NQ + 6: the whole skull larger or smaller about its centre
    return coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(3 * len(X), NU)).tocsr()
# --- regular terms (fixed): stiff edges, shape, mirror, seams, hinge
RW, RR = [], []                                                     # rows (sparse) and right-hand sides of  || RW z - RR ||^2
def add(Mx, rhs, sig): RW.append(Mx / sig); RR.append(np.asarray(rhs, float).ravel() / sig)
def row_block(entries, n):                                          # entries: (row, col, val) lists
    r, c, v = zip(*entries); return coo_matrix((np.concatenate(v), (np.concatenate(r), np.concatenate(c))), shape=(n, NU)).tocsr()
ent = []; rhs = []; k_ = 0; elen = []; PAIR = []
for b in hb:
    nd = nodes_of[b]
    if len(nd) < 2: continue
    d = np.linalg.norm(G[nd][:, None] - G[nd][None], axis=2); nb = np.argsort(d, 1)[:, 1:min(4, len(nd))]
    for i in range(len(nd)):
        for jn in nb[i]:
            a_, b_ = nd[i], nd[jn]; dg = G[b_] - G[a_]; L = np.linalg.norm(dg)       # (A_a - I) dg + t_a - t_b = 0, in units of the edge's strain
            for ax in range(3):
                ent.append((np.full(5, k_), np.r_[12 * a_ + 3 * ax + np.arange(3), 12 * a_ + 9 + ax, 12 * b_ + 9 + ax], np.r_[dg, 1, -1] / (SIG_S * L))); k_ += 1; PAIR.append((a_, b_))
            elen.append(L)
            if a_ < b_ or a_ not in nd[nb[jn]]:                    # neighbours in a bone also turn and stretch alike (bone does not twist: without this a node could spin about its edge)
                ent.append((np.repeat(k_ + np.arange(9), 2), np.c_[12 * a_ + np.arange(9), 12 * b_ + np.arange(9)].ravel(), np.tile([1.0, -1.0], 9) / SIG_S)); k_ += 9; PAIR += [(a_, b_)] * 9
STIFF = row_block(ent, k_); PAIR = np.array(PAIR)       # kept apart from the fixed terms: these rows are loosened per node (below)
n_stiff = k_; ent = []; k_ = 0; SYM = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)]
for j in range(M):                                                  # stretch of every node: the symmetric part of A_j - I
    for a, q in SYM:
        if a == q: ent.append((np.array([k_]), np.array([12 * j + 3 * a + q]), np.array([1.0])))
        else: ent.append((np.full(2, k_), np.array([12 * j + 3 * a + q, 12 * j + 3 * q + a]), np.array([0.5, 0.5])))
        k_ += 1
SHAPE = row_block(ent, k_); SH_NODE = np.repeat(np.arange(M), 6); ent = []; k_ = 0; Mx = np.array([-1.0, 1, 1]); elm = float(np.median(elen)) if elen else d0
for a_, b_ in MIR:                                                  # mirror: A_b = Mx A_a Mx, t_b = Mx t_a  (a node on the midline with itself)
    for ax in range(3):
        for q in range(3):
            sg = Mx[ax] * Mx[q]
            if a_ == b_:
                if sg < 0: ent.append((np.array([k_]), np.array([12 * a_ + 3 * ax + q]), np.array([2.0 / SIG_S]))); k_ += 1
            else: ent.append((np.full(2, k_), np.array([12 * b_ + 3 * ax + q, 12 * a_ + 3 * ax + q]), np.array([1.0, -sg]) / SIG_S)); k_ += 1
        if a_ == b_:
            if Mx[ax] < 0: ent.append((np.array([k_]), np.array([12 * a_ + 9 + ax]), np.array([2.0 / (SIG_S * elm)]))); k_ += 1
        else: ent.append((np.full(2, k_), np.array([12 * b_ + 9 + ax, 12 * a_ + 9 + ax]), np.array([1.0, -Mx[ax]]) / (SIG_S * elm))); k_ += 1
add(row_block(ent, k_), np.zeros(k_), 1.0)
def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i); return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]
SIG_SEAM = SJ['joint_gap_mm']['median'] / 1000 / s0; trees = {}; seams = []; hinge = []; rng = np.random.default_rng(0); linked = {b: {b} for b in hb}
for a, b in K['contact_pairs']:
    a, b = int(a), int(b)
    if a not in ext or b not in ext: continue
    pts = []
    for u_, v_ in ((a, b), (b, a)):
        if v_ not in trees: trees[v_] = cKDTree(V0[vid_b[v_]])
        dd = trees[v_].query(V0[vid_b[u_]])[0]; l = np.log(dd + 1e-9); pts.append(V0[vid_b[u_]][l < otsu(l)] if len(l) > 2 else V0[vid_b[u_]])
    P = np.vstack(pts)
    if not len(P): continue
    if jaw(a) != jaw(b):
        if 'Temporal' in names[a] + names[b]: hinge.append((a, b, P.mean(0, keepdims=True)))      # the jaw joint: only the middle of the contact is held
        continue                                                    # other jaw-skull contacts (teeth meeting) do not tie the jaw
    P = P[rng.choice(len(P), min(len(P), 40), replace=False)]; seams.append((a, b, P)); linked[a].add(b); linked[b].add(a)
if not hinge:                                                       # jaw joints: where the mandible comes nearest each temporal bone (the joint space is
    mb = byname['Mandible']; Pm = V0[vid_b[mb]]                    # wider than the contact threshold, so no pair was listed); the nearest tenth of its
    for side in ('.r', '.l'):                                       # nearest-distance points, and their middle
        tb_ = byname.get('Temporal bone' + side)
        if tb_ is None: continue
        dd = cKDTree(V0[vid_b[tb_]]).query(Pm)[0]; near = Pm[dd <= np.sort(dd)[max(1, len(dd) // 100)]]; hinge.append((mb, tb_, near.mean(0, keepdims=True)))
for a, b, P in seams + hinge: add(Bmat(P, a) - Bmat(P, b), np.zeros(3 * len(P)), SIG_SEAM)
comp = {}                                                           # bones with no seam to the rest of their group: tied to the nearest node of that group (stiff)
for grp in (False, True):
    bs = [b for b in hb if jaw(b) == grp]; root = max(bs, key=lambda b: ext[b]); seen = {root}; stack = [root]
    while stack:
        for o in linked[stack.pop()]:
            if o not in seen and jaw(o) == grp: seen.add(o); stack.append(o)
    main = np.concatenate([nodes_of[b] for b in seen]); tm = cKDTree(G[main]); loose = [b for b in bs if b not in seen]
    for b in loose:
        gj = nodes_of[b][0]; gi = main[tm.query(G[gj])[1]]; P = G[gj][None]; add(Bmat(P, b) - Bmat(P, GB[gi]), np.zeros(3), SIG_S * np.linalg.norm(G[gj] - G[gi]) + 1e-9)
        for gj in nodes_of[b]:                                      # ... and turns and stretches with it (a point tie alone left the bone free to spin: a cheekbone turned 57 degrees)
            add(row_block([(np.repeat(np.arange(9), 2), np.c_[12 * gj + np.arange(9), 12 * gi + np.arange(9)].ravel(), np.tile([1.0, -1.0], 9))], 9), np.zeros(9), SIG_S * (NOSE_X if b in nose else 1.0))
    comp[grp] = len(loose)
log('edges: %d stiff rows, %d seams (%d contact points), %d jaw-joint points, %d + %d bones tied to their group by a link; seam scale %.2f mm' % (n_stiff, len(seams), sum(len(p[2]) for p in seams), len(hinge), comp[False], comp[True], 1000 * s0 * SIG_SEAM))
# --- data: bone curves under the lines, CT depth sites
hm = np.isin(OB, hb); bidx = np.flatnonzero(hm); bt = cKDTree(V0[bidx]); MVm = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); vid = np.load(os.path.join(MD, 'parts_zoom', 'head_vids.npy'))
HE = MVm[vid]; HN = np.load(os.path.join(MD, 'parts', 'orig_normals.npz'))['N'][vid].astype(np.float64); ht = cKDTree(HE); eE = hst
AV = np.load(os.path.join(AH, 'parts', 'orig_colours.npz'))['V']; AF = np.load(os.path.join(AH, 'parts', 'orig_normals.npz'))['F']; eA = float(np.median(np.linalg.norm(AV[AF[:, 0]] - AV[AF[:, 1]], axis=1)))
def line(P, step): L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()); tck, _ = splprep(P.T, s=0, k=min(3, len(P) - 1)); return np.array(splev(np.linspace(0, 1, max(8, int(L / step) + 1)), tck)).T
import face_lines; FL_ = face_lines.load(MD)
RUNS = {n: l['points'] for n, l in FL_.items() if l['under_skin']}; LINES = {}   # the lines the naming step says lie directly on bone or cartilage
WHOLE = {n for n, l in FL_.items() if l['under_skin'] and l['cartilage']}      # lines over the nose's cartilages count along their whole length: nasal bones and cartilages are both in the fit
for n, r in RUNS.items():
    La = line(XA[list(r)], eA); dA, kA = bt.query(La); bv = bidx[kA]; Le = line(XE[list(r)], eE); Nn = HN[ht.query(Le)[1]]; Nn /= np.linalg.norm(Nn, axis=1, keepdims=True) + 1e-12
    ua = np.linspace(0, 1, len(La)); ue = np.linspace(0, 1, len(Le)); bn = np.array([re.sub(r'\.[lr]$', '', names[b]) for b in OB[bv]]); un, cn = np.unique(bn, return_counts=True); main = un[cn.argmax()]
    im = np.arange(len(bn)) if n in WHOLE else np.flatnonzero(bn == main); i0, i1 = im.min(), im.max(); ke = (ue >= ua[i0]) & (ue <= ua[i1])                                           # the stretch of the line over its main bone, on both heads
    dep = np.interp(ue, ua, dA) * s0; tgt = to_atlas(Le - dep[:, None] * Nn)[ke]; bv, dA = bv[i0:i1 + 1], dA[i0:i1 + 1]                          # her line, moved inward by the atlas's depth there
    log('line %s: over %s for %.0f%% of its length; the rest (ends over %s) is left out' % (n, main, 100 * (ua[i1] - ua[i0]), ', '.join(x for x in un if x != main) or 'nothing'))
    LINES[n] = dict(bv=bv, tgt=tgt, B=Bmat(V0[bv], OB[bv]), X=V0[bv], depth=dA)
SIG_L = float(np.median(list(L1['lines + bones inside']['line_distance_mm'].values()))) / 1000 / s0
ZL_ = {k_: Z[k_] for k_ in Z.files}                                 # the saved model lines; the nose lines are built here
for n in WHOLE: ZL_['line_model_' + n.replace(' ', '_')] = line(XE[list(RUNS[n])], eE)
mid = lambda a, b: (XE[a] + XE[b]) / 2
SITES = {'nose root (n)': ('n', XE[27]), 'between the brows (g)': ('g', mid(21, 22)), 'above the right eye (mso)': ('mso R', XE[19]), 'above the left eye (mso)': ('mso L', XE[24]), 'under the nose (sn)': ('sn', XE[33]),
         'mid upper lip (mp)': ('mp', mid(33, 51)), 'upper lip (ls)': ('ls', XE[51]), 'lower lip (li)': ('li', XE[57]), 'chin (gn)': ('gn', XE[8])}
SITE = {k: dict(P=to_atlas(v[1][None])[0], tau=CTS[v[0]]['median_mm'] / 1000, sig=CTS[v[0]]['robust_sd_mm'] / 1000) for k, v in SITES.items()}   # atlas space is adult-sized: the CT depth itself
# eyes: the bones round each eye socket carry the atlas's eye centre (on its skin); it must lie under her eye centre. Only
# across and up-down count (the atlas's x and z): how deep an eye sits is not known. Bones round the socket = those with a
# point within one eye width of the eye centre.
EYE = []
for r_ in (range(36, 42), range(42, 48)):
    eA = XA[list(r_)].mean(0); eE = to_atlas(XE[list(r_)].mean(0)[None])[0]; wid = float(np.linalg.norm(XA[r_[0]] - XA[r_[3]])); dd, kk = bt.query(eA, k=min(4000, len(bidx)), distance_upper_bound=wid)
    kk = kk[np.isfinite(dd)]; bb, first = np.unique(OB[bidx[kk]], return_index=True); xv_ = bidx[kk[first]]; EYE.append(dict(eA=eA, eE=eE, xv=xv_, bones=[names[b] for b in bb]))
log('eyes: socket bones %s; her eye centres against the atlas eyes after stage 1 (mm, across / up): %s' % (sorted({re.sub(r'\.[lr]$', '', n) for E_ in EYE for n in E_['bones']}), ', '.join('%.1f / %.1f' % (1000 * s0 * (E_['eE'] - E_['eA'])[0], 1000 * s0 * (E_['eE'] - E_['eA'])[2]) for E_ in EYE)))
SIG_MED = float(np.median([S_['sig'] for S_ in SITE.values()]))
# the philtrum as an area (face/philtrum_area.py: Jev's box on her face): the CT study measured the tissue at single points
# down the midline (under the nose, mid upper lip, upper lip); her philtrum is covered with such points at the spacing
# those sites have on her own face, each with the mid upper lip's depth and spread. Soft, like the lip sites.
pf_ = os.path.join(MD, 'parts_zoom', 'philtrum.npz'); N_AREA = 0
if os.path.exists(pf_):
    PP = MVm[np.load(pf_)['vids']]; sp_ = float(np.linalg.norm(XE[33] - mid(33, 51))); pick = [int(np.argmin(np.linalg.norm(PP - PP.mean(0), axis=1)))]; dd_ = np.linalg.norm(PP - PP[pick[0]], axis=1)
    while dd_.max() >= sp_: pick.append(int(dd_.argmax())); dd_ = np.minimum(dd_, np.linalg.norm(PP - PP[pick[-1]], axis=1))
    for i_, k_ in enumerate(pick): SITE['philtrum area %d' % (i_ + 1)] = dict(P=to_atlas(PP[k_][None])[0], tau=CTS['mp']['median_mm'] / 1000, sig=CTS['mp']['robust_sd_mm'] / 1000)
    N_AREA = len(pick); log('philtrum area: %d of her surface points, covered by %d depth sites %.1f mm apart' % (len(PP), N_AREA, 1000 * sp_))   # sites steadier than the median site (nose root, between the brows, above the eyes) get the hard bound;
log('hard depth bound at: %s; soft only at: %s' % (', '.join(k.split(' (')[0] for k, S_ in SITE.items() if S_['sig'] < SIG_MED), ', '.join(k.split(' (')[0] for k, S_ in SITE.items() if S_['sig'] >= SIG_MED)))   # lips, under the nose and chin vary more between people: soft pull only
BONE_ = np.flatnonzero(~np.isin(OB[bidx], nose))                       # CT tissue depth is measured to bone: the nose's cartilages do not count for the depth sites
hull = []
for b in hb:
    ii = vid_b[b]
    try: hull.append(ii[np.unique(ConvexHull(V0[ii]).vertices)] if len(ii) >= 8 else ii)
    except Exception: hull.append(ii)
hull = np.concatenate(hull); BH = Bmat(V0[hull], OB[hull]); BALL = Bmat(V0[bidx], OB[bidx])
MAT = np.load(os.path.join(MD, 'contacts', 'material.npz')); bare = (MAT['material'][vid] == [str(m) for m in MAT['materials']].index('flesh'))
TAU_MIN = float(json.load(open(os.path.join(CT, 'summary.json'))).get('thinnest_p5_mm', 0)) / 1000 * s0   # world metres
def phi(Xw, grad=False):
    c = ((Xw - glo) / hst).T; v = ndimage.map_coordinates(sdf, c, order=1, mode='nearest')
    if not grad: return v
    g = np.stack([ndimage.map_coordinates(np.gradient(sdf, hst, axis=a), c, order=1, mode='nearest') for a in range(3)], 1) if not hasattr(phi, 'G') else np.stack([ndimage.map_coordinates(phi.G[a], c, order=1, mode='nearest') for a in range(3)], 1)
    return v, g
phi.G = [np.gradient(sdf, hst, axis=a) for a in range(3)]
def nearest_on(P, Q):
    a, b = Q[:-1], Q[1:]; d = b - a; u = np.clip(((P[:, None] - a) * d).sum(2) / ((d * d).sum(1) + 1e-30), 0, 1); C = a + u[..., None] * d; j = np.linalg.norm(P[:, None] - C, axis=2).argmin(1); i = np.arange(len(P)); return C[i, j], j, u[i, j]
def checks(z):
    Dall = V0[bidx] + (BALL @ z).reshape(-1, 3); Dw = to_world(Dall); tb = cKDTree(Dall[BONE_]); out = {'lines_mm': {}, 'sites_mm': {}}
    for n, Ln in LINES.items(): Dl = Ln['X'] + (Ln['B'] @ z).reshape(-1, 3); out['lines_mm'][n] = 1000 * s0 * float(np.median(np.linalg.norm(Dl - nearest_on(Dl, Ln['tgt'])[0], axis=1)))
    for k, S_ in SITE.items(): out['sites_mm'][k] = {'depth': 1000 * s0 * float(tb.query(S_['P'])[0]), 'ct': 1000 * s0 * S_['tau'], 'ct_spread': 1000 * s0 * S_['sig']}
    ph = phi(Dw); out['outside_share'] = float((ph < 0).mean()); out['outside_worst_mm'] = 1000 * float(max(0, -ph.min()))
    gaps = np.concatenate([np.linalg.norm(((Bmat(P, a) - Bmat(P, b)) @ z).reshape(-1, 3), axis=1) for a, b, P in seams]); out['seam_gap_mm'] = {'median': 1000 * s0 * float(np.median(gaps)), 'max': 1000 * s0 * float(gaps.max())}
    out['jaw_joint_gap_mm'] = [1000 * s0 * float(np.linalg.norm((Bmat(P, a) - Bmat(P, b)) @ z)) for a, b, P in hinge]
    A = z[:NQ].reshape(M, 12)[:, :9].reshape(M, 3, 3) + np.eye(3); sv = np.linalg.svd(A, compute_uv=False); out['stretch'] = {'median_abs_pct': 100 * float(np.median(np.abs(np.log(sv)))), 'max_abs_pct': 100 * float(np.abs(np.log(sv)).max())}
    out['size_pct'] = 100 * float(z[NQ + 6]); jm = int(np.abs(np.log(sv)).max(1).argmax()); out['stretch']['where'] = names[GB[jm]]; om = np.linalg.norm(np.stack([A[:, 2, 1] - A[:, 1, 2], A[:, 0, 2] - A[:, 2, 0], A[:, 1, 0] - A[:, 0, 1]], 1) / 2, axis=1); out['turn_deg'] = {'median': float(np.degrees(np.median(om))), 'max': float(np.degrees(om.max())), 'where': names[GB[int(om.argmax())]]}
    jn = np.concatenate([nodes_of[b] for b in hb if jaw(b)]); out['jaw_move_mm'] = 1000 * s0 * float(np.median(np.linalg.norm(z[:NQ].reshape(M, 12)[jn, 9:], axis=1))); out['pose'] = {'turn_deg': np.degrees(z[NQ:NQ + 3]).tolist(), 'shift_mm': (1000 * s0 * z[NQ + 3:NQ + 6]).tolist()}
    return out
def show(lab, c):
    log('%s: bone curve to her line (mm): %s; outside %.2f%% (worst %.1f mm); seams median %.2f max %.2f mm; jaw joints %s mm; stretch median %.1f%% max %.1f%% (%s); largest turn %.1f deg (%s); jaw moved %.1f mm; whole skull turned %s deg, shifted %s mm, size %+.1f%%' % (lab, ', '.join('%s %.1f' % kv for kv in c['lines_mm'].items()),
        100 * c['outside_share'], c['outside_worst_mm'], c['seam_gap_mm']['median'], c['seam_gap_mm']['max'], [round(g, 2) for g in c['jaw_joint_gap_mm']], c['stretch']['median_abs_pct'], c['stretch']['max_abs_pct'], c['stretch']['where'], c['turn_deg']['max'], c['turn_deg']['where'], c['jaw_move_mm'], np.round(c['pose']['turn_deg'], 1).tolist(), np.round(c['pose']['shift_mm'][:3], 1).tolist(), c['size_pct']))
    log('   depth to bone at CT sites, hers / CT median +- spread (mm): ' + '; '.join('%s %.1f / %.1f +- %.1f' % (k.split(' (')[0], v['depth'], v['ct'], v['ct_spread']) for k, v in c['sites_mm'].items()))
z = np.zeros(NU); rep = {'stage1': checks(z)}; show('stage 1', rep['stage1']); REST_W = svstack(RW).tocsr(); REST_R = np.concatenate(RR); hist = []
# --- per-part loosening. The stretch limits above are what real skulls show; her face is outside that range, so some parts
# stop short of their targets while others are already there or already against the surface. Rule: after the fit settles,
# every node whose own targets are still unmet (a line farther than the detector's noise, a CT depth off by more than its
# spread) gets its stiffness and its stretch cap doubled, with its mirror partner; nodes that have met their standard keep
# theirs; a node stops being loosened once loosening no longer brings it closer. The inside rule and the seams never loosen.
from scipy.sparse import diags
ST = json.load(open(os.path.join(AH, 'selftest.json')))['similar']; NOISE = ST['detected_vs_true_landmark_mm']['median'] / 1000 / ST['scale_true_vs_found'][0]   # detector noise, atlas units
SCL = np.ones(M); SCL[np.isin(GB, nose)] = NOSE_X; log('nose cartilage: %.1f times as free to stretch as bone (nose proportions vary %.1f%% between people, skulls %.1f%%)' % (NOSE_X, 100 * NOSE_X * SIG_S, 100 * SIG_S)); node_of_rows = lambda Bm: [np.unique(Bm[3 * i:3 * i + 3].indices[Bm[3 * i:3 * i + 3].indices < NQ] // 12) for i in range(Bm.shape[0] // 3)]
for Ln in LINES.values(): Ln['nodes'] = node_of_rows(Ln['B'])
def unmet(z):
    """per node: how far its own targets are from their standard, in units of that standard (<= 1: met)"""
    vals = [[] for _ in range(M)]; Dall = V0[bidx] + (BALL @ z).reshape(-1, 3); tb = cKDTree(Dall[BONE_]); sv_ = np.zeros(M)
    for Ln in LINES.values():
        Dl = Ln['X'] + (Ln['B'] @ z).reshape(-1, 3); d = np.linalg.norm(Dl - nearest_on(Dl, Ln['tgt'])[0], axis=1) / NOISE
        for i, nd in enumerate(Ln['nodes']):
            for j in nd: vals[j].append(d[i])
    u = np.array([np.median(v) if v else 0.0 for v in vals]); AREA_NODES.clear()
    for k_, S_ in SITE.items():
        d, i = tb.query(S_['P']); xv = bidx[BONE_][i]; zz = abs(d - S_['tau']) / S_['sig']
        for j in [q for q in np.unique(Bmat(V0[xv][None], OB[xv]).indices // 12) if q < M]:
            sv_[j] = max(sv_[j], zz)
            if k_.startswith('philtrum area'): AREA_NODES.add(int(j))
    return np.maximum(u, sv_)
prev_u = np.full(M, np.inf); frozen = np.zeros(M, bool); stages = []
AREA_NODES = set()   # the nodes that carry the bone under the philtrum area (Jev's box): her face asks more of this bone than any real skull gives, and it is to follow -
# these nodes go on being loosened for as long as their depth is unmet, with no stop for small gains (their mirror partners with them)
for stage in range(0 if os.environ.get('S2_HOLD_WHOLE') else -1, 8):                                          # stage -1: the skull as a whole first (pose and size, no bending); then it bends
    REG_W = svstack([diags(1 / np.maximum(SCL[PAIR[:, 0]], SCL[PAIR[:, 1]])) @ STIFF, diags(1 / (SIG_S * SCL[SH_NODE])) @ SHAPE, REST_W]).tocsr(); REG_R = np.r_[np.zeros(STIFF.shape[0] + SHAPE.shape[0]), REST_R]
    for rnd in range(12):
        Dall = V0[bidx] + (BALL @ z).reshape(-1, 3); tb = cKDTree(Dall[BONE_]); DW, DR, SG, SH = [], [], [], []
        for n, Ln in LINES.items():                                 # bone curve <-> her line at depth, both ways, sliding
            sgl = SIG_L * (NOSE_X if stage < 0 and n in WHOLE else 1.0)    # while the skull only moves as a whole, a nose line may miss by as much more as noses vary
            Dl = Ln['X'] + (Ln['B'] @ z).reshape(-1, 3); c, _, _ = nearest_on(Dl, Ln['tgt']); DW.append(Ln['B'] / sgl); DR.append((c - Ln['X']).ravel() / sgl)
            _, j, u = nearest_on(Ln['tgt'], Dl); Bi = Ln['B']; n3 = len(Ln['tgt']); r_ = np.arange(3 * n3); a_ = (3 * j[:, None] + np.arange(3)).ravel(); b_ = a_ + 3
            Wm = coo_matrix((np.r_[np.repeat(1 - u, 3), np.repeat(u, 3)], (np.r_[r_, r_], np.r_[a_, b_])), shape=(3 * n3, Bi.shape[0])).tocsr(); Xi = (Wm @ Ln['X'].ravel())
            DW.append((Wm @ Bi) / sgl); DR.append((Ln['tgt'].ravel() - Xi) / sgl)
        for k, S_ in SITE.items():                                  # depth at the CT sites: along the line from the skin point to its nearest bone point
            d, i = tb.query(S_['P']); xv = bidx[BONE_][i]; nh = (Dall[BONE_][i] - S_['P']) / (d + 1e-12); Bi = Bmat(V0[xv][None], OB[xv]); DW.append(coo_matrix(nh[None] @ Bi.toarray()).tocsr() / S_['sig']); DR.append(np.array([S_['tau'] - nh @ (V0[xv] - S_['P'])]) / S_['sig'])
            if S_['sig'] < SIG_MED: SG.append(coo_matrix(nh[None] @ Bi.toarray()).tocsr()); SH.append(S_['tau'] - S_['sig'] - nh @ (V0[xv] - S_['P']))   # hard, at the steady sites only: depth >= CT median - one spread
            if k.startswith('philtrum area'): SG.append(coo_matrix(-nh[None] @ Bi.toarray()).tocsr()); SH.append(-(S_['tau'] + S_['sig']) + nh @ (V0[xv] - S_['P']))   # hard, under the philtrum area: the bone follows her face - depth <= CT median + one spread
        SGm = svstack(SG).tocsr() if SG else coo_matrix((0, NU)).tocsr(); SHv = np.array(SH)
        Dh = V0[hull] + (BH @ z).reshape(-1, 3); ph, gw = phi(to_world(Dh), True); ga = gw @ (s0 * R0)                                       # inside: phi + grad . step >= 0
        tau = np.where(bare[ht.query(to_world(Dh))[1]], TAU_MIN, 0.0); ph = ph - tau                                                        # under bare skin: at least the thinnest measured tissue
        act = ph < 3 * SIG_L * s0; ii = np.flatnonzero(act); Gm = coo_matrix((ga[ii].ravel(), (np.repeat(np.arange(len(ii)), 3), (3 * ii[:, None] + np.arange(3)).ravel())), shape=(len(ii), 3 * len(hull))).tocsr() @ BH
        for E_ in EYE:                                              # eye centre carried by the socket bones = her eye centre (across and up-down)
            Bm = sum(Bmat(V0[v][None], OB[v]) for v in E_['xv']) / len(E_['xv']); DW.append(Bm[[0, 2]] / NOISE); DR.append((E_['eE'] - E_['eA'])[[0, 2]] / NOISE)
        Aw = svstack([REG_W] + DW + [BH / (SIG_L * 2)]).tocsr(); br = np.concatenate([REG_R] + DR + [(BH @ z) / (SIG_L * 2)])
        zv = cp.Variable(NU); sl = cp.Variable(len(ii), nonneg=True); s2 = cp.Variable(len(SHv), nonneg=True)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(Aw @ zv - br) + (cp.sum(sl) + cp.sum(s2)) / (hst / s0) * 1e3), [ph[ii] + Gm @ (zv - z) + sl >= 0, SGm @ zv + s2 >= SHv, cp.abs(SHAPE @ zv) <= 3 * SIG_S * SCL[SH_NODE], zv[NQ + 1] == 0, zv[NQ + 2] == 0, zv[NQ + 3] == 0] + ([zv[:NQ] == 0] if stage < 0 else []) + ([zv[NQ:] == 0] if os.environ.get('S2_HOLD_WHOLE') else []))   # S2_HOLD_WHOLE: a test - the skull may only bend, not move or change size as a whole   # stretch cap: three spreads, times the node's loosening
        prob.solve(solver=cp.CLARABEL)
        if zv.value is None: log('stage %d round %d: solver gave no answer (%s)' % (stage, rnd, prob.status)); break
        zn = zv.value; move = float(np.linalg.norm(((BH @ (zn - z)).reshape(-1, 3)), axis=1).max()) * s0; z = zn
        if move < hst / 10: break
    if stage < 0: c = checks(z); show('whole skull first (pose and size only)', c); rep['whole_first'] = c; continue
    c = checks(z); u = unmet(z); newly = np.isfinite(prev_u) & ~frozen & (prev_u - u < 1)   # a gain smaller than the standard itself (the detector's noise, one CT spread) is no gain:
    for a_, b_ in MIR:
        if a_ in AREA_NODES or b_ in AREA_NODES: AREA_NODES.update((a_, b_))
    newly[list(AREA_NODES)] = False
    for a_, b_ in MIR: newly[a_] = newly[b_] = (newly[a_] or newly[b_]) and not (frozen[a_] or frozen[b_])   # that node's last loosening is taken back and it is left alone
    SCL[newly] /= 2; frozen |= newly; lo = (u > 1) & ~frozen
    for a_, b_ in MIR: lo[a_] = lo[b_] = (lo[a_] or lo[b_]) and not (frozen[a_] or frozen[b_])
    if AREA_NODES: log('   philtrum bone nodes: %s, loosened x%s, unmet %s' % (sorted({names[GB[j]] for j in AREA_NODES}), sorted({float(SCL[j]) for j in AREA_NODES}), np.round([u[j] for j in sorted(AREA_NODES)], 1).tolist()))
    stages.append({'stage': stage, 'rounds': rnd + 1, 'unmet_nodes': int((u > 1).sum()), 'loosened_now': int(lo.sum()), 'stopped_no_gain': int(frozen.sum()), 'lines_mm': c['lines_mm'], 'outside_worst_mm': c['outside_worst_mm']})
    log('stage %d: %d rounds; lines %s mm; nodes with unmet targets %d of %d; loosening %d (x2), stopped for no gain %d; outside worst %.2f mm; stretch max %.1f%% (%s) (%.0f s)' % (stage, rnd + 1, ', '.join('%.1f' % v for v in c['lines_mm'].values()),
        (u > 1).sum(), M, lo.sum(), frozen.sum(), c['outside_worst_mm'], c['stretch']['max_abs_pct'], c['stretch']['where'], time.time() - t00))
    if not lo.any() and not newly.any(): break
    prev_u[lo] = u[lo]; SCL[lo] *= 2
u = unmet(z); Dh = V0[hull] + (BH @ z).reshape(-1, 3); marg = phi(to_world(Dh)) - np.where(bare[ht.query(to_world(Dh))[1]], TAU_MIN, 0.0); touch = {b for b in np.unique(OB[hull][marg < hst])}
by_bone = {}
for b in hb:
    nd = nodes_of[b]
    if SCL[nd].max() > 1 or u[nd].max() > 1: by_bone[names[b]] = {'loosened_x': float(SCL[nd].max()), 'unmet': float(u[nd].max()), 'at_surface': bool(b in touch)}
log('per bone (loosening factor, how far from its standard, touching the surface): ' + '; '.join('%s x%.0f %.1f %s' % (k, v['loosened_x'], v['unmet'], 'touch' if v['at_surface'] else 'free') for k, v in sorted(by_bone.items(), key=lambda kv: -kv[1]['unmet'])[:14]))
rep['stage2'] = checks(z); rep['eyes_mm'] = [dict(bones=E_['bones'], off_stage1=(1000 * s0 * (E_['eE'] - E_['eA'])[[0, 2]]).tolist(), off_stage2=(1000 * s0 * ((E_['eE'] - E_['eA']) - (sum(Bmat(V0[v][None], OB[v]) for v in E_['xv']) / len(E_['xv']) @ z))[[0, 2]]).tolist()) for E_ in EYE]; log('eyes after stage 2 (mm, across / up): ' + ', '.join('%.1f / %.1f' % tuple(e_['off_stage2']) for e_ in rep['eyes_mm'])); rep['stages'] = stages; rep['bones'] = by_bone; rep['detector_noise_mm'] = 1000 * s0 * NOISE; rep['nodes'] = int(M); rep['scales'] = {'strain': SIG_S, 'seam_mm': 1000 * s0 * SIG_SEAM, 'line_mm': 1000 * s0 * SIG_L}; show('stage 2', rep['stage2'])
Vout = K['V_rigid'].astype(np.float64).copy(); V1 = Vout.copy(); V1[bidx] = to_world(V0[bidx]); Vout[bidx] = to_world(V0[bidx] + (BALL @ z).reshape(-1, 3))
for b in hc:                                                        # cartilage rides with the bone nearest to it (it is fitted to her nose in a later step)
    ii = vid_b[b]; nb = OB[bidx[bt.query(V0[ii])[1]]]; V1[ii] = to_world(V0[ii]); Vout[ii] = to_world(V0[ii] + (Bmat(V0[ii], nb) @ z).reshape(-1, 3))
cm = np.isin(OB, hc)
np.savez(os.path.join(FD, 'skull_stage2.npz'), V=Vout.astype(np.float32), V_stage1=V1.astype(np.float32), V_fit=K['V_rigid'], F_head=FB[hm[FB].all(1)], F_cartilage=FB[cm[FB].all(1)], z=z, nodes=G, node_bone=GB, points_model=XE,
         **{'line_model_' + n.replace(' ', '_'): ZL_['line_model_' + n.replace(' ', '_')] for n in RUNS}, **{'line_target_' + n.replace(' ', '_'): to_world(L['tgt']) for n, L in LINES.items()},
         **{'line_bone_' + n.replace(' ', '_'): to_world(L['X'] + (L['B'] @ z).reshape(-1, 3)) for n, L in LINES.items()})
json.dump(rep, open(os.path.join(FD, 'skull_stage2.json'), 'w'), indent=1); log('done (%.0f s)' % (time.time() - t00))
