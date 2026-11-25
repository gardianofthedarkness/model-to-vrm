"""E-PACK v0, left thigh: one conforming tet mesh with a region per bone, per muscle and for fat, from the warped atlas and
the elf's own skin (plan.md §4.1, pack_format.md).  Every size and cut comes from the data:
  region   the elf's left-leg skin piece (its own surface, flesh/islands.npz) inside a crop box: from the top of the left hip
           bone down to the middle of the left tibia (so the knee can bend inside the pack), and from the body midline
           (half-way between the left and right hip bones) outwards; the box is the skin's extent plus one fat element
  members  every left bone and muscle of the atlas that reaches into the box (fascia, septa, bursae, sheaths left out: they
           become shells in v1); clipped at the box
  sizes    per region, half the structure's mean thickness (volume / area, the rule of muscle_sim.py), so each is at least
           two tets thick; fat: half the median skin-to-structure depth; outside the body: the largest of these
  overlaps the atlas, warped into the elf, overlaps itself (thigh_overlap_survey.py: 97 pairs).  simwild meshes all inputs at
           once with a tag per input; then every muscle pair sharing tets is resolved to the nearer muscle (resolve_overlaps);
           then each tet takes one label: outside skin or box -> dropped; any bone -> that bone; a muscle -> that muscle;
           else fat.  A final simwild remeshing pass restores element quality with the labels fixed.
Writes pack_thigh_l.npz (pack_format.md) and pack_thigh_l.json (inputs, sizes, volumes before/after, timings).
  E:\\splat\\.venv-uipc\\Scripts\\python.exe epack_thigh.py <anatomy_dense dir> <model dir> <out dir> [threads]"""
import sys, os, re, json, time, hashlib, numpy as np, wildmeshing as wm
from scipy.spatial import cKDTree
FD, MD, OUT = sys.argv[1:4]; NT = int(sys.argv[4]) if len(sys.argv) > 4 else os.cpu_count()
os.makedirs(OUT, exist_ok=True); log = dict(inputs={}, steps={}); t00 = time.time()
def say(*a): print('EP', *a, flush=True)
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
from scipy.sparse import csr_matrix
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
meshes, inputs, tag_of, sizing = {}, [], {}, []
def add(tag, P, G, length):
    meshes[tag] = {'V': np.ascontiguousarray(P), 'cells': {'triangle': {'connectivity': np.ascontiguousarray(G, dtype=np.int64)}}}
    inputs.append(tag); sizing.append({'tags': tag, 'length': float(length)})
add('skin', SP, SG, h_fat); add('box', BX, BG, h_fat)
for k, i in enumerate(mem):
    tag = ('b%d' if system[i] == 'bone' else 'm%d') % k; tag_of[tag] = names[i]; add(tag, *S[i], thick[names[i]])
h_out = max(s['length'] for s in sizing)
sizing = [{'tags': '_', 'length': h_out}, {'tags': 'skin & box', 'length': h_fat}] + [s for s in sizing if s['tags'] not in ('skin', 'box')]
log['inputs'] = dict(leg_island=leg, box=dict(lo=lo.tolist(), hi=hi.tolist()), fat_element_m=h_fat, outside_element_m=h_out,
                     members={tag_of[t]: dict(tag=t, element_m=thick[tag_of[t]], source_volume_cm3=round(abs(vol_area(*S[nid[tag_of[t]]])[0]) * 1e6, 3)) for t in tag_of},
                     atlas_sha1=hashlib.sha1(open(os.path.join(FD, 'muscle_repair.npz'), 'rb').read(1 << 24)).hexdigest(), wildmeshing=getattr(wm, '__version__', '2.0.0'))
for t in sorted(tag_of, key=lambda t: thick[tag_of[t]])[:8]: say('  finest: %-40s element %.2f mm' % (tag_of[t], 1e3 * thick[tag_of[t]]))
if os.environ.get('DRY') == '1':
    for t in tag_of: say('  %-44s %s element %.2f mm' % (tag_of[t], t, 1e3 * thick[tag_of[t]]))
    say('outside element %.1f mm' % (1e3 * h_out)); sys.exit(0)
# 1 mesh everything at once
t0 = time.time()
out, rep = wm.simwild({'input': inputs, 'input_names': inputs, 'length': h_out, 'sizing_field': sizing, 'num_threads': NT, 'max_iterations': 80,
                       'log_file': os.path.join(OUT, 'simwild_1_mesh.log')}, meshes)
M = out['mesh']; log['steps']['mesh_s'] = time.time() - t0
lab = list(M['cells']['tetrahedron']['attributes']['tags']['labels']); val = np.asarray(M['cells']['tetrahedron']['attributes']['tags']['values'])
say('meshed in %.0f s: %d nodes, %d tets' % (log['steps']['mesh_s'], len(M['V']), len(val)))
# 2 muscle-muscle overlaps to the nearer muscle
mt = [t for t in lab if t.startswith('m')]; Mv = val[:, [lab.index(t) for t in mt]]
pairs = sorted({(mt[a], mt[b]) for row in np.unique(Mv[Mv.sum(1) > 1], axis=0) for a in np.flatnonzero(row) for b in np.flatnonzero(row) if a < b})
log['steps']['muscle_overlap_pairs'] = [[tag_of[a], tag_of[b]] for a, b in pairs]; say('%d overlapping muscle pairs' % len(pairs))
if pairs:
    t0 = time.time()
    out, rep = wm.simwild({'input': ['m'], 'operation': 'resolve_overlaps', 'resolve_overlaps_tags': [[a, b] for a, b in pairs], 'length': h_out,
                           'sizing_field': sizing, 'num_threads': NT, 'log_file': os.path.join(OUT, 'simwild_2_resolve.log')}, {'m': M})
    M = out['mesh']; log['steps']['resolve_s'] = time.time() - t0
    lab = list(M['cells']['tetrahedron']['attributes']['tags']['labels']); val = np.asarray(M['cells']['tetrahedron']['attributes']['tags']['values'])
# 3 one label per tet
col = {t: lab.index(t) for t in lab}; inside = (val[:, col['skin']] == 1) & (val[:, col['box']] == 1)
region_tags = [t for t in tag_of if t.startswith('b')] + [t for t in tag_of if t.startswith('m')]
final = np.full(len(val), -1)
for t in region_tags:                                                              # muscles (resolved, so at most one each)
    if t in col and t.startswith('m'): final[(val[:, col[t]] == 1) & inside & (final < 0)] = region_tags.index(t)
for t in region_tags:                                                              # bones override muscles
    if t in col and t.startswith('b'): final[(val[:, col[t]] == 1) & inside] = region_tags.index(t)
final[inside & (final < 0)] = len(region_tags)                                     # fat
outside_skin = {tag_of[t]: int(((val[:, col[t]] == 1) & (val[:, col['box']] == 1) & (val[:, col['skin']] == 0)).sum()) for t in region_tags if t in col}
log['steps']['tets_of_structures_outside_skin'] = {k: v for k, v in outside_skin.items() if v}
keep = final >= 0; X = M['V']; T = M['cells']['tetrahedron']['connectivity'][keep]; reg = final[keep]
used = np.unique(T); remap = np.full(len(X), -1); remap[used] = np.arange(len(used)); X = X[used]; T = remap[T]
rnames = [tag_of[t] for t in region_tags] + ['fat']
# 4 quality pass with the labels fixed
oh = np.zeros((len(T), len(rnames)), np.uint8); oh[np.arange(len(T)), reg] = 1
tagged = {'V': X, 'cells': {'tetrahedron': {'connectivity': T.astype(np.int64), 'attributes': {'tags': {'values': oh, 'labels': ['r%d' % i for i in range(len(rnames))]}}}}}
t0 = time.time()
out, rep = wm.simwild({'input': ['m'], 'operation': 'remeshing', 'length': h_out, 'num_threads': NT, 'max_iterations': 40,
                       'sizing_field': [{'tags': 'r%d' % i, 'length': (h_fat if n == 'fat' else thick[n])} for i, n in enumerate(rnames)],
                       'log_file': os.path.join(OUT, 'simwild_4_quality.log')}, {'m': tagged})
M = out['mesh']; log['steps']['quality_s'] = time.time() - t0
lab = list(M['cells']['tetrahedron']['attributes']['tags']['labels']); val = np.asarray(M['cells']['tetrahedron']['attributes']['tags']['values'])
X = M['V'].astype(np.float64); T = M['cells']['tetrahedron']['connectivity'].astype(np.int64)
rc = [lab.index('r%d' % i) if 'r%d' % i in lab else -1 for i in range(len(rnames))]
reg = np.full(len(T), -1);
for i, c in enumerate(rc):
    if c >= 0: reg[val[:, c] == 1] = i
keepT = reg >= 0; T, reg = T[keepT], reg[keepT]
D = np.stack([X[T[:, i]] - X[T[:, 0]] for i in (1, 2, 3)], 2); sv = np.linalg.det(D); T = np.where((sv < 0)[:, None], T[:, [0, 1, 3, 2]], T)
vols = {n: round(float(np.abs(sv[reg == i]).sum() / 6 * 1e6), 3) for i, n in enumerate(rnames)}
kind = ['bone' if system[nid[n]] == 'bone' else 'muscle' for n in rnames[:-1]] + ['fat']
src = {n: log['inputs']['members'][n]['source_volume_cm3'] for n in rnames[:-1]}
log['volumes_cm3'] = {n: dict(pack=vols[n], source=src.get(n)) for n in rnames}; log['nodes'] = int(len(X)); log['tets'] = int(len(T)); log['total_s'] = time.time() - t00
np.savez(os.path.join(OUT, 'pack_thigh_l.npz'), X=X, T=T, region=reg.astype(np.int32), region_names=np.array(rnames), region_kind=np.array(kind),
         interfaces=json.dumps([]), source=json.dumps(log['inputs']), source_volume_cm3=json.dumps(src))
json.dump(log, open(os.path.join(OUT, 'pack_thigh_l.json'), 'w'), indent=1, default=float)
say('pack: %d nodes, %d tets, %d regions, %.0f s total' % (len(X), len(T), len(rnames), log['total_s']))
for n in rnames: say('  %-42s %8.2f cm3 (atlas %s)' % (n, vols[n], src.get(n, '-')))
