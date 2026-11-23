"""T0a pelvis measurement, runs ON a private Hugging Face CPU Job.  Numbers only: no images are written.  args: <repo id> [limit]

Selection (census/label_extents.csv): both hip bones and the sacrum present and not touching any edge of the volume;
adults only (meta.csv age >= 18; unknown age is kept and counted).
Levels (the pelvis's own critical slices, per person, from the labels): axial planes (scanner axis = the body's long
  axis on the table) at five landmarks, top to bottom:
    crest     top of the hip bones (iliac crests)
    s1        top of the sacrum
    head_top  top of the femoral heads (mean of both sides)
    sac_end   bottom of the sacrum
    ischium   bottom of the hip bones (ischial tuberosities)
  plus the midpoint between each consecutive pair: 9 levels. Comparable across people because each is an anatomical
  landmark, not a distance. The tilt of the pelvis on the table is recorded (angle of the line from the sacrum's top to the
  pubic end of the hip bones against the table plane), not corrected.
Rays: 72 even angles in the axial plane from the centre of the body section (the patient's own section: the connected
  piece of body that holds the sacrum, holes filled so that gas in the bowel is inside the body); theta = 0 towards
  anterior, increasing towards the patient's left.
Per ray, from the skin inward:
  skin      the last body sample along the ray (body/air split = Otsu of the plane, as in the thigh job)
  dermis    from the skin to the outermost fat run of at least two samples
  fat       the subcutaneous fat run; adipose -190..-30 HU, lean -29..150 HU (cited conventions: Kvist 1988; Yoshizumi 1999,
            Radiology 211:283; Mitsiopoulos 1998, J Appl Physiol 85:115)
  fascia    stand-in: the outer end of the first lean run of at least two samples inside the subcutaneous fat; "clear" when the
            HU step exceeds twice the pooled SD of the two sides
  muscle    the first labelled pelvic muscle along the ray (gluteus maximus, medius, minimus, iliopsoas, autochthon): an
            independent check of the fascia stand-in where those labels exist
  bone      the first bone label along the ray (hip bones, sacrum, femur, L5, S1), and its name; a ray that meets no bone
            before the section's centre (the front of the abdomen) is recorded as "no bone", an observation in its own right
Rays crossing bone-bright tissue that no pelvic label covers, outside the body wall (between the skin and the fascia
stand-in), are dropped: an arm or a hand lying against the side. Such tissue inside the wall (contrast in the bowel,
calcified vessels, a lumbar vertebra) is kept and only counted.
Flags per plane: body touching the edge of the image (rays reaching the edge are dropped one by one, not the plane);
arms beside the pelvis (humerus label, or a bone-bright piece outside the hull of the labelled pelvic bones, larger than a
one-voxel-radius disc; pieces inside the hull are contrast, calcified vessels, a vertebra or the coccyx); metal (12-bit
ceiling); contrast study.
Repeatability: on the first NREP_CASES people, 10 re-extractions with every landmark moved by up to one voxel
(uniform, the labels' own resolution) and the labels eroded, dilated or kept; spread = SD over re-extractions per ray.
Outputs (private repo, numbers only): t0a_pelvis/rays.csv.gz, planes.csv, summary.json."""
# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel"]
# ///
import os, sys, io, csv, json, gzip, time, zipfile, collections, numpy as np
from multiprocessing import Pool
FAT, LEAN, CEIL, NRAY, NREP, NREP_CASES = (-190, -30), (-29, 150), 3071, 72, 10, 40
BONES = ('hip_left', 'hip_right', 'sacrum', 'femur_left', 'femur_right', 'vertebrae_L5', 'vertebrae_S1')
MUSCLES = ('gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left', 'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right',
           'iliopsoas_left', 'iliopsoas_right', 'autochthon_left', 'autochthon_right')
LEVELS = ('crest', 'crest-s1', 's1', 's1-head_top', 'head_top', 'head_top-sac_end', 'sac_end', 'sac_end-ischium', 'ischium')
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def otsu(v):
    v = np.asarray(v, float); h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    return float(c[np.nanargmax(s[:-1])])
_Z = None
def _init(zp):
    global _Z; _Z = zipfile.ZipFile(zp)
def load(name):
    import nibabel as nib
    img = nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(name))); return np.asanyarray(img.dataobj), img.affine

def landmarks(lab, aff):
    """z (mm, world, superior = larger z) of the five landmarks from the label volumes, and the pelvic tilt in degrees."""
    def zr(m):
        k = np.flatnonzero(np.asarray(m).any((0, 1)));
        if not len(k): return None
        z = (aff @ np.c_[np.zeros((2, 2)), k[[0, -1]], np.ones(2)].T)[2]; return float(z.min()), float(z.max())
    H = np.asarray(lab['hip_left']) | np.asarray(lab['hip_right']); S = np.asarray(lab['sacrum'])
    hz, sz = zr(H), zr(S)
    heads = []
    for side in ('femur_left', 'femur_right'):
        if side in lab and np.asarray(lab[side]).any(): heads.append(zr(lab[side])[1])
    if hz is None or sz is None or not heads: return None, None
    L = dict(crest=hz[1], s1=sz[1], head_top=float(np.mean(heads)), sac_end=sz[0], ischium=hz[0])
    # tilt: sacrum top centre against the most anterior point of the hip bones (the pubic end), in the sagittal plane
    st = np.argwhere(S); sc = (aff @ np.c_[st, np.ones(len(st))].T)[:3].T; top = sc[sc[:, 2] > np.percentile(sc[:, 2], 95)].mean(0)
    hp = np.argwhere(H); hc = (aff @ np.c_[hp, np.ones(len(hp))].T)[:3].T; pub = hc[hc[:, 1] >= np.percentile(hc[:, 1], 99)].mean(0)   # RAS: +y anterior
    tilt = float(np.degrees(np.arctan2(top[2] - pub[2], pub[1] - top[1])))
    return L, tilt

def level_z(L):
    k = ['crest', 's1', 'head_top', 'sac_end', 'ischium']; z = []
    for i, n in enumerate(k):
        z.append((n, L[n]))
        if i + 1 < len(k): z.append(('%s-%s' % (n, k[i + 1]), (L[n] + L[k[i + 1]]) / 2))
    return z

def measure_plane(CT, labs, aff, z):
    """72 rays in the axial plane at world height z.  Returns (rays, flags)."""
    from scipy.ndimage import label as cc, binary_fill_holes
    inv = np.linalg.inv(aff); kz = (inv @ np.array([0, 0, z, 1.]))[2]; k = int(round(kz))
    if k < 0 or k >= CT.shape[2]: return None, dict(outside=True)
    img = np.asarray(CT[:, :, k], np.float32); vox = float(np.sqrt(abs(np.linalg.det(aff[:2, :2]))))
    t_air = otsu(np.clip(img, -1024, 200)); body = img > t_air
    comp, n = cc(body); sl = {nm: np.asarray(m[:, :, k]) > 0 for nm, m in labs.items()}
    sac = sl.get('sacrum', np.zeros_like(body)); hips = sl.get('hip_left', np.zeros_like(body)) | sl.get('hip_right', np.zeros_like(body))
    seed = sac if sac.any() else hips
    if not seed.any():                                         # no pelvic bone in this plane: take the largest piece
        if n == 0: return None, dict(empty=True)
        lab_ = int(np.argmax(np.bincount(comp.ravel())[1:])) + 1
    else:
        ids = comp[seed]; ids = ids[ids > 0]
        if not len(ids): return None, dict(empty=True)
        lab_ = int(np.bincount(ids).argmax())
    pat = binary_fill_holes(comp == lab_)
    cut = bool(pat[0, :].any() or pat[-1, :].any() or pat[:, 0].any() or pat[:, -1].any())
    hi = img[pat & (img > LEAN[1])]                              # above the soft-tissue range: bone, contrast, metal
    t_bone = otsu(np.clip(hi, LEAN[1], CEIL)) if hi.size > 16 else float(LEAN[1])
    known = np.zeros_like(pat)
    for nm in BONES: known |= sl.get(nm, False)
    # bone-bright pieces that no pelvic label covers: inside the hull of the labelled pelvic bones they are contrast agent,
    # calcified vessels, a lumbar vertebra or the coccyx, which do not disturb the soft tissue; outside it they are an arm
    # or a hand lying beside the pelvis, which does
    bc, nb = cc(pat & (img > t_bone) & ~known); extra = 0; kp = np.argwhere(known)
    if nb and len(kp) > 3:
        from scipy.spatial import Delaunay
        try: hull = Delaunay(kp)
        except Exception: hull = None
        for i in range(1, nb + 1):
            m_ = bc == i
            if m_.sum() <= np.pi: continue                       # not larger than a one-voxel-radius disc
            if hull is None or hull.find_simplex(np.argwhere(m_).mean(0)[None])[0] < 0: extra += 1
    arms = bool(sl.get('humerus_left', np.zeros(1)).any() or sl.get('humerus_right', np.zeros(1)).any())
    flags = dict(t_air=round(t_air), t_bone=round(t_bone), skin_cut=cut, arms=arms, extra_bone=extra, metal=bool(img.max() >= CEIL))
    ij = np.argwhere(pat); c = ij.mean(0)
    # in-plane directions: RAS axes of the voxel grid; theta 0 anterior (+y), towards the patient's left (+x)
    ax_i = aff[:3, 0] / np.linalg.norm(aff[:3, 0]); ax_j = aff[:3, 1] / np.linalg.norm(aff[:3, 1])
    step = 0.5; R = float(np.linalg.norm(pat.shape))
    from scipy.ndimage import map_coordinates
    bone_map = np.zeros(pat.shape, np.int16)
    for i_, nm in enumerate(BONES, 1): bone_map[sl.get(nm, np.zeros_like(pat))] = i_
    mus = np.zeros_like(pat)
    for nm in MUSCLES: mus |= sl.get(nm, False)
    out = []
    for r_ in range(NRAY):
        th = 2 * np.pi * r_ / NRAY; dirw = np.array([np.sin(th), np.cos(th), 0.])        # world: x left, y anterior
        di, dj = dirw @ ax_i, dirw @ ax_j; d = np.array([di, dj]); d /= np.linalg.norm(d)
        t = np.arange(0, R, step); P = c[None] + t[:, None] * d[None]
        inb = (P[:, 0] >= 0) & (P[:, 0] <= pat.shape[0] - 1) & (P[:, 1] >= 0) & (P[:, 1] <= pat.shape[1] - 1); P = P[inb]; t = t[inb]
        hu = map_coordinates(img, P.T, order=1); pb = map_coordinates(pat.astype(np.float32), P.T, order=0) > 0
        bn = map_coordinates(bone_map, P.T, order=0).astype(int); mm = map_coordinates(mus.astype(np.float32), P.T, order=0) > 0
        rr = t * vox; o = dict(ray=r_)
        if not pb.any(): out.append(o); continue
        isk = int(np.flatnonzero(pb)[-1])
        if isk >= len(pb) - 1: o['skin_cut'] = True; out.append(o); continue
        o['r_skin'] = rr[isk]
        # inward from the skin
        seg = hu[:isk + 1][::-1]; F = (seg >= FAT[0]) & (seg <= FAT[1]); L = (seg >= LEAN[0]) & (seg <= LEAN[1])
        bseg = bn[:isk + 1][::-1]; mseg = mm[:isk + 1][::-1]
        hb = np.flatnonzero(bseg > 0)
        if len(hb): o['d_SB'] = hb[0] * step * vox; o['bone'] = BONES[bseg[hb[0]] - 1]
        else: o['no_bone'] = True
        hm = np.flatnonzero(mseg)
        if len(hm): o['d_SM'] = hm[0] * step * vox
        f2 = np.flatnonzero(F[1:] & F[:-1])
        if not len(f2): o['no_fat'] = True; out.append(o); continue
        fs = int(f2[0]); o['dermis'] = fs * step * vox                                  # outer end of the outermost fat run
        i = fs
        while i + 2 < len(seg) and not (L[i + 1] and L[i + 2]): i += 1
        if i + 2 >= len(seg): o['fascia_missing'] = True; out.append(o); continue
        fi = i + 1; o['d_SF'] = (fi - 0.5) * step * vox; o['d_fat'] = float(F[fs:fi].sum() * step * vox)
        if np.any((seg[:fi] > t_bone) & (bseg[:fi] == 0)): o['foreign_bone'] = True       # bone outside the body wall on this ray: an arm or a hand lying against the side
        a_ = seg[max(fs, fi - 5):fi]; b_ = seg[fi:fi + 5]
        sd = np.sqrt((np.var(a_) + np.var(b_)) / 2) if len(a_) > 1 and len(b_) > 1 else np.inf
        o['fascia_clear'] = bool(np.mean(b_) - np.mean(a_) > 2 * sd)
        out.append(o)
    radii = np.array([o_.get('r_skin', np.nan) for o_ in out])
    if np.isfinite(radii).sum() > 4:
        q1, q3 = np.nanpercentile(radii, [25, 75]); fence = q3 + 1.5 * (q3 - q1)
        for o_ in out:
            if o_.get('r_skin', 0) > fence: o_['skin_outlier'] = True
    return out, flags

def measure_case(args):
    case, nrep, seed = args
    from scipy.ndimage import binary_erosion, binary_dilation
    CT, aff = load('%s/ct.nii.gz' % case)
    labs0 = {}
    for nm in BONES + MUSCLES + ('humerus_left', 'humerus_right'):
        try: labs0[nm] = np.asarray(load('%s/segmentations/%s.nii.gz' % (case, nm))[0]) > 0
        except KeyError: pass
    L0, tilt = landmarks(labs0, aff)
    if L0 is None: return dict(case=case, failed='landmarks')
    vz = abs(aff[2, 2]); rng = np.random.default_rng(seed); reps = []
    for rep in range(nrep + 1):
        labs = labs0
        if rep:
            op = rep % 3
            labs = {k: (binary_erosion(v) if op == 1 else binary_dilation(v) if op == 2 else v) for k, v in labs0.items()}
        L = L0 if rep == 0 else {k: v + rng.uniform(-vz, vz) for k, v in L0.items()}
        planes = []
        for nm, z in level_z(L):
            rays, flags = measure_plane(CT, labs, aff, z); planes.append(dict(level=nm, z=z, flags=flags, rays=rays))
        reps.append(planes)
    return dict(case=case, tilt=tilt, landmarks=L0, vox=float(abs(np.linalg.det(aff[:3, :3])) ** (1 / 3)), reps=reps)

if __name__ == '__main__':
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); repo = sys.argv[1]; limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    def put(data, name): api.upload_file(path_or_fileobj=data.encode() if isinstance(data, str) else data, path_in_repo=name, repo_id=repo, repo_type='dataset')
    ext = list(csv.DictReader(open(hf_hub_download(repo, 'census/label_extents.csv', repo_type='dataset'))))
    meta = {r['image_id']: r for r in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    by = collections.defaultdict(dict)
    for r in ext: by[r['case']][r['label']] = r
    whole = lambda r: bool(r) and r['present'] == '1' and not any(r[k] == '1' for k in ('cut_inf', 'cut_sup', 'cut_side', 'cut_ap'))
    sel = sorted(c for c, d in by.items() if whole(d.get('hip_left')) and whole(d.get('hip_right')) and whole(d.get('sacrum')))
    age = lambda c: float(meta[c]['age']) if meta.get(c, {}).get('age') else None
    minors = [c for c in sel if age(c) is not None and age(c) < 18]; noage = [c for c in sel if age(c) is None]; sel = [c for c in sel if c not in minors]
    if limit: sel = sel[:limit]
    log('%d people with a whole pelvis (both hip bones and the sacrum away from the volume edges); %d minors excluded; %d of unknown age kept' % (len(sel), len(minors), len(noage)))
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); log('zip ready')
    first = [i.filename for i in zipfile.ZipFile(zp).infolist() if i.filename.endswith('/ct.nii.gz')][0]
    pre = first[:-len('/ct.nii.gz')].rsplit('/', 1)[0] + '/' if first.count('/') > 1 else ''
    import zlib
    args = [(pre + c, NREP if i < NREP_CASES else 0, zlib.crc32(c.encode())) for i, c in enumerate(sel)]
    res = []
    with Pool(4, _init, (zp,)) as p:
        for k, r in enumerate(p.imap_unordered(measure_case, args)):
            res.append(r)
            if (k + 1) % 20 == 0 or k + 1 == len(args): log('measured %d / %d' % (k + 1, len(args)))
    put(gzip.compress(__import__('pickle').dumps(res)), 't0a_pelvis/raw_measurements.pkl.gz'); log('raw measurements saved')
    # ---- tables
    Q = ('r_skin', 'dermis', 'd_fat', 'd_SF', 'd_SM', 'd_SB')
    rays_rows, plane_rows = [], []
    for r in res:
        if 'failed' in r: continue
        c = r['case'].rsplit('/', 1)[-1]; base = r['reps'][0]
        for pi, pl in enumerate(base):
            fl = pl['flags'] or {}; plane_rows.append(dict(case=c, level=pl['level'], z=round(pl['z'], 1), tilt=round(r['tilt'], 1), sex=meta.get(c, {}).get('gender', ''), age=meta.get(c, {}).get('age', ''),
                                                       contrast='angio' in meta.get(c, {}).get('study_type', ''), **{k: fl.get(k) for k in ('skin_cut', 'arms', 'extra_bone', 'metal', 'empty', 'outside')}))
            for o in pl['rays'] or []:
                row = dict(case=c, level=pl['level'], ray=o['ray'], **{q: o.get(q) for q in Q}, bone=o.get('bone'), no_bone=o.get('no_bone', False),
                           fascia_clear=o.get('fascia_clear'), skin_outlier=o.get('skin_outlier', False), skin_cut=o.get('skin_cut', False), foreign_bone=o.get('foreign_bone', False))
                if len(r['reps']) > 1:
                    for q in Q:
                        v = [rp[pi]['rays'][o['ray']].get(q) for rp in r['reps'][1:] if pi < len(rp) and rp[pi]['rays']]
                        v = [x for x in v if x is not None]; row[q + '_sd'] = float(np.std(v)) if len(v) > 2 else None
                rays_rows.append(row)
    def csvtext(rows):
        keys = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 999); b = io.StringIO(); w = csv.DictWriter(b, keys); w.writeheader(); w.writerows(rows); return b.getvalue()
    put(gzip.compress(csvtext(rays_rows).encode()), 't0a_pelvis/rays.csv.gz'); put(csvtext(plane_rows), 't0a_pelvis/planes.csv')
    # ---- summary: a plane counts when its section is whole (not cut), no arms beside it, no unlabelled bone, no metal
    okp = {(p['case'], p['level']) for p in plane_rows if not any(p.get(k) for k in ('arms', 'metal', 'empty', 'outside'))}   # a section cut by the image edge keeps its other rays: each ray that reaches the edge is dropped on its own
    good = [r for r in rays_rows if (r['case'], r['level']) in okp and not r['skin_cut'] and not r['skin_outlier'] and not r['foreign_bone']]
    sector = lambda ray: 'front' if ray < 9 or ray >= 63 else 'back' if 27 <= ray < 45 else 'side'
    summ = {}
    for lv in LEVELS:
        for sec in ('front', 'side', 'back'):
            rs = [r for r in good if r['level'] == lv and sector(r['ray']) == sec]; d = {}
            for q in Q:
                per = collections.defaultdict(list)
                for r in rs:
                    if r[q] is not None: per[r['case']].append(r[q])
                med = [np.median(v) for v in per.values()]
                sd = [r.get(q + '_sd') for r in rs if r.get(q + '_sd') is not None]
                d[q] = dict(people=len(med), median_mm=round(float(np.median(med)), 1) if med else None, iqr_mm=[round(float(x), 1) for x in np.percentile(med, [25, 75])] if med else None,
                            repeat_sd_median_mm=round(float(np.median(sd)), 2) if sd else None)
            d['fascia_clear_share'] = round(float(np.mean([bool(r['fascia_clear']) for r in rs if r['fascia_clear'] is not None])), 3) if rs else None
            d['no_bone_share'] = round(float(np.mean([r['no_bone'] for r in rs])), 3) if rs else None
            fm = [r for r in rs if r['d_SF'] is not None and r['d_SM'] is not None]
            d['fascia_minus_muscle_label_mm'] = round(float(np.median([r['d_SF'] - r['d_SM'] for r in fm])), 2) if fm else None
            summ['%s/%s' % (lv, sec)] = d
    people_by_level = {lv: len({p['case'] for p in plane_rows if p['level'] == lv and (p['case'], lv) in okp}) for lv in LEVELS}
    problems = {k: sum(1 for p in plane_rows if p.get(k)) for k in ('skin_cut', 'arms', 'metal')}; problems['extra_bone'] = sum(1 for p in plane_rows if p.get('extra_bone'))
    out = dict(people=len({r['case'] for r in res if 'failed' not in r}), failed=[r['case'] for r in res if 'failed' in r], minors_excluded=len(minors), unknown_age=len(noage),
               planes=len(plane_rows), planes_ok=len(okp), people_usable_by_level=people_by_level, plane_problems=problems,
               tilt_deg=dict(median=round(float(np.median([r['tilt'] for r in res if 'failed' not in r])), 1), iqr=[round(float(x), 1) for x in np.percentile([r['tilt'] for r in res if 'failed' not in r], [25, 75])]),
               layers=summ)
    put(json.dumps(out, indent=1), 't0a_pelvis/summary.json'); log('SUMMARY ' + json.dumps({k: v for k, v in out.items() if k != 'layers'}))
