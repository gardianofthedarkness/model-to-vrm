"""T0a thigh measurement, runs ON a private Hugging Face CPU Job.  Numbers only: no images are written.  args: <repo id>

Selection (from census/label_extents.csv, the exact per-label census):
  femur label present, not touching any edge of the volume, and long: the femur lengths are bimodal (label fragments at the
  hip vs whole bones); the split is Otsu over all present femur lengths.  Adults only (meta.csv age >= 18).
Frame: femur axis = first principal axis of the femur label voxels (mm, world frame); axis points proximal -> distal;
  fraction 0 = most proximal label point, 1 = most distal.
Planes: the shaft is the contiguous run of the cross-section-area profile below its Otsu split that touches neither end of
  the profile (runs at the ends are the tapering tips of the label) and holds the smallest area among such runs.
  The comparable region is [median start, median end] over the selected femurs (fractions); the number of planes is
  round(median shaft length / median shaft diameter), with diameter from the minimum area (planes about one diameter apart),
  placed at the centres of equal sub-intervals.
Rays: 72 even angles (the project's polar convention, za_slices.py), theta = 0 towards anterior, increasing towards lateral
  (mirrored for the left leg), from the femur-label centroid in the plane, sampled every half voxel (Nyquist) with linear
  interpolation.
Per ray: bone boundary from the label (last label sample), bone intensity edge (half-maximum between the cortical peak
  and the lean level: the usual FWHM edge convention), skin (first body->air crossing outward; body/air split = Otsu of the
  plane's histogram), subcutaneous fat run, deep-fascia stand-in.  HU conventions (cited, not tuned): adipose -190..-30
  (Kvist 1988; Yoshizumi 1999, Radiology 211:283); skeletal muscle -29..150 (Mitsiopoulos 1998, J Appl Physiol 85:115).
  Subcutaneous fat starts at the outermost fat run of at least two samples (one sample is partial volume at the skin).
  Fascia stand-in = outer end of the outermost lean run at least two samples long (shorter lean runs inside fat are septa
  below the resolution) beneath the subcutaneous fat; clear when the HU step across it exceeds twice the pooled SD of the
  two sides (two-sided 95 %) and no fat run between it and the bone is as long as the subcutaneous fat run.
Flags: ray reaches the volume edge before air (skin cut); skin radius beyond the Tukey upper fence (Q3 + 1.5 IQR) of the
  plane (touching leg, arm, table or blanket); other bone-intensity component in the thigh's body component (arm or other
  leg); the other femur label inside the thigh's body component (legs touching); samples at the 12-bit CT ceiling
  (>= 3071 HU, metal); study type with contrast (angiography).
Repeatability (R_p): 20 re-extractions, each with the femur label eroded, dilated or kept (one voxel, the label's own
  resolution) and its voxels bootstrap-resampled, then axis, length, planes and rays recomputed; spread = SD over the
  re-extractions per ray, summarised per quantity.
"""
# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "nibabel"]
# ///
import os, sys, io, csv, json, gzip, time, zipfile, collections, numpy as np
from multiprocessing import Pool
from huggingface_hub import HfApi, hf_hub_download
api = HfApi(); t0 = time.time(); repo = sys.argv[1]
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
def put(data, name): api.upload_file(path_or_fileobj=data.encode() if isinstance(data, str) else data, path_in_repo=name, repo_id=repo, repo_type='dataset')
def otsu(v):
    v = np.asarray(v, float); h, e = np.histogram(v, 256); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); m = np.cumsum(h * c)
    with np.errstate(divide='ignore', invalid='ignore'): s = (m[-1] * w / w[-1] - m) ** 2 / (w * (w[-1] - w))
    return float(c[np.nanargmax(s[:-1])])
FAT, LEAN, CEIL, NRAY, NREP = (-190, -30), (-29, 150), 3071, 72, 20

_Z = None
def _init(zp):
    global _Z; _Z = zipfile.ZipFile(zp)
def load(name):
    import nibabel as nib
    img = nib.Nifti1Image.from_bytes(gzip.decompress(_Z.read(name))); return np.asanyarray(img.dataobj), img.affine

def femur_frame(pts):
    """pts: femur voxel centres in mm (N,3).  Returns centroid, unit axis (proximal->distal), s range."""
    c = pts.mean(0); u = np.linalg.svd(pts - c, full_matrices=False)[2][0]
    if u[2] > 0: u = -u                                   # world +z is superior; distal is -z
    s = (pts - c) @ u; return c, u, s.min(), s.max()

def shaft_profile(pts, c, u, s0, s1, vox):
    s = (pts - c) @ u; nb = max(8, int((s1 - s0) / vox)); h, e = np.histogram(s, nb, (s0, s1))
    area = h * vox ** 3 / ((s1 - s0) / nb)                    # mm^2 per bin
    t = otsu(area); low = area < t
    # the low-area runs that touch either end of the profile are the tapering tips of the label (head, condyles), not the
    # shaft: the shaft is the low run that touches neither end; among such runs, the one holding the smallest area
    tr = np.flatnonzero(np.diff(np.r_[0, low.astype(int), 0])); runs = [(int(a_), int(b_) - 1) for a_, b_ in zip(tr[0::2], tr[1::2])]
    inner = [(a_, b_) for a_, b_ in runs if a_ > 0 and b_ < nb - 1]
    if not inner: return np.nan, np.nan, np.nan
    a, b = min(inner, key=lambda r_: area[r_[0]:r_[1] + 1].min()); k = a + int(np.argmin(area[a:b + 1]))
    return a / nb, (b + 1) / nb, float(area[k])

def rays_in_plane(CT, inv, FL, other, BONEALL, c, u, side, frac, s0, s1, vox):
    """Measure the 72 rays of one plane.  Returns per-ray dict list and plane flags."""
    from scipy.ndimage import map_coordinates, label as cc
    p0 = c + u * (s0 + frac * (s1 - s0))
    ant = np.array([0, 1, 0.]) - u * u[1]; ant /= np.linalg.norm(ant)
    lat = np.cross(u, ant); lat /= np.linalg.norm(lat)
    if (lat[0] > 0) == (side == 'right'): lat = -lat          # +x is the patient's left in RAS: lateral of the right leg is -x
    step = vox / 2; R = 0.5 * float(np.linalg.norm(np.array(CT.shape) * vox))   # out to half the volume's diagonal
    def samp(vol, pts, order=1):
        ijk = (inv[:3, :3] @ pts.T + inv[:3, 3:4]); return map_coordinates(vol, ijk, order=order, mode='constant', cval=np.nan if order else 0)
    # plane image (for body component and Otsu split) on a grid in plane coordinates
    gs = vox; g = np.arange(-R, R + gs, gs); GX, GY = np.meshgrid(g, g, indexing='ij')   # plane image at the voxel size
    P = p0 + GX[..., None] * ant + GY[..., None] * lat
    img = samp(CT, P.reshape(-1, 3)).reshape(GX.shape); inside = np.isfinite(img)
    if not inside.any(): return None, dict(empty=True)
    t_air = otsu(np.clip(img[inside], -1024, 200)); body = inside & (img > t_air)
    fl = samp(FL, P.reshape(-1, 3), 0).reshape(GX.shape) > 0
    if not fl.any(): return None, dict(no_femur=True)
    ci = np.argwhere(fl).mean(0); cpt = p0 + (g[0] + ci[0] * gs) * ant + (g[0] + ci[1] * gs) * lat
    comp, _ = cc(body); lab = comp[int(round(ci[0])), int(round(ci[1]))]
    thigh = comp == lab if lab else body
    oth = samp(other, P.reshape(-1, 3), 0).reshape(GX.shape) > 0 if other is not None else np.zeros_like(fl)
    t_bone = otsu(np.clip(img[body], -200, 2000))
    bone_cc, nb = cc(thigh & (img > t_bone) & ~fl)
    extra_bone = [i for i in range(1, nb + 1) if (bone_cc == i).sum() * gs ** 2 > np.pi * (vox) ** 2]   # larger than a one-voxel-radius disc
    flags = dict(t_air=round(t_air), t_bone=round(t_bone), legs_touching=bool((thigh & oth).any()), other_bone_in_thigh=len(extra_bone),
                 metal=bool(np.nanmax(img) >= CEIL))
    out = []
    rr = np.arange(0, R, step)
    for k in range(NRAY):
        th = 2 * np.pi * k / NRAY; d = np.cos(th) * ant + np.sin(th) * lat
        pts = cpt + rr[:, None] * d; hu = samp(CT, pts); f = samp(FL, pts, 0) > 0
        r = dict(ray=k)
        if not f.any(): out.append(r); continue
        ib = int(np.flatnonzero(f)[-1]); r['r_bone_label'] = rr[ib]
        # half-maximum cortical edge
        seg = hu[:ib + 3]; ip = int(np.nanargmax(seg)) if np.isfinite(seg).any() else ib
        after = hu[ib + 1:]; lean_lv = np.nanmedian(after[(after >= LEAN[0]) & (after <= LEAN[1])]) if np.any((after >= LEAN[0]) & (after <= LEAN[1])) else np.nan
        half = (hu[ip] + lean_lv) / 2 if np.isfinite(lean_lv) else np.nan
        j = ip
        while np.isfinite(half) and j + 1 < len(hu) and hu[j + 1] >= half: j += 1
        if np.isfinite(half) and j + 1 < len(hu) and np.isfinite(hu[j + 1]):
            r['r_bone_edge'] = rr[j] + step * (hu[j] - half) / max(hu[j] - hu[j + 1], 1e-6)
        # skin: first body->air crossing outward from the bone
        bodyray = np.isfinite(hu) & (hu > t_air); j = ib
        while j + 1 < len(hu) and bodyray[j + 1]: j += 1
        if j + 1 >= len(hu) or not np.isfinite(hu[j + 1]): r['skin_cut'] = True; out.append(r); continue
        is_ = j; r['r_skin'] = rr[is_] + step * (hu[is_] - t_air) / max(hu[is_] - hu[is_ + 1], 1e-6)
        # classes between bone and skin
        seg = hu[ib + 1: is_ + 1]; F = (seg >= FAT[0]) & (seg <= FAT[1]); L = (seg >= LEAN[0]) & (seg <= LEAN[1])
        n = len(seg)
        if n == 0: out.append(r); continue
        fat_idx = np.flatnonzero(F[1:] & F[:-1])                # fat runs of >= 2 samples (single samples are partial volume at the skin)
        if fat_idx.size == 0: r['no_fat'] = True; out.append(r); continue
        sf = int(fat_idx[-1]) + 1                               # outer end of the outermost fat run; dermis = samples outside it
        r['dermis'] = (n - 1 - sf) * step
        # walk inward through the subcutaneous layer to the first lean run of >= 2 samples
        i = sf
        while i - 1 >= 0 and not (L[i - 1] and i - 2 >= 0 and L[i - 2]): i -= 1
        if i - 1 < 0: r['fascia_missing'] = True; out.append(r); continue
        fi = i - 1                                              # last lean sample (outer end of the lean run)
        r['r_fascia'] = rr[ib + 1 + fi] + step / 2
        sub = slice(fi + 1, sf + 1); r['d_fat'] = float(F[sub].sum() * step)
        lean_side = seg[max(0, fi - 4): fi + 1]; fat_side = seg[fi + 1: fi + 6]
        sd = np.sqrt((np.nanvar(lean_side) + np.nanvar(fat_side)) / 2) if lean_side.size > 1 and fat_side.size > 1 else np.inf
        step_hu = np.nanmean(lean_side) - np.nanmean(fat_side)
        tr = np.flatnonzero(np.diff(np.r_[0, F[:fi + 1].astype(int), 0])); runs = (tr[1::2] - tr[0::2]) if tr.size else np.array([0])   # fat runs inside the lean compartment
        r['fascia_clear'] = bool(step_hu > 2 * sd and runs.max() * step < r['d_fat'])
        out.append(r)
    radii = np.array([o.get('r_skin', np.nan) for o in out], float)
    if np.isfinite(radii).sum() > 4:
        q1, q3 = np.nanpercentile(radii, [25, 75]); fence = q3 + 1.5 * (q3 - q1)
        for o in out:
            if o.get('r_skin', 0) > fence: o['skin_outlier'] = True
    for o in out:
        if 'r_skin' in o:
            o['d_SB'] = o['r_skin'] - o['r_bone_label']
            if 'r_fascia' in o: o['d_SF'] = o['r_skin'] - o['r_fascia']
    return out, flags

def measure_side(args):
    case, side, other_present, fracs, contrast = args
    from scipy.ndimage import binary_erosion, binary_dilation
    CT, aff = load('%s/ct.nii.gz' % case); CT = np.asarray(CT, np.float32); inv = np.linalg.inv(aff)
    FL, _ = load('%s/segmentations/femur_%s.nii.gz' % (case, side)); FL = (np.asarray(FL) > 0).astype(np.uint8)
    osd = 'left' if side == 'right' else 'right'
    OT = (np.asarray(load('%s/segmentations/femur_%s.nii.gz' % (case, osd))[0]) > 0).astype(np.uint8) if other_present else None
    vox = float(np.abs(np.linalg.det(aff[:3, :3])) ** (1 / 3))
    import zlib; rng = np.random.default_rng(zlib.crc32(('%s|%s' % (case, side)).encode()))
    res = []
    for rep in range(NREP + 1):
        M = FL
        if rep:
            op = rep % 3
            M = binary_erosion(FL).astype(np.uint8) if op == 1 else binary_dilation(FL).astype(np.uint8) if op == 2 else FL
        idx = np.argwhere(M)
        if rep: idx = idx[rng.integers(0, len(idx), len(idx))]
        pts = idx @ aff[:3, :3].T + aff[:3, 3]
        c, u, s0, s1 = femur_frame(pts)
        planes = []
        for pi, fr in enumerate(fracs):
            out, flags = rays_in_plane(CT, inv, M, OT, None, c, u, side, fr, s0, s1, vox)
            planes.append(dict(plane=pi, frac=fr, flags=flags, rays=out))
        res.append(dict(rep=rep, length=float(s1 - s0), planes=planes))
    return dict(case=case, side=side, contrast=contrast, vox=vox, reps=res)

def frame_only(args):
    case, side = args
    FL, aff = load('%s/segmentations/femur_%s.nii.gz' % (case, side)); idx = np.argwhere(np.asarray(FL) > 0)
    pts = idx @ aff[:3, :3].T + aff[:3, 3]; c, u, s0, s1 = femur_frame(pts)
    vox = float(np.abs(np.linalg.det(aff[:3, :3])) ** (1 / 3))
    a, b, amin = shaft_profile(pts, c, u, s0, s1, vox)
    return dict(case=case, side=side, length=float(s1 - s0), shaft0=a, shaft1=b, min_area=amin)

if __name__ == '__main__':
    ext = list(csv.DictReader(open(hf_hub_download(repo, 'census/label_extents.csv', repo_type='dataset'))))
    meta = {r['image_id']: r for r in csv.DictReader(open(hf_hub_download(repo, 'census/meta.csv', repo_type='dataset'), encoding='utf-8-sig'), delimiter=';')}
    fem = [r for r in ext if r['label'] in ('femur_left', 'femur_right') and r['present'] == '1']
    split = otsu([float(r['length_mm']) for r in fem])
    sel = [r for r in fem if float(r['length_mm']) > split and not any(r[k] == '1' for k in ('cut_inf', 'cut_sup', 'cut_side', 'cut_ap'))]
    minors = [r for r in sel if meta[r['case']]['age'] and float(meta[r['case']]['age']) < 18]
    sel = [r for r in sel if r not in minors]
    present = {(r['case'], r['label'][6:]) for r in fem}
    log('femur length split %.1f mm; %d whole femurs from %d people; %d minors excluded' % (split, len(sel), len({r['case'] for r in sel}), len(minors)))
    zp = hf_hub_download(repo, 'raw/Totalsegmentator_dataset_v201.zip', repo_type='dataset', local_dir='.'); log('zip ready')
    first = [i.filename for i in zipfile.ZipFile(zp).infolist() if i.filename.endswith('/ct.nii.gz')][0]
    pre = first[:-len('/ct.nii.gz')].rsplit('/', 1)[0] + '/' if first.count('/') > 1 else ''
    items = [(pre + r['case'], r['label'][6:]) for r in sel]
    with Pool(4, _init, (zp,)) as p: frames = p.map(frame_only, items)
    bad = [f for f in frames if not np.isfinite(f['shaft0'])]; frames = [f for f in frames if np.isfinite(f['shaft0'])]; log('shaft found in %d femurs, not in %d' % (len(frames), len(bad)))
    f0 = float(np.median([f['shaft0'] for f in frames])); f1 = float(np.median([f['shaft1'] for f in frames]))
    diam = float(np.median([2 * np.sqrt(f['min_area'] / np.pi) for f in frames])); L = float(np.median([f['length'] for f in frames]))
    npl = max(1, int(round((f1 - f0) * L / diam))); fracs = [f0 + (f1 - f0) * (i + 0.5) / npl for i in range(npl)]
    design = dict(length_split_mm=round(split, 1), shaft_fraction=[round(f0, 3), round(f1, 3)], median_femur_mm=round(L, 1),
                  median_shaft_diameter_mm=round(diam, 1), planes=npl, plane_fractions=[round(x, 3) for x in fracs], rays=NRAY, reps=NREP,
                  hu_fat=FAT, hu_lean=LEAN, minors_excluded=len(minors))
    log('DESIGN ' + json.dumps(design))
    args = [(pre + r['case'], r['label'][6:], (r['case'], 'left' if r['label'].endswith('right') else 'right') in present, fracs,
             'angio' in meta[r['case']]['study_type']) for r in sel]
    results = []
    with Pool(3, _init, (zp,), maxtasksperchild=2) as p:
        for k, res in enumerate(p.imap_unordered(measure_side, args)):
            results.append(res); log('measured %d / %d (%s %s)' % (k + 1, len(args), res['case'].split('/')[-1], res['side']))
    import pickle; put(gzip.compress(pickle.dumps(dict(design=design, results=results))), 't0a_v2/raw_measurements.pkl.gz'); log('raw measurements saved')
    # ---- per-ray table: base extraction + repeatability SD over re-extractions
    Q = ['r_bone_label', 'r_bone_edge', 'r_skin', 'r_fascia', 'd_SB', 'd_SF', 'd_fat', 'dermis']
    rows, plane_rows = [], []
    for res in results:
        base = res['reps'][0]; Lb = base['length']
        for pl in base['planes']:
            fl = pl['flags']; plane_rows.append(dict(case=res['case'].split('/')[-1], side=res['side'], plane=pl['plane'], frac=round(pl['frac'], 3), femur_mm=round(Lb, 1), contrast=res['contrast'], **fl))
            if not pl['rays']: continue
            for o in pl['rays']:
                row = dict(case=res['case'].split('/')[-1], side=res['side'], plane=pl['plane'], ray=o['ray'], femur_mm=round(Lb, 1))
                for q in Q:
                    v = o.get(q); row[q] = round(float(v), 2) if v is not None else ''
                    reps = [rp['planes'][pl['plane']]['rays'][o['ray']].get(q) for rp in res['reps'][1:] if rp['planes'][pl['plane']]['rays']]
                    reps = [x for x in reps if x is not None]
                    row[q + '_sd'] = round(float(np.std(reps, ddof=1)), 3) if len(reps) > 2 else ''
                for fk in ('skin_cut', 'skin_outlier', 'fascia_clear', 'fascia_missing', 'no_fat'): row[fk] = int(bool(o.get(fk)))
                rows.append(row)
    def csvtext(rs):
        keys = list(dict.fromkeys(k for r in rs for k in r)); b = io.StringIO(); w = csv.DictWriter(b, keys, restval=''); w.writeheader(); w.writerows(rs); return b.getvalue()
    put(gzip.compress(csvtext(rows).encode()), 't0a_v2/thigh_rays.csv.gz'); put(csvtext(plane_rows), 't0a_v2/thigh_planes.csv')
    # ---- usable thighs and summaries
    pb = collections.defaultdict(list)
    for r in plane_rows: pb[(r['case'], r['side'])].append(r)
    rb = collections.defaultdict(list)
    for r in rows: rb[(r['case'], r['side'])].append(r)
    usable = {}
    for key, pls in pb.items():
        rs = rb[key]; bad_plane = any(p.get('legs_touching') or p.get('other_bone_in_thigh') or p.get('metal') or p.get('empty') or p.get('no_femur') for p in pls)
        cut = any(r['skin_cut'] for r in rs); ok = [r for r in rs if not r['skin_cut'] and not r['skin_outlier'] and r['d_SB'] != '']
        usable[key] = dict(plane_problem=bad_plane, skin_cut=cut, valid_ray_share=round(len(ok) / max(1, len(rs)), 3),
                           fascia_clear_share=round(sum(r['fascia_clear'] for r in ok) / max(1, len(ok)), 3), usable=not bad_plane and not cut)
    people = collections.defaultdict(list)
    for (c, s), u_ in usable.items():
        if u_['usable']: people[c].append(s)
    summ = {}
    good = [r for r in rows if usable[(r['case'], r['side'])]['usable'] and not r['skin_cut'] and not r['skin_outlier']]
    for q in ['d_SB', 'd_SF', 'd_fat', 'dermis', 'r_skin', 'r_bone_label']:
        per = collections.defaultdict(list)
        for r in good:
            if r[q] != '': per[(r['case'], r['side'])].append(r[q])
        med = np.array([np.median(v) for v in per.values()]) if per else np.array([np.nan])
        prop = np.array([np.median(v) / float(next(rr['femur_mm'] for rr in good if (rr['case'], rr['side']) == k)) for k, v in per.items()]) if per else np.array([np.nan])
        sds = [r[q + '_sd'] for r in good if r.get(q + '_sd') not in ('', None)]
        summ[q] = dict(n_thighs=len(per), median_mm=round(float(np.median(med)), 2), iqr_mm=[round(float(x), 2) for x in np.percentile(med, [25, 75])],
                       median_prop=round(float(np.median(prop)), 4), iqr_prop=[round(float(x), 4) for x in np.percentile(prop, [25, 75])],
                       repeat_sd_median_mm=round(float(np.median(sds)), 3) if sds else None, repeat_sd_p90_mm=round(float(np.percentile(sds, 90)), 3) if sds else None)
    bone_gap = [r['r_bone_edge'] - r['r_bone_label'] for r in good if r['r_bone_edge'] != '' and r['r_bone_label'] != '']
    out = dict(design=design, femurs_measured=len(usable), people_measured=len({c for c, _ in usable}),
               usable_thighs=sum(u_['usable'] for u_ in usable.values()), people_usable_one_side=len(people),
               people_usable_both=sum(1 for v in people.values() if len(v) == 2),
               problems=dict(plane_problem=sum(u_['plane_problem'] for u_ in usable.values()), skin_cut=sum(u_['skin_cut'] for u_ in usable.values()),
                             legs_touching=sum(1 for k, pls in pb.items() if any(p.get('legs_touching') for p in pls)),
                             other_bone_in_thigh=sum(1 for k, pls in pb.items() if any(p.get('other_bone_in_thigh') for p in pls)),
                             metal=sum(1 for k, pls in pb.items() if any(p.get('metal') for p in pls)),
                             contrast_study=sum(1 for r in results if r['contrast'])),
               rays_valid_share_median=round(float(np.median([u_['valid_ray_share'] for u_ in usable.values()])), 3),
               fascia_clear_share_of_valid_rays=round(sum(r['fascia_clear'] for r in good) / max(1, len(good)), 3),
               fascia_missing_share=round(sum(r['fascia_missing'] for r in good) / max(1, len(good)), 3),
               bone_edge_minus_label_mm=dict(median=round(float(np.median(bone_gap)), 2), iqr=[round(float(x), 2) for x in np.percentile(bone_gap, [25, 75])]) if bone_gap else None,
               quantities=summ, per_thigh=usable and {'%s_%s' % k: v for k, v in usable.items()})
    put(json.dumps(out, indent=1), 't0a_v2/summary.json')
    log('SUMMARY ' + json.dumps({k: v for k, v in out.items() if k != 'per_thigh'}))
