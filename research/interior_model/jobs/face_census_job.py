# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "pynrrd"]
# ///
"""How fine must the face grid be? HaN-Seg's 42 head-and-neck CTs (CC BY-ND 4.0, private use; numbers only).
Per expert structure (30 organs at risk: mandible, eyes, lips, oral cavity, glands, optic nerves, ...): volume and
thickness (twice the mean distance from inside to the boundary, mm), and the share of its voxels that a token grid of
size s would leave as 'pure' (all voxels of one 2x2x2 sub-block, at labels of s/4) for s = 4, 6, 8, 12 mm.
Also the CT's own voxel size, the face soft tissue in front of bone (already in the census), and the head window's
size in tokens at each s. args: <repo of HaN-Seg>"""
import os, sys, io, json, time, zipfile, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
_Z = None
def _init(zp):
    global _Z; _Z = zipfile.ZipFile(zp)
def one(arg):
    import nrrd
    from scipy.ndimage import distance_transform_edt
    case, files = arg
    try:
        out = dict(case=case, structures={})
        ct_name = [f for f in files if f.endswith('_IMG_CT.nrrd')][0]
        data, hdr = nrrd.read(io.BytesIO(_Z.read(ct_name))) if False else (None, None)
        p = '/tmp/%s_ct.nrrd' % case; open(p, 'wb').write(_Z.read(ct_name)); hdr = nrrd.read_header(p); os.remove(p)
        sp = np.linalg.norm(np.array(hdr['space directions'], float), axis=1); out['voxel_mm'] = [round(float(x), 2) for x in sp]; out['shape'] = [int(x) for x in hdr['sizes']]
        body_lo, body_hi = None, None
        for f in files:
            if '_OAR_' not in f or not f.endswith('.seg.nrrd'): continue
            name = f.split('_OAR_')[1][:-len('.seg.nrrd')]; q = '/tmp/%s_%s.nrrd' % (case, name); open(q, 'wb').write(_Z.read(f)); m, h = nrrd.read(q); os.remove(q)
            m = np.asarray(m) > 0
            if m.ndim == 4: m = m.any(0)
            if not m.any(): continue
            ii = np.argwhere(m); lo = np.maximum(ii.min(0) - 2, 0); hi = ii.max(0) + 3; m = m[tuple(slice(a, b) for a, b in zip(lo, hi))]   # crop to the structure, so memory stays small
            msp = np.linalg.norm(np.array(h['space directions'], float)[-3:], axis=1) if 'space directions' in h else sp
            d = distance_transform_edt(m, sampling=msp)
            out['structures'][name] = dict(volume_ml=round(float(m.sum() * np.prod(msp) / 1000), 3), thickness_mm=round(float(2 * d[m].mean()), 2), max_inradius_mm=round(float(d.max()), 2))
        return out
    except Exception as e: return dict(case=case, failed='%s %s' % (type(e).__name__, str(e)[:200]))

if __name__ == '__main__':
    from multiprocessing import Pool
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi(); repo = sys.argv[1]
    zp = hf_hub_download(repo, 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.'); names = zipfile.ZipFile(zp).namelist()
    cases = sorted({n.split('/')[-2] for n in names if n.endswith('_IMG_CT.nrrd')}); log(len(cases), 'cases')
    try: ram = int(open('/sys/fs/cgroup/memory.max').read()) / 1e9
    except Exception: ram = 16
    with Pool(max(1, min(8, int(ram // 6))), _init, (zp,), maxtasksperchild=4) as p: rows = list(p.imap_unordered(one, [(c, [n for n in names if '/%s/' % c in n]) for c in cases]))
    ok = [r for r in rows if 'failed' not in r]; log('ok', len(ok), 'failed', [r for r in rows if 'failed' in r][:2])
    names_ = sorted({k for r in ok for k in r['structures']})
    per = {n: dict(people=sum(1 for r in ok if n in r['structures']), volume_ml=round(float(np.median([r['structures'][n]['volume_ml'] for r in ok if n in r['structures']])), 2),
                   thickness_mm=round(float(np.median([r['structures'][n]['thickness_mm'] for r in ok if n in r['structures']])), 2)) for n in names_}
    th = np.array([v['thickness_mm'] for v in per.values()])
    need = {('%d mm token' % s): dict(structures_at_least_one_token_thick=int((th >= s).sum()), of=len(th), head_window_tokens=int(150 * 200 * 150 / s ** 3)) for s in (3, 4, 6, 8, 12)}
    res = dict(people=len(ok), ct_voxel_mm_median=np.median([r['voxel_mm'] for r in ok], 0).round(2).tolist(), structures=per, thickness_summary_mm=dict(min=float(th.min()), p25=float(np.percentile(th, 25)), median=float(np.median(th)), p75=float(np.percentile(th, 75))), tokens=need)
    os.makedirs('face', exist_ok=True); json.dump(dict(summary=res, rows=rows), open('face/face_census.json', 'w'), indent=1)
    api.upload_folder(folder_path='face', path_in_repo='census/face_sizes', repo_id=repo, repo_type='dataset', commit_message='face size census')
    log('SUMMARY ' + json.dumps(res))
