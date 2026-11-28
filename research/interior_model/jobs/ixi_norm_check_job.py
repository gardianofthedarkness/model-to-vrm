# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "monai==1.5.1", "matplotlib"]
# ///
"""Does brightness normalisation (ixi_norm_job.py: N4 + histogram standardisation to HaN-Seg) fix the IXI labelling?
GPU (Colab). The all-heads MRI labeller (mri_game/segresnet_all.pt, unchanged) on K IXI heads spread over the list, on the
old input (ixi_prep_v1) and the normalised one (ixi_norm_v1). The network's own 'outside' class is kept (no body-mask
override: that drew the black stripes at the nose).
Check, per head: compartment volumes (ml); a head passes when its brain and eye volumes lie inside the range of the
HaN-Seg CT truth (face_v1 labels mapped to compartments, 42 adults; the range is the data's min..max). Pass rates and
median volumes old vs new. Pictures (PRIVATE: real people) for the same 8 heads as step 1: MRI old | prediction old |
MRI normalised | prediction normalised, to <ixi repo>/ixi_norm_v1/check/ only.
env: COMMIT, K (heads for the volume check, default 80), N (picture heads = 2N, default 4), HANSEG_REPO. args: <ixi repo>"""
import os, sys, json, time, urllib.request, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
if __name__ == '__main__':
    try: import monai
    except ImportError: import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'monai==1.5.1'], check=True)
    import torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from monai.networks.nets import SegResNet
    from monai.inferers import sliding_window_inference
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; user = api.whoami()['name']; C = os.environ.get('COMMIT', 'main')
    K = int(os.environ.get('K', '80')); N = int(os.environ.get('N', '4')); hrepo = os.environ.get('HANSEG_REPO', user + '/hanseg-ct-mr')
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/game.py' % C, 'game.py')
    sys.path.insert(0, '.'); import game
    names = json.load(open(hf_hub_download(hrepo, 'mri_game/mri_game.json', repo_type='dataset', local_dir='m')))['classes']; NCL = len(names)
    net = SegResNet(spatial_dims=3, in_channels=1, out_channels=NCL, init_filters=16).cuda()
    net.load_state_dict(torch.load(hf_hub_download(hrepo, 'mri_game/segresnet_all.pt', repo_type='dataset', local_dir='m'), map_location='cuda')); net.eval()
    log('gpu', torch.cuda.get_device_name(0))
    # QC range from the HaN-Seg CT truth
    fd = snapshot_download(hrepo, repo_type='dataset', allow_patterns=['face_v1/*'], local_dir='w')
    truth = {}
    for f in sorted(os.listdir(os.path.join(fd, 'face_v1'))):
        if not f.endswith('.npz'): continue
        z = np.load(os.path.join(fd, 'face_v1', f)); T = game.LUT[z['labels']]; vml = abs(np.linalg.det(z['affine'][:3, :3])) / 1000
        truth[f[:-4]] = {n: float((T == game.GNAMES.index(n)).sum() * vml) for n in ('brain', 'eyes', 'skull')}
    rng = {n: (min(v[n] for v in truth.values()), max(v[n] for v in truth.values())) for n in ('brain', 'eyes')}
    log('CT truth range ml from', len(truth), 'heads', {k: (round(a), round(b)) for k, (a, b) in rng.items()})
    files = api.list_repo_files(repo, repo_type='dataset')
    have = sorted(f.split('/')[-1][:-4] for f in files if f.startswith('ixi_norm_v1/') and f.endswith('.npz'))
    allprep = sorted(f.split('/')[-1][:-4] for f in files if f.startswith('ixi_prep_v1/') and f.endswith('.npz'))
    pics = [s for s in (allprep[i] for i in np.linspace(0, len(allprep) - 1, 2 * N).astype(int)) if s in have]   # step 1's picture heads
    sel = sorted(set(pics) | {have[i] for i in np.linspace(0, len(have) - 1, min(K, len(have))).astype(int)})
    d = snapshot_download(repo, repo_type='dataset', allow_patterns=['ixi_prep_v1/%s.npz' % s for s in sel] + ['ixi_norm_v1/%s.npz' % s for s in sel], local_dir='in', max_workers=16)
    def pred(M):
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
            lg = sliding_window_inference(torch.tensor(M[None, None], device='cuda'), (96, 96, 96), 4, net, overlap=0.5)
        return lg.float().argmax(1)[0].cpu().numpy().astype(np.uint8)
    COL = {'outside': (0, 0, 0), 'fat_layer': (0.98, 0.84, 0.45), 'skull': (0.93, 0.89, 0.78), 'mandible': (0.85, 0.75, 0.55), 'teeth': (1.0, 1.0, 0.92), 'eyes': (0.6, 0.9, 1.0),
           'brain': (0.92, 0.62, 0.70), 'chew_left': (0.80, 0.22, 0.20), 'chew_right': (0.62, 0.15, 0.30), 'tongue': (0.95, 0.40, 0.45), 'airway': (0.30, 0.60, 0.90),
           'spine': (0.70, 0.70, 0.62), 'other_bone': (0.60, 0.55, 0.45), 'other_soft': (0.72, 0.45, 0.40), 'canal_left': (0, 1, 0), 'canal_right': (0, 1, 0)}
    pal = np.array([COL[n] for n in names] + [(0.25, 0.25, 0.25)] * (256 - NCL))
    os.makedirs('check', exist_ok=True); vols = {}; vml = 1.5 ** 3 / 1000
    for s in sel:
        Mo = np.load(os.path.join(d, 'ixi_prep_v1', s + '.npz'))['M'].astype(np.float32); Mn = np.load(os.path.join(d, 'ixi_norm_v1', s + '.npz'))['M'].astype(np.float32)
        Po, Pn = pred(Mo), pred(Mn)
        vols[s] = {k: {n: round(float((P == names.index(n)).sum() * vml), 1) for n in names[1:]} for k, P in (('old', Po), ('new', Pn))}
        for k in ('old', 'new'): vols[s][k]['pass'] = bool(all(rng[n][0] <= vols[s][k][n] <= rng[n][1] for n in rng))
        if s in pics:
            e = Pn == names.index('eyes'); ce = np.argwhere(e).mean(0).astype(int) if e.any() else np.array(Pn.shape) // 2
            cols = [(Mo, 'mri', 'MRI as before'), (Po, 'lab', 'prediction before'), (Mn, 'mri', 'MRI normalised'), (Pn, 'lab', 'prediction normalised')]
            fig, axs = plt.subplots(3, 4, figsize=(16.8, 12))
            for r, (sl, nm) in enumerate([((ce[0], slice(None), slice(None)), 'axial'), ((slice(None), ce[1], slice(None)), 'coronal'), ((slice(None), slice(None), ce[2]), 'sagittal')]):
                for c, (A, kind, t) in enumerate(cols):
                    a = A[sl]; ax = axs[r, c]
                    ax.imshow(a, cmap='gray', vmin=0, vmax=1, origin='lower' if r else 'upper') if kind == 'mri' else ax.imshow(pal[a], origin='lower' if r else 'upper')
                    ax.set_title('%s, %s' % (t, nm), fontsize=9); ax.axis('off')
            fig.suptitle('%s  brain ml %s -> %s' % (s, vols[s]['old']['brain'], vols[s]['new']['brain']), fontsize=11); plt.tight_layout(); plt.savefig('check/%s.png' % s, dpi=60); plt.close(fig)
        log(s, 'brain ml', vols[s]['old']['brain'], '->', vols[s]['new']['brain'], 'pass', vols[s]['old']['pass'], '->', vols[s]['new']['pass'])
    site = lambda s: s.split('-')[1]
    summ = {}
    for k in ('old', 'new'):
        summ[k] = dict(pass_rate=round(float(np.mean([v[k]['pass'] for v in vols.values()])), 3),
                       pass_rate_by_site={st: round(float(np.mean([v[k]['pass'] for s, v in vols.items() if site(s) == st])), 3) for st in sorted({site(s) for s in vols})},
                       median_ml={n: round(float(np.median([v[k][n] for v in vols.values()])), 1) for n in names[1:]})
    summ['ct_truth_range_ml'] = rng; summ['heads'] = len(vols)
    json.dump(dict(summary=summ, per_head=vols), open('check/check.json', 'w'), indent=1)
    api.upload_folder(folder_path='check', path_in_repo='ixi_norm_v1/check', repo_id=repo, repo_type='dataset', commit_message='IXI normalisation check (private)')
    log('SUMMARY ' + json.dumps(summ)); log('SUMMARY done')
