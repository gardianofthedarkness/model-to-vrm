# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "SimpleITK", "torch", "monai==1.5.1", "matplotlib"]
# ///
"""Pictures for the user's eye check of IXI step 1 (user, 2026-11-27: scores miss flaws eyes catch). PRIVATE: real people.
(a) HaN-Seg: for N held-out people (one per fold), MRI | CT truth compartments | MRI-only prediction by the fold model that
    never saw them, three slices through the eyes' centre. (b) IXI: for N prepared subjects, MRI | prediction of the
    all-heads model. PNGs go to the private repos only (<hanseg repo>/mri_game/pics, <ixi repo>/ixi_label_v1/pics) and
    to the private gallery; never to a report or a public page.
env: COMMIT, N (default 4), IXI_REPO (default <user>/ixi-t1-t2). args: <hanseg repo>"""
import os, sys, json, time, zipfile, urllib.request, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
if __name__ == '__main__':
    try: import monai
    except ImportError: import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'monai==1.5.1', 'SimpleITK'], check=True)
    import torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from monai.networks.nets import SegResNet
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; user = api.whoami()['name']; C = os.environ.get('COMMIT', 'main'); N = int(os.environ.get('N', '4'))
    for f in ('jobs/mri_game_job.py', 'game.py'):
        urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/%s' % (C, f), os.path.basename(f))
    sys.path.insert(0, '.'); import game, mri_game_job as mg
    names = json.load(open(hf_hub_download(repo, 'mri_game/mri_game.json', repo_type='dataset', local_dir='m')))['classes']; NCL = len(names)
    COL = {'outside': (0, 0, 0), 'fat_layer': (0.98, 0.84, 0.45), 'skull': (0.93, 0.89, 0.78), 'mandible': (0.85, 0.75, 0.55), 'teeth': (1.0, 1.0, 0.92), 'eyes': (0.6, 0.9, 1.0),
           'brain': (0.92, 0.62, 0.70), 'chew_left': (0.80, 0.22, 0.20), 'chew_right': (0.62, 0.15, 0.30), 'tongue': (0.95, 0.40, 0.45), 'airway': (0.30, 0.60, 0.90),
           'spine': (0.70, 0.70, 0.62), 'other_bone': (0.60, 0.55, 0.45), 'other_soft': (0.72, 0.45, 0.40), 'canal_left': (0, 1, 0), 'canal_right': (0, 1, 0)}
    pal = np.array([COL[n] for n in names] + [(0.25, 0.25, 0.25)] * (256 - NCL))       # 255 unknown: dark grey
    def net_from(path):
        n = SegResNet(spatial_dims=3, in_channels=1, out_channels=NCL, init_filters=16).cuda(); n.load_state_dict(torch.load(path, map_location='cuda')); return n
    def sheet(cols, titles, ce, out, head):
        """cols: list of (array (z,y,x), kind) ; three slices through ce"""
        fig, axs = plt.subplots(3, len(cols), figsize=(4.2 * len(cols), 12))
        for r, (sl, nm) in enumerate([((ce[0], slice(None), slice(None)), 'axial'), ((slice(None), ce[1], slice(None)), 'coronal'), ((slice(None), slice(None), ce[2]), 'sagittal')]):
            for c, ((A, kind), t) in enumerate(zip(cols, titles)):
                a = A[sl]; ax = axs[r, c]
                if kind == 'mri': ax.imshow(a, cmap='gray', origin='lower' if r else 'upper', vmin=0, vmax=1)
                else: ax.imshow(pal[a], origin='lower' if r else 'upper')
                ax.set_title('%s, %s' % (t, nm), fontsize=9); ax.axis('off')
        fig.suptitle(head, fontsize=11); plt.tight_layout(); plt.savefig(out, dpi=60); plt.close(fig)
    os.makedirs('pics', exist_ok=True)
    # (a) HaN-Seg held-out people
    zp = hf_hub_download(repo, 'raw/HaN-Seg.zip', repo_type='dataset', local_dir='.'); os.makedirs(mg.W, exist_ok=True)
    snapshot_download(repo, repo_type='dataset', allow_patterns=['face_v1/*'], local_dir=mg.W)
    ids = json.load(open('%s/face_v1/face_ids.json' % mg.W))['ids']
    cases = sorted({n.split('/')[-2] for n in zipfile.ZipFile(zp).namelist() if n.endswith('_IMG_MR_T1.nrrd')}); cases = [c for c in cases if os.path.exists('%s/face_v1/%s.npz' % (mg.W, c))]
    import SimpleITK as sitk; sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
    mg._init(zp, game.LUT, (ids['auditory_canal_left'], ids['auditory_canal_right']), len(game.GNAMES))
    pick = [[c for i, c in enumerate(cases) if i % 5 == f][1] for f in range(min(N, 5))]      # the second person of each fold
    for f, c in enumerate(pick):
        r = mg.stage1(c)[1]
        if r[0]: log('failed', c, r[0]); continue
        z = np.load('%s/%s.npz' % (mg.W, c)); M, T = z['M'].astype(np.float32), z['T']
        P = mg.predict(net_from(hf_hub_download(repo, 'mri_game/segresnet_fold%d.pt' % f, repo_type='dataset', local_dir='m')), M)
        P[T == 255] = 255; ce = np.argwhere(T == names.index('eyes')).mean(0).astype(int) if (T == names.index('eyes')).any() else np.array(T.shape) // 2
        sheet([(M, 'mri'), (T, 'lab'), (P, 'lab')], ['MRI', 'CT truth', 'MRI-only prediction'], ce, 'pics/hanseg_%s.png' % c, '%s (fold %d model never saw this person)' % (c, f)); log('picture', c)
    api.upload_folder(folder_path='pics', path_in_repo='mri_game/pics', repo_id=repo, repo_type='dataset', commit_message='step 1 pictures (private)')
    # (b) IXI subjects with the all-heads model
    ixi = os.environ.get('IXI_REPO', user + '/ixi-t1-t2'); net = net_from(hf_hub_download(repo, 'mri_game/segresnet_all.pt', repo_type='dataset', local_dir='m'))
    subs = sorted(f.split('/')[-1][:-4] for f in api.list_repo_files(ixi, repo_type='dataset') if f.startswith('ixi_prep_v1/') and f.endswith('.npz'))
    sel = [subs[i] for i in np.linspace(0, len(subs) - 1, 2 * N).astype(int)]              # spread over sites
    os.makedirs('pics_ixi', exist_ok=True)
    for s in sel:
        z = np.load(hf_hub_download(ixi, 'ixi_prep_v1/%s.npz' % s, repo_type='dataset', local_dir='ixi')); M = z['M'].astype(np.float32); sh = tuple(z['shape'])
        body = np.unpackbits(z['body'])[:np.prod(sh)].reshape(sh).astype(bool); P = mg.predict(net, M); P[~body] = 0
        e = P == names.index('eyes'); ce = np.argwhere(e).mean(0).astype(int) if e.any() else np.array(P.shape) // 2
        sheet([(M, 'mri'), (P, 'lab')], ['IXI MRI', 'prediction'], ce, 'pics_ixi/%s.png' % s, s); log('picture', s)
    api.upload_folder(folder_path='pics_ixi', path_in_repo='ixi_label_v1/pics', repo_id=ixi, repo_type='dataset', commit_message='IXI label pictures (private)')
    log('SUMMARY done')
