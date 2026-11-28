# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "safetensors", "matplotlib"]
# ///
"""MARROW-1.0-head (the public release, unchanged) on heads it never saw: PDDCA's 48 full-face CTs (user 2026-11-28,
option 1: test before any retraining). GPU (Colab).
Protocol, the same for every set: the head's own shell (truth labels: 0 outside, 255 unknown, anything else inside) goes
through the release's marrow_head.generate (seed 0, 16 steps, as the card); the compartment with the highest probability
is compared with the truth compartments (game.LUT) on voxels whose truth is known. Per compartment: Dice and volume ratio
(predicted / true). Sets:
  control  the release's own 15 held-out test heads (TotalSegmentator + HaN-Seg test splits, game.assemble): must
           reproduce the card's scores, otherwise the protocol differs and the PDDCA numbers are not comparable;
  pddca    all 48 PDDCA heads (face_v1_headframe_s1), also split by PDDCA's own train / val / test lists.
Also a check that PDDCA's truth looks like the training data's truth: compartment volumes per set (ml).
PRIVATE pictures (real people) of 4 PDDCA heads, truth | prediction, to <repo>/marrow1_eval/pics only. Numbers to
<repo>/marrow1_eval/eval.json.
env: SETS (default control,pddca), LIMIT (heads per set, smoke), MODEL (default gardianofthedarkness/MARROW-1.0-head),
     COMMIT. args: <pddca repo>"""
import os, sys, json, time, urllib.request, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
if __name__ == '__main__':
    try: import safetensors
    except ImportError: import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'safetensors'], check=True)
    import torch, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    repo = sys.argv[1]; api = HfApi(); C_ = os.environ.get('COMMIT', 'main'); LIMIT = int(os.environ.get('LIMIT', '0'))
    SETS = os.environ.get('SETS', 'control,pddca').split(','); MODEL = os.environ.get('MODEL', 'gardianofthedarkness/MARROW-1.0-head')
    urllib.request.urlretrieve('https://raw.githubusercontent.com/gardianofthedarkness/model-to-vrm/%s/research/interior_model/game.py' % C_, 'game.py')
    wd = snapshot_download(MODEL, local_dir='marrow1'); sys.path.insert(0, wd); sys.path.insert(0, '.')
    import marrow_head as mh, game
    dev = torch.device('cuda'); cfg, ae, net = mh.load(wd, dev); log('gpu', torch.cuda.get_device_name(0), 'model', cfg['registry_code'])
    assert list(mh.NAMES) == list(game.GNAMES), 'compartment order differs'
    heads = {}                                                                          # name -> (path, set, split)
    if 'control' in SETS:
        os.makedirs('/content/g' if os.path.isdir('/content') else '/data/g', exist_ok=True); gd = '/content/g' if os.path.isdir('/content') else '/data/g'
        hs = game.assemble(gd); ctl = sorted(c for c, v in hs.items() if v['split'] == 'test')
        for c in ctl[:LIMIT or None]: heads[c] = (os.path.join(gd, c + '.npz'), 'control', 'test')
    if 'pddca' in SETS:
        pi = json.load(open(hf_hub_download(repo, 'face_v1_headframe_s1/index_pddca_s1.json', repo_type='dataset')))['heads']
        pd = sorted(pi)[:LIMIT or None]
        d = snapshot_download(repo, repo_type='dataset', allow_patterns=['face_v1_headframe_s1/%s.npz' % c for c in pd], local_dir='pd', max_workers=16)
        for c in pd: heads[c] = (os.path.join(d, 'face_v1_headframe_s1', c + '.npz'), 'pddca', pi[c]['split'])
    log(len(heads), 'heads', {s: sum(1 for v in heads.values() if v[1] == s) for s in SETS})
    N = game.GNAMES; rows = {}; pics = [c for c in sorted(heads) if heads[c][1] == 'pddca'][::12][:4]; os.makedirs('pics', exist_ok=True)
    for c, (p, st, sp) in heads.items():
        lab = np.load(p)['labels']; t1 = time.time()
        P, body = mh.generate(lab, cfg, ae, net, dev, seed=0, steps=cfg.get('sampling_steps', 16), log=lambda *a: None)
        pred = P.astype(np.float32).argmax(0).astype(np.uint8); X, Y, Z = pred.shape
        truth = game.LUT[lab[:X, :Y, :Z]]; known = truth != game.UNK
        r = dict(set=st, split=sp, dice={}, vol_ratio={}, true_ml={}, seconds=round(time.time() - t1))
        for k, n in enumerate(N):
            if k == 0: continue
            a, b = (pred == k) & known, (truth == k) & known; s = int(a.sum()) + int(b.sum())
            if b.sum(): r['dice'][n] = round(float(2 * (a & b).sum() / s), 4); r['vol_ratio'][n] = round(float(a.sum() / b.sum()), 3); r['true_ml'][n] = round(float(b.sum()) / 1000, 1)
        rows[c] = r; log(c, st, sp, '%ds' % r['seconds'], 'skull', r['dice'].get('skull'), 'mandible', r['dice'].get('mandible'), 'brain', r['dice'].get('brain'))
        if c in pics:
            pal = np.array([(0, 0, 0), (0.98, 0.84, 0.45), (0.93, 0.89, 0.78), (0.85, 0.75, 0.55), (1, 1, 0.92), (0.6, 0.9, 1.0), (0.92, 0.62, 0.70), (0.80, 0.22, 0.20),
                            (0.62, 0.15, 0.30), (0.95, 0.40, 0.45), (0.30, 0.60, 0.90), (0.70, 0.70, 0.62), (0.60, 0.55, 0.45), (0.72, 0.45, 0.40)] + [(0.3, 0.3, 0.3)] * 242)
            e = np.argwhere(truth == N.index('eyes')); ce = e.mean(0).astype(int) if len(e) else np.array(truth.shape) // 2
            fig, axs = plt.subplots(2, 3, figsize=(13, 9))
            for j, (A, t) in enumerate([(truth, 'truth'), (pred * body[:X, :Y, :Z], 'MARROW-1.0')]):
                for i, (sl, nm) in enumerate([((ce[0], slice(None), slice(None)), 'sagittal'), ((slice(None), slice(None), ce[2]), 'axial'), ((slice(None), ce[1], slice(None)), 'coronal')]):
                    axs[j, i].imshow(pal[A[sl]].transpose(1, 0, 2), origin='lower'); axs[j, i].set_title('%s, %s' % (t, nm), fontsize=9); axs[j, i].axis('off')
            fig.suptitle('%s (PDDCA, unseen)  skull %.2f  mandible %.2f' % (c, r['dice'].get('skull', 0), r['dice'].get('mandible', 0))); plt.tight_layout(); plt.savefig('pics/%s.png' % c, dpi=60); plt.close(fig)
    def agg(sel):
        out = {}
        for n in N[1:]:
            d = [rows[c]['dice'][n] for c in sel if n in rows[c]['dice']]; v = [rows[c]['vol_ratio'][n] for c in sel if n in rows[c]['vol_ratio']]; m = [rows[c]['true_ml'][n] for c in sel if n in rows[c]['true_ml']]
            if d: out[n] = dict(dice_mean=round(float(np.mean(d)), 3), dice_median=round(float(np.median(d)), 3), vol_ratio_median=round(float(np.median(v)), 3), true_ml_median=round(float(np.median(m)), 1), heads=len(d))
        return out
    summ = {s: agg([c for c in rows if rows[c]['set'] == s]) for s in SETS}
    if 'pddca' in SETS:
        for sp in ('train', 'val', 'test'): summ['pddca_' + sp] = agg([c for c in rows if rows[c]['set'] == 'pddca' and rows[c]['split'] == sp])
    os.makedirs('ev', exist_ok=True); json.dump(dict(model=cfg['registry_code'], protocol='seed 0, argmax, known truth voxels', summary=summ, per_head=rows), open('ev/eval.json', 'w'), indent=1)
    dest = 'marrow1_eval' + ('_smoke' if LIMIT else '')
    api.upload_folder(folder_path='ev', path_in_repo=dest, repo_id=repo, repo_type='dataset', commit_message='MARROW-1.0 on unseen PDDCA heads')
    if pics: api.upload_folder(folder_path='pics', path_in_repo=dest + '/pics', repo_id=repo, repo_type='dataset', commit_message='MARROW-1.0 PDDCA pictures (private)')
    for s in summ: log('SUMMARY', s, json.dumps({n: (v['dice_mean'], v['dice_median']) for n, v in summ[s].items()}))
    log('SUMMARY done')
