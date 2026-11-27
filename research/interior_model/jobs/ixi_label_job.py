# /// script
# dependencies = ["huggingface_hub>=1.0", "hf_xet", "numpy", "scipy", "torch", "monai==1.5.1"]
# ///
"""IXI step 2b (GPU, Colab): the MRI-to-compartment labeller trained on all 42 HaN-Seg people (mri_game/segresnet_all.pt)
run on every prepared IXI T1 (ixi_prep_v1, from ixi_prep_job.py on HF CPU). Private outputs only (real people).
Output <repo>/ixi_label_v1/<id>.npz: labels (uint8: the 14 compartments, then canal_left, canal_right; outside the MRI
body = 0), conf (uint8: the winning class's probability x 255), plus the prep's grid (shape, spacing, origin, direction).
Uploads in batches; a restart skips subjects already labelled.
env: LIMIT (smoke), MODEL_REPO (default <user>/hanseg-ct-mr), MODEL (default mri_game/segresnet_all.pt). args: <repo (ixi-t1-t2)>"""
import os, sys, json, time, numpy as np
t0 = time.time()
def log(*a): print('TS %5.0fs' % (time.time() - t0), *a, flush=True)
if __name__ == '__main__':
    try: import monai
    except ImportError: import subprocess; subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'monai==1.5.1'], check=True)
    import torch
    from monai.networks.nets import SegResNet
    from monai.inferers import sliding_window_inference
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download
    api = HfApi(); repo = sys.argv[1]; user = api.whoami()['name']; OUT = 'ixi_label_v1'; os.makedirs(OUT, exist_ok=True)
    mrepo = os.environ.get('MODEL_REPO', user + '/hanseg-ct-mr'); mpath = os.environ.get('MODEL', 'mri_game/segresnet_all.pt')
    classes = json.load(open(hf_hub_download(mrepo, 'mri_game/mri_game.json', repo_type='dataset', local_dir='m')))['classes']
    net = SegResNet(spatial_dims=3, in_channels=1, out_channels=len(classes), init_filters=16).cuda()
    net.load_state_dict(torch.load(hf_hub_download(mrepo, mpath, repo_type='dataset', local_dir='m'), map_location='cuda')); net.eval()
    files = api.list_repo_files(repo, repo_type='dataset')
    prepped = sorted(f.split('/')[-1][:-4] for f in files if f.startswith('ixi_prep_v1/') and f.endswith('.npz'))
    done = {f.split('/')[-1][:-4] for f in files if f.startswith(OUT + '/') and f.endswith('.npz')}
    todo = [s for s in prepped if s not in done]
    if os.environ.get('LIMIT'): todo = todo[:int(os.environ['LIMIT'])]
    log(len(prepped), 'prepared,', len(done), 'already labelled,', len(todo), 'to do; classes', classes)
    def flush():
        for a in range(5):
            try: api.upload_folder(folder_path=OUT, path_in_repo=OUT, repo_id=repo, repo_type='dataset', allow_patterns=['*.npz'], commit_message='IXI labels batch'); break
            except Exception as e: log('upload retry', a, type(e).__name__); time.sleep(60)
        for f in os.listdir(OUT):
            if f.endswith('.npz'): os.remove(os.path.join(OUT, f))
    vols = {}
    for i in range(0, len(todo), 25):
        part = todo[i:i + 25]
        d = snapshot_download(repo, repo_type='dataset', allow_patterns=['ixi_prep_v1/%s.npz' % s for s in part], local_dir='in', max_workers=8)
        for s in part:
            z = np.load(os.path.join(d, 'ixi_prep_v1', s + '.npz')); shape = tuple(z['shape']); body = np.unpackbits(z['body'])[:np.prod(shape)].reshape(shape).astype(bool)
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16):
                lg = sliding_window_inference(torch.tensor(z['M'].astype(np.float32)[None, None], device='cuda'), (96, 96, 96), 4, net, overlap=0.5)
                p = torch.softmax(lg.float(), 1)[0]; conf, lab = p.max(0)
            lab = lab.cpu().numpy().astype(np.uint8); conf = (conf.cpu().numpy() * 255).astype(np.uint8); lab[~body] = 0; conf[~body] = 255
            np.savez_compressed(os.path.join(OUT, s + '.npz'), labels=lab, conf=conf, shape=z['shape'], spacing=z['spacing'], origin=z['origin'], direction=z['direction'])
            vols[s] = {classes[k]: round(float((lab == k).sum()) * 1.5 ** 3 / 1000, 1) for k in range(1, len(classes))}
            os.remove(os.path.join(d, 'ixi_prep_v1', s + '.npz'))
        log(min(i + 25, len(todo)), 'of', len(todo), 'example volumes ml', vols[part[-1]]); flush()
    json.dump(dict(volumes_ml=vols, classes=classes, model=mrepo + '/' + mpath), open('label_index_%d.json' % int(time.time()), 'w'), indent=1)
    for f in os.listdir('.'):
        if f.startswith('label_index_'): api.upload_file(path_or_fileobj=f, path_in_repo=OUT + '/' + f, repo_id=repo, repo_type='dataset', commit_message='IXI label volumes')
    if vols:
        med = {k: round(float(np.median([v[k] for v in vols.values()])), 1) for k in classes[1:]}; log('SUMMARY median volumes ml', med)
    log('SUMMARY done')
