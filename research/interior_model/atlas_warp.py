"""Atlas-warp baseline for interior-from-surface: a different structure from the token transformer.
Input is only the target's body shell. The K training people whose bodies overlap it most (same ranking as the
nearest-person baseline) are each warped onto the target with an existing registration tool (ANTs SyN, default
settings) driven only by the two body shapes (signed distance to the body surface); the warped label maps then vote.
Because every vote is a real person's anatomy moved smoothly, organs keep their shape, count and side.
Scores: per-class Dice for K = 1, 3, 5 ... (prefixes of the ranking) and for the unwarped nearest person.
Runs on CPU. Output (private, CT-derived): <out>/pred_<part>_K<k>_<person>.npz and <out>/atlas_warp_<part>_<split>.json
args: <interior_data root> <face|pelvis> <val|test> <out folder> [limit people] [K max]"""
import os, sys, json, time, numpy as np, ants
from scipy.ndimage import distance_transform_edt as edt
root, part, which, out = sys.argv[1:5]; limit = int(sys.argv[5]) if len(sys.argv) > 5 else 0; KMAX = int(sys.argv[6]) if len(sys.argv) > 6 else 5
UNK = 255; os.makedirs(out, exist_ok=True)
fidx = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); pidx = json.load(open(os.path.join(root, 'pelvis_v1', 'index.json')))
hs = sorted(c for c, v in fidx['heads'].items() if v['source'] == 'hanseg')                      # split exactly as transformer_smoke.py
split = {c: ('test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train') for i, c in enumerate(hs)}
split.update({c: v['split'] for c, v in fidx['heads'].items() if v['source'] == 'ts'}); split.update({c: v['split'] for c, v in pidx['people'].items()})
folder, members, fill, mm, names = (('face_v1_headframe', fidx['heads'], 3, 1.0, {v: k for k, v in fidx['ids'].items()}) if part == 'face'
                                    else ('pelvis_v1', pidx['people'], 5, 3.0, dict(enumerate(pidx['classes']))))
REG_MM = max(mm, 2.0)                                                                             # register on a grid no finer than 2 mm (shape only); labels are warped at full resolution
load = lambda c: np.load(os.path.join(root, folder, c + '.npz'))['labels']
train = sorted(c for c in members if split.get(c) == 'train'); targets = sorted(c for c in members if split.get(c) == which)
if limit: targets = targets[:: max(1, len(targets) // limit)][:limit]
bodies = {}
for c in train: L = load(c); b = (L > 0) & (L < UNK); bodies[c] = (np.packbits(b, axis=-1), b.shape)
print(part, which, len(targets), 'targets;', len(train), 'training people', flush=True)

def sdf_image(body):                                                                              # signed distance to the body surface, mm, as an ANTs image
    d = (edt(~body) - edt(body)).astype(np.float32) * mm
    im = ants.from_numpy(d, spacing=(mm,) * 3)
    return ants.resample_image(im, (REG_MM,) * 3, use_voxels=False, interp_type=0) if REG_MM > mm else im
def dice_table(P, T):
    o = {}
    for k in np.unique(T):
        if k == 0 or k == UNK: continue
        a, b = P == k, T == k
        if b.sum() >= 10: o[names.get(int(k), str(k))] = float(2 * (a & b).sum() / (a.sum() + b.sum()))
    return o
def vote(stack):                                                                                  # per-voxel most common label among the first rows; ties go to the better-ranked person
    best = stack[0].copy(); bc = np.zeros(best.shape, np.uint8)
    for i in range(len(stack)):
        c = np.zeros(best.shape, np.uint8)
        for j in range(len(stack)): c += stack[i] == stack[j]
        up = c > bc; best[up] = stack[i][up]; bc[up] = c[up]
    return best

KS = [k for k in (1, 3, 5, 7, 9) if k <= KMAX]; rows = {('K%d' % k): [] for k in KS}; rows['nearest'] = []; secs = []
for t in targets:
    T = load(t); body = (T > 0) & (T < UNK); Lt = np.where(T == UNK, 0, T); t0 = time.time()
    def iou(q):
        pk, sh = bodies[q]; bq = np.unpackbits(pk, axis=-1, count=sh[-1]).astype(bool); s = tuple(slice(0, min(x, y)) for x, y in zip(sh, T.shape))
        return (bq[s] & body[s]).sum() / max(1, (bq[s] | body[s]).sum())
    rank = sorted(train, key=iou, reverse=True)[:KMAX]
    fix = sdf_image(body); ref = ants.from_numpy(np.zeros(T.shape, np.float32), spacing=(mm,) * 3); warped = []
    for q in rank:
        Q = load(q); Q = np.where(Q == UNK, 0, Q); mov = sdf_image(Q > 0)
        reg = ants.registration(fixed=fix, moving=mov, type_of_transform='SyN')
        W = ants.apply_transforms(fixed=ref, moving=ants.from_numpy(Q.astype(np.float32), spacing=(mm,) * 3), transformlist=reg['fwdtransforms'], interpolator='nearestNeighbor')
        warped.append(W.numpy().round().astype(np.uint8))
        for f in reg['fwdtransforms'] + reg['invtransforms']:
            try: os.remove(f)
            except OSError: pass
    def finish(P): P = P.copy(); P[~body] = 0; P[body & (P == 0)] = fill; return P
    Q = load(rank[0]); s = tuple(slice(0, min(x, y)) for x, y in zip(Q.shape, T.shape)); N = np.zeros_like(T); N[s] = np.where(Q[s] == UNK, 0, Q[s])
    rows['nearest'].append(dice_table(finish(N), Lt))
    for k in KS:
        P = finish(vote(warped[:k])); rows['K%d' % k].append(dice_table(P, Lt)); np.savez_compressed(os.path.join(out, 'pred_%s_K%d_%s.npz' % (part, k, t)), labels=P)
    secs.append(time.time() - t0)
    print(t, '%.0f s' % secs[-1], {k: round(float(np.median(list(v[-1].values()))), 3) for k, v in rows.items()}, flush=True)
    summ = {}
    for who, rr in rows.items():
        keys = sorted({k for r in rr for k in r}); per = {k: round(float(np.median([r[k] for r in rr if k in r])), 3) for k in keys}
        summ[who] = dict(median_over_classes=round(float(np.median(list(per.values()))), 3), per_class=per)
    json.dump(dict(part=part, split=which, people=targets[:len(secs)], seconds_per_person=round(float(np.mean(secs)), 1), reg_mm=REG_MM, results=summ),
              open(os.path.join(out, 'atlas_warp_%s_%s.json' % (part, which)), 'w'), indent=1)
print('SUMMARY', {k: v['median_over_classes'] for k, v in summ.items()})
