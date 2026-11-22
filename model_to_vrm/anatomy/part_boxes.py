"""Where the body's intricate ends are - head, hands, feet - for zoomed views and detection:
  E:\\splat\\.venv\\Scripts\\python.exe part_boxes.py <fits dir with skeleton_fit.npz + atlas_warp.npz> <model dir>
A model vertex belongs to the region of the fitted bone nearest to it (bones: skeleton_fit.npz; a bone's region: the
segment that lists it in segments.py, an unlisted bone that of its nearest listed bone, as in atlas_warp.py).
Writes <model dir>/parts_zoom/boxes.json: per region centre, half size (the owned vertices' bounding box) and the
owned vertex ids (parts_zoom/<region>_vids.npy), so the zoomed renders can show the region alone, unoccluded."""
import sys, os, re, json, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segments import SEG, ORDER
FD, MD = sys.argv[1], sys.argv[2]; REG = ['head', 'hand_r', 'hand_l', 'foot_r', 'foot_l']
W = np.load(os.path.join(FD, 'atlas_warp.npz')); V0, OB = W['V0'], W['obj']; names = [str(x) for x in W['names']]; sysn = [str(x) for x in W['system']]
K = np.load(os.path.join(FD, 'skeleton_fit.npz'), allow_pickle=True); bones = [int(b) for b in K['bones']]; Vr = K['V_rigid']
def bone_names(spec):
    return [n for n in spec if n in names] if isinstance(spec, list) else [n for n, s in zip(names, sysn) if s == 'bone' and re.search(spec, n)]
listed = {}
for s in ORDER:
    for n in bone_names(SEG[s]['bones']): listed.setdefault(n, s)
lb = [b for b in bones if names[b] in listed]; lv = np.flatnonzero(np.isin(OB, lb)); ltree = cKDTree(V0[lv])
seg_of = {b: listed[names[b]] if names[b] in listed else listed[names[OB[lv[ltree.query(V0[OB == b].mean(0))[1]]]]] for b in bones}
bv = np.flatnonzero(np.isin(OB, bones))[::4]; reg = np.array([seg_of[o] for o in OB[bv]])     # every 4th bone vertex: dense enough for ownership
MV = np.load(os.path.join(MD, 'parts', 'orig_colours.npz'))['V']; own = reg[cKDTree(Vr[bv]).query(MV)[1]]
od = os.path.join(MD, 'parts_zoom'); os.makedirs(od, exist_ok=True); out = {}
for r in REG:
    ids = np.flatnonzero(own == r); P = MV[ids]; lo, hi = P.min(0), P.max(0)
    out[r] = {'centre': ((lo + hi) / 2).tolist(), 'half': ((hi - lo) / 2).tolist(), 'vertices': int(len(ids))}; np.save(os.path.join(od, r + '_vids.npy'), ids)
    print('PB %s: %d vertices, box %s m' % (r, len(ids), np.round(hi - lo, 3)))
json.dump(out, open(os.path.join(od, 'boxes.json'), 'w'), indent=1)
