"""Starting frames for the CT slicing from the measured keypoints alone (no body model fitted):
  E:\\splat\\.venv\\Scripts\\python.exe kp_frames.py <model dir> [atlas.npz]
The multi-view triangulated keypoints (body/joints3d.npz, COCO-WholeBody) give the joints the segments need, under the
joint names segments.py uses (so fit_slices.py reads them like a skeleton):
  limbs     shoulder, elbow, wrist, hip, knee, ankle keypoints; hand: thumb base and first joint, index / middle / ring /
            little finger knuckles; foot: ball = middle of the big and small toe, heel
  head      head joint = middle of the ears; eyes; nose
  trunk     neck base = middle of the shoulders; the spine base and the levels between come from the atlas's own
            proportions along hip-middle -> neck base (segments.Atlas axes: where the pelvis segment ends on the way
            from the hips to the neck), so no number is chosen by hand
  helpers   joints that only name a bone for ownership (forearm twist, foot sub-joints) lie on the line between their
            measured neighbours at the atlas's own fractions where the atlas has them, else at the middle
These are starting frames only: after the bone fit the joints are taken from the fitted bones (bone_joints.py) and the
slicing is redone. Writes body/frames.npz (heads, parents, labels, source per joint)."""
import sys, os, numpy as np
R = sys.argv[1]; ap = sys.argv[2] if len(sys.argv) > 2 else r'E:\figures\models\zanatomy\export\atlas.npz'
J = np.load(os.path.join(R, 'body', 'joints3d.npz')); X = J['X'].astype(np.float64)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from segments import Atlas
AT = Atlas(ap); ax = AT.axes
kp = lambda *i: np.nanmean(X[list(i)], 0)
lp = lambda a, b, f: a + (b - a) * f
pel = np.linalg.norm(ax['pelvis']['J1'] - ax['pelvis']['J0']); tr = np.linalg.norm(ax['trunk']['J1'] - ax['trunk']['J0'])
f_sp0 = pel / (pel + tr)                                            # the spine base's place between hips and neck base, in the atlas
hipm, neck = kp(11, 12), kp(5, 6); sp0 = lp(hipm, neck, f_sp0)
J_ = {'c_spine0': (sp0, None, 'atlas proportion hips -> neck'), 'c_neck': (neck, 'c_spine3', 'shoulders middle'),
      'c_spine1': (lp(sp0, neck, 1 / 4), 'c_spine0', 'helper'), 'c_spine2': (lp(sp0, neck, 2 / 4), 'c_spine1', 'helper'), 'c_spine3': (lp(sp0, neck, 3 / 4), 'c_spine2', 'helper'),
      'c_head': (kp(3, 4), 'c_neck', 'ears middle'), 'c_jaw': (kp(0), 'c_head', 'nose'), 'c_teeth': (kp(0), 'c_jaw', 'nose'),
      'r_eye': (kp(2), 'c_head', 'keypoint'), 'l_eye': (kp(1), 'c_head', 'keypoint')}
for i in range(5): J_['c_tongue%d' % i] = (kp(0), 'c_teeth' if i == 0 else 'c_tongue%d' % (i - 1), 'nose')
for s, (sho, elb, wri, hip, kne, ank, bt, st, hl, h0) in (('l', (5, 7, 9, 11, 13, 15, 17, 18, 19, 91)), ('r', (6, 8, 10, 12, 14, 16, 20, 21, 22, 112))):
    g = lambda nm: s + '_' + nm
    J_[g('clavicle')] = (lp(neck, kp(sho), 0.5), 'c_spine3', 'helper'); J_[g('uparm')] = (kp(sho), g('clavicle'), 'keypoint')
    J_[g('lowarm')] = (kp(elb), g('uparm'), 'keypoint'); J_[g('wrist_twist')] = (lp(kp(elb), kp(wri), 0.5), g('lowarm'), 'helper')
    J_[g('wrist')] = (kp(wri), g('wrist_twist'), 'keypoint')
    J_[g('thumb0')] = (kp(h0 + 1), g('wrist'), 'keypoint'); J_[g('thumb1')] = (kp(h0 + 2), g('thumb0'), 'keypoint')
    for nm, o in (('index1', 5), ('middle1', 9), ('ring1', 13)): J_[g(nm)] = (kp(h0 + o), g('wrist'), 'keypoint')
    J_[g('pinky0')] = (lp(kp(wri), kp(h0 + 17), 0.5), g('wrist'), 'helper'); J_[g('pinky1')] = (kp(h0 + 17), g('pinky0'), 'keypoint')
    J_[g('upleg')] = (kp(hip), 'c_spine0', 'keypoint'); J_[g('lowleg')] = (kp(kne), g('upleg'), 'keypoint')
    J_[g('foot')] = (lp(kp(kne), kp(ank), 0.5), g('lowleg'), 'helper'); J_[g('talocrural')] = (kp(ank), g('foot'), 'keypoint')
    ball = kp(bt, st); J_[g('subtalar')] = (kp(hl), g('talocrural'), 'heel keypoint')
    J_[g('transversetarsal')] = (lp(kp(ank), ball, 0.5), g('subtalar'), 'helper'); J_[g('ball')] = (ball, g('transversetarsal'), 'toes middle')
labels = list(J_); order, seen = [], set()                          # parents before children
def add(n):
    if n in seen: return
    p = J_[n][1]
    if p is not None: add(p)
    seen.add(n); order.append(n)
for n in labels: add(n)
H = np.array([J_[n][0] for n in order]); par = np.array([order.index(J_[n][1]) if J_[n][1] is not None else -1 for n in order])
bad = [n for n, h in zip(order, H) if not np.isfinite(h).all()]
if bad: print('KF missing keypoints for: %s' % ', '.join(bad))
np.savez(os.path.join(R, 'body', 'frames.npz'), heads=H, parents=par, labels=np.array(order), source=np.array([J_[n][2] for n in order]))
print('KF %d joints from keypoints (%d measured, %d helpers); spine base at %.2f of hips -> neck (atlas)' % (
    len(order), sum(J_[n][2] != 'helper' for n in order), sum(J_[n][2] == 'helper' for n in order), f_sp0))
