"""Stick skeleton from the MHR pose fit (its joints matched the triangulated keypoints to ~5 mm; its surface is not used):
  python skeleton.py <mhr_fit.npz> <out.npz>
Keeps the anatomical joints and drops MHR's helper joints (twist '_proc', end '_null', the world joint), re-linking
each kept joint to its nearest kept ancestor. Writes heads (joint positions), parents, labels."""
import sys, numpy as np
Z = np.load(sys.argv[1]); H, par, lab = Z['heads'], Z['parents'], list(Z['labels'])
keep = [i for i, l in enumerate(lab) if not (l.endswith('_proc') or l.endswith('_null') or l == 'body_world')]
idx = {o: n for n, o in enumerate(keep)}
def up(i):
    p = par[i]
    while p >= 0 and p not in idx: p = par[p]
    return idx.get(p, -1)
np.savez(sys.argv[2], heads=H[keep], parents=np.array([up(i) for i in keep]), labels=np.array([lab[i] for i in keep]))
print('SK %d joints kept of %d' % (len(keep), len(lab)))
