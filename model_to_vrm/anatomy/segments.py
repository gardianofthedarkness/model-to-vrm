"""Body segments for the CT-like atlas slicing, both in the Z-Anatomy atlas and in a model's stick skeleton (body/skeleton.npz).
Anatomy, not tuned numbers: which bones meet at each end, which named skin regions belong to the segment, which landmark
sets the roll, which skeleton bones own the model's surface.

  atlas side   prox / dist   joint = mean of [(bones A, bones B), ...] facing-vertex centres (see joint())
               dist 'top'    the head: axis continues the neck's, length = the head skin's farthest point along it
               bones         the segment's own bones (drawn ivory, rigid in the sim)
               roll          ('bones', names) centroid | ('skin', regex) centroid | ('front',) the body's front
               skin          regex on the atlas skin-region names (limbs: this side only)
               ext           (lo, hi): let the slices run past the joint to the end of the segment's own skin
                             (heel behind the ankle, toes past the ball)
  model side   j0 / j1       joint names averaged; j1 'top' for the head (length from the eyes, see fit_slices.py)
               roll          ('joints', names) mean | ('front',)
               own           a skeleton bone belongs to the segment when its child joint is listed here
  front        cross(spine up, right shoulder - left shoulder), the same formula in the atlas and the model, so the
               two frames aim the same way whatever the model's pose"""
import re, json, os, numpy as np
from scipy.spatial import cKDTree

def _limbs(s):
    n = lambda *xs: [x + '.' + s for x in xs]; o = lambda *xs: [s + '_' + x for x in xs]
    ordn = ['First', 'Second', 'Third', 'Fourth', 'Fifth']
    carp = n('Scaphoid bone', 'Lunate bone', 'Triquetrum bone', 'Pisiform bone', 'Trapezium bone', 'Trapezoid bone', 'Capitate bone', 'Hamate bone')
    mc = n(*['%s metacarpal bone' % x for x in ordn]); pph = n(*['Proximal phalanx of %s finger of hand' % x.lower() for x in ordn[1:]])
    tars = n('Talus', 'Calcaneus', 'Navicular bone', 'Cuboid bone', 'Medial cuneiform bone', 'Intermediate cuneiform bone', 'Lateral cuneiform bone')
    mt = n(*['%s metatarsal bone' % x for x in ordn]); fph = [p for x in ordn for p in n(*['%s phalanx of %s finger of foot' % (k, x.lower()) for k in ('Proximal', 'Middle', 'Distal')])]
    pft = n(*['Proximal phalanx of %s finger of foot' % x.lower() for x in ordn])
    side = r'.*\.%s$' % s
    return {
     'uparm_' + s: dict(prox=[(n('Scapula'), n('Humerus'))], dist=[(n('Humerus'), n('Radius', 'Ulna'))], bones=n('Humerus'), roll=('front',),
                        skin=r'(region of arm|deltoid|bicipital|elbow|cubital)' + side, ext=(False, False),
                        model=dict(j0=o('uparm'), j1=o('lowarm'), roll=('front',), own=o('lowarm'))),
     'forearm_' + s: dict(prox=[(n('Humerus'), n('Radius', 'Ulna'))], dist=[(n('Radius', 'Ulna'), n('Scaphoid bone', 'Lunate bone', 'Triquetrum bone'))],
                          bones=n('Radius', 'Ulna'), roll=('bones', n('First metacarpal bone')), skin=r'(forearm|elbow|wrist|cubital|olecranon)' + side, ext=(False, False),
                          model=dict(j0=o('lowarm'), j1=o('wrist'), roll=('joints', o('thumb0', 'thumb1')), own=o('wrist_twist', 'wrist'))),
     'hand_' + s: dict(prox=[(n('Radius', 'Ulna'), n('Scaphoid bone', 'Lunate bone', 'Triquetrum bone'))], dist=[(mc[1:], pph)], bones=carp + mc,
                       roll=('bones', n('First metacarpal bone')), skin=r'(palm|dorsum of hand|wrist)' + side, ext=(False, False),
                       model=dict(j0=o('wrist'), j1=o('index1', 'middle1', 'ring1', 'pinky1'), roll=('joints', o('thumb0', 'thumb1')),
                                  own=o('index1', 'middle1', 'ring1', 'pinky0', 'pinky1', 'thumb0', 'thumb1'))),
     'thigh_' + s: dict(prox=[(n('Hip bone'), n('Femur'))], dist=[(n('Femur'), n('Tibia'))], bones=n('Femur'), roll=('bones', n('Patella')),
                        skin=r'(thigh|femoral|knee|popliteal)' + side, ext=(False, False),
                        model=dict(j0=o('upleg'), j1=o('lowleg'), roll=('knee',), own=o('lowleg'))),
     'calf_' + s: dict(prox=[(n('Femur'), n('Tibia'))], dist=[(n('Tibia', 'Fibula'), n('Talus'))], bones=n('Tibia', 'Fibula'), roll=('bones', n('Second metatarsal bone')),
                       skin=r'(region of leg|knee|popliteal|malleol|ankle)' + side, ext=(False, False),
                       model=dict(j0=o('lowleg'), j1=o('talocrural'), roll=('joints', o('ball')), own=o('foot', 'talocrural'))),
     'foot_' + s: dict(prox=[(n('Tibia', 'Fibula'), n('Talus'))], dist=[(mt, pft)], bones=tars + mt + fph, roll=('bones', n('Tibia')),
                       skin=r'(foot|sole|heel|ankle|malleol|hallucial|metatarsal region)' + side, ext=(True, True),
                       model=dict(j0=o('talocrural'), j1=o('ball'), roll=('joints', o('lowleg', 'talocrural')), own=o('subtalar', 'transversetarsal', 'ball')))}

_V = lambda *xs: ['Vertebra ' + x for x in xs]
CENTRE = {
 'pelvis': dict(prox=[(['Hip bone.r'], ['Femur.r']), (['Hip bone.l'], ['Femur.l'])], dist=[(_V('L5'), ['Sacrum'])],
                bones=['Hip bone.r', 'Hip bone.l', 'Sacrum', 'Coccyx'], roll=('front',),
                skin=r'(gluteal region|sacral|hip region|inguinal|hypogastric|urogenital|anal region)', ext=(False, False),   # below the hips the section is only the crotch ring: blend zone
                model=dict(j0=['r_upleg', 'l_upleg'], j1=['c_spine0'], roll=('front',), own=['r_upleg', 'l_upleg', 'c_spine0'])),
 'trunk': dict(prox=[(_V('L5'), ['Sacrum'])], dist=[(_V('C7'), _V('T1'))],
               bones=r'^(Vertebra [TL]\d+|.* rib\.[rl]|Costal cartilage.*|Body of sternum|Manubrium of sternum|Xiphoid process|Clavicle\.[rl]|Scapula\.[rl])$', roll=('front',),
               skin=r'(pectoral|mammary|presternal|epigastric|hypochondriac|umbilic|region of abdomen|lumbar|thorax|scapular|auscultation|infraclavicular|vertebral region)', ext=(False, False),
               model=dict(j0=['c_spine0'], j1=['c_neck'], roll=('front',), own=['c_spine1', 'c_spine2', 'c_spine3', 'r_clavicle', 'l_clavicle', 'c_neck'])),
 'neck': dict(prox=[(_V('C7'), _V('T1'))], dist=[(['Atlas (C1)'], ['Occipital bone'])],
              bones=r'^(Vertebra C\d|Atlas \(C1\)|Axis \(C2\)|Hyoid bone|Thyroid cartilage|Cricoid cartilage)$', roll=('front',),
              skin=r'(region of neck|carotid|muscular triangle|sternocleidomastoid|submandibular|submental|supraclavicular)', ext=(False, False),
              model=dict(j0=['c_neck'], j1=['c_head'], roll=('front',), own=['c_head'])),
 'head': dict(prox=[(['Atlas (C1)'], ['Occipital bone'])], dist='top',
              bones=r'^(Frontal bone|Parietal bone.*|Occipital bone|Temporal bone.*|Sphenoid bone|Ethmoid bone|Mandible|Maxilla.*|Zygomatic bone.*|Nasal bone.*|Lacrimal bone.*|Palatine bone.*|Vomer|.*(incisor|canine|premolar|molar).*)$',
              roll=('skin', r'^Orbital region'),
              skin=r'^(frontal|parietal|temporal|occipital|orbital|infra-orbital|nasal|oral|buccal|zygomatic|mental|parotid|auricular|mastoid|eyebrow|angle of mouth|labial|philtrum|tubercle of upper lip|sulcus|nasolabial|mentolabial|helix|antihelix|tragus|antitragus|concha|cavity of concha|cymba|scapha|lobule|apex of auricle|crura|eminentia|intertragic|fossa antihelica|triangular fossa|anterior notch|posterior auricular)', ext=(False, False),
              model=dict(j0=['c_head'], j1='top', roll=('joints', ['r_eye', 'l_eye']), own=['c_head', 'c_jaw', 'c_teeth', 'c_tongue0', 'c_tongue1', 'c_tongue2', 'c_tongue3', 'c_tongue4', 'r_eye', 'l_eye']))}
SEG = dict(CENTRE)
for _s in ('r', 'l'): SEG.update(_limbs(_s))
ORDER = ['pelvis', 'trunk', 'neck', 'head'] + [p + '_' + s for s in ('r', 'l') for p in ('uparm', 'forearm', 'hand', 'thigh', 'calf', 'foot')]

def otsu(v):
    v = np.sort(np.asarray(v, float)); m = len(v); c = np.cumsum(v); i = np.arange(1, m); m0, m1 = c[:-1] / i, (c[-1] - c[:-1]) / (m - i)
    return v[np.argmax(i * (m - i) * (m0 - m1) ** 2)]

class Atlas:
    """the atlas arrays plus every segment's joints and axis (cached next to atlas.npz)"""
    def __init__(self, path):
        Z = np.load(path); self.V, self.F, self.OB = Z['V'].astype(np.float64), Z['F'], Z['obj']
        self.names = [str(x) for x in Z['names']]; self.sys = [str(x) for x in Z['system']]; self.idx = {n: i for i, n in enumerate(self.names)}
        cp = os.path.join(os.path.dirname(path), 'atlas_axes.json')
        if os.path.exists(cp) and os.path.getmtime(cp) > os.path.getmtime(path) and os.path.getmtime(cp) > os.path.getmtime(__file__):
            self.axes = {k: {a: np.array(b) for a, b in v.items()} for k, v in json.load(open(cp)).items()}
        else:
            self.axes = self._axes(); json.dump({k: {a: b.tolist() for a, b in v.items()} for k, v in self.axes.items()}, open(cp, 'w'), indent=1)
    def verts(self, ns): return self.V[np.isin(self.OB, [self.idx[n] for n in ns])]
    def bone_names(self, spec):
        return [n for n in spec if n in self.idx] if isinstance(spec, list) else [n for n, s in zip(self.names, self.sys) if s == 'bone' and re.search(spec, n)]
    def skin_ids(self, rx): return [i for i, n in enumerate(self.names) if self.sys[i] == 'skin' and re.search(rx, n, re.I) and 'eyelash' not in n.lower()]
    def joint(self, pairs):
        """mean of each pair's facing vertices: each group's vertices closer to the other group than the Otsu split of the log distances"""
        out = []
        for a, b in pairs:
            A, B = self.verts(a), self.verts(b); da, db = cKDTree(B).query(A)[0], cKDTree(A).query(B)[0]
            la, lb = np.log(da + 1e-9), np.log(db + 1e-9); out.append(np.r_[A[la < otsu(la)], B[lb < otsu(lb)]].mean(0))
        return np.mean(out, 0)
    def _axes(self):
        ax = {}
        for k in ORDER:
            d = SEG[k]; J0 = self.joint(d['prox'])
            if d['dist'] == 'top':                                # the head: continue the neck's axis to the head skin's farthest point
                u = ax['neck']['J1'] - ax['neck']['J0']; u /= np.linalg.norm(u)
                P = self.V[np.isin(self.OB, self.skin_ids(d['skin']))]; J1 = J0 + u * float(((P - J0) @ u).max())
            else: J1 = self.joint(d['dist'])
            ax[k] = dict(J0=J0, J1=J1)
        return ax
    def front(self):
        a = self.axes; f = np.cross(a['trunk']['J1'] - a['trunk']['J0'], a['uparm_r']['J0'] - a['uparm_l']['J0']); return f / np.linalg.norm(f)
    def roll_target(self, seg):
        r = SEG[seg]['roll']; J0 = self.axes[seg]['J0']
        if r[0] == 'front': return J0 + self.front()
        if r[0] == 'bones': return self.verts(r[1]).mean(0)
        return self.V[np.isin(self.OB, self.skin_ids(r[1]))].mean(0)
    def nearest_axis(self, P):
        """index into ORDER of each point's nearest segment axis (the atlas's own stick skeleton)"""
        D = []
        for k in ORDER:
            a, b = self.axes[k]['J0'], self.axes[k]['J1']; d = b - a; u = np.clip(((P - a) @ d) / (d @ d), 0, 1); D.append(np.linalg.norm(P - a - u[:, None] * d, axis=1))
        return np.argmin(np.array(D), 0)

def model_frame(seg, H, lab, head_len=None):
    """J0, J1, roll target point, own bone child indices for a model skeleton (heads, labels)"""
    m = SEG[seg]['model']; h = lambda ns: np.mean([H[lab.index(n)] for n in ns], 0)
    J0 = h(m['j0'])
    if m['j1'] == 'top':
        u = h(['c_head']) - h(['c_neck']); u /= np.linalg.norm(u); J1 = J0 + u * head_len
    else: J1 = h(m['j1'])
    r = m['roll']
    if r[0] == 'front':
        f = np.cross(h(['c_neck']) - h(['c_spine0']), h(['r_uparm']) - h(['l_uparm'])); tgt = J0 + f / np.linalg.norm(f)
    elif r[0] == 'knee':                                          # the front of the knee: away from the shin plus along the foot
        s = seg[-1]; ank, ball = h([s + '_talocrural']), h([s + '_ball']); ax = (J1 - J0) / np.linalg.norm(J1 - J0)
        pp = lambda v: v - ax * (v @ ax)
        e = -pp(ank - J1) / np.linalg.norm(ank - J1) + pp(ball - ank) / np.linalg.norm(ball - ank); tgt = J0 + e
    else: tgt = h(r[1])
    return J0, J1, tgt, [lab.index(n) for n in m['own']]
