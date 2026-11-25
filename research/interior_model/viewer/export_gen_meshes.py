"""Surfaces of a generated head interior for the frame-check page: one mesh per named structure, grouped.
Writes <viewer>/elf/<tag>.json. args: <interior_data root> <generated npz (flow_sample.py)> <viewer dir> <tag> [draw index=0]"""
import os, sys, json, re, numpy as np
from skimage import measure
root, genf, viewer, tag = sys.argv[1:5]; di = int(sys.argv[5]) if len(sys.argv) > 5 else 0
fi = json.load(open(os.path.join(root, 'face_v1_headframe', 'index.json'))); ids = fi['ids']; lo = np.array(fi['window_lo_mm'])
GROUPS = [('Bones and teeth', r'^(skull|mandible|vertebrae_|teeth_|hard_palate)'), ('Eyes and optic nerves', r'^(eye_|optic_nerve|Glnd_Lacrimal|OpticChiasm)'),
          ('Brain and spinal cord', r'^(brain|Brainstem|SpinalCord|Pituitary|Cochlea)'), ('Muscles, glands, tongue', r'^(masseter|lateral_pterygoid|medial_pterygoid|temporalis|digastric|parotid|submandibular|Glnd_Thyroid|tongue|Lips|BuccalMucosa|soft_palate|Cricopharyngeus|A_Carotid)'),
          ('Airways and cavities', r'^(sinus_|nasal_cavity|nasopharynx|oropharynx|hypopharynx|Cavity_Oral|trachea|Larynx|Glottis|Arytenoid|auditory_canal|Esophagus)')]
P = np.load(genf)['draws'][di]; rng = np.random.default_rng(11); out = []
for name, k in sorted(ids.items(), key=lambda x: x[1]):
    g = next((gn for gn, pat in GROUPS if re.match(pat, name)), None); m = P == k; n = int(m.sum())
    if g is None or n < 200: continue
    ii = np.argwhere(m); a = np.maximum(ii.min(0) - 2, 0); b = ii.max(0) + 3; sub = np.pad(m[a[0]:b[0], a[1]:b[1], a[2]:b[2]], 2).astype(np.float32)
    v, f, _, _ = measure.marching_cubes(sub, 0.5, step_size=3 if n > 150000 else 2 if n > 8000 else 1); v = v - 2 + a + lo + 0.5
    col = {'Bones and teeth': [0.92, 0.9, 0.82], 'Eyes and optic nerves': [1.0, 0.85, 0.2], 'Brain and spinal cord': [0.95, 0.55, 0.65]}.get(g, (rng.random(3) * 0.6 + 0.35).tolist())
    out.append(dict(name=name, group=g, ml=round(n / 1000, 1), colour=col, v=np.round(v).astype(int).ravel().tolist(), f=f.ravel().tolist()))
os.makedirs(os.path.join(viewer, 'elf'), exist_ok=True); json.dump(out, open(os.path.join(viewer, 'elf', tag + '.json'), 'w'))
print(tag, len(out), 'structures', {g: sum(o['ml'] for o in out if o['group'] == g) for g, _ in GROUPS}, 'file MB', round(os.path.getsize(os.path.join(viewer, 'elf', tag + '.json')) / 1e6, 1))
