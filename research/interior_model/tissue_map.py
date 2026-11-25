"""Face labels -> tissue types (user direction 2026-11-25: the model predicts tissue; names come later from fitting
Z-Anatomy parts to the predicted field). Same id space as face_v1 (so every downstream script works unchanged):
  0 outside, 1 fat under the skin, 2 other fat, 3 lean (muscle, gland, nerve, vessel, cartilage, mucosa),
  4 air inside the body, 5 bone (dense), plus  organ (id of brain): brain, brainstem, eyes and lenses as one soft mass.
What matters is muscle, fat, bone, air (and later fascia and tendon); organ insides do not (user, 2026-11-25).
Named structures become the tissue they are mostly made of. Airway outlines (sinuses, nasal cavity, pharynx, ear
canals, trachea) are counted as air although their walls are not; the oral cavity outline holds the tongue and is lean.
Unknown (255) stays unknown. Used when env TISSUE=1 (face_compress.py, prep_v3.py)."""
import numpy as np
BONE = ['mandible', 'teeth_lower', 'teeth_upper', 'skull', 'hard_palate', 'Cochlea_L', 'Cochlea_R'] + ['vertebrae_C%d' % i for i in range(1, 8)]
AIR = ['sinus_maxillary', 'sinus_frontal', 'nasal_cavity_left', 'nasal_cavity_right', 'nasopharynx', 'oropharynx', 'hypopharynx',
       'auditory_canal_left', 'auditory_canal_right', 'trachea']
EYE = ['eye_left', 'eye_right', 'eye_lens_left', 'eye_lens_right']
BRAIN = ['brain', 'Brainstem']

def table(ids):
    t = np.arange(256, dtype=np.uint8)
    for n, k in ids.items():
        if k < 6: continue
        t[k] = 5 if n in BONE else 4 if n in AIR else ids['brain'] if n in EYE + BRAIN else 3
    t[6] = 3                                                                                   # 'another body structure' -> lean
    return t

def remap(L, ids, _cache={}):
    key = id(ids)
    if key not in _cache: _cache[key] = table(ids)
    return _cache[key][L]
