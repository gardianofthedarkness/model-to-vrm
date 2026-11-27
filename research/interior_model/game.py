"""Stage 1 of the cascade (plan v8 14.11-14.12, game branch): named physics compartments from the face label set
(face_v1_headframe: 64 named structures over basic tissue classes 0-6), boundaries smoothed to about 4 mm, plus the hair
and cloth augmentation of the shell. Labels stay exact (no smoothing). Used by field_exp / field_prep / field_flow / ae_check when CLASSES=game.
Compartments (physics units: one material and one set of attachments each): rigid parts (skull, mandible, teeth, spine, other bone), the eyes, the brain, chewing muscles per side
(masseter, temporalis, both pterygoids), the tongue, the airway (sinuses, nasal cavity, pharynx, trachea, ear canals,
air), the fat layer under the skin, and all other soft tissue."""
import numpy as np, torch, torch.nn.functional as F_
GNAMES = ['outside', 'fat_layer', 'skull', 'mandible', 'teeth', 'eyes', 'brain', 'chew_left', 'chew_right', 'tongue', 'airway', 'spine', 'other_bone', 'other_soft']
GNC = len(GNAMES); UNK = 255
_SRC = {  # face_v1_headframe label ids -> compartment
    'fat_layer': [1], 'other_soft': [2, 3, 6] + list(range(20, 26)) + [33] + list(range(44, 49)) + [50, 51, 52, 53, 54, 55, 56, 57, 58, 60, 61],
    'airway': [4, 13, 14, 26, 27, 28, 29, 30, 31, 32, 59, 66], 'other_bone': [5],
    'skull': [12, 34], 'mandible': [10], 'teeth': [11, 15], 'eyes': [16, 17, 18, 19], 'brain': [49, 62, 63, 65],
    'chew_left': [36, 38, 40, 42], 'chew_right': [35, 37, 39, 41], 'tongue': [43], 'spine': [64] + list(range(67, 74))}
LUT = np.full(256, GNAMES.index('other_soft'), np.uint8); LUT[0] = 0; LUT[UNK] = UNK
for name, src in _SRC.items(): LUT[src] = GNAMES.index(name)

def compartments(labels):
    return LUT[labels]

def load_game_head(path):
    """face_v1_headframe npz -> the keys the field pipeline uses: cls (compartments, 255 unknown), unk, outside, hard (= cls)"""
    c = compartments(np.load(path)['labels']); return dict(cls=c, hard=c, unk=c == UNK, outside=c == 0, shape=c.shape)

def game_targets(cls):
    """(B, X, Y, Z) compartment ids (255 unknown) -> (B, GNC, X, Y, Z) exact one-hot (unknown voxels all zero; losses mask
    them). No smoothing at the data level (user, 2026-11-27): coarseness comes from the autoencoder's token size."""
    c = cls.long(); known = (c != UNK).float()
    return F_.one_hot(c.clamp(max=GNC - 1), GNC).permute(0, 4, 1, 2, 3).float() * known[:, None]

def assemble(out, ts_repo='gardianofthedarkness/totalsegmentator-ct-v201', hs_repo='gardianofthedarkness/hanseg-ct-mr', limit=0):
    """the 103 named-label heads (TotalSegmentator 61 + HaN-Seg 42, F1's set) in one folder with index.json; splits as F1
    (TotalSegmentator: its own; HaN-Seg: i % 5 test / val / train, as every earlier run)"""
    import os, json, shutil
    from huggingface_hub import hf_hub_download, snapshot_download
    os.makedirs(out, exist_ok=True)
    fi = json.load(open(hf_hub_download(ts_repo, 'face_v1_headframe/index.json', repo_type='dataset')))
    ts = sorted(c for c, v in fi['heads'].items() if v['source'] == 'ts'); hs = sorted(c for c, v in fi['heads'].items() if v['source'] == 'hanseg')
    if limit: ts, hs = ts[:limit], hs[:limit]
    d1 = snapshot_download(ts_repo, repo_type='dataset', allow_patterns=['face_v1_headframe/%s.npz' % c for c in ts], local_dir='/data/g_ts', max_workers=16)
    d2 = snapshot_download(hs_repo, repo_type='dataset', allow_patterns=['face_v1_headframe/%s.npz' % c for c in hs], local_dir='/data/g_hs', max_workers=16)
    heads = {}
    for c in ts:
        shutil.copy(os.path.join(d1, 'face_v1_headframe', c + '.npz'), os.path.join(out, c + '.npz')); heads[c] = dict(source='ts', split=fi['heads'][c].get('split') or 'train')
    for i, c in enumerate(hs):
        shutil.copy(os.path.join(d2, 'face_v1_headframe', c + '.npz'), os.path.join(out, c + '.npz')); heads[c] = dict(source='hanseg', split='test' if i % 5 == 0 else 'val' if i % 5 == 1 else 'train')
    json.dump(dict(heads=heads, names=GNAMES, window_lo_mm=fi['window_lo_mm'], window_hi_mm=fi['window_hi_mm'], voxel_mm=1.0), open(os.path.join(out, 'index.json'), 'w'), indent=1)
    return heads

def hair_shell(outside, unk, rng, max_ml=2500.0, origin_vox=(108, 168, 144)):
    """augmented 'outside' for the SHELL only: extra outer volume on the scalp, the back of the head and the neck, as hair or
    cloth would add (the elf: 9.0 L against 6.0-7.4 L for real heads). Interior targets are unchanged (that space stays
    'outside'). Thickness and coverage random; capped at max_ml."""
    from scipy import ndimage as ndi
    body = ~outside & ~unk; d = ndi.distance_transform_edt(~body)                       # mm from the body
    X, Y, Z = np.ogrid[:outside.shape[0], :outside.shape[1], :outside.shape[2]]
    region = (Z > origin_vox[2] - rng.uniform(0, 40)) | (Y < origin_vox[1] - rng.uniform(-20, 20))   # top of the head, or behind the ears
    noise = ndi.gaussian_filter(rng.standard_normal(outside.shape).astype(np.float32), 12.0); cover = noise > np.quantile(noise, rng.uniform(0.0, 0.6))
    thick = rng.uniform(4, 30); extra = outside & ~unk & region & cover & (d > 0) & (d <= thick)
    if extra.sum() > max_ml * 1000:                                                     # keep the closest voxels up to the cap
        lim = np.quantile(d[extra], max_ml * 1000 / extra.sum()); extra &= d <= lim
    return extra
