"""Render one HaN-Seg head's TotalSegmentator labels as 3D surfaces (inner structures only: no skin surface).
args: <case> <out png>"""
import sys, json, numpy as np
sys.path.insert(0, r'E:\code\model-to-vrm\model_to_vrm\common')
from vault import get
from huggingface_hub import hf_hub_download
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
case, out = sys.argv[1], sys.argv[2]; tok = get('huggingface'); repo = 'gardianofthedarkness/hanseg-ct-mr'
L = np.load(hf_hub_download(repo, 'head_v1/%s.npz' % case, repo_type='dataset', token=tok))
fc = json.load(open(hf_hub_download(repo, 'census/face_sizes/face_census.json', repo_type='dataset', token=tok)))
sp = np.array([r for r in fc['rows'] if r['case'] == case][0]['voxel_mm'])
tasks = json.load(open(hf_hub_download(repo, 'head_v1/hanseg_shard_0_of_2.json', repo_type='dataset', token=tok)))['tasks']
GROUPS = [  # (task, class names, colour, alpha)
    ('craniofacial_structures', ['skull'], '#c8c2b4', 0.10), ('craniofacial_structures', ['mandible'], '#d9cfb8', 0.55),
    ('craniofacial_structures', ['teeth_upper', 'teeth_lower'], '#ffffff', 0.9), ('craniofacial_structures', ['sinus_maxillary', 'sinus_frontal'], '#7fb3d5', 0.5),
    ('head_muscles', ['masseter_left', 'masseter_right', 'temporalis_left', 'temporalis_right'], '#c0392b', 0.6),
    ('head_muscles', ['lateral_pterygoid_left', 'lateral_pterygoid_right', 'medial_pterygoid_left', 'medial_pterygoid_right', 'digastric_left', 'digastric_right'], '#e67e22', 0.7),
    ('head_muscles', ['tongue'], '#e8a0a8', 0.7), ('head_glands_cavities', ['eye_left', 'eye_right'], '#f4f6f7', 0.9),
    ('head_glands_cavities', ['optic_nerve_left', 'optic_nerve_right'], '#f1c40f', 1.0),
    ('head_glands_cavities', ['parotid_gland_left', 'parotid_gland_right', 'submandibular_gland_left', 'submandibular_gland_right'], '#27ae60', 0.7),
    ('head_glands_cavities', ['nasal_cavity_left', 'nasal_cavity_right', 'nasopharynx', 'oropharynx', 'hypopharynx'], '#5dade2', 0.35)]
meshes = []
for task, names, col, al in GROUPS:
    inv = {v: int(k) for k, v in tasks[task].items()}; m = np.isin(L[task], [inv[n] for n in names if n in inv])
    if m.sum() < 20: continue
    ii = np.argwhere(m); lo = np.maximum(ii.min(0) - 2, 0); hi = ii.max(0) + 3; sub = m[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
    st = max(1, int(round(1.2 / sp[:2].min()))) if names != ['skull'] else max(1, int(round(2.5 / sp[:2].min())))
    sub = sub[::st, ::st, :]; s2 = sp * np.array([st, st, 1])
    v, f, *_ = marching_cubes(sub.astype(np.float32), 0.5, spacing=tuple(s2)); v += lo * sp
    meshes.append((v, f, col, al, names[0]))
allv = np.concatenate([m[0] for m in meshes]); c = allv.mean(0); r = (allv.max(0) - allv.min(0)).max() / 2
fig = plt.figure(figsize=(15, 5.6), facecolor='#111418')
for k, (el, az, title) in enumerate([(5, -90, 'front'), (5, 0, 'side'), (25, -55, 'oblique')]):
    ax = fig.add_subplot(1, 3, k + 1, projection='3d', facecolor='#111418')
    for v, f, col, al, nm in meshes:
        ax.add_collection3d(Poly3DCollection(v[f], facecolor=col, alpha=al, edgecolor='none', linewidth=0))
    ax.set_xlim(c[0] - r, c[0] + r); ax.set_ylim(c[1] - r, c[1] + r); ax.set_zlim(c[2] - r, c[2] + r); ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=el, azim=az); ax.set_axis_off(); ax.set_title(title, color='w')
fig.suptitle('HaN-Seg %s: TotalSegmentator head labels (bones pale, jaw muscles red/orange, glands green, airways blue, eyes white, optic nerves yellow; skin left out)' % case, color='w', fontsize=10)
plt.tight_layout(); plt.savefig(out, dpi=110, facecolor=fig.get_facecolor()); print('saved', out, len(meshes), 'groups', sp)
