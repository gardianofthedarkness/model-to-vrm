"""Writes the runnable notebooks (re-run this after changing a step):  python build_notebooks.py
  pipeline.ipynb        one model in -> skeleton, materials, contacts, flesh islands; every step's pictures inline
  anatomy_slices.ipynb  Z-Anatomy atlas -> critical cross-sections -> fitted to every body segment of a model
Open in VS Code, pick the kernel E:\\splat\\.venv\\Scripts\\python.exe, run cells top to bottom. Each step cell only runs the
step when its outputs are missing (set FORCE = True in a cell to redo it); logs go to <out>/logs/<step>.log."""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
def nb(cells):
    return {'cells': [{'cell_type': t, 'metadata': {}, 'source': src.strip('\n').splitlines(True),
                       **({'outputs': [], 'execution_count': None} if t == 'code' else {})} for t, src in cells],
            'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3 (splat .venv)', 'language': 'python'},
                         'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
HELPERS = r'''
import os, sys, glob, subprocess, html, base64
from IPython.display import display, HTML
REPO = r'E:\code\model-to-vrm'
PY = r'E:\splat\.venv\Scripts\python.exe'; PYB = r'E:\splat\.venv-body\Scripts\python.exe'; PYS = r'E:\splat\.venv-sim\Scripts\python.exe'
BLENDER = r'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe'
def sh(cmd, log=None):
    """run a command, print its last lines; stop the notebook on failure"""
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=REPO)
    out = (r.stdout or '') + (r.stderr or '')
    if log: open(log, 'a', encoding='utf-8').write(out)
    print('\n'.join(l for l in out.splitlines()[-15:] if 'it/s]' not in l))
    if r.returncode: raise RuntimeError('failed: %s' % ' '.join(cmd[:4]))
def show(*patterns, width=320):
    """show images side by side (glob patterns), newest render of each file"""
    files = [f for p in patterns for f in sorted(glob.glob(p))]
    if not files: print('no pictures for', patterns); return
    tags = []
    for f in files:
        ext = os.path.splitext(f)[1][1:].replace('jpg', 'jpeg'); b = base64.b64encode(open(f, 'rb').read()).decode()
        tags.append('<figure style="display:inline-block;margin:4px"><img src="data:image/%s;base64,%s" width="%d"><figcaption style="font-size:11px">%s</figcaption></figure>' % (ext, b, width, html.escape(os.path.basename(f))))
    display(HTML(''.join(tags)))
'''
P = [('markdown', '''
# Model to VRM: pipeline
One textured model in. Out: renders, keypoints, MHR body fit and stick skeleton, SAM 3 materials per vertex, contacts
labelled by the LLM (real contact or fold), and the flesh cut into islands. Every step shows its pictures below it.

**Run:** VS Code, kernel `E:\\splat\\.venv\\Scripts\\python.exe`, cells top to bottom. A step runs only when its outputs
are missing, so re-running is cheap; set `FORCE = True` in a step's cell to redo it. Steps that call the LLM read the
key from the vault and refuse to start if they would make more than `LLM_MAX_CALLS` calls.
'''), ('code', HELPERS + r'''
MODEL = r'E:\figures\model_r\test model R.glb'      # the input model (.glb, textured)
OUT = r'E:\figures\model_r'                   # everything for this model goes here
FACE = True                                 # also run the face-organ steps
os.environ['LLM_MAX_CALLS'] = '200'
def step(name, force=False):
    cmd = [PY, os.path.join(REPO, 'model_to_vrm', 'pipeline.py'), MODEL, OUT] + (['--face'] if FACE else []) + (['--only', name] if force else ['--step', name])
    sh(cmd)
'''), ('markdown', '## 1. Renders around the model (8 directions, textured and bare) and what each camera sees'),
('code', '''FORCE = False
step('views', FORCE); step('vis', FORCE)
show(OUT + r'\\body\\tex_a*.png', width=180); show(OUT + r'\\body\\clay_a*.png', width=180)'''),
('markdown', '## 2. Per-vertex texture colour, normals, triangles'), ('code', "FORCE = False\nstep('colours', FORCE); step('normals', FORCE)"),
('markdown', '## 3. Keypoints (RTMW on every view, triangulated) → MHR body fit → stick skeleton'),
('code', '''FORCE = False
step('joints', FORCE); show(OUT + r'\\body\\joints_check.jpg', width=900)
step('mhr', FORCE); step('skeleton', FORCE); step('sticks', FORCE)
show(OUT + r'\\body\\sticks_*.png', width=240)'''),
('markdown', '## 4. SAM 3 masks → shares per vertex; then the vision LLM labels each segment (the model\'s own parts) skin / cloth / hair / prop'),
('code', '''FORCE = False
step('masks', FORCE); step('vote', FORCE); step('materials', FORCE); step('seglabel', FORCE)
print(open(OUT + r'\\sam_body\\materials.json').read())
show(OUT + r'\\sam_body\\seg_cards\\seg_p[0-5].jpg', width=480)'''),
('markdown', '## 5. Contacts: a ray along every normal → patches → split by material pair → a Blender card each'),
('code', '''FORCE = False
for s in ('rays', 'patches', 'combine', 'split', 'subrender', 'cards'): step(s, FORCE)
show(OUT + r'\\contacts\\sub_cards\\sub_[0-5].jpg', width=480)'''),
('markdown', '## 6. LLM per contact (parallel): real contact or a fold; its material flags → fixes'),
('code', '''FORCE = False
step('sublabel', FORCE); step('matfix', FORCE)
import json; L = json.load(open(OUT + r'\\contacts\\sub_labels.json'))
from collections import Counter; print(Counter(v.get('kind') for v in L.values())); print(open(OUT + r'\\contacts\\sub_fix.json').read())'''),
('markdown', '## 7. Skin/cloth label per vertex: materials re-spread with the LLM fixes, specks and holes cleaned (used by the CT slice fit)'),
('code', '''FORCE = False
step('flesh', FORCE)
step('show_contacts', FORCE)
show(OUT + r'\\contacts\\sub_final\\all_*.png', width=240)'''),
('markdown', '## 8. Face organs from bare geometry (only with FACE = True)'),
('code', '''FORCE = False
if FACE:
    for s in ('geo_views', 'geo_masks', 'geo_vote', 'geo_paint'): step(s, FORCE)
    show(OUT + r'\\geo\\organs_*.png', width=320)''')]
A = [('markdown', '''
# Anatomy slices: CT-like cross-sections from the Z-Anatomy atlas, fitted to a model
Z-Anatomy (CC BY-SA 4.0, BodyParts3D-based) gives bones, muscles, skin and muscle insertions. Each body segment (pelvis,
trunk, neck, head; upper arm, forearm, hand, thigh, calf, foot on both sides; see `model_to_vrm/anatomy/segments.py`) is cut at
its own **critical levels** (peaks and valleys of the skin section's size, off-centre shift and flattening along the
bone, ripples dropped by persistence), then each slice's outline is fitted to the model's own skin
(5 numbers per slice: shift, roll, width, depth); the atlas muscles and bones ride along. At joint zones (hip, shoulder,
crotch) a slice owns only part of the skin ring: slices stop where the ring coverage drops, and the zone between two
segments' end slices is left to a blend.

**Needs:** the model's pipeline run up to step 7 (skeleton, materials, flesh). Kernel `E:\\splat\\.venv\\Scripts\\python.exe`.
'''), ('code', HELPERS + r'''
ATLAS_BLEND = r'E:\figures\models\zanatomy\blend\Startup.blend'
ATLAS = r'E:\figures\models\zanatomy\export'      # atlas.npz goes here
SLICES = r'E:\figures\anatomy_test'               # atlas slices per segment
MODEL_DIR = r'E:\figures\model_r'                   # a model already through pipeline.ipynb
sys.path.insert(0, os.path.join(REPO, 'model_to_vrm', 'anatomy')); from segments import ORDER
SEGMENTS = ORDER                                   # every body segment, both sides
A = lambda *p: os.path.join(REPO, 'model_to_vrm', 'anatomy', *p)
'''), ('markdown', '## 1. Export the atlas once (bones, muscles, skin, insertions → arrays)'),
('code', '''FORCE = False
if FORCE or not os.path.exists(ATLAS + r'\\atlas.npz'):
    sh([BLENDER, '-b', '-Y', ATLAS_BLEND, '-P', A('za_export.py'), '--', ATLAS])
print('atlas ready:', os.path.exists(ATLAS + r'\\atlas.npz'))'''),
('markdown', '## 2. Critical slices of each segment (bone tan, muscles red/purple, skin outline)'),
('code', '''FORCE = False
for sg in SEGMENTS:
    if FORCE or not os.path.exists(os.path.join(SLICES, sg + '.npz')):
        sh([PY, '-W', 'ignore', A('za_slices.py'), ATLAS + r'\\atlas.npz', sg, SLICES])
    show(os.path.join(SLICES, sg + '_sheet.png'), width=900)'''),
('markdown', '## 3. Fit the slices to the model (dots: the model\'s surface — pink skin, blue cloth, green prop; red: fitted outline)'),
('code', '''FORCE = False
for sg in SEGMENTS:
    out = os.path.join(MODEL_DIR, 'anatomy')
    if FORCE or not os.path.exists(os.path.join(out, sg + '_fit.npz')):
        sh([PY, '-W', 'ignore', A('fit_slices.py'), MODEL_DIR, sg, os.path.join(SLICES, sg + '.npz'), out])
    show(os.path.join(out, sg + '_fit_sheet.png'), width=900)'''),
('markdown', '## 3b. All fitted slices inside the model, in Blender (red: fitted skin outline, purple: muscles, tan: bones, blue: segment axis)'),
('code', '''FORCE = False
out = os.path.join(MODEL_DIR, 'anatomy'); pre = os.path.join(out, 'slices3d_all')
fits = [os.path.join(out, sg + '_fit.npz') for sg in SEGMENTS if os.path.exists(os.path.join(out, sg + '_fit.npz'))]
if FORCE or not os.path.exists(pre + '_body_front.png'):
    sh([BLENDER, '-b', '-P', A('show_slices3d.py'), '--', sorted(glob.glob(os.path.join(MODEL_DIR, '*.glb')))[0], pre] + fits)
show(pre + '_body_front.png', pre + '_body_side.png', pre + '_body_back.png', width=300)'''),
('markdown', '## 4. Build the real forearm (atlas bones + muscles warped in, closed skin) and move it with Newton flesh (bend the elbow, twist). Cut: pink skin, yellow sticky layer, red muscle, ivory bone'),
('code', '''FORCE = False; BEND, TWIST = 45, 60          # degrees
out = os.path.join(MODEL_DIR, 'anatomy'); limb = os.path.join(out, 'forearm_r_limb.npz'); sim = os.path.join(out, 'forearm_r_sim.npz')
if FORCE or not os.path.exists(limb):
    sh([PY, '-W', 'ignore', A('limb_build.py'), ATLAS + r'\\atlas.npz', os.path.join(SLICES, 'forearm_r.npz'), os.path.join(out, 'forearm_r_fit.npz'), limb])
if FORCE or not os.path.exists(sim):          # ~7 min on this laptop
    sh([PYS, '-W', 'ignore', A('limb_sim.py'), limb, os.path.join(MODEL_DIR, 'body', 'skeleton.npz'), sim, str(BEND), str(TWIST)])
    sh([BLENDER, '-b', '-P', A('show_limb_sim.py'), '--', sim, os.path.join(out, 'forearm_sim')])
show(os.path.join(out, 'forearm_sim_*_a.png'), width=300); show(os.path.join(out, 'forearm_sim_*_b.png'), width=300)
show(os.path.join(out, 'forearm_sim_cut_*.png'), width=400)''')]
for name, cells in (('pipeline.ipynb', P), ('anatomy_slices.ipynb', A)):
    json.dump(nb(cells), open(os.path.join(HERE, name), 'w', encoding='utf-8'), indent=1); print('wrote', name)
