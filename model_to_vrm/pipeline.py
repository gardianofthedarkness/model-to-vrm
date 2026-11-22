"""One model in, everything so far out - no hand steps:
  python pipeline.py <model.glb> <out dir> [--anatomy] [--face] [--from STEP] [--only STEP] [--step STEP] [--list]
Each step is skipped when its outputs already exist (so a rerun resumes); --from/--only force steps to run again;
--step runs just that one step, and only if its outputs are missing (what the notebook cells use).
Logs: <out>/logs/<step>.log. Tools from the environment (defaults are this machine's):
  BLENDER, PY (main venv: SAM 3, RTMW, numpy), PY_BODY (MHR / pymomentum venv), MHR_ASSETS, LLM_MAX_CALLS (budget guard:
  a per-item LLM step refuses to start when it would make more calls than this; default 200).
Stages (out dir layout the step scripts expect):
  body/     8 renders + cameras, per-view visibility, RTMW keypoints triangulated, MHR fit, stick skeleton
  parts/    per-vertex texture colour, normals, triangles (import order)
  sam_body/ SAM 3 masks voted per vertex -> shares, top concept; LLM names each concept's material; the vision LLM
            labels each segment (the model's own parts, or colour pieces) flesh/cloth/hair/prop - this wins over the vote
  contacts/ normal rays -> contact patches -> split by material pair -> Blender cards -> LLM: real contact or fold
            (parallel), LLM flags turned into material fixes
  anatomy/  (ct_fit) Z-Anatomy atlas cross-sections fitted to all 16 body segments, drawn inside the model
  flesh/    materials re-spread with the fixes, specks/holes/borders cleaned -> contacts/material.npz (the skin/cloth
            label the CT slice fit uses; the flesh islands it also writes are no longer used downstream)
  geo/      (--face) bare-geometry head views -> SAM 3 -> face organs
  anatomy_dense/, parts_zoom/, lines/, loose/   (--anatomy) the atlas fitted inside the model: dense slice fit (Hugging Face
            jobs), atlas warp, skeleton fit, named points and joints (LLM), close-ups of head, hands and feet with their
            keypoints, toes and eyes (SAM 3), skull placed and bent (CT tissue depths, Jev's philtrum box), hand and foot
            bones, whole-body assembly, loose clothing and hair (Jev's boxes), muscles carried, kept inside, set to depth
            and repaired -> anatomy_dense/muscle_repair.npz and result.jpg"""
import sys, os, json, time, subprocess, shutil
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
BL = os.environ.get('BLENDER', r'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe')
PY = os.environ.get('PY', r'E:\splat\.venv\Scripts\python.exe'); PYB = os.environ.get('PY_BODY', r'E:\splat\.venv-body\Scripts\python.exe')
MHR = os.environ.get('MHR_ASSETS', r'E:\figures\models\mhr\assets'); BUDGET = int(os.environ.get('LLM_MAX_CALLS', '200'))
args = [a for a in sys.argv[1:] if not a.startswith('--')]; flags = sys.argv[1:]
def opt(name):
    return flags[flags.index(name) + 1] if name in flags else None
if '--list' not in flags and len(args) < 2: sys.exit(__doc__)
M, R = (os.path.abspath(args[0]), os.path.abspath(args[1])) if len(args) >= 2 else ('model.glb', 'out')
P = lambda *p: os.path.join(R, *p); S = lambda *p: os.path.join(HERE, *p); A = lambda *p: os.path.join(ROOT, 'agents', *p)
def bl(script, *a): return [BL, '-b', '-P', script, '--'] + list(a)
def n_subs(): return len(json.load(open(P('contacts', 'sub.json'))))
STEPS = [  # name, outputs, command, optional guard (returns a reason to stop, or None)
 ('views',     [P('body', 'cameras.json')],              bl(S('rig', 'body_views.py'), M, P('body'))),
 ('vis',       [P('body', 'vis.npz')],                   bl(S('parts', 'views_vis.py'), M, P('body'))),
 ('colours',   [P('parts', 'orig_colours.npz')],         bl(S('parts', 'part_colours.py'), M, P('parts', 'orig_colours.npz'))),
 ('normals',   [P('parts', 'orig_normals.npz')],         bl(S('parts', 'part_normals.py'), M, P('parts', 'orig_normals.npz'))),
 ('joints',    [P('body', 'joints3d.npz')],              [PY, S('rig', 'joints3d.py'), P('body'), 'tex']),
 ('mhr',       [P('body', 'mhr_fit.npz')],               [PYB, S('rig', 'fit_mhr.py'), M, P('body', 'joints3d.npz'), MHR, P('body', 'mhr_fit.npz')]),
 ('skeleton',  [P('body', 'skeleton.npz')],              [PYB, S('rig', 'skeleton.py'), P('body', 'mhr_fit.npz'), P('body', 'skeleton.npz')]),
 ('sticks',    [P('body', 'sticks_front.png')],          bl(S('rig', 'show_sticks.py'), M, P('body', 'skeleton.npz'), P('body', 'sticks'))),
 ('masks',     [P('body', 'masks', 'tex_a315.npz')],     [PY, S('parts', 'sam3_masks.py'), P('body')]),
 ('vote',      [P('sam_body', 'top_concept.npz'), P('sam_body', 'shares.npz')], [PY, S('parts', 'body_vote.py'), P('body'), P('parts', 'orig_colours.npz'), '-', P('sam_body')]),
 ('materials', [P('sam_body', 'materials.json')],        [PY, A('concept_material.py'), P('sam_body', 'shares.npz'), P('sam_body', 'materials.json')]),
 ('seglabel',  [P('sam_body', 'segment_material.npz')],  [PY, A('segment_label.py'), R]),
 ('rays',      [P('contacts', 'rays.npz')],              bl(S('parts', 'contacts_find.py'), M, P('contacts', 'rays.npz'))),
 ('patches',   [P('contacts', 'patches.npz')],           [PY, S('parts', 'contacts_patches.py'), P('contacts', 'rays.npz'), P('parts', 'orig_colours.npz'), P('parts', 'orig_normals.npz'), P('body'), P('body', 'skeleton.npz'), P('contacts'), P('sam_body', 'shares.npz')]),
 ('combine',   [P('contacts', 'combined.npz')],          [PY, S('parts', 'contacts_combine.py'), R]),
 ('split',     [P('contacts', 'sub.json')],              [PY, S('parts', 'contacts_split.py'), R]),
 ('subrender', [P('contacts', 'sub_renders', 'done')],   bl(S('parts', 'show_subcontacts.py'), M, P('contacts'))),
 ('cards',     [P('contacts', 'sub_cards', 'done')],     [PY, S('parts', 'contacts_split.py'), R, 'cards']),
 ('sublabel',  [P('contacts', 'sub_labels.json')],       [PY, A('subcontact_label.py'), P('contacts')],
               lambda: ('%d sub-contacts > LLM_MAX_CALLS %d' % (n_subs(), BUDGET)) if n_subs() > BUDGET else None),
 ('matfix',    [P('contacts', 'sub_fix.json')],          [PY, A('material_fix.py'), P('contacts')]),
 ('flesh',     [P('flesh', 'islands.npz'), P('contacts', 'material.npz')],              [PY, S('parts', 'flesh_islands.py'), R]),
 ('ct_fit',    [P('anatomy', 'ct_fit.json'), P('anatomy', 'slices3d_body_front.png')], [PY, S('ct_fit_all.py'), M, R]),
 ('show_contacts', [P('contacts', 'sub_final', 'all_front.png')], bl(S('parts', 'show_subcontact_final.py'), M, P('contacts'))),
]
FACE = [
 ('geo_views', [P('geo', 'cameras.json')],               bl(S('parts', 'geo_views.py'), M, P('geo'))),
 ('geo_masks', [P('geo', 'masks')],                      [PY, S('parts', 'sam3_masks.py'), P('geo')]),
 ('geo_vote',  [P('geo', 'organs.npz')],                 [PY, S('parts', 'geo_vote.py'), P('geo')]),
 ('geo_paint', [P('geo', 'organs_face_a+00_e00.png')],                 bl(S('parts', 'geo_paint.py'), M, P('geo', 'organs.npz'), P('geo'), P('geo', 'organs.png'))),
]
if '--face' in flags: STEPS += FACE
# --- anatomy: bones, skull, muscles inside the model (--anatomy). Needs the one-time atlas data (docs/anatomy-sources.md):
#   ATLAS          the exported Z-Anatomy atlas (za_export.py)                 atlas.npz
#   ATLAS_PLANES   its dense slice planes (hf_fit_jobs.py slice)               and their name in the jobs repo: ATLAS_SET
#   ATLAS_HEAD     the atlas head with its own views, face points and named lines
#   CT_DEPTH       the CT tissue-depth statistics (face_skull_stats.py)
ATLAS = os.environ.get('ATLAS', r'E:\figures\models\zanatomy\export\atlas.npz'); PLANES = os.environ.get('ATLAS_PLANES', r'E:\figures\anatomy_dense_planes'); ASET = os.environ.get('ATLAS_SET', 'planes1')
AHEAD = os.environ.get('ATLAS_HEAD', r'E:\figures\atlas_head'); CTD = os.environ.get('CT_DEPTH', r'E:\figures\models\tissue_depth_ct')
FD = P('anatomy_dense'); F = lambda *p: os.path.join(FD, *p); AN = S('anatomy'); AGD = os.path.join(ROOT, 'agents'); Z = lambda *p: P('parts_zoom', *p)
an = lambda script, *a: [PY, '-u', '-W', 'ignore', os.path.join(AN, script)] + list(a)
ag = lambda script, *a: [PY, '-W', 'ignore', os.path.join(AGD, script)] + list(a)
TOE_VIEWS = 'tex_e+90_a000,tex_e+45_a000,tex_e+45_a060,tex_e+45_a120,tex_e+45_a180,tex_e+45_a240,tex_e+45_a300'      # the views from above (part_views.py), where toes show
COPY = 'import shutil, sys, os; os.makedirs(os.path.dirname(sys.argv[2]), exist_ok=True); shutil.copyfile(sys.argv[1], sys.argv[2])'
ANAT = [
 ('kp_frames',    [P('body', 'frames.npz')],             an('kp_frames.py', R), None, dict(cwd=AN)),
 ('slice_fit',    [F('trunk_fit.npz'), F('head_fit.npz')], an('hf_fit_jobs.py', R, '--atlas-set=' + ASET, FD), None, dict(cwd=AN)),      # Hugging Face CPU jobs, one per body segment
 ('atlas_warp',   [F('atlas_warp.npz')],                 an('atlas_warp.py', ATLAS, PLANES, R, FD), None, dict(cwd=AN)),
 ('skeleton_fit', [F('skeleton_fit.npz')],               an('skeleton_fit.py', F('atlas_warp.npz'), R), None, dict(cwd=AN)),
 ('part_boxes',   [Z('boxes.json')],                     an('part_boxes.py', FD, R), None, dict(cwd=AN)),
] + [('zoom_views_' + r, [Z(r, 'cameras.json')],         bl(os.path.join(AN, 'part_views.py'), M, Z(), r)) for r in ('head', 'hand_r', 'hand_l', 'foot_r', 'foot_l')] + [
 ('zoom_points_' + r, [Z(r, 'points3d.npz')],            an('zoom_joints3d.py', Z(), r), None, dict(cwd=AN)) for r in ('hand_r', 'hand_l', 'head')] + [
 ('line_names',   [P('lines', 'answers.json')],          [PY, '-c', COPY, os.path.join(AHEAD, 'lines', 'answers.json'), P('lines', 'answers.json')]),   # the face lines were named once, on the atlas head (agents/landmark_lines.py ask)
 ('line_check',   [P('lines', 'check.json')],            ag('landmark_lines.py', R, F('atlas_warp.npz'), 'check', F('skeleton_fit.npz')), None, dict(cwd=AGD)),
 ('point_ask',    [P('lines', 'points_answer.json')],    ag('point_names.py', R, F('atlas_warp.npz'), F('skeleton_fit.npz'), 'ask'), None, dict(cwd=AGD)),          # 1 LLM call
 ('point_names',  [P('lines', 'points_named.json'), F('bone_gaps.npz')], ag('point_names.py', R, F('atlas_warp.npz'), F('skeleton_fit.npz'), 'check'), None, dict(cwd=AGD)),
 ('joint_ask',    [F('joints_answer.json')],             ag('joint_names.py', FD, F('atlas_warp.npz'), 'ask'), None, dict(cwd=AGD)),                                  # 1 LLM call
 ('joint_names',  [F('joints_named.json')],              ag('joint_names.py', FD, F('atlas_warp.npz'), 'check'), None, dict(cwd=AGD)),
] + [('toe_masks_' + r, [Z(r, 'masks', 'tex_e+45_a300.npz')], [PY, S('parts', 'part_concept_masks.py'), Z(), r, TOE_VIEWS, 'toe']) for r in ('foot_r', 'foot_l')] + [
 ('toe_lines_' + r, [Z(r, 'toes.npz')],                  an('toe_lines.py', R, r), None, dict(cwd=AN)) for r in ('foot_r', 'foot_l')] + [
 ('eye_masks',    [Z('head', 'masks', 'tex_e+00_a000.npz')],  [PY, S('parts', 'part_concept_masks.py'), Z(), 'head', 'tex_e+00_a000', 'eye,iris,eyeball']),
 ('eye_masks_clay', [Z('head', 'masks', 'clay_e+00_a000.npz')], [PY, S('parts', 'part_concept_masks.py'), Z(), 'head', 'clay_e+00_a000', 'eye,eyeball,eyelid']),
 ('eye_geo',      [Z('eye_geo.json')],                   [PY, S('face', 'eye_geo.py'), R]),
 ('skull_lines',  [F('skull_lines.npz')],                an('skull_line_fit.py', FD, R, AHEAD), None, dict(cwd=AN)),
 ('philtrum',     [Z('philtrum.npz')],                   [PY, S('face', 'philtrum_area.py'), R]),                                                                    # up to 4 Jev calls
 ('skull_bend',   [F('skull_stage2.npz')],               an('skull_stage2.py', FD, R, AHEAD, CTD), None, dict(cwd=AN)),
] + [('bones_' + r, [F(r + '_bones.npz')],               an('part_bone_fit.py', FD, R, r), None, dict(cwd=AN)) for r in ('hand_l', 'hand_r', 'foot_l', 'foot_r')] + [
 ('knee_mask',    [P('body', 'masks', 'clay_a000.npz')], [PY, S('parts', 'part_concept_masks.py'), R, 'body', 'clay_a000', 'knee,kneecap']),
 ('body_bones',   [F('body_assembly.npz')],              an('body_assembly.py', FD, R, ATLAS), None, dict(cwd=AN)),
 ('loose_names',  [P('loose', 'names.json')],            [PY, S('parts', 'loose_names.py'), R]),                                                                     # 8 Jev calls
 ('loose_vote',   [P('loose', 'loose.npz')],             [PY, S('parts', 'loose_vote.py'), R]),
 ('muscle_carry', [F('muscle_carry.npz')],               an('muscle_carry.py', FD, R), None, dict(cwd=AN)),
 ('muscle_inside', [F('muscle_inside.npz')],             an('muscle_inside.py', FD, R), None, dict(cwd=AN)),
 ('muscle_depth', [F('muscle_depth.npz')],               an('muscle_depth.py', FD, R), None, dict(cwd=AN)),
 ('repair',       [F('muscle_repair.npz')],              an('muscle_inside.py', FD, R, 'muscle_depth', 'muscle_repair', 'skin', CTD), None, dict(cwd=AN)),
 ('show_result',  [F('result.jpg')],                     bl(os.path.join(AN, 'show_meshes.py'), M, F('result'), F('muscle_repair.npz') + ':V:F_bone:0.9,0.88,0.8,1', F('muscle_repair.npz') + ':V:F_muscle:0.75,0.2,0.18,1')),
]
if '--anatomy' in flags:
    STEPS = [st for st in STEPS if st[0] != 'ct_fit']             # the dense slice fit (slice_fit) replaces the coarse local one
    STEPS = [((st[0], st[1], st[2][:3] + ['16'], None, dict(env=dict(LABEL_JUDGE='pi'), cwd=AGD)) if st[0] == 'seglabel' else st) for st in STEPS] + ANAT   # the labelling as it was run for the reference figure: Pi judge, 16 workers
# what each step reads (paths in the out dir; 'model.glb' = the input model) - for the node graph and for checking order
READS = {
 'views': ['model.glb'], 'vis': ['model.glb', 'body/cameras.json'], 'colours': ['model.glb'], 'normals': ['model.glb'],
 'joints': ['body/cameras.json'], 'mhr': ['model.glb', 'body/joints3d.npz'], 'skeleton': ['body/mhr_fit.npz'],
 'sticks': ['model.glb', 'body/skeleton.npz'], 'masks': ['body/cameras.json'],
 'vote': ['body/masks/tex_a315.npz', 'body/vis.npz', 'parts/orig_colours.npz'], 'materials': ['sam_body/shares.npz'],
 'seglabel': ['parts/orig_colours.npz', 'parts/orig_normals.npz', 'body/vis.npz', 'body/cameras.json', 'sam_body/shares.npz'],
 'rays': ['model.glb'],
 'patches': ['contacts/rays.npz', 'parts/orig_colours.npz', 'parts/orig_normals.npz', 'body/vis.npz', 'body/skeleton.npz', 'sam_body/shares.npz'],
 'combine': ['sam_body/top_concept.npz', 'sam_body/materials.json', 'parts/orig_normals.npz', 'contacts/patches.npz', 'contacts/rays.npz', 'body/vis.npz'],
 'split': ['contacts/combined.npz', 'parts/orig_normals.npz', 'contacts/patches.npz', 'contacts/rays.npz', 'body/skeleton.npz', 'body/vis.npz'],
 'subrender': ['model.glb', 'contacts/combined.npz', 'contacts/sub.json'], 'cards': ['contacts/sub.json', 'contacts/sub_renders/done'],
 'sublabel': ['contacts/sub.json', 'contacts/sub_cards/done'], 'matfix': ['contacts/sub.json', 'contacts/sub_labels.json'],
 'flesh': ['sam_body/top_concept.npz', 'sam_body/materials.json', 'sam_body/segment_material.npz', 'contacts/combined.npz', 'contacts/sub.json', 'contacts/sub_labels.json', 'contacts/sub_fix.json', 'body/skeleton.npz'],
 'ct_fit': ['model.glb', 'parts/orig_colours.npz', 'parts/orig_normals.npz', 'contacts/material.npz', 'body/skeleton.npz'],
 'show_contacts': ['model.glb', 'contacts/combined.npz', 'contacts/sub.json', 'contacts/sub_labels.json'],
 'geo_views': ['model.glb'], 'geo_masks': ['geo/cameras.json'], 'geo_vote': ['geo/masks'], 'geo_paint': ['model.glb', 'geo/organs.npz'],
}
if '--list' in flags:
    for s in STEPS: print('%-14s -> %s' % (s[0], ', '.join(os.path.relpath(o, R) for o in s[1])))
    sys.exit()
if '--graph' in flags:                                            # JSON for the web app's node view
    import json as _j
    print(_j.dumps([{'name': s[0], 'writes': [os.path.relpath(o, R).replace('\\', '/') for o in s[1]], 'reads': READS.get(s[0], []),
                     'tool': 'blender' if s[2] and s[2][0] == BL else 'llm' if s[2] and any('agents' in str(x) for x in s[2]) else 'python'} for s in STEPS]))
    sys.exit()
names = [s[0] for s in STEPS]; frm, only, one = opt('--from'), opt('--only'), opt('--step')
for x in (frm, only, one):
    if x and x not in names: sys.exit('no step %s; steps: %s' % (x, ' '.join(names)))
os.makedirs(P('logs'), exist_ok=True)
def run(cmd, log, how=None):
    how = how or {}
    with open(log, 'a', encoding='utf-8') as f:
        f.write('\n$ %s\n' % ' '.join(cmd)); f.flush()
        return subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=how.get('cwd', ROOT), env=dict(os.environ, **how.get('env', {}))).returncode
t_all = time.time()
for i, st in enumerate(STEPS):
    name, outs, cmd = st[:3]; guard = st[3] if len(st) > 3 else None
    force = (only == name) or (frm is not None and i >= names.index(frm))
    if (only and only != name) or (one and one != name): continue
    if not force and all(os.path.exists(o) for o in outs): print('[skip] %-13s outputs exist' % name); continue
    for o in outs: os.makedirs(os.path.dirname(o), exist_ok=True)
    if guard and guard(): sys.exit('[stop] %s: %s' % (name, guard()))
    log = P('logs', name + '.log'); t = time.time(); print('[run ] %-13s ...' % name, flush=True)
    rc = run(cmd, log, st[4] if len(st) > 4 else None)
    if rc == 0 and name == 'show_result':                         # the four views side by side
        from PIL import Image
        im = [Image.open(F('result_%s.png' % v)).convert('RGB') for v in ('front', 'back', 'side', 'three_quarter')]; w, h = im[0].size; sheet = Image.new('RGB', (2 * w, 2 * h), 'white')
        for k, a in enumerate(im): sheet.paste(a, ((k % 2) * w, (k // 2) * h))
        sheet.save(outs[0], quality=88)
    if rc == 0 and name in ('subrender', 'cards'): open(outs[0], 'w').write('ok')
    missing = [o for o in outs if not os.path.exists(o)]
    if rc or missing: sys.exit('[FAIL] %s (exit %s, missing %s) - see %s' % (name, rc, [os.path.relpath(m, R) for m in missing], log))
    print('[ ok ] %-13s %.0f s' % (name, time.time() - t), flush=True)
print('pipeline done in %.0f min: %s' % ((time.time() - t_all) / 60, R))
