"""Pipeline step: CT-like atlas slices fitted to every body segment of a model, then drawn inside it from Blender:
  python ct_fit_all.py <model.glb> <out dir>          -> <out>/anatomy/<segment>_fit.npz/.json/_fit_sheet.png,
                                                          <out>/anatomy/ct_fit.json (per segment: slices, residuals),
                                                          <out>/anatomy/slices3d_body_<view>.png and close-ups
Uses the anatomy scripts handed over from the test model R research (model_to_vrm/anatomy: segments.ORDER, za_slices.py,
fit_slices.py, show_slices3d.py). Atlas slices are atlas-only: cut once into ATLAS_SLICES and reused for every model.
Inputs from earlier steps: parts/orig_colours.npz, parts/orig_normals.npz, contacts/material.npz, body/skeleton.npz.
A segment that fails is reported and skipped; the step fails only if no segment fits."""
import sys, os, json, subprocess, time
HERE = os.path.dirname(os.path.abspath(__file__)); AN = os.path.join(HERE, 'anatomy'); sys.path.insert(0, AN)
import segments
M, R = sys.argv[1:3]; od = os.path.join(R, 'anatomy'); os.makedirs(od, exist_ok=True)
PY = os.environ.get('PY', r'E:\splat\.venv\Scripts\python.exe'); BL = os.environ.get('BLENDER', r'C:\Program Files\Blender Foundation\Blender 5.3\blender.exe')
ATLAS = os.environ.get('ATLAS', r'E:\figures\models\zanatomy\export\atlas.npz'); SL = os.environ.get('ATLAS_SLICES', r'E:\figures\anatomy_test')
def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='replace', cwd=AN)
    return r.returncode, (r.stdout + r.stderr)[-600:]
summary, fits = {}, []
for seg in segments.ORDER:
    t = time.time(); sp = os.path.join(SL, seg + '.npz'); fp = os.path.join(od, seg + '_fit.npz')
    if not os.path.exists(sp):
        rc, out = run([PY, '-W', 'ignore', os.path.join(AN, 'za_slices.py'), ATLAS, seg, SL])
        if rc: summary[seg] = {'error': 'atlas slices: ' + out}; print('CT %-10s atlas slices failed' % seg, flush=True); continue
    if not os.path.exists(fp):
        rc, out = run([PY, '-W', 'ignore', os.path.join(AN, 'fit_slices.py'), R, seg, sp, od])
        if rc or not os.path.exists(fp): summary[seg] = {'error': 'fit: ' + out}; print('CT %-10s fit failed: %s' % (seg, out[-200:].replace('\n', ' ')), flush=True); continue
    js = os.path.join(od, seg + '_fit.json'); info = json.load(open(js)) if os.path.exists(js) else {}
    rv = info.get('median_flesh_residual_m'); rv = [x for x in (rv if isinstance(rv, list) else [rv]) if x is not None]
    summary[seg] = {'fit': os.path.basename(fp), 'seconds': round(time.time() - t), 'length_m': info.get('length_m'),
                    'skin_residual_mm_median': round(1000 * sorted(rv)[len(rv) // 2], 2) if rv else None,
                    'slices_seeing_skin': len(rv)}
    fits.append(fp); print('CT %-10s ok (%.0f s)' % (seg, time.time() - t), flush=True)
json.dump(summary, open(os.path.join(od, 'ct_fit.json'), 'w'), indent=1)
if not fits: sys.exit('CT no segment fitted')
rc, out = subprocess.run([BL, '-b', '-P', os.path.join(AN, 'show_slices3d.py'), '--', M, os.path.join(od, 'slices3d')] + fits, capture_output=True, text=True, encoding='utf-8', errors='replace').returncode, ''
print('CT done: %d of %d segments fitted; 3D render %s' % (len(fits), len(segments.ORDER), 'ok' if rc == 0 else 'failed'))
