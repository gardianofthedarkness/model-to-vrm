"""Small test: the detector's face points smoothed into lines, mapped to anatomical terms and atlas bones by a language
model (text only, one call), and checked on the atlas's own head, where the bones under the skin are known.
  E:\\splat\\.venv\\Scripts\\python.exe landmark_lines.py <head dir> <atlas_warp.npz> ask|check
  lines    the 68 face points come in ordered runs (jaw outline 0-16, brows, nose bridge, nose base, eye outlines, lip
           outlines): each run -> a cubic spline through its 3D points, sampled at the surface's median edge spacing and
           put on the skin (nearest surface vertex). A point may slide along its line; the line itself is the feature.
  ask      one question: for each line (described in words, no picture) the anatomical name of what it follows and
           which bone(s) lie directly under it, chosen from the dictionary = the atlas's own bone names found under this
           head's skin; or soft tissue. Cached in lines/answers.json.
  check    truth from the atlas: under each line's samples, the nearest bone (share per bone) and the depth to it.
           Reported per line: share of the line lying over a bone the model named, and the depth where the model said
           bone lies directly under the skin against where it said soft tissue.
(A first try took lines from the surface relief - raised vertices above the Tukey fence. On the atlas head that found
only the ears: its skin is too smooth for bony ridges to stand out, so that route was dropped.)"""
import sys, os, re, json, subprocess, numpy as np
from scipy.interpolate import splprep, splev
from scipy.spatial import cKDTree
D, WP, STEP = sys.argv[1:4]; OD = os.path.join(D, 'lines'); os.makedirs(OD, exist_ok=True)
RUN = r'E:\code\novel-ontology-pi\engine\characters\pipeline\pi_run.py'; PY = r'E:\splat\.venv\Scripts\python.exe'
RUNS = {'jaw outline': (range(0, 17), 'from in front of one ear, down along the edge of the face, under the chin, up to in front of the other ear'),
        'right eyebrow': (range(17, 22), 'along the right eyebrow, outer end to inner end'), 'left eyebrow': (range(22, 27), 'along the left eyebrow, inner end to outer end'),
        'nose bridge': (range(27, 31), 'down the midline of the nose from between the eyes to the tip'), 'nose base': (range(31, 36), 'across the bottom of the nose, from one nostril wing under the tip to the other'),
        'right eye outline': (list(range(36, 42)) + [36], 'around the opening of the right eye along the eyelid edges'), 'left eye outline': (list(range(42, 48)) + [42], 'around the opening of the left eye along the eyelid edges'),
        'outer lip outline': (list(range(48, 60)) + [48], 'around the outer edge of the lips'), 'inner lip outline': (list(range(60, 68)) + [60], 'around the inner edge of the lips where they meet')}
V = np.load(os.path.join(D, 'parts', 'orig_colours.npz'))['V'].astype(np.float64); F = np.load(os.path.join(D, 'parts', 'orig_normals.npz'))['F']; X = np.load(os.path.join(D, 'parts_zoom', 'head', 'points3d.npz'))['X'][:68]
edge = float(np.median(np.linalg.norm(V[F[:, 0]] - V[F[:, 1]], axis=1))); st = cKDTree(V); LN = {}
for nm, (ix, _) in RUNS.items():
    P = X[list(ix)]; P = P[np.isfinite(P).all(1)]; L = float(np.linalg.norm(np.diff(P, axis=0), axis=1).sum()); tck, _ = splprep(P.T, s=0, k=min(3, len(P) - 1))
    Q = np.array(splev(np.linspace(0, 1, max(8, int(L / edge))), tck)).T; LN[nm] = V[st.query(Q)[1]]
W = np.load(WP); V0 = W['V0'].astype(np.float64); OB = W['obj']; names = [str(x) for x in W['names']]; sysn = np.array([str(x) for x in W['system']]); isb = sysn[OB] == 'bone'; bob = OB[isb]
VB = np.load(sys.argv[4], allow_pickle=True)['V_rigid'].astype(np.float64) if len(sys.argv) > 4 else V0; bt = cKDTree(VB[isb])   # a model: its own fitted bones (same atlas vertices, placed in the model)
hv = os.path.join(D, 'parts_zoom', 'head_vids.npy'); VH = V[np.load(hv)] if os.path.exists(hv) else V   # the head's own surface, when the mesh is a whole body
base = lambda n: re.sub(r'\.[lr]$', '', n); VOC = sorted({base(names[b]) for b in np.unique(bob[bt.query(VH)[1]])})   # the dictionary: atlas bones found under this head's skin
if STEP == 'ask':
    Q = ('A face landmark detector gives ordered runs of points on the skin of a human face. Each run, joined up, is a line on the skin. For each line below, give the anatomical name of the '
         'structure it follows and the bone or bones lying directly under it, in order along the line. Choose bones ONLY from this list of the anatomical model own names: %s. If no bone lies '
         'close under the skin along a line (lips, eyelids), say so. Lines: %s. Answer with JSON only: {"lines": [{"line": "name as given", "anatomical_name": "...", "bones": ["names from the list, '
         'in order along the line"], "bone_directly_under_skin": true or false, "why": "one sentence"}]}') % ('; '.join(VOC), '; '.join('%s - %s' % (k, v[1]) for k, v in RUNS.items()))
    r = subprocess.run([PY, RUN, '--no-session', '-p', Q], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=900); m = re.search(r'\{.*\}', r.stdout, re.S)
    if not m: sys.exit('LL no answer: ' + (r.stdout[-300:] + ' / ' + r.stderr[-300:]))
    A = json.loads(m.group(0)); json.dump({'question': Q, 'answer': A}, open(os.path.join(OD, 'answers.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    for a in A['lines']: print('LL %-18s %-44s bones %s; under skin: %s' % (a['line'], str(a['anatomical_name'])[:44], a['bones'], a['bone_directly_under_skin']), flush=True)
if STEP == 'check':
    A = {a['line']: a for a in json.load(open(os.path.join(OD, 'answers.json'), encoding='utf-8'))['answer']['lines']}; rows = []; dall = bt.query(VH)[0]
    for nm, P in LN.items():
        d, k = bt.query(P); nb = np.array([base(names[b]) for b in bob[k]]); u, c = np.unique(nb, return_counts=True); o = np.argsort(-c); a = A.get(nm, {}); said = [str(b) for b in a.get('bones', [])]
        cov = float(np.isin(np.char.lower(nb), [s.lower() for s in said]).mean()); extra = [s for s in said if s.lower() not in [x.lower() for x in u]]
        rows.append(dict(line=nm, anatomical_name=a.get('anatomical_name'), said=said, under_skin=a.get('bone_directly_under_skin'), true={u[j]: float(c[j] / c.sum()) for j in o[:4]}, share_over_named_bones=cov,
                         named_but_absent=extra, depth_median_mm=1000 * float(np.median(d)), depth_p90_mm=1000 * float(np.percentile(d, 90))))
        print('LL %-18s -> %s' % (nm, str(a.get('anatomical_name'))[:60])); print('      said %s (under skin: %s)' % (said, a.get('bone_directly_under_skin')))
        print('      atlas %s; depth median %.1f mm; line over named bones %.0f%%; named but not under the line: %s' % (', '.join('%s %.0f%%' % (u[j], 100 * c[j] / c.sum()) for j in o[:4]), rows[-1]['depth_median_mm'], 100 * cov, extra))
    yes = [r['depth_median_mm'] for r in rows if r['under_skin'] is True]; no = [r['depth_median_mm'] for r in rows if r['under_skin'] is False]
    print('LL summary: %d lines; mean share of a line over its named bones %.0f%%; depth where the model said bone under the skin: %s mm (%d lines), where it said soft: %s mm (%d lines); whole head median %.1f mm' % (
        len(rows), 100 * np.mean([r['share_over_named_bones'] for r in rows]), '%.1f' % np.median(yes) if yes else '-', len(yes), '%.1f' % np.median(no) if no else '-', len(no), 1000 * np.median(dall)))
    json.dump(rows, open(os.path.join(OD, 'check.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False); np.savez(os.path.join(OD, 'lines.npz'), **{k.replace(' ', '_'): v for k, v in LN.items()})
