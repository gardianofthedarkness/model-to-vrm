"""The face lines the skull fits use, as the naming step answered them (agents/landmark_lines.py -> <model dir>/lines/
answers.json): which runs of the 68 detected face points lie on bone directly under the skin, and which structures lie
under each. The point numbers of each run are the published 68-point scheme (iBUG 300-W), the same table the naming
step shows the language model."""
import os, json
SCHEME = {'jaw outline': range(0, 17), 'right eyebrow': range(17, 22), 'left eyebrow': range(22, 27), 'nose bridge': range(27, 31), 'nose base': range(31, 36),
          'right eye outline': list(range(36, 42)) + [36], 'left eye outline': list(range(42, 48)) + [42], 'outer lip outline': list(range(48, 60)) + [48], 'inner lip outline': list(range(60, 68)) + [60]}
def load(MD):
    """-> {line: dict(points, under_skin, bones, cartilage)}; cartilage = the answer lists a cartilage under the line."""
    f = os.path.join(MD, 'lines', 'answers.json')
    if not os.path.exists(f): raise SystemExit('face lines are not named yet: run agents/landmark_lines.py (it writes %s)' % f)
    A = json.load(open(f, encoding='utf-8')); A = A.get('answer', A); out = {}
    for l in A['lines']:
        if l['line'] in SCHEME: out[l['line']] = dict(points=SCHEME[l['line']], under_skin=bool(l.get('bone_directly_under_skin')), bones=list(l.get('bones') or []), cartilage=any('cartilage' in str(b).lower() for b in (l.get('bones') or [])))
    return out
