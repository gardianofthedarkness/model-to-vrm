"""Which touching bones really form a joint? The measured list (bone_gaps.npz: gaps in the lower group of an Otsu split)
also holds bones that merely lie close - calf bone and heel bone, calf bone and thigh bone - and tying those together
over-constrains the skeleton. A language model (text only, ONE call) says for each measured pair whether the two bones
articulate, and how; the measurement stays the hard limit.
  E:\\splat\\.venv\\Scripts\\python.exe joint_names.py <fits dir> <atlas_warp.npz> ask|check
  ask     every touching pair once (left and right asked as one): is there an anatomical joint between the two, its name,
          and its kind: fused (suture, tooth socket, growth-plate union: no movement), fibrous or cartilaginous (slight
          movement), ball, hinge, pivot, saddle, condyloid, plane (synovial), or none (no joint: only neighbours).
          Cached in <fits dir>/joints_answer.json.
  check   hard rules: R1 only pairs measured as touching can be joints (the model cannot add one); R2 a pair it did not
          answer stays a joint (the measurement stands) and is reported; R3 left and right get the same answer.
          Writes <fits dir>/joints_named.json: per atlas bone pair - joint or not, name, kind."""
import sys, os, re, json, subprocess, numpy as np
FD, WP, STEP = sys.argv[1:4]; HERE = os.path.dirname(os.path.abspath(__file__)); RUN = os.path.join(HERE, '..', 'model_to_vrm', 'common', 'pi_run.py'); PY = sys.executable
W = np.load(WP); names = [str(x) for x in W['names']]; G = np.load(os.path.join(FD, 'bone_gaps.npz')); base = lambda n: re.sub(r'\.[lr]$', '', n)
T = [(int(a), int(b)) for (a, b), g in zip(G['pairs'], G['rel_gap']) if g < float(G['threshold'])]; U = sorted({tuple(sorted((base(names[a]), base(names[b])))) for a, b in T})
KINDS = ['fused', 'fibrous', 'cartilaginous', 'ball', 'hinge', 'pivot', 'saddle', 'condyloid', 'plane', 'none']
if STEP == 'ask':
    Q = ('In a 3D anatomical model of the human skeleton these pairs of bones (or cartilages) lie very close together. For each pair say whether the two actually form an anatomical joint '
         '(articulate with each other), or merely lie next to each other without a joint between them. "A | A" means the left and right bone of that name. '
         'Kinds: "fused" (suture, tooth in its socket, parts of one bone: no movement), "fibrous" (syndesmosis: slight movement), "cartilaginous" (disc, symphysis, costal cartilage), '
         'the synovial kinds "ball", "hinge", "pivot", "saddle", "condyloid", "plane", and "none" (no joint between these two). '
         'Pairs: %s. Answer with JSON only, one entry per pair, in the same order: {"pairs": [{"i": number, "kind": "...", "name": "the joint name, or empty"}]}') % '; '.join('%d: %s | %s' % (i, a, b) for i, (a, b) in enumerate(U))
    qf = os.path.join(FD, 'joints_question.txt'); open(qf, 'w', encoding='utf-8').write(Q)                # too long for a command line: passed as a file
    r = subprocess.run([PY, RUN, '--no-session', '-p', '@' + qf, 'Answer the question in the attached file. JSON only.'], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1800); m = re.search(r'\{.*\}', r.stdout, re.S)
    if not m: sys.exit('JN no answer: ' + (r.stdout[-300:] + ' / ' + r.stderr[-300:]))
    A = json.loads(m.group(0)); json.dump({'pairs_asked': U, 'answer': A}, open(os.path.join(FD, 'joints_answer.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False); print('JN %d pairs asked, %d answered' % (len(U), len(A['pairs'])))
if STEP == 'check':
    J = json.load(open(os.path.join(FD, 'joints_answer.json'), encoding='utf-8')); ans = {}
    for e in J['answer']['pairs']:
        if isinstance(e.get('i'), int) and 0 <= e['i'] < len(J['pairs_asked']) and e.get('kind') in KINDS: ans[tuple(J['pairs_asked'][e['i']])] = e
    out = []; cnt = {}; missing = set()
    for a, b in T:
        k = tuple(sorted((base(names[a]), base(names[b])))); e = ans.get(k)
        if e is None: missing.add(k); e = {'kind': 'unanswered', 'name': ''}
        out.append(dict(a=names[a], b=names[b], joint=e['kind'] != 'none', kind=e['kind'], name=e.get('name', ''))); cnt[e['kind']] = cnt.get(e['kind'], 0) + 1
    print('JN %d touching pairs: %s' % (len(out), ', '.join('%s %d' % kv for kv in sorted(cnt.items(), key=lambda kv: -kv[1]))))
    if missing: print('JN not answered (kept as joints, R2): %d, e.g. %s' % (len(missing), sorted(missing)[:5]))
    no = sorted({tuple(sorted((base(o['a']), base(o['b'])))) for o in out if not o['joint']}); print('JN not joints (%d): %s' % (len(no), '; '.join('%s | %s' % p for p in no)))
    json.dump(out, open(os.path.join(FD, 'joints_named.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
