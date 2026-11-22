"""Whole body: every detector point named in anatomical terms by a language model (text only, ONE call), then checked by
hard rules against the atlas, so each point becomes a usable target for the bones - or is set aside as not bone.
  E:\\splat\\.venv\\Scripts\\python.exe point_names.py <model dir> <atlas_warp.npz> <skeleton_fit.npz> ask|check
The same idea as landmark_lines.py (face), for any part: points -> lines -> names -> rules.
  points   the detectors' own point lists: body and feet (COCO-WholeBody, Jin et al. 2020: 17 body + 6 foot points) and the
           hand (21 points). Left and right are asked once.
  ask      for each point: its anatomical name, its kind, and the bones concerned, chosen ONLY from the atlas's own bone
           names. Kinds: joint (the centre of a joint: the bones that meet there), bone_end (the free end of a bone under
           soft tissue: a fingertip), bone_surface (bone right under the skin: the heel), soft (no bone: nose tip, ear).
           And lines: ordered runs of points that follow consecutive bones, with the bone between each two points.
           Cached in <model dir>/lines/points_answer.json.
  check    hard rules, all from the atlas (nothing from the language model is used unchecked):
             R1  every bone named exists in the atlas (per side).
             R2  the atlas files cartilage with the bones: anything with "cartilage" in its name is not bone, and a point
                 of kind soft gives no bone target.
             R3  a joint's bones must touch in the atlas (gap, as a share of the smaller bone's size, in the lower group of
                 an Otsu split over all nearby bone pairs; at least one pair of the named bones). The joint's place in the
                 atlas is where all its touching bones come together (on each bone, the points nearest the farthest
                 of the others), not the average of the pairs: two of the bones may touch elsewhere as well.
             R4  on a line, the bone between two points must be one of the bones of both (it joins them); a stretch that
                 fails (wrist to knuckle crosses the wrist bones too) is not one bone and is not used as a bone's axis.
             R5  a bone_end or bone_surface point lies on its own bone, at the end farthest from its neighbour on the line.
           A point failing a rule is reported and not used. Written to <model dir>/lines/points_named.json:
           per point and side - kind, bones, atlas position, the rule it failed (if any); and the lines."""
import sys, os, re, json, subprocess, numpy as np
from scipy.spatial import cKDTree
D, WP, SF, STEP = sys.argv[1:5]; OD = os.path.join(D, 'lines'); os.makedirs(OD, exist_ok=True); HERE = os.path.dirname(os.path.abspath(__file__))
RUN = os.path.join(HERE, '..', 'model_to_vrm', 'common', 'pi_run.py'); PY = sys.executable
BODY = ['nose', 'eye', 'ear', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'big toe', 'small toe', 'heel']                 # COCO-WholeBody, one side
HAND = ['hand wrist'] + ['%s %d' % (f, k) for f in ('thumb', 'index finger', 'middle finger', 'ring finger', 'little finger') for k in (1, 2, 3, 4)]
W = np.load(WP); V0 = W['V0'].astype(np.float64); OB = W['obj']; names = [str(x) for x in W['names']]; sysn = np.array([str(x) for x in W['system']])
bones = [int(b) for b in np.unique(OB[sysn[OB] == 'bone'])]; base = lambda n: re.sub(r'\.[lr]$', '', n); VOC = sorted({base(names[b]) for b in bones}); low = {v.lower(): v for v in VOC}
if STEP == 'ask':
    Q = ('Pose detectors give named points on a human body. For each point below say what it is anatomically, so that a skeleton can be fitted to the points. '
         'Body points (one side; the other side is the mirror): %s. Hand points (one hand): "hand wrist", then for each of the five fingers four points numbered 1 to 4 from the base of the finger '
         'towards its tip, 4 being the fingertip (for the thumb, 1 is the point nearest the wrist): %s. '
         'For each point give: "anatomical_name"; "kind", one of: "joint" (the point is the centre of a joint), "bone_end" (the free end of a bone covered by soft tissue), '
         '"bone_surface" (a place where a bone lies right under the skin), "soft" (soft tissue or cartilage, no bone there); and "bones": for a joint ALL the bones that meet in it, '
         'for bone_end or bone_surface the one bone, for soft an empty list. Choose bones ONLY from this list, the names used by the anatomical model (in it the fingers and the toes are '
         'numbered first to fifth; the first finger of the hand is the thumb, the first finger of the foot is the big toe): %s. '
         'Then give "lines": ordered runs of these points that follow consecutive bones (an arm, a leg, each finger, the foot), each with "points" in order and "bones_between": '
         'for each two neighbouring points the one bone that runs between them (same list of names), so bones_between has one entry fewer than points. '
         'Answer with JSON only: {"points": [{"point": "name as given", "anatomical_name": "...", "kind": "...", "bones": ["..."], "why": "one short sentence"}], '
         '"lines": [{"line": "...", "points": ["..."], "bones_between": ["..."]}]}') % ('; '.join(BODY), '; '.join(HAND), '; '.join(VOC))
    r = subprocess.run([PY, RUN, '--no-session', '-p', Q], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1200); m = re.search(r'\{.*\}', r.stdout, re.S)
    if not m: sys.exit('PN no answer: ' + (r.stdout[-300:] + ' / ' + r.stderr[-300:]))
    A = json.loads(m.group(0)); json.dump({'question': Q, 'answer': A}, open(os.path.join(OD, 'points_answer.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
    for a in A['points']: print('PN %-16s %-12s %-46s %s' % (a['point'], a['kind'], str(a['anatomical_name'])[:46], a['bones']), flush=True)
    for l in A['lines']: print('PN line %-14s %s | %s' % (l['line'], ' > '.join(l['points']), ' > '.join(l['bones_between'])), flush=True)
if STEP == 'check':
    A = json.load(open(os.path.join(OD, 'points_answer.json'), encoding='utf-8'))['answer']; nid = {n: i for i, n in enumerate(names)}; pts = lambda b: V0[OB == b]
    gf = os.path.join(os.path.dirname(os.path.abspath(SF)), 'bone_gaps.npz')                                    # gap between every two nearby atlas bones (cached: the atlas does not change)
    if not os.path.exists(gf):
        from skimage.filters import threshold_otsu
        Pb = {b: pts(b) for b in bones}; Tb = {b: cKDTree(Pb[b]) for b in bones}; lo = {b: Pb[b].min(0) for b in bones}; hi = {b: Pb[b].max(0) for b in bones}; sz = {b: float(np.linalg.norm(hi[b] - lo[b])) for b in bones}; pr, gp = [], []
        for i_, a_ in enumerate(bones):
            for b_ in bones[i_ + 1:]:
                m_ = min(sz[a_], sz[b_])
                if (lo[a_] - m_ > hi[b_]).any() or (lo[b_] - m_ > hi[a_]).any(): continue
                pr.append((a_, b_)); gp.append(float(Tb[b_].query(Pb[a_])[0].min()) / m_)                       # gap as a share of the smaller bone's size
        gp = np.array(gp); th = float(np.exp(threshold_otsu(np.log(np.maximum(gp, 1e-6))))); np.savez(gf, pairs=np.array(pr), rel_gap=gp, threshold=th)
    G = np.load(gf); CP = {tuple(sorted(map(int, p))) for p, g in zip(G['pairs'], G['rel_gap']) if g < float(G['threshold'])}
    print('PN touching: gap under %.3f of the smaller bone (Otsu split of the log gaps of all nearby bone pairs): %d pairs' % (float(G['threshold']), len(CP)), flush=True)
    def bone_id(n, sd):                                              # R1, R2: an atlas bone of that side (or a midline bone), not cartilage
        v = low.get(str(n).lower())
        if v is None: return None, 'R1 "%s" is not an atlas bone name' % n
        if 'cartilage' in v.lower(): return None, 'R2 "%s" is cartilage' % v
        i = nid.get(v + sd, nid.get(v)); return (i, None) if i is not None else (None, 'R1 "%s" has no %s side' % (v, sd))
    def contact(a, b):                                               # R3: middle of the contact points of two touching bones
        A_, B_ = pts(a), pts(b); d, i = cKDTree(B_).query(A_); k = np.argsort(d)[:max(3, len(A_) // 100)]; return (A_[k].mean(0) + B_[i[k]].mean(0)) / 2
    def meet(ids):                                                   # where ALL the bones of a joint come together: on each bone, the points nearest to
        Pb = [pts(b) for b in ids]; Tb = [cKDTree(P_) for P_ in Pb]; out_ = []        # the farthest of the others (two bones may also touch elsewhere: shin and calf bone, radius and ulna)
        for i_, P_ in enumerate(Pb):
            sc = np.max([Tb[j_].query(P_)[0] for j_ in range(len(ids)) if j_ != i_], axis=0); out_.append(P_[np.argsort(sc)[:max(3, len(P_) // 100)]].mean(0))
        return np.mean(out_, axis=0)
    def far_end(b, x): P = pts(b); d = np.linalg.norm(P - x, axis=1); return P[np.argsort(d)[-max(3, len(P) // 100):]].mean(0)
    out = {}; L_out = []; n_fail = 0
    for sd in ('.l', '.r'):
        P = {}
        for a in A['points']:
            e = dict(point=a['point'], side=sd, anatomical_name=a.get('anatomical_name'), kind=a.get('kind'), bones=[], atlas=None, failed=None); P[a['point']] = e
            if e['kind'] == 'soft': e['failed'] = 'R2 soft tissue: no bone target'; continue
            ids = []
            for n in a.get('bones', []):
                i, why = bone_id(n, sd)
                if i is None: e['failed'] = why
                else: ids.append(i)
            e['bones'] = [names[i] for i in ids]; e['_ids'] = ids
            if e['kind'] == 'joint' and not e['failed']:
                pr = [(x, y) for k_, x in enumerate(ids) for y in ids[k_ + 1:] if tuple(sorted((x, y))) in CP]
                if not pr: e['failed'] = 'R3 the bones named do not touch in the atlas'
                else: e['atlas'] = meet(sorted({b_ for pr_ in pr for b_ in pr_})).tolist(); e['touching'] = [[names[x], names[y]] for x, y in pr]
        for l in A['lines']:                                         # R4, R5
            pp = [P.get(p) for p in l['points']]; bb = l['bones_between']; row = dict(line=l['line'], side=sd, points=l['points'], bones=[], failed=[])
            if None in pp or len(bb) != len(pp) - 1: row['failed'].append('points or bones do not match the lists'); L_out.append(row); continue
            for k_, n in enumerate(bb):
                i, why = bone_id(n, sd); p, q = pp[k_], pp[k_ + 1]; row['bones'].append(names[i] if i is not None else None)
                if i is None: row['failed'].append(why); continue
                for e in (p, q):
                    if e['kind'] != 'soft' and i not in e.get('_ids', []): row['failed'].append('R4 %s is not a bone of point "%s"' % (names[i], e['point']))
                for e, o in ((p, q), (q, p)):                       # R5: free ends, placed from the bone's joint on this line
                    if e['kind'] in ('bone_end', 'bone_surface') and e['atlas'] is None and not e['failed'] and o['atlas'] is not None and e.get('_ids'): e['atlas'] = far_end(e['_ids'][0], np.array(o['atlas'])).tolist()
            L_out.append(row)
        for e in P.values():
            if e['kind'] in ('bone_end', 'bone_surface') and e['atlas'] is None and not e['failed']: e['failed'] = 'R5 no line gives this bone a joint to measure its far end from'
            e.pop('_ids', None); out['%s%s' % (e['point'], sd)] = e; n_fail += e['failed'] is not None and e['kind'] != 'soft'
            print('PN %-18s %-12s %-40s %s' % (e['point'] + sd, e['kind'], ' + '.join(e['bones'])[:40], 'ok' if e['atlas'] is not None else 'NOT USED: ' + str(e['failed'])), flush=True)
    for r in L_out:
        if r['failed']: print('PN line %s%s: %s' % (r['line'], r['side'], '; '.join(sorted(set(r['failed'])))), flush=True)
    use = sum(e['atlas'] is not None for e in out.values()); soft = sum(e['kind'] == 'soft' for e in out.values())
    print('PN summary: %d points (both sides); %d usable bone targets, %d soft (set aside), %d failed a rule; %d of %d lines pass' % (len(out), use, soft, n_fail, sum(not r['failed'] for r in L_out), len(L_out)))
    json.dump({'points': out, 'lines': L_out}, open(os.path.join(OD, 'points_named.json'), 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
