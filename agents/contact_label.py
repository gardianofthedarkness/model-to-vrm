"""Ask the vision LLM (DeepSeek through the Pi agent, key from the vault) what each contact patch's two sides are:
  python contact_label.py <contacts dir>   ->  <dir>/labels.json   (re-run: patches already labelled are kept)
One call per patch with its card (side A magenta, side B cyan on the model's own textured renders) and the measured
hints (texture colour, nearest skeleton joint, SAM 3 concepts when present). The model answers JSON only:
  {"A": {"material": "flesh|cloth|hair|prop", "what": "..."}, "B": {...}, "pair": "flesh-flesh|flesh-cloth|...",
   "real_contact": true|false, "why": "..."}
real_contact = false when the two sides are one thing folded on itself or texture noise rather than two things pressing."""
import sys, os, json, re, subprocess
D = sys.argv[1]; P = json.load(open(os.path.join(D, 'patches.json'))); out_p = os.path.join(D, 'labels.json')
done = json.load(open(out_p)) if os.path.exists(out_p) else {}
RUN = r'E:\code\novel-ontology-pi\engine\characters\pipeline\pi_run.py'; PY = r'E:\splat\.venv\Scripts\python.exe'
Q = ('The picture shows a 3D model of a person from three directions. Two surfaces of the model touch each other: side A '
     'is painted MAGENTA, side B is painted CYAN. Measured hints: {hints}. For each side say what material it is: '
     'flesh (bare skin), cloth (a worn garment: shirt, skirt, sock, shoe...), hair, or prop (a carried thing like a bag '
     'and its strap), and name it. Then name the pair (e.g. flesh-cloth). Say real_contact=false if A and B are one thing '
     'folded on itself or noise rather than two things pressing together. Answer with JSON only, no other text: '
     '{{"A": {{"material": "...", "what": "..."}}, "B": {{"material": "...", "what": "..."}}, "pair": "...", "real_contact": true, "why": "..."}}')
for p in P:
    k = str(p['patch'])
    if k in done: continue
    hints = '; '.join('side %s: colour %s, near joint %s%s' % (s, p[s]['colour'], p[s]['near_joint'], (', SAM 3 says ' + p[s]['sam3']) if 'sam3' in p[s] else '') for s in ('A', 'B') if p[s])
    card = os.path.join(D, 'cards', 'patch_%s.jpg' % k)
    r = subprocess.run([PY, RUN, '--no-session', '-p', '@' + card, Q.format(hints=hints)], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
    m = re.search(r'\{.*\}', r.stdout, re.S)
    try: ans = json.loads(m.group(0)) if m else {'error': r.stdout[-300:] + r.stderr[-300:]}
    except Exception as e: ans = {'error': 'bad json: ' + m.group(0)[:300]}
    done[k] = ans; json.dump(done, open(out_p, 'w'), indent=1)
    print('CL patch %s: %s' % (k, ans.get('pair', ans.get('error', '?'))[:120]), flush=True)
print('CL done: %d patches labelled' % len(done))
