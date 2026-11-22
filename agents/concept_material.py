"""Ask the LLM (DeepSeek through Pi, key from the vault) which material each SAM 3 concept name is - so no hand table:
  python concept_material.py <shares.npz> <out.json>
Answer per concept: flesh (bare skin of the body), cloth (anything worn), hair, or prop (a carried thing)."""
import sys, json, re, subprocess, numpy as np
sp, out = sys.argv[1:3]; names = [str(x) for x in np.load(sp)['concepts']]
RUN = r'E:\code\novel-ontology-pi\engine\characters\pipeline\pi_run.py'; PY = r'E:\splat\.venv\Scripts\python.exe'
Q = ('These are names of regions found by a segmenter on a 3D model of a dressed person: %s. For each name say which '
     'material that region is on a person: "flesh" (bare skin of the body), "cloth" (anything worn: garments, shoes, '
     'accessories worn on the body), "hair", or "prop" (a carried object such as a bag). Answer with JSON only, one key '
     'per name, e.g. {"name": "flesh", ...}.') % ', '.join(names)
r = subprocess.run([PY, RUN, '--no-session', '-p', Q], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
m = re.search(r'\{.*\}', r.stdout, re.S); M = json.loads(m.group(0))
miss = [n for n in names if M.get(n) not in ('flesh', 'cloth', 'hair', 'prop')]
if miss: sys.exit('CM no valid material for %s: %s' % (miss, r.stdout[-300:]))
json.dump({n: M[n] for n in names}, open(out, 'w'), indent=1); print('CM', {n: M[n] for n in names})
