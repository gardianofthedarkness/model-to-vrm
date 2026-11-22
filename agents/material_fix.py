"""Turn the labelling agent's free-text material flags (sub_labels.json 'material_wrong', e.g. "B: bag handle strap") into
material fixes per sub-contact side, with one text call to the LLM (DeepSeek through Pi) - no hand word list:
  python material_fix.py <contacts dir>     ->  <dir>/sub_fix.json  {"<sub>": {"A": "prop"}, ...}
A flag that only renames within the same material (shirt, not dress) gives no fix."""
import sys, os, json, re, subprocess
D = sys.argv[1]; G = {str(g['sub']): g for g in json.load(open(os.path.join(D, 'sub.json')))}; L = json.load(open(os.path.join(D, 'sub_labels.json')))
RUN = r'E:\code\novel-ontology-pi\engine\characters\pipeline\pi_run.py'; PY = r'E:\splat\.venv\Scripts\python.exe'
items = ['contact %s: segmenter says A is %s, B is %s; reviewer says side A is "%s", side B is "%s", and flags: "%s"' %
         (k, G[k]['A']['material'], G[k]['B']['material'], v.get('A', ''), v.get('B', ''), v['material_wrong']) for k, v in L.items() if v.get('material_wrong')]
Q = ('A segmenter gave each side of some contacts on a 3D model of a dressed person a material: flesh (bare skin), cloth '
     '(anything worn), hair, or prop (a carried thing such as a bag and its strap). A reviewer looked at pictures and flagged '
     'some sides. For each contact give the corrected material of each flagged side, only where the material itself '
     'changes (a different garment of the same material is no change). Contacts:\n%s\nJSON only: {"<contact>": {"A": '
     '"<material>"}, ...}; leave out contacts with no material change.') % '\n'.join(items)
r = subprocess.run([PY, RUN, '--no-session', '-p', Q], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=600)
m = re.search(r'\{.*\}', r.stdout, re.S); F = json.loads(m.group(0))
F = {k: {s: v for s, v in d.items() if s in ('A', 'B') and v in ('flesh', 'cloth', 'hair', 'prop') and v != G[k][s]['material']} for k, d in F.items() if k in G}
F = {k: d for k, d in F.items() if d}; json.dump(F, open(os.path.join(D, 'sub_fix.json'), 'w'), indent=1)
print('MF %d flags -> %d fixes:' % (len(items), sum(len(d) for d in F.values())), F)
