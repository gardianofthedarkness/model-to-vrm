"""Record what a unit is:  python tools/set.py <unit id> key=value [key=value ...]
Keys: name, category (skin|hair|cloth|leather|rigid accessory|eye|mouth|other), material, stiffness (soft|medium|stiff),
colour (a hex from the unit's candidates or its parent's), shade (hex list, comma-separated), movable (yes|no), notes.
A colour not among the measured candidates is refused."""
import sys, ws
uid, kv = sys.argv[1], dict(a.split('=', 1) for a in sys.argv[2:]); s = ws.sheet(); o = ws.owner()
if uid not in set(o.tolist()): sys.exit('%s: not a current unit (see tools/show.py)' % uid)
e = s.setdefault(uid, {}); par = s.get(e.get('parent', ''), {})
allowed = {c['hex'].lower() for c in e.get('candidates', []) + par.get('candidates', [])}
for k, v in kv.items():
    if k == 'colour' and allowed and v.lower() not in allowed: sys.exit('%s: colour %s is not a measured candidate %s (make its card first: tools/card.py)' % (uid, v, sorted(allowed)))
    e[k] = v.split(',') if k == 'shade' else v
ws.save_sheet(s); print('%s: %s' % (uid, {k: e.get(k) for k in ('name', 'category', 'material', 'stiffness', 'colour', 'movable')}))
