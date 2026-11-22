"""List the current units and what is recorded for each:  python tools/show.py [unit id ...]"""
import sys, numpy as np, ws
o = ws.owner(); s = ws.sheet(); units = sys.argv[1:] or sorted(set(o.tolist()), key=lambda u: [int(x) for x in u.split('.')])
done = 0
for u in units:
    e = s.get(u, {}); n = int((o == u).sum()); done += bool(e.get('colour'))
    print('%-6s %6d pts  %-22s %-10s %-12s %-7s %-8s move=%-3s %s' % (u, n, e.get('name', '-'), e.get('category', '-'), e.get('material', '-'), e.get('stiffness', '-'),
          e.get('colour', '-'), e.get('movable', '-'), ('parent ' + e['parent']) if 'parent' in e else ''))
if not sys.argv[1:]: print('%d units, %d with a colour, %d without' % (len(units), done, len(units) - done))
