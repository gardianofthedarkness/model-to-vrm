"""Show the tree: python tools/tree.py   (per node: kind, its own parts and points, and its one colour = the
point-weighted median of its parts' measured colours)"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from common import *
t = tree(); A = all_parts()
def show(n, d):
    ids = [int(p) for p, v in t['assign'].items() if v == n]; c = entity_colour(t, n, A); nd = t['nodes'][n]
    print('%s%s [%s]%s  parts %d (%d pts)%s  %s' % ('  ' * d, n, nd.get('kind', ''), ' movable' if nd.get('movable') == 'yes' else '', len(ids),
          sum(A[i]['points'] for i in ids), '  colour ' + hexc(c) if c else '', ids if len(ids) <= 12 else ids[:12] + ['...']))
    for k in sorted(k for k, v in t['nodes'].items() if v['parent'] == n): show(k, d + 1)
show('body', 0); print('unassigned parts: %d' % sum(str(i) not in t['assign'] for i in A))
