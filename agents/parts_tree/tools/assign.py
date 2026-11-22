"""Put parts into a node: python tools/assign.py <node> <part id> [...]   (a part belongs to exactly one node; assigning
again moves it)"""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from common import *
t = tree(); A = all_parts(); n = sys.argv[1]
if n not in t['nodes']: sys.exit('no node %s; make it with node.py first' % n)
for a in sys.argv[2:]:
    if int(a) not in A: print('no part', a); continue
    t['assign'][str(int(a))] = n
save_tree(t); print('%s now holds %d parts' % (n, sum(v == n for v in t['assign'].values())))
