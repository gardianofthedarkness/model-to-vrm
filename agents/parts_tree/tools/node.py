"""Tree nodes: python tools/node.py <node> parent=<node> kind=<entity|garment|subpart|detail> movable=yes|no notes="..."
A node is a thing in the character: skin, hair, one garment, a sleeve, an eye, an iris... Every node gets ONE colour,
computed from the parts assigned to it (never typed in). Put things that are the same stuff in the same node (all skin
in one 'skin' node, every cardigan piece in 'cardigan'); make a child node only for something that really is a separate
thing on it (a stripe, a button, a pocket trim, an iris).
python tools/node.py --delete <node>   removes a node with no children and no parts."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from common import *
t = tree(); a = sys.argv[1:]
if a[0] == '--delete':
    n = a[1]
    if any(v['parent'] == n for v in t['nodes'].values()): sys.exit('node has children')
    if n in t['assign'].values(): sys.exit('node has parts')
    del t['nodes'][n]; save_tree(t); print('deleted', n); sys.exit()
n = a[0]; kv = dict(x.split('=', 1) for x in a[1:]); node = dict(t['nodes'].get(n, {})); node.update(kv)
if node.get('parent') not in t['nodes'] or node.get('parent') == n: sys.exit('parent must be an existing node: ' + ', '.join(t['nodes']))
t['nodes'][n] = node; save_tree(t); print(n, node)
