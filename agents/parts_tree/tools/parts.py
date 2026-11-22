"""List the parts: python tools/parts.py [unassigned]
Each line: id, kind (colour region = big part from colour differencing; sam3 detail = small part SAM 3 found), points,
median colour, centre (x, y, z; z is up, the model is about 0.98 tall, front faces -y), the hint (Tripo piece name, or the
SAM 3 concept and what it sits inside), and the node it is assigned to."""
import sys, os; sys.path.insert(0, os.path.dirname(__file__)); from common import *
A = all_parts(); t = tree(); only = len(sys.argv) > 1
for i, p in sorted(A.items(), key=lambda kv: -kv[1]['points']):
    if only and str(i) in t['assign']: continue
    hint = ('on Tripo piece: %s' % p['tripo_part_name']) if p['kind'] == 'colour region' else ('SAM3 "%s", inside %s, on region %s' % (p['concept'], p['inside_concept'], p['on_colour_region']))
    print('%4d  %-13s %7d pts  %s  centre %s  %s  -> %s' % (i, p['kind'], p['points'], hexc(p['colour']), [round(x, 2) for x in p['centre']], hint, t['assign'].get(str(i), '-')))
print('%d parts, %d unassigned' % (len(A), sum(str(i) not in t['assign'] for i in A)))
