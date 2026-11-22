"""Undo a split, or join units into one:  python tools/merge.py <unit> [<unit> ...]
With one id (e.g. 6): every region carved from it (6.1, 6.2, ...) goes back into 6. With several ids: all of them join the
first one (its record is kept)."""
import sys, numpy as np, ws
o = ws.owner().astype('<U12'); s = ws.sheet(); ids = sys.argv[1:]
if len(ids) == 1:
    u = ids[0]; m = np.char.startswith(o, u + '.'); o[m] = u; s = {k: v for k, v in s.items() if not k.startswith(u + '.')}; s.get(u, {}).pop('split_into', None)
else:
    for u in ids[1:]: o[o == u] = ids[0]; s.pop(u, None)
ws.save_owner(o); ws.save_sheet(s); print('units now: %d' % len(set(o.tolist())))
