"""Make the card of a unit and list its candidate colours:  python tools/card.py <unit id> [<unit id> ...]   (e.g. 6, 6.2)
Prints: unit, points, card path, candidate colours (hex, share). The candidates are stored in sheet.json."""
import sys, json, numpy as np, ws
o = ws.owner(); s = ws.sheet()
for uid in sys.argv[1:]:
    idx = np.flatnonzero(o == uid)
    if len(idx) == 0: print('%s: no such unit (units: see tools/show.py)' % uid); continue
    p = ws.card(uid, idx); c = ws.candidates(idx); s.setdefault(uid, {})['candidates'] = c; s[uid]['points'] = int(len(idx))
    print('%s: %d points; card %s; candidates %s' % (uid, len(idx), p, ', '.join('%s (%.0f%%)' % (x['hex'], 100 * x['share']) for x in c)))
ws.save_sheet(s)
