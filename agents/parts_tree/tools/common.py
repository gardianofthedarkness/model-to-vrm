"""Shared loading for the parts-tree tools. Everything model-specific comes from <workspace>/config.json:
  model      the GLB             colours   orig_colours.npz (V, C per vertex, import order)
  views      a views dir with cameras.json (+ vis.npz: views x vertices visibility, names)   front/back: camera names
  closeups   optional list of other views dirs (each with cameras.json + head_points.npz or vis.npz)"""
import os, json, numpy as np
WS = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG = json.load(open(os.path.join(WS, 'config.json')))
def parts():
    return json.load(open(os.path.join(WS, 'part_list.json')))
def all_parts():
    P = parts(); d = {p['id']: dict(p, kind='colour region') for p in P['colour_regions']}
    d.update({p['id']: dict(p, kind='sam3 detail') for p in P['sam3_details']}); return d
def tree():
    f = os.path.join(WS, 'tree.json')
    return json.load(open(f)) if os.path.exists(f) else {'nodes': {'body': {'parent': None, 'kind': 'whole', 'notes': 'the whole character'}}, 'assign': {}}
def save_tree(t): json.dump(t, open(os.path.join(WS, 'tree.json'), 'w'), indent=1)
def hexc(c): return '#%02x%02x%02x' % tuple(int(255 * min(max(x, 0), 1)) for x in c)
def entity_colour(t, node, A):
    """One colour per entity: the point-weighted median (by brightness order) of the measured texture colours of its
    parts. A subpart (face, sleeve, pocket) is the same stuff as its entity, so it takes the entity's colour, computed
    over the entity and all its subparts; a detail (eye, button, stripe) keeps its own colour."""
    N = t['nodes']; top = node
    while N[top].get('kind') == 'subpart' and N[top].get('parent') in N and N[N[top]['parent']].get('kind') != 'whole': top = N[top]['parent']
    group = {top}; grow = True
    while grow:
        new = {k for k, v in N.items() if v.get('parent') in group and v.get('kind') == 'subpart'} - group; grow = bool(new); group |= new
    ids = [int(p) for p, n in t['assign'].items() if n in group]
    if not ids: return None
    w = np.array([A[i]['points'] for i in ids], float); c = np.array([A[i]['colour'] for i in ids])
    o = np.argsort(c.sum(1)); cw = np.cumsum(w[o]); return c[o][np.searchsorted(cw, cw[-1] / 2)].tolist()
def views():
    """[(views dir, camera name, camera, visibility per vertex)] over the main views and the close-ups"""
    out = []
    for d in [CFG['views']] + CFG.get('closeups', []):
        cams = json.load(open(os.path.join(d, 'cameras.json'))); vf = os.path.join(d, 'vis.npz'); hf = os.path.join(d, 'head_points.npz')
        if os.path.exists(vf):
            Z = np.load(vf); vis = dict(zip([str(n) for n in Z['names']], Z['vis']))
            out += [(d, n, cams[n], vis[n]) for n in sorted(vis)]
        elif os.path.exists(hf):
            Z = np.load(hf); n_all = len(np.load(CFG['colours'])['V'])
            for n, cm in sorted(cams.items()):
                vv = np.zeros(n_all, bool); vv[Z['idx']] = Z['vis'][cm['view']]; out.append((d, n, cm, vv))
    return out
