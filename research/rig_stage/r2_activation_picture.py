"""R2 check picture: muscle effort over time for the main left-leg muscles (after activation dynamics, and the static
optimisation's before it, thin), left-foot stance shaded (ARDY's labels), estimated vertical ground reaction and what the
clip leaves unexplained (in body weights); optional published EMG on-windows per muscle as bars (json: muscle -> list of
[start, end] in % of the left gait cycle, from heel strike), drawn only on strides with a flight phase (running strides).
  E:\\splat\\.venv\\Scripts\\python.exe r2_activation_picture.py <act npz> <out png> [emg windows json]"""
import sys, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = np.load(sys.argv[1]); OUT = sys.argv[2]; EMG = {k: v for k, v in json.load(open(sys.argv[3])).items() if not k.startswith('_')} if len(sys.argv) > 3 else {}
A, S, n = D['activation'], D['activation_static'], [str(x) for x in D['names']]; fps = float(D['fps']); t = np.arange(len(A)) / fps
fc = D['foot_contact']; L = fc[:len(A), :3].any(1) if fc.ndim == 2 else fc[:len(A)]
FC = np.load(sys.argv[1])  # same file
fcl = None
MUS = [('vaslat_l', 'vastus lateralis'), ('recfem_l', 'rectus femoris'), ('bflh_l', 'biceps femoris (long)'), ('semiten_l', 'semitendinosus'),
       ('glmax2_l', 'gluteus maximus'), ('glmed1_l', 'gluteus medius'), ('gasmed_l', 'gastrocnemius (medial)'), ('soleus_l', 'soleus'), ('tibant_l', 'tibialis anterior')]
fig, ax = plt.subplots(len(MUS) + 1, 1, figsize=(11, 1.25 * (len(MUS) + 1) + 1), sharex=True)
def shade(a):
    on = np.r_[False, L, False].astype(int); e = np.flatnonzero(np.diff(on))
    for s_, e_ in zip(e[::2], e[1::2]): a.axvspan(t[s_], t[min(e_, len(t) - 1)], color='#e8e4dc', lw=0)
hs = [k for k in range(1, len(L)) if L[k] and not L[k - 1]]                     # left heel strikes (stance onsets)
air = ~(fc[:len(A)].any(1) if fc.ndim == 2 else fc[:len(A)])                     # no foot on the ground: a running stride has flight
runs = [(s0, s1) for s0, s1 in zip(hs[:-1], hs[1:]) if air[s0:s1].any()]
for a, (k, lab) in zip(ax, MUS):
    shade(a); i = n.index(k); a.plot(t, S[:, i], color='#9aa5b1', lw=0.8); a.plot(t, A[:, i], color='#2b5d8a', lw=1.6)
    a.set_ylim(0, 1.05); a.set_ylabel(lab, rotation=0, ha='right', va='center', fontsize=8); a.tick_params(labelsize=7)
    for w in EMG.get(k, []):
        for s0, s1 in runs:
            T0, T1 = t[s0], t[s1]; a.plot([T0 + w[0] / 100 * (T1 - T0), T0 + w[1] / 100 * (T1 - T0)], [1.0, 1.0], color='#c0392b', lw=4, solid_capstyle='butt')
a = ax[-1]; shade(a); bw = float(D['body_weight_N']); a.plot(t, D['grf'][:, 2] / bw, color='#333', lw=1.4, label='vertical ground reaction (estimated)')
a.plot(t, D['unexplained_N'] / bw, color='#c0392b', lw=1, label='unexplained by the clip (flight)'); a.set_ylabel('x body weight', fontsize=8); a.legend(fontsize=7, loc='upper right'); a.set_xlabel('s')
fig.suptitle('R2: muscle effort from ARDY motion alone (blue: with activation dynamics, grey: static optimisation; shaded: left foot planted%s)'
             % ('; red bars: published EMG on-windows' if EMG else ''), fontsize=9)
plt.tight_layout(); plt.savefig(OUT, dpi=110); print('wrote', OUT)
