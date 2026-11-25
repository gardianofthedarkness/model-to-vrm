"""R2 validation picture: per runner, stride-averaged vertical ground reaction (plates vs R2) and, per muscle, EMG envelope
(mean +- sd, scaled to its own peak) vs R2 activation (scaled to its own peak), 0-100 % of the gait cycle from foot strike.
Shapes only: amplitudes are compared in the json where MVC trials exist.  Uses measured data: keep the picture private.
  E:\\splat\\.venv\\Scripts\\python.exe r2_validate_picture.py <out dir> <png> <subject> [<subject> ...]"""
import sys, os, json, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
OUT, PNG, subs = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
MUS = ['Sol', 'GastrocMed', 'GastrocLat', 'Tibialis', 'VastusMed', 'VastusLat', 'RectusFem', 'BicepsFem', 'Semitendi/membra', 'GlutMax', 'GlutMed']
fig, ax = plt.subplots(len(subs), len(MUS) + 1, figsize=(2.0 * (len(MUS) + 1), 2.0 * len(subs) + 0.6), squeeze=False)
x = np.arange(101)
for r_, S in enumerate(subs):
    D = np.load(os.path.join(OUT, 'Sub_%02d_r2.npz' % S)); J = json.load(open(os.path.join(OUT, 'Sub_%02d_r2.json' % S))); st = D['strikes']
    def cyc(v): return np.array([np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, b - a + 1), v[a:b + 1]) for a, b in zip(st[:-1], st[1:])])
    a = ax[r_, 0]; Fm, Fe = cyc(D['Fm']), cyc(D['Fe'])
    a.fill_between(x, Fm.mean(0) - Fm.std(0), Fm.mean(0) + Fm.std(0), color='#cfcfcf', lw=0); a.plot(x, Fm.mean(0), color='#333', lw=1.5, label='plates')
    a.plot(x, Fe.mean(0), color='#2b5d8a', lw=1.5, label='R2'); a.set_title('runner %d (%s, %s y)\nvertical GRF, BW  r=%.2f' % (S, J['sex'], J['age'], J['grf']['r']), fontsize=7)
    if r_ == 0: a.legend(fontsize=6)
    for c_, k in enumerate(MUS):
        a = ax[r_, c_ + 1]; E, M, Ms = D['curve_' + k.replace('/', '_')]; sc = M.max()
        a.fill_between(x, (M - Ms) / sc, (M + Ms) / sc, color='#e6cfcf', lw=0); a.plot(x, M / sc, color='#a33', lw=1.4, label='EMG')
        a.plot(x, E / max(E.max(), 1e-9), color='#2b5d8a', lw=1.4, label='R2'); a.set_ylim(-0.05, 1.6)
        a.set_title('%s  r=%.2f' % (k, J['muscles'][k]['r']), fontsize=7)
        if r_ == 0 and c_ == 0: a.legend(fontsize=6)
    for a in ax[r_]: a.tick_params(labelsize=6)
for a in ax[-1]: a.set_xlabel('% gait cycle', fontsize=7)
fig.suptitle('R2 against measured running at 3 m/s (Van Hooren & Meijer 2024): R2 sees only the measured motion and contact timing', fontsize=9)
plt.tight_layout(); plt.savefig(PNG, dpi=110); print('wrote', PNG)
