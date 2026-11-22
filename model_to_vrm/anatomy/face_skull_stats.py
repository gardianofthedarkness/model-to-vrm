"""A small statistical model, in proportions, of what lies under a face: from the distances between 12 face points to the
skull's own measurements and the tissue depth at 41 sites (FBI ReFace CT data, 388 living adults, CC BY 4.0, Mendeley
Data pss59h29nb / cnc2j4h26b / byr94xy7mv; Simmons-Ehrhardt, Monson et al.):
  E:\\splat\\.venv\\Scripts\\python.exe face_skull_stats.py <data dir with the three .xlsx>
  input     the 66 distances between 12 skin points measured on each person's CT face (sellion, outer eye corners, nose
            tip, alar curvatures, subnasale, upper / lower lip, stomion, mouth corners) - all of them findable among the 68
            detected face points. Face size = their geometric mean; features = log(distance / face size): shape only.
  targets   log(skull measurement / face size) and log(tissue depth / face size): proportions, never millimetres, so the
            model applies to a head of any size
  models    the mean proportion (no model: "a scaled human"); ridge regression (penalty by leave-one-out, sklearn
            RidgeCV); partial least squares (components by cross-validation). 10-fold cross-validated error per target, as a
            relative error (SD of log residual, in %), and how much of the spread between people the face explains (R2)
  also      whether depths follow face size at all: error of "mean depth in mm" against "mean proportion x face size"
Writes <data dir>/face_skull_model.npz (feature names, means, the chosen model's coefficients, per-target cross-validated
error) and face_skull_model.json (the table)."""
import sys, os, re, json, zipfile, numpy as np, xml.etree.ElementTree as ET
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, cross_val_predict
D = sys.argv[1]; NS = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
def read(f):                                                        # .xlsx with the standard library (no spreadsheet package needed)
    z = zipfile.ZipFile(f); ss = [''.join(t.text or '' for t in si.iter('{%s}t' % NS['m'])) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', NS)]
    names = [s.get('name') for s in ET.fromstring(z.read('xl/workbook.xml')).find('m:sheets', NS)]; out = {}
    for i, nm in enumerate(names):
        rows = []
        for r in ET.fromstring(z.read('xl/worksheets/sheet%d.xml' % (i + 1))).iter('{%s}row' % NS['m']):
            row = {}
            for c in r.findall('m:c', NS):
                k = 0
                for ch in re.match('[A-Z]+', c.get('r')).group(): k = k * 26 + ord(ch) - 64
                v = c.find('m:v', NS); row[k - 1] = None if v is None else (ss[int(v.text)] if c.get('t') == 's' else v.text)
            rows.append([row.get(j) for j in range(max(row) + 1)] if row else [])
        out[nm] = rows
    return out
def num(v):
    try: return float(v)
    except Exception: return np.nan
isid = lambda s: bool(re.match(r'^[A-Za-z]{3}[0-9]+$', str(s)))
T = read(os.path.join(D, 'tissue_depth.xlsx'))['Tissue Depths_ReFace']; sites = [str(h) for h in T[0][2:]]
TD = {str(r[0]).upper(): np.array([num(r[j]) if j < len(r) else np.nan for j in range(2, len(T[0]))]) for r in T[1:] if r and isid(r[0])}
CR = {}; CK = None
for nm, sh in read(os.path.join(D, 'cranial_ild.xlsx')).items():
    if nm == 'Ref': continue
    h = [str(c) for c in sh[0]]
    for r in sh[1:]:
        if r and isid(r[0]): CR[str(r[0]).upper()] = {h[j]: num(r[j]) for j in range(len(r)) if j < len(h) and re.match('^[A-Z]{3,4}$', h[j]) and h[j] != 'Sex'}
CK = sorted(set.intersection(*[set(v) for v in CR.values()])); FA = {}; FN = None
for nm, sh in read(os.path.join(D, 'facial_ild.xlsx')).items():
    if not nm.endswith('_k'): continue                              # k = measured on the person's own CT face
    FN = [str(r[0]) for r in sh[1:] if r and r[0]]
    for j in range(1, len(sh[0])):
        if sh[0][j]: FA[str(sh[0][j]).upper().replace('CAU', 'CAU')] = np.array([num(r[j]) if j < len(r) else np.nan for r in sh[1:] if r and r[0]])
ids_t = set(TD); print('FS people: tissue depths %d, skull measurements %d, face distances %d (%d distances between 12 points)' % (len(TD), len(CR), len(FA), len(FN)))
print('FS id match: face-depth %d, face-skull %d (examples %s | %s | %s)' % (len(set(FA) & ids_t), len(set(FA) & set(CR)), sorted(FA)[:2], sorted(TD)[:2], sorted(CR)[:2]))
def table(ids, Y, ynames, label):
    X = np.array([FA[i] for i in ids]); ok = np.isfinite(X).all(1) & (X > 0).all(1); X, Y = X[ok], Y[ok]
    size = np.exp(np.log(X).mean(1)); Xf = np.log(X) - np.log(size)[:, None]; rows = {}; kf = KFold(10, shuffle=True, random_state=0)
    for j, yn in enumerate(ynames):
        y = Y[:, j]; m = np.isfinite(y) & (y > 0)
        if m.sum() < 100: continue
        ly = np.log(y[m]); t = ly - np.log(size[m]); Xm = Xf[m]
        base_abs = float(np.std(ly - ly.mean())); base = float(np.std(t - t.mean()))          # mean mm / mean proportion (leave-out bias is negligible at n > 100)
        pr = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 25)), Xm, t, cv=kf); e_r = float(np.std(t - pr))
        best = (np.inf, 0)
        for nc in range(1, 9):
            pp = cross_val_predict(PLSRegression(nc), Xm, t, cv=kf).ravel(); best = min(best, (float(np.std(t - pp)), nc))
        rows[yn] = {'n': int(m.sum()), 'median_mm': float(np.median(y[m])), 'median_share_of_face_size': float(np.exp(np.median(t))), 'err_mean_mm_pct': 100 * base_abs, 'err_mean_proportion_pct': 100 * base,
                    'err_ridge_pct': 100 * e_r, 'err_pls_pct': 100 * best[0], 'pls_components': best[1], 'r2_ridge': 1 - (e_r / base) ** 2}
    print('FS %s  (relative error in %%: mean mm | mean proportion | ridge | PLS;  R2 of ridge)' % label)
    for yn, r in sorted(rows.items(), key=lambda z: -z[1]['r2_ridge']):
        print('FS   %-7s n %3d  median %6.1f mm = %.3f of face size   %5.1f | %5.1f | %5.1f | %5.1f (%d)   R2 %+.2f' % (yn, r['n'], r['median_mm'], r['median_share_of_face_size'], r['err_mean_mm_pct'],
              r['err_mean_proportion_pct'], r['err_ridge_pct'], r['err_pls_pct'], r['pls_components'], r['r2_ridge']))
    return rows, (X, size, Xf, ok)
ids1 = sorted(set(FA) & set(CR)); R1, _ = table(ids1, np.array([[CR[i][k] for k in CK] for i in ids1]), CK, 'skull measurements from face shape')
ids2 = sorted(set(FA) & ids_t); R2, (X2, size2, Xf2, ok2) = table(ids2, np.array([TD[i] for i in ids2]), sites, 'tissue depths from face shape')
print('FS face size (geometric mean of the 66 distances): median %.1f mm, SD %.1f%%' % (np.median(size2), 100 * np.std(np.log(size2))))
# the fitted models on everyone, for use on a figure: ridge per target
def fit_all(ids, Y, ynames, rows):
    X = np.array([FA[i] for i in ids]); ok = np.isfinite(X).all(1) & (X > 0).all(1); X, Y = X[ok], Y[ok]; size = np.exp(np.log(X).mean(1)); Xf = np.log(X) - np.log(size)[:, None]; W = {}
    for j, yn in enumerate(ynames):
        if yn not in rows: continue
        y = Y[:, j]; m = np.isfinite(y) & (y > 0); t = np.log(y[m]) - np.log(size[m]); mdl = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xf[m], t)
        W[yn] = (mdl.coef_, float(mdl.intercept_), float(t.mean()))
    return W, Xf.mean(0)
W1, mu1 = fit_all(ids1, np.array([[CR[i][k] for k in CK] for i in ids1]), CK, R1); W2, mu2 = fit_all(ids2, np.array([TD[i] for i in ids2]), sites, R2)
np.savez(os.path.join(D, 'face_skull_model.npz'), features=np.array(FN), skull_names=np.array(list(W1)), skull_coef=np.array([W1[k][0] for k in W1]), skull_icpt=np.array([W1[k][1] for k in W1]), skull_mean=np.array([W1[k][2] for k in W1]),
         skull_err=np.array([R1[k]['err_ridge_pct'] / 100 for k in W1]), skull_err_mean=np.array([R1[k]['err_mean_proportion_pct'] / 100 for k in W1]),
         depth_names=np.array(list(W2)), depth_coef=np.array([W2[k][0] for k in W2]), depth_icpt=np.array([W2[k][1] for k in W2]), depth_mean=np.array([W2[k][2] for k in W2]),
         depth_err=np.array([R2[k]['err_ridge_pct'] / 100 for k in W2]), depth_err_mean=np.array([R2[k]['err_mean_proportion_pct'] / 100 for k in W2]), feat_mean=mu2, feat_sd=Xf2.std(0))
json.dump({'skull': R1, 'depth': R2}, open(os.path.join(D, 'face_skull_model.json'), 'w'), indent=1)
