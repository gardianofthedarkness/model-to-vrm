"""R2 validation on measured running (Van Hooren & Meijer 2024, OSF 7qbxc; data kept private, outside the repo): feed R2 only
the measured motion (their OpenSim inverse kinematics) and foot-contact timing, then compare what R2 estimates with what was
measured: vertical ground reaction (two force plates in a treadmill) and muscle effort against EMG.

  motion   their IK coordinates drive myoLegs directly (same OpenSim coordinate names); OpenSim axes (X forward, Y up,
           Z right) -> MuJoCo (x left, y back, z up); pelvis Z-X-Y rotations; lumbar through one ball joint at L5/S1.
           Model masses and inertias scaled to the runner's body mass (muscle strengths are the model's own).
  contact  stance = total vertical plate force, low-passed at 50 Hz, above 50 N (the upper end of the usual 20-50 N range:
           this treadmill's flight noise reaches ~90 N before filtering, so 20 N gives one-frame false contacts); the stance foot is the
           one whose heel and metatarsal markers are lower.  R2 gets these labels (heel and toe points both allowed).
  R2       r2_core.muscle_effort: exactly the pipeline's code (ground reaction from the motion, static optimisation,
           activation dynamics).  Vertical alignment: the stance foot touches the floor (same rule as the ARDY pipeline).
  EMG      SENIAM-style linear envelope: band-pass 20-450 Hz (4th-order Butterworth), full-wave rectified, low-pass 6 Hz;
           divided by the peak envelope of the muscle's own maximum voluntary contraction trial where the dataset has one,
           otherwise by the trial's own peak (then only the shape is compared).  The EMG side is the leg whose stance holds
           the soleus bursts.
  scores   per muscle: Pearson r between the stride-averaged curves (0-100 % of the cycle from foot strike), the lag that
           maximises r within +-60 ms, peak timing difference, and the amplitude ratio where MVC exists.  Ground reaction:
           RMSE and peak difference in body weights, r.
  E:\\splat\\.venv-myo\\Scripts\\python.exe r2_validate.py <raw dir> <subject nr> <out dir> [start s] [seconds]"""
import sys, os, json, glob, math, warnings, numpy as np, mujoco, myo_sim, ezc3d
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation as Rot
from myosuite.physics.joint_coupling import JointCouplings, project_joint_couplings
from myosuite.biomechanics.inverse_dynamics import clip_kinematics
from myosuite.biomechanics.static_optimization import StaticOptimizationConfig
from r2_core import muscle_effort
warnings.filterwarnings('ignore')
RAW, SUB, OUT = sys.argv[1], int(sys.argv[2]), sys.argv[3]; T0 = float(sys.argv[4]) if len(sys.argv) > 4 else None; DUR = float(sys.argv[5]) if len(sys.argv) > 5 else 10.0
os.makedirs(OUT, exist_ok=True); tag = 'Sub_%02d' % SUB
def say(*a): print('RV', *a, flush=True)
# ---------- subject, EMG channel map (sheet 1 of the dataset's spreadsheets: electrode numbers)
import zipfile, xml.etree.ElementTree as ET
def sheet_rows(path, k=1):
    ns = {'m': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}; z = zipfile.ZipFile(path); ss = []
    if 'xl/sharedStrings.xml' in z.namelist():
        for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('m:si', ns): ss.append(''.join(t.text or '' for t in si.iter('{%s}t' % ns['m'])))
    rows = []
    for row in ET.fromstring(z.read('xl/worksheets/sheet%d.xml' % k)).iter('{%s}row' % ns['m']):
        vals = []
        for c in row.findall('m:c', ns):
            v = c.find('m:v', ns)
            if v is not None: vals.append(ss[int(v.text)] if c.get('t') == 's' else v.text)
        rows.append(vals)
    return rows
subj = {r[0]: r for r in sheet_rows(os.path.join(RAW, 'subjects.xlsx')) if r and r[0].isdigit()}[str(SUB)]
mass = float(subj[4]); sex = subj[1]; height = float(subj[3]); age = subj[5]
ch = sheet_rows(os.path.join(RAW, 'emg_channels.xlsx'))
head = ch[0]; row = next(r for r in ch if r[:2] == [str(SUB), '3 m/s'])
EMGCH = {head[i]: int(row[i]) for i in range(2, 15)}
say('%s: %s, %.1f cm, %.1f kg, age %s; EMG electrodes %s' % (tag, sex, height, mass, age, EMGCH))
# ---------- C3D: plates, EMG, markers
C = ezc3d.c3d(os.path.join(RAW, tag + '_3ms.c3d'), extract_forceplat_data=True); P = C['parameters']
rp, ra = float(P['POINT']['RATE']['value'][0]), float(P['ANALOG']['RATE']['value'][0]); AL = P['ANALOG']['LABELS']['value']
Fz = sum(pl['force'][2] for pl in C['data']['platform']); an = C['data']['analogs'][0]; ta = np.arange(an.shape[1]) / ra
ML = P['POINT']['LABELS']['value']; PTS = C['data']['points'][:3] / 1000.0; tp = np.arange(PTS.shape[2]) / rp
def emg_env(x, fs):
    b, a = butter(4, [20 / (fs / 2), 450 / (fs / 2)], 'band'); y = np.abs(filtfilt(b, a, x - np.mean(x))); b, a = butter(4, 6 / (fs / 2)); return filtfilt(b, a, y)
def emg_of(name, c3d_an, labels): return c3d_an[labels.index('Voltage.emg%d' % EMGCH[name])]
# MVC normalisers
MVCFILE = {'Sol': 'Sol', 'GastrocMed': 'GastrocMedLat', 'GastrocLat': 'GastrocMedLat', 'Tibialis': 'TibAnt', 'VastusMed': 'VasMedLat', 'VastusLat': 'VasMedLat',
           'RectusFem': 'RecFem', 'BicepsFem': 'HamstringMedLat', 'Semitendi/membra': 'HamstringMedLat', 'GlutMax': 'GlutMax', 'GlutMed': 'GlutMed'}
mvc = {}
for mname, f in MVCFILE.items():
    fp = os.path.join(RAW, '%s_EMG_MVC_%s.c3d' % (tag, f))
    if os.path.exists(fp):
        M_ = ezc3d.c3d(fp); fs_ = float(M_['parameters']['ANALOG']['RATE']['value'][0]); lab_ = M_['parameters']['ANALOG']['LABELS']['value']
        try: mvc[mname] = float(emg_env(emg_of(mname, M_['data']['analogs'][0], lab_), fs_).max())
        except (ValueError, KeyError): pass
say('MVC available for', sorted(mvc))
# ---------- IK window
hdr = open(os.path.join(RAW, tag + '_3ms_IK.mot')).read().split('endheader')[1].split('\n')[1].split('\t')
IK = np.loadtxt(os.path.join(RAW, tag + '_3ms_IK.mot'), skiprows=11); col = {h.strip(): i for i, h in enumerate(hdr)}
t_ik = IK[:, 0]; T0 = t_ik[0] + 5 if T0 is None else T0; sel = (t_ik >= T0) & (t_ik < T0 + DUR); IK = IK[sel]; t_ik = t_ik[sel]; fps = 1 / np.median(np.diff(t_ik)); NT = len(IK)
# ---------- contact labels from the plates, stance foot from the markers
bf, af = butter(4, 50 / (ra / 2)); Fz_f = filtfilt(bf, af, Fz)                     # plate force low-passed at 50 Hz before event detection
Fz_ik = np.interp(t_ik, ta, Fz_f); stance = Fz_ik > 50.0
def mk(n): return np.stack([np.interp(t_ik, tp, PTS[k, ML.index(n)]) for k in range(3)], 1)
zL = 0.5 * (mk('LHEE')[:, 2] + mk('LMT2')[:, 2]); zR = 0.5 * (mk('RHEE')[:, 2] + mk('RMT2')[:, 2])
left = np.zeros(NT, bool)                                                         # each footfall (contiguous stance) goes to the foot lower on average over it
on = np.flatnonzero(np.diff(np.r_[0, stance.astype(int), 0]))
for a_, b_ in zip(on[::2], on[1::2]): left[a_:b_] = np.mean(zL[a_:b_]) < np.mean(zR[a_:b_])
right = stance & ~left
labels = np.zeros((NT, 6), bool); labels[:, 0] = labels[:, 1] = left; labels[:, 3] = labels[:, 4] = right
# ---------- model, scaled to the runner's mass
spec = myo_sim.load_spec('myolegs'); jb = spec.body('lumbar5').add_joint(); jb.type = mujoco.mjtJoint.mjJNT_BALL; jb.name = 'trunk_ball'
for g in spec.geoms:
    if g.parent.name in ('tibia_l', 'tibia_r', 'femur_l', 'femur_r'): g.contype = 0; g.conaffinity = 0
m = spec.compile(); d = mujoco.MjData(m)
mujoco.mj_setConst(m, d); s_m = mass / float(m.body_subtreemass[1]); m.body_mass[:] *= s_m; m.body_inertia[:] *= s_m; mujoco.mj_setConst(m, d)
jid = lambda n: m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)]; bid = lambda n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n)
mb = myo_sim.load('myolegs')[0]; key = m.qpos0.copy()
for j in range(mb.njnt):
    n = mujoco.mj_id2name(mb, mujoco.mjtObj.mjOBJ_JOINT, j); a, k = mb.jnt_qposadr[j], {0: 7, 1: 4}.get(int(mb.jnt_type[j]), 1); key[jid(n):jid(n) + k] = mb.key_qpos[0][a:a + k]
d.qpos[:] = key; mujoco.mj_kinematics(m, d); S_sac = d.xmat[bid('sacrum')].reshape(3, 3).copy()
Cm = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]], float)          # OpenSim (X fwd, Y up, Z right) -> MuJoCo (x left, y back, z up)
def osim_R(z, x, y): return Rot.from_euler('ZXY', [z, x, y]).as_matrix()
cpl = JointCouplings.from_model(m); rad = np.radians
JN = ['hip_flexion', 'hip_adduction', 'hip_rotation', 'knee_angle', 'ankle_angle', 'subtalar_angle', 'mtp_angle']
Q = np.zeros((NT, m.nq))
for t in range(NT):
    r = IK[t]; q = key.copy()
    q[:3] = Cm @ np.array([r[col['pelvis_tx']], r[col['pelvis_ty']], r[col['pelvis_tz']]])
    q[3:7] = np.roll(Rot.from_matrix(Cm @ osim_R(rad(r[col['pelvis_tilt']]), rad(r[col['pelvis_list']]), rad(r[col['pelvis_rotation']])) @ Cm.T).as_quat(), 1)
    RL = Cm @ osim_R(rad(r[col['lumbar_extension']]), rad(r[col['lumbar_bending']]), rad(r[col['lumbar_rotation']])) @ Cm.T
    jt = jid('trunk_ball'); q[jt:jt + 4] = np.roll(Rot.from_matrix(S_sac.T @ RL @ S_sac).as_quat(), 1)
    for s in 'lr':
        for jn in JN:
            nm = '%s_%s' % (jn, s)
            if nm in col and mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, nm) >= 0: q[jid(nm)] = rad(r[col[nm]])
    q2 = q[None].copy(); project_joint_couplings(cpl, q2); Q[t] = q2[0]
# floor: a plane; each stance frame puts the stance foot on it, flight frames interpolate (as the ARDY pipeline)
FOOTG = {s: [i for i in range(m.ngeom) if m.geom_bodyid[i] in (bid('calcn_' + s), bid('toes_' + s)) and m.geom_contype[i]] for s in 'lr'}
def lowest(q, s):
    d.qpos[:] = q; mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
    return min(d.geom_xpos[i][2] - (m.geom_size[i][0] if m.geom_type[i] in (2, 3) else np.abs(d.geom_xmat[i].reshape(3, 3)[2]) @ m.geom_size[i]) for i in FOOTG[s])
corr = np.full(NT, np.nan)
for t in range(NT):
    if left[t]: corr[t] = lowest(Q[t], 'l')
    elif right[t]: corr[t] = lowest(Q[t], 'r')
ok = np.isfinite(corr); Q[:, 2] -= np.interp(np.arange(NT), np.flatnonzero(ok), corr[ok])
kin = clip_kinematics(m, Q, 1 / fps, lowpass_hz=6.0)
say('%d frames at %.0f Hz from t=%.1f s; stance %d frames (left %d, right %d); model mass scaled x%.3f to %.1f kg' % (NT, fps, T0, stance.sum(), left.sum(), right.sum(), s_m, mass))
R2 = muscle_effort(m, kin, labels, fps); A = R2['activation']; names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(m.nu)]
bw = mass * 9.81
# ---------- EMG envelopes on the IK clock, side
ENV = {k: np.interp(t_ik, ta, emg_env(emg_of(k, an, AL), ra)) for k in EMGCH if k not in ('ErSpin', 'AbObl')}
sol_peak_stance = [left[i] for i in range(NT) if ENV['Sol'][i] > np.percentile(ENV['Sol'], 90)]
side = 'l' if np.mean(sol_peak_stance) > 0.5 else 'r'; st = left if side == 'l' else right
strikes = [i for i in range(1, NT) if st[i] and not st[i - 1]]
say('EMG side: %s (share of top-decile soleus EMG during left stance %.2f); %d strides' % (side, np.mean(sol_peak_stance), len(strikes) - 1))
MAP = {'Sol': ['soleus'], 'GastrocMed': ['gasmed'], 'GastrocLat': ['gaslat'], 'Tibialis': ['tibant'], 'VastusMed': ['vasmed'], 'VastusLat': ['vaslat'], 'RectusFem': ['recfem'],
       'BicepsFem': ['bflh'], 'Semitendi/membra': ['semiten', 'semimem'], 'GlutMax': ['glmax1', 'glmax2', 'glmax3'], 'GlutMed': ['glmed1', 'glmed2', 'glmed3']}
def cyc(x):
    rows = []
    for a_, b_ in zip(strikes[:-1], strikes[1:]): rows.append(np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, b_ - a_ + 1), x[a_:b_ + 1]))
    return np.array(rows)
scores = {}; curves = {}
for k, ms in MAP.items():
    est = np.mean([A[:, names.index(x + '_' + side)] for x in ms], 0); meas = ENV[k] / mvc[k] if k in mvc else ENV[k] / ENV[k].max()
    E, M_ = cyc(est).mean(0), cyc(meas).mean(0); curves[k] = (E, M_, cyc(meas).std(0))
    r0 = float(np.corrcoef(E, M_)[0, 1]); per = np.mean(np.diff(strikes)) / fps; best = (r0, 0)
    for lag in range(-6, 7):                                                     # +-60 ms at 100 Hz on the raw clock
        e2 = np.roll(est, lag); rr = float(np.corrcoef(cyc(e2).mean(0), M_)[0, 1]); best = max(best, (rr, lag))
    scores[k] = dict(r=round(r0, 3), best_r=round(best[0], 3), best_lag_ms=int(1000 * best[1] / fps), peak_est_pct=int(np.argmax(E)), peak_emg_pct=int(np.argmax(M_)),
                     amp_est_peak=round(float(E.max()), 3), amp_emg_peak_mvc=round(float(M_.max()), 3) if k in mvc else None)
Fm = Fz_ik / bw; Fe = R2['GRF'][:, 2] / bw
grf = dict(rmse_bw=round(float(np.sqrt(np.mean((Fm - Fe) ** 2))), 3), r=round(float(np.corrcoef(Fm, Fe)[0, 1]), 3),
           peak_meas_bw=round(float(cyc(Fm).mean(0).max()), 2), peak_est_bw=round(float(cyc(Fe).mean(0).max()), 2), unexplained_stance_bw=round(float(np.median(R2['unexplained'][stance]) / bw), 3),
           unexplained_flight_bw=round(float(np.median(R2['unexplained'][~stance]) / bw), 3))
res = dict(subject=SUB, sex=sex, mass_kg=mass, height_cm=height, age=age, window_s=[T0, T0 + DUR], fps=fps, emg_side=side, strides=len(strikes) - 1, mvc=sorted(mvc), grf=grf, muscles=scores)
json.dump(res, open(os.path.join(OUT, '%s_r2.json' % tag), 'w'), indent=1)
np.savez_compressed(os.path.join(OUT, '%s_r2.npz' % tag), t=t_ik, Fm=Fm, Fe=Fe, A=A, names=np.array(names), stance=stance, left=left, strikes=np.array(strikes),
                    **{'curve_' + k.replace('/', '_'): np.stack(v) for k, v in curves.items()})
say(json.dumps(res, indent=1))
