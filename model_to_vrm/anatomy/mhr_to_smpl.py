"""Convert the model's fitted MHR body (body/mhr_fit.npz) to a SMPL body, so SMPL-based anatomy models (HIT) can run on it:
  E:\\splat\\.venv-body\\Scripts\\python.exe mhr_to_smpl.py <model dir> <mhr assets dir> <SMPL model dir> <female|male> [out.npz]
  conversion  Meta's own MHR -> SMPL tool (facebookresearch/MHR, tools/mhr_smpl_conversion, PyTorch fitting): the MHR
              body is rebuilt from the saved identity / model parameters at LOD 1 (the mesh the tool's surface mapping
              is made for), in MHR's own frame (cm, Y up), and SMPL is fitted to it
  frames      world (the model's, metres, Z up) = SMPL (metres, Y up) @ toB.T + trans, the same toB / trans as fit_mhr.py
  writes      anatomy/smpl_fit.npz: betas, body_pose, global_orient, transl (SMPL frame), toB, trans, gender, SMPL vertices
              and faces in the world, the MHR LOD-1 vertices in the world, per-vertex fitting error"""
import sys, os, numpy as np, torch, smplx
from pathlib import Path
R, ad, smpl_dir, gender = sys.argv[1:5]; out = sys.argv[5] if len(sys.argv) > 5 else os.path.join(R, 'anatomy', 'smpl_fit.npz')
TOOL = r'E:\code\external\MHR\tools\mhr_smpl_conversion'; sys.path.insert(0, TOOL)
from mhr.mhr import MHR
from conversion import Conversion
os.makedirs(os.path.dirname(out), exist_ok=True)
Z = np.load(os.path.join(R, 'body', 'mhr_fit.npz'))
toB = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float); tr = Z['trans'].astype(float)   # as in rig/fit_mhr.py
m = MHR.from_files(folder=Path(ad), device=torch.device('cpu'), lod=1)
with torch.no_grad():
    v, _ = m(torch.tensor(Z['identity']), torch.tensor(Z['model_params']), torch.zeros(1, 72))
mv = v.numpy()                                                    # (1, n, 3) cm, MHR frame
mw = mv[0] / 100.0 @ toB.T + tr
print('MS MHR LOD1: %d vertices; world height %.3f m' % (len(mw), np.ptp(mw[:, 2])), flush=True)
smpl = smplx.SMPL(model_path=os.path.join(smpl_dir, 'SMPL_%s.pkl' % gender.upper()), gender=gender)
out = os.path.abspath(out); os.chdir(TOOL)                       # the tool loads its assets from ./assets
conv = Conversion(mhr_model=m, smpl_model=smpl, method='pytorch')
res = conv.convert_mhr2smpl(mhr_vertices=mv, single_identity=True, return_smpl_parameters=True, return_smpl_vertices=True, return_fitting_errors=True)
P = {k: (v_.detach().cpu().numpy() if torch.is_tensor(v_) else np.asarray(v_)) for k, v_ in res.result_parameters.items()}
sv = np.asarray(res.result_vertices)[0]; sw = sv @ toB.T + tr       # SMPL vertices (m, SMPL frame) -> world
err = np.asarray(res.result_errors).ravel()
print('MS SMPL params: %s' % {k: v_.shape for k, v_ in P.items()}, flush=True)
print('MS fitting error: %s' % np.round(err[:10], 4), flush=True)
np.savez(out, **{k: v_ for k, v_ in P.items()}, toB=toB, trans=tr, gender=gender, V=sw, F=np.asarray(smpl.faces), mhr_V=mw, errors=err)
print('MS wrote', out)
