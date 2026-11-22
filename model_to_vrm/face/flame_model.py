"""FLAME 2023 Open (CC-BY-4.0, Li et al. 2017) as a small torch model, in the twin's axes (z up, facing -y).
  load()            arrays from the official pickle (numpy/scipy only; other classes stubbed)
  Flame(...)(...)   vertices and joints for shape beta (first 300 of shapedirs), expression psi (last 100), jaw and eye turns
Joints: 0 global, 1 neck, 2 jaw, 3 left eye, 4 right eye (FLAME's order)."""
import pickle, numpy as np, torch
PKL = 'E:/figures/models/flame2023open/flame2023_Open.pkl'
AX = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)                              # FLAME (y up, facing +z) -> twin (z up, facing -y)
def load():
    class Any:
        def __init__(self, *a, **k): pass
        def __setstate__(self, s): self.__dict__.update(s if isinstance(s, dict) else {'state': s})
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod.startswith(('numpy', 'scipy', 'collections', 'builtins', 'copyreg', '_codecs')): return super().find_class(mod, name)
            return Any
    d = U(open(PKL, 'rb'), encoding='latin1').load()
    return {'v': np.asarray(d['v_template']) @ AX.T, 'f': np.asarray(d['f']).astype(np.int64), 'sd': np.einsum('ij,vjk->vik', AX, np.asarray(d['shapedirs'])),
            'pd': np.asarray(d['posedirs']), 'jr': np.asarray(d['J_regressor'].todense()), 'w': np.asarray(d['weights']), 'par': np.asarray(d['kintree_table'])[0]}
def rodr(r):
    th = r.norm(dim=-1, keepdim=True).clamp_min(1e-8); k = r / th; K = torch.zeros(*r.shape[:-1], 3, 3, device=r.device)
    K[..., 0, 1], K[..., 0, 2], K[..., 1, 0], K[..., 1, 2], K[..., 2, 0], K[..., 2, 1] = -k[..., 2], k[..., 1], k[..., 2], -k[..., 0], -k[..., 1], k[..., 0]
    th = th[..., None]; return torch.eye(3, device=r.device) + torch.sin(th) * K + (1 - torch.cos(th)) * K @ K
class Flame(torch.nn.Module):
    def __init__(self, d, dev='cuda'):
        super().__init__(); t = lambda a: torch.tensor(a, dtype=torch.float32, device=dev)
        self.v, self.sd, self.jr, self.w = t(d['v']), t(d['sd']), t(d['jr']), t(d['w']); self.f = d['f']; self.par = d['par']; self.dev = dev
        # pose correctives: FLAME's posedirs are in its own axes; rotate them into ours
        pd = np.einsum('ij,vjk->vik', AX, d['pd']); self.pd = t(pd.reshape(-1, pd.shape[-1]))
    def forward(self, beta, psi, jaw=None, eyes=None, neck=None):
        B = torch.cat([beta, psi]); v = self.v + torch.einsum('vik,k->vi', self.sd[:, :, :B.shape[0]], B)
        J = self.jr @ v; z = torch.zeros(3, device=self.dev)
        rots = [z, neck if neck is not None else z, jaw if jaw is not None else z] + ([eyes[0], eyes[1]] if eyes is not None else [z, z])
        R = rodr(torch.stack(rots))
        feat = (R[1:] - torch.eye(3, device=self.dev)).reshape(-1)                        # FLAME: 4 joints x 9 = 36 pose features
        v = v + (self.pd @ feat).reshape(-1, 3)
        G, Jw = [None] * 5, [None] * 5                                                    # x' = G_j (x - J_j) + Jw_j
        for j in range(5):
            p = int(self.par[j])
            if j == 0 or p < 0 or p > 4: G[j] = R[j]; Jw[j] = J[j]
            else: G[j] = G[p] @ R[j]; Jw[j] = Jw[p] + G[p] @ (J[j] - J[p])
        Gs = torch.stack(G); Ts = torch.stack([Jw[j] - G[j] @ J[j] for j in range(5)])
        vp = torch.einsum('vj,jab,vb->va', self.w, Gs, v) + self.w @ Ts
        return vp, torch.stack(Jw)
