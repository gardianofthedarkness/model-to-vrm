# /// script
# dependencies = ["torch==2.4.1", "numpy", "packaging", "ninja", "einops", "transformers", "pip", "setuptools", "wheel"]
# ///
"""Transformer against Mamba at whole-body token counts: time and peak GPU memory of one training step
(forward + backward, batch 1, bfloat16), 8 layers, width 384. Random tokens; this measures cost only, not quality.
Transformer: pre-norm blocks with scaled-dot-product attention (flash kernel) and a 4x MLP.
Mamba: bidirectional (one Mamba forward, one on the reversed sequence, summed) per layer, from the mamba-ssm
package, installed at run time; if it cannot be installed the error is reported and only the transformer is timed."""
import sys, json, time, subprocess, torch, torch.nn as nn, torch.nn.functional as F
dev = 'cuda'; D, LAYERS = 384, 8
def log(*a): print('TS', *a, flush=True)
class TBlock(nn.Module):
    def __init__(s):
        super().__init__(); s.n1 = nn.LayerNorm(D); s.qkv = nn.Linear(D, 3 * D); s.o = nn.Linear(D, D); s.n2 = nn.LayerNorm(D); s.mlp = nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))
    def forward(s, x):
        B, N, _ = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, N, 3, 6, D // 6).permute(2, 0, 3, 1, 4)
        x = x + s.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)); return x + s.mlp(s.n2(x))
def bench(make, n, steps=3):
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); m = make().to(dev); opt = torch.optim.AdamW(m.parameters(), 1e-4); x = torch.randn(1, n, D, device=dev); ts = []
        for i in range(steps + 1):
            torch.cuda.synchronize(); t = time.time()
            with torch.autocast('cuda', dtype=torch.bfloat16): loss = m(x).float().pow(2).mean()
            opt.zero_grad(); loss.backward(); opt.step(); torch.cuda.synchronize(); ts.append(time.time() - t)
        r = dict(seconds_per_step=round(min(ts[1:]), 3), peak_gb=round(torch.cuda.max_memory_allocated() / 1e9, 2), params_m=round(sum(p.numel() for p in m.parameters()) / 1e6, 1))
        del m, opt, x; return r
    except torch.OutOfMemoryError: return dict(oom=True)
    except Exception as e: return dict(error='%s %s' % (type(e).__name__, str(e)[:150]))
out = dict(gpu=torch.cuda.get_device_name(0), width=D, layers=LAYERS, transformer={}, mamba={})
NS = [5000, 20000, 43000, 80000, 160000, 350000]
import shutil; log("nvcc", shutil.which("nvcc"))
for n in NS:
    out['transformer'][n] = bench(lambda: nn.Sequential(*[TBlock() for _ in range(LAYERS)]), n); log('transformer', n, out['transformer'][n])
    if out['transformer'][n].get('oom') or out['transformer'][n].get('seconds_per_step', 0) > 60: break
r = subprocess.run([sys.executable, '-m', 'pip', 'install', 'mamba-ssm', '--no-build-isolation'], capture_output=True, text=True)
if r.returncode != 0:
    r2 = subprocess.run(['uv', 'pip', 'install', '--python', sys.executable, 'mamba-ssm', '--no-build-isolation'], capture_output=True, text=True); r = r2 if r2.returncode == 0 else r
log('mamba install rc', r.returncode, ((r.stderr or '') + (r.stdout or ''))[-900:] if r.returncode else '')
try:
    from mamba_ssm import Mamba
    class MBlock(nn.Module):
        def __init__(s):
            super().__init__(); s.n = nn.LayerNorm(D); s.f = Mamba(d_model=D, d_state=16, d_conv=4, expand=2); s.b = Mamba(d_model=D, d_state=16, d_conv=4, expand=2)
        def forward(s, x):
            h = s.n(x); return x + s.f(h) + s.b(h.flip(1)).flip(1)
    for n in NS:
        out['mamba'][n] = bench(lambda: nn.Sequential(*[MBlock() for _ in range(LAYERS)]), n); log('mamba', n, out['mamba'][n])
        if out['mamba'][n].get('oom'): break
except Exception as e: out['mamba_error'] = '%s %s' % (type(e).__name__, str(e)[:300]); log('mamba unavailable', out['mamba_error'])
log('SUMMARY ' + json.dumps(out))
