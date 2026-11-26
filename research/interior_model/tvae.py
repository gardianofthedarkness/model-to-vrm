"""Direction D2, first form (plan v8 14.7, after MAGI-1): a patch transformer VAE for tissue labels.
Patches of 8^3 voxels (8 mm) are the tokens, exactly the flow transformer's tokens, so the generator is unchanged. A 3D
conv patch embedding, pre-norm transformer blocks with QK-norm and 3D rotary positions (relative, so the code of a place
does not depend on where the crop starts), LayerNorm + linear to the latent mean and log-variance (C channels; KL term),
and a symmetric decoder ending in a per-token linear "pixel shuffle" to 8^3 voxels x classes plus one 3x3x3 conv.
Same interface as the conv autoencoder: encode(x) -> (B, C, gx, gy, gz) means; decode(z) -> (B, NC, 8gx, 8gy, 8gz) logits;
forward(x) samples the latent in training and keeps the KL in self.kl."""
import math, torch, torch.nn as nn, torch.nn.functional as F_

def rope_3d(q, k, g):
    """rotate q, k (B, H, N, D) by token position on the grid g = (gx, gy, gz); D split into three axis parts"""
    D = q.shape[-1]; per = (D // 3) // 2 * 2; dev = q.device
    ii = torch.stack(torch.meshgrid(*[torch.arange(n, device=dev, dtype=torch.float32) for n in g], indexing='ij'), -1).reshape(-1, 3)
    out = []
    for t in (q, k):
        parts = []
        for a in range(3):
            x = t[..., a * per:(a + 1) * per]; f = 1.0 / (100.0 ** (torch.arange(0, per, 2, device=dev, dtype=torch.float32) / per))
            ang = ii[:, a:a + 1] * f[None]; c, s = ang.cos()[None, None].to(t.dtype), ang.sin()[None, None].to(t.dtype)
            x1, x2 = x[..., 0::2], x[..., 1::2]; parts.append(torch.stack([x1 * c - x2 * s, x1 * s + x2 * c], -1).flatten(-2))
        parts.append(t[..., 3 * per:]); out.append(torch.cat(parts, -1))
    return out

class Block(nn.Module):
    def __init__(s, d, h):
        super().__init__(); s.h = h; s.n1 = nn.LayerNorm(d); s.qkv = nn.Linear(d, 3 * d); s.qn = nn.RMSNorm(d // h); s.kn = nn.RMSNorm(d // h)
        s.proj = nn.Linear(d, d); s.n2 = nn.LayerNorm(d); s.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
    def forward(s, x, g):
        B, N, D = x.shape; q, k, v = s.qkv(s.n1(x)).view(B, N, 3, s.h, D // s.h).permute(2, 0, 3, 1, 4)
        q, k = rope_3d(s.qn(q), s.kn(k), g); a = F_.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D)
        x = x + s.proj(a); return x + s.mlp(s.n2(x))

class TVAE(nn.Module):
    def __init__(s, NC, C=8, P=8, d=256, depth=8, heads=8, soft=True):
        super().__init__(); s.NC, s.C, s.P, s.soft = NC, C, P, soft; s.kl = torch.zeros(())
        s.emb = None if soft else nn.Embedding(NC + 1, 16)
        s.patch = nn.Conv3d(NC if soft else 16, d, P, stride=P)
        s.enc = nn.ModuleList([Block(d, heads) for _ in range(depth)]); s.en = nn.LayerNorm(d); s.to_lat = nn.Linear(d, 2 * C)
        s.from_lat = nn.Linear(C, d); s.dec = nn.ModuleList([Block(d, heads) for _ in range(depth)]); s.dn = nn.LayerNorm(d)
        s.shuffle = nn.Linear(d, NC * P ** 3); s.refine = nn.Conv3d(NC, NC, 3, padding=1)
    def _lat(s, x):
        h = s.patch(x if s.soft else s.emb(x).permute(0, 4, 1, 2, 3)); B, D = h.shape[:2]; g = tuple(h.shape[2:]); h = h.flatten(2).transpose(1, 2)
        for b in s.enc: h = b(h, g)
        mu, lv = s.to_lat(s.en(h)).chunk(2, -1); return mu, lv.clamp(-30, 20), g
    def encode(s, x):
        mu, lv, g = s._lat(x); return mu.transpose(1, 2).reshape(mu.shape[0], s.C, *g)
    def decode(s, z):
        B = z.shape[0]; g = tuple(z.shape[2:]); h = s.from_lat(z.flatten(2).transpose(1, 2))
        for b in s.dec: h = b(h, g)
        P = s.P; y = s.shuffle(s.dn(h)).view(B, *g, s.NC, P, P, P).permute(0, 4, 1, 5, 2, 6, 3, 7).reshape(B, s.NC, g[0] * P, g[1] * P, g[2] * P)
        return y + s.refine(y)
    def forward(s, x):
        mu, lv, g = s._lat(x)
        z = mu + torch.randn_like(mu) * (0.5 * lv).exp() if s.training else mu
        s.kl = 0.5 * (mu.float() ** 2 + lv.float().exp() - 1 - lv.float()).sum(-1).mean()
        return s.decode(z.transpose(1, 2).reshape(z.shape[0], s.C, *g))
