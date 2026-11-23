import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import compress_job as J
rng = np.random.default_rng(0)
def vol():
    X, Y, Z = np.meshgrid(np.arange(80) - 40, np.arange(64) - 32, np.arange(72), indexing='ij'); a = rng.uniform(28, 36); r = np.sqrt((X / a) ** 2 + (Y / (a * .7)) ** 2)
    L = np.zeros(X.shape, np.uint8); L[r < 1] = 1; L[r < .8] = 3; L[np.hypot(X - 12, Y) < 5] = 40; L[(np.abs(np.hypot(X + 12, Y) - 7) < 1)] = 41   # a filled bone and a thin shell
    return L
tr, va, te = [vol() for _ in range(12)], [vol() for _ in range(2)], [vol() for _ in range(2)]
res = J.run(tr, va, te, {1: 'subcutaneous fat', 3: 'other lean', 40: 'femur_left', 41: 'rib_left_1'}, 'cuda', 6, variants=((4, 8),))
r = res['F4_C8']; print({k: r[k] for k in ('token_mm', 'tokens_for_75_litre_body', 'median_dice', 'by_thickness_mm', 'subcut_fat_thickness_err_mm')}); print(r['worst'])
