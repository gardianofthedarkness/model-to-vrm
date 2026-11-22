"""NVIDIA ARDY text-to-motion on this laptop: the text encoder on the CPU (once per prompt, cached), the motion model on the GPU.

  python ardy_run.py encode "<prompt>" ["<prompt>" ...]      CPU, bf16, ~16 GB RAM; writes one cached vector per prompt
  python ardy_run.py gen "<prompt>" [generate.py flags]       GPU; reads the cached vector (encode it first), runs ARDY's own
                                                              scripts/generate.py unchanged (e.g. --model soma --duration 5)

Run with ARDY's own environment (E:\\splat\\.venv-ardy; it needs numpy<2, unlike the pipeline's). The encoder is the exact
one ARDY was trained with (LLM2Vec: Llama-3-8B-Instruct + the McGill-NLP MNTP and supervised adapters); a generic Llama
embedding would not match. The Hugging Face token (for the gated Llama weights) comes from the vault and is handed only to
this process's environment. Two processes, never one: the 16 GB encoder is gone from memory before the motion model loads."""
import hashlib, os, sys
from pathlib import Path
import numpy as np

ARDY = Path(os.environ.get('ARDY_DIR', r'E:\code\ardy'))
CACHE = Path(os.environ.get('ARDY_TEXT_CACHE', r'E:\figures\ardy\text_cache'))
key = lambda t: hashlib.sha1(t.strip().encode('utf-8')).hexdigest()


def encode(prompts):
    sys.path.insert(0, str(Path(__file__).parent))
    import vault, torch
    os.environ['HF_TOKEN'] = vault.get('huggingface')
    torch.set_num_threads(os.cpu_count())
    from ardy.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
    from ardy.model.load_model import TEXT_ENCODER_PRESETS
    kw = dict(TEXT_ENCODER_PRESETS['llm2vec']['kwargs']); kw['device'] = 'cpu'          # bf16 as ARDY's default
    enc = LLM2VecEncoder(**kw); CACHE.mkdir(parents=True, exist_ok=True)
    for t in prompts:
        f = CACHE / (key(t) + '.npy')
        if f.exists(): print('AR cached   ', t); continue
        v, _ = enc(t); np.save(f, v.float().cpu().numpy())                                  # (1, 4096)
        (CACHE / (key(t) + '.txt')).write_text(t.strip(), encoding='utf-8'); print('AR encoded  ', t)


class CachedEncoder:
    """Stands in for ARDY's text encoder: same call and return as LLM2VecEncoder, read from the cache."""
    def __init__(self): self.device = 'cpu'
    def to(self, device=None, dtype=None):
        if device is not None: self.device = device
        return self
    def eval(self): return self
    def __call__(self, texts):
        import torch
        one = isinstance(texts, str); texts = [texts] if one else texts; out = []
        for t in texts:
            if not t.strip(): out.append(np.zeros((1, 4096), np.float32)); continue        # ARDY zeroes empty prompts itself
            f = CACHE / (key(t) + '.npy')
            if not f.exists(): raise SystemExit('AR prompt not encoded yet, run: ardy_run.py encode "%s"' % t)
            out.append(np.load(f))
        v = torch.tensor(np.stack(out)).to(self.device); n = [1] * len(texts)
        return (v[0], n[0]) if one else (v, n)


def gen(argv):
    sys.path.insert(0, str(ARDY / 'scripts'))
    import generate, ardy.model as am
    real = generate.load_model
    generate.load_model = lambda *a, **k: real(*a, **{**k, 'text_encoder': CachedEncoder()})
    sys.argv = ['generate.py'] + argv; generate.main()


if __name__ == '__main__':
    cmd, rest = sys.argv[1], sys.argv[2:]
    encode(rest) if cmd == 'encode' else gen(rest)
