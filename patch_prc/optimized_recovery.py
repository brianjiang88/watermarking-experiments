"""Scoped execution optimizations for a fixed FP32 pipeline on one GPU.

Prompt cache is per pipeline instance. Create a new instance if text-encoder
weights, tokenizer, device, or dtype change. No global no_grad surrounds recovery.
"""
from contextlib import contextmanager
from unittest.mock import patch
from weakref import WeakKeyDictionary

_PROMPT_CACHE=WeakKeyDictionary()


@contextmanager
def optimized_recovery(pipe, cache_prompt=True):
    import torch
    params=list(pipe.vae.parameters())
    flags=[p.requires_grad for p in params]
    versions=[p._version for p in params]
    assert all(p.dtype==torch.float32 for p in params)
    assert not pipe.vae.training, 'Require the pretrained evaluation-mode VAE'
    cached=None
    if cache_prompt:
        if pipe not in _PROMPT_CACHE:
            with torch.no_grad():
                _PROMPT_CACHE[pipe]=tuple(t.detach() if t is not None else None for t in pipe.encode_prompt('', 'cuda', 1, True, None))
        cached=_PROMPT_CACHE[pipe]
    original_encode=pipe.encode_prompt
    def encode_prompt(*args,**kwargs):
        if args==('', 'cuda', 1, True, None) and not kwargs:return cached
        return original_encode(*args,**kwargs)
    def decode(latents):
        # Same scaling, sample slicing, decode, and concatenation as the released
        # decode_image_for_gradient_float; only the deepcopy is removed.
        scaled_latents=1/0.18215*latents
        return torch.cat([pipe.vae.decode(scaled_latents[i:i+1]).sample for i in range(len(latents))])
    try:
        for p in params:p.requires_grad_(False)
        with patch.object(pipe,'decode_image_for_gradient_float',decode):
            if cache_prompt:
                with patch.object(pipe,'encode_prompt',encode_prompt):yield
            else:yield
    finally:
        for p,flag in zip(params,flags):p.requires_grad_(flag)
        assert [p._version for p in params]==versions, 'VAE weights were modified'
        assert [p.requires_grad for p in params]==flags, 'Parameter flags were not restored'
