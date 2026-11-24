# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Dependency bindings for the test-only upstream extraction, never production."""

import os

import torch
import triton.language as tl
import triton.language.extra.cuda.libdevice as libdevice

FLA_CHUNK_SIZE = 64
RCP_LN2 = 1.4426950216
log = libdevice.fast_logf if os.environ.get("FLA_USE_FAST_OPS", "0") == "1" else tl.log


def cdiv(a, b):
    return -(a // -b)


def prepare_chunk_indices(cu_seqlens, chunk_size):
    """Synchronous test shim of upstream index.py:25-33 (positive lengths).

    The target host/kernel definitions are unchanged. vLLM's async H2D and
    tensor_cache are intentionally not emulated; this helper is never timed.
    Like that upstream helper, zero-length sequences are outside its contract.
    """
    lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
    if not lengths or min(lengths) <= 0:
        raise NotImplementedError("Upstream index shim requires positive lengths")
    indices = torch.cat([torch.arange(cdiv(n, chunk_size)) for n in lengths])
    pairs = torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1)
    return pairs.to(device=cu_seqlens.device, dtype=cu_seqlens.dtype)
