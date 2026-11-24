# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Independent Torch reference and deterministic synthetic inputs; test-only."""

from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F

from .support import RCP_LN2

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class Case:
    name: str
    T: int = 129
    B: int = 1
    H: int = 4
    D: int = 128
    chunk: int = 64
    lengths: tuple = ()

    def describe(self):
        return {"provenance": "synthetic, not captured model shapes", **asdict(self)}


CORRECTNESS_CASES = (
    Case("single", T=1),
    Case("short", T=63),
    Case("full", T=64),
    Case("tail", T=65),
    Case("multi", T=129),
    Case("batch_feature_tail", T=129, B=2, H=3, D=80),
    Case("chunk32", T=97, chunk=32),
    Case("chunk128", T=257, chunk=128),
    Case("varlen", T=193, lengths=(1, 63, 64, 65)),
)

PERFORMANCE_CASES = (
    Case("tiny", T=64),
    Case("tail", T=65),
    Case("medium", T=512),
    Case("prefill2k", T=2048),
    Case("prefill8k", T=8192),
    Case("prefill16k", T=16384),
    Case("wide_heads", T=2048, H=16),
    Case("varlen_even", T=8192, lengths=(4096, 4096)),
    Case("varlen_ragged", T=2177, lengths=(1, 63, 65, 2048)),
)


def make_inputs(
    case, dtype=torch.bfloat16, bias=True, safe=False, output=torch.float32
):
    generator = torch.Generator(device="cuda").manual_seed(20261027)
    g = torch.randn(
        (case.B, case.T, case.H, case.D), device="cuda", generator=generator
    ).to(dtype)
    a = torch.randn(case.H, device="cuda", generator=generator) * 0.3 - 1
    a = a.view(1, 1, case.H, 1)
    b = (
        torch.randn(case.H * case.D, device="cuda", generator=generator) * 0.2
        if bias
        else None
    )
    kwargs = dict(
        g_bias=b,
        chunk_size=case.chunk,
        safe_gate=safe,
        output_dtype=output,
    )
    if case.lengths:
        assert case.B == 1 and sum(case.lengths) == case.T
        boundaries = [0]
        indices = []
        for seq, length in enumerate(case.lengths):
            boundaries.append(boundaries[-1] + length)
            indices.extend(
                (seq, c) for c in range((length + case.chunk - 1) // case.chunk)
            )
        kwargs["cu_seqlens"] = torch.tensor(
            boundaries, device="cuda", dtype=torch.int64
        )
        kwargs["chunk_indices"] = torch.tensor(
            indices, device="cuda", dtype=torch.int64
        ).reshape(-1, 2)
    return (g, a), kwargs


def torch_reference(
    raw_g,
    A_log,
    g_bias=None,
    beta=1.0,
    threshold=20.0,
    cu_seqlens=None,
    chunk_indices=None,
    chunk_size=64,
    output_dtype=torch.float32,
    safe_gate=False,
    lower_bound=-5.0,
):
    """Gate then inclusive cumsum, independently of Triton's triangular dot."""
    B, T, H, D = raw_g.shape
    x = raw_g.float()
    if g_bias is not None:
        x = x + g_bias.float().view(1, 1, H, D)
    a = A_log.float().view(1, 1, H, 1).exp()
    gate = (
        lower_bound * torch.sigmoid(a * x)
        if safe_gate
        else -a * F.softplus(x, beta=beta, threshold=threshold)
    )
    y = torch.empty_like(gate)
    if cu_seqlens is None:
        for start in range(0, T, chunk_size):
            y[:, start : start + chunk_size] = gate[
                :, start : start + chunk_size
            ].cumsum(1)
    else:
        assert B == 1
        boundaries = cu_seqlens.cpu().tolist()
        for bos, eos in zip(boundaries, boundaries[1:]):
            for start in range(bos, eos, chunk_size):
                stop = min(start + chunk_size, eos)
                y[:, start:stop] = gate[:, start:stop].cumsum(1)
    return (y * RCP_LN2).to(output_dtype or raw_g.dtype)


def assert_close(actual, expected):
    if actual.dtype == torch.float32:
        rtol, atol = 1e-4, 1e-5
    elif actual.dtype == torch.float16:
        rtol, atol = 2e-3, 2e-3
    else:
        rtol, atol = 1e-2, 1e-2
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol, equal_nan=True)
