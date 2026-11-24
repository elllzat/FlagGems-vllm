# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
# Copyright 2026 FlagOS Contributors
# Original flash-linear-attention code: MIT, Copyright (c) 2023-2025,
# Songlin Yang, Yu Zhang.
"""Experimental native Triton scan; not a registered FlagGems operator.

Copied from installed vLLM 0.32.0 kernels.py:763-900, SHA256
6f5865164682e2ea475995fbdeb1df6942baecfb06abeee3a4468c7c6344775e.
Only the triangular dot block changes to masked FP32 cumsum.
Native wrapper, API, heuristics and autotune are retained verbatim.
Finite-gate accuracy is tested; NaN/Inf propagation is NOT fully equivalent.
"""

import torch

from vllm.third_party.flash_linear_attention.ops.index import prepare_chunk_indices
from vllm.third_party.flash_linear_attention.ops.op import log
from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import RCP_LN2, cdiv


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BD": BD}, num_warps=num_warps)
        for BD in [32, 64]
        for num_warps in [2, 4, 8]
    ],
    key=["H", "D", "BT", "IS_VARLEN"],
)
@triton.jit(do_not_specialize=["T"])
def kda_gate_cumsum_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    cu_seqlens,
    chunk_indices,
    cumsum_scale,
    beta,
    threshold,
    SAFE_GATE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    T,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_d, i_t, i_bh = tl.program_id(0), tl.program_id(1).to(tl.int64), tl.program_id(2)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BT)
    o_d = i_d * BD + tl.arange(0, BD)
    m_t = o_t < T
    m_g = m_t[:, None] & (o_d[None, :] < D)
    p_g = g + (bos * H + i_h) * D + o_t[:, None] * (H * D) + o_d[None, :]
    p_y = y + (bos * H + i_h) * D + o_t[:, None] * (H * D) + o_d[None, :]

    b_g = tl.load(p_g, mask=m_g, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_bias = tl.load(g_bias + i_h * D + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = tl.exp(b_a) if SAFE_GATE else -tl.exp(b_a)
    if SAFE_GATE:
        # y = lower_bound * sigmoid(exp(A) * (g + g_bias)), bounded to
        # (lower_bound, 0) for safe-gate checkpoints.
        b_gate = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_g)))
    else:
        b_g_scaled = b_g * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_g,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = b_a * b_softplus

    # Inclusive FP32 scan, with invalid lanes zeroed before accumulation.
    b_gate = tl.where(m_g, b_gate, 0.0)
    b_y = tl.cumsum(b_gate, axis=0, dtype=tl.float32) * cumsum_scale
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_g)


def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = FLA_CHUNK_SIZE,
    output_dtype: torch.dtype | None = torch.float,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert raw_g.shape[0] == 1, (
            "Only batch size 1 is supported when cu_seqlens are provided"
        )
    B, T, H, D = raw_g.shape
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = cdiv(T, chunk_size) if cu_seqlens is None else len(chunk_indices)

    A_log = A_log.reshape(-1)
    if g_bias is not None:
        g_bias = g_bias.reshape(-1)
    y = torch.empty_like(raw_g, dtype=output_dtype or raw_g.dtype)

    def grid(meta):
        return (cdiv(meta["D"], meta["BD"]), NT, B * H)

    kda_gate_cumsum_fwd_kernel[grid](
        g=raw_g,
        A=A_log,
        y=y,
        g_bias=g_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        # RCP_LN2 folds in the natural-log -> log2 conversion so downstream
        # exp2-based kernels reproduce exp(g). Keep this in sync with the
        # `use_exp2=True` path in `_chunk_kda_fwd_with_cumulative_g`.
        cumsum_scale=RCP_LN2,
        beta=beta,
        threshold=threshold,
        SAFE_GATE=safe_gate,
        LOWER_BOUND=lower_bound,
        T=T,
        H=H,
        D=D,
        BT=chunk_size,
    )
    return y
