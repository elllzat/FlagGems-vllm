# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Ascend gate plus sequence/chunk-local FP32 scan; no external-op fallback."""

import math
import numbers
import statistics

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.ops.fused_kda_gate_chunk_cumsum import (
    _FLOAT_DTYPES,
    RCP_LN2,
    _check_tensor,
)
from flaggems_vllm.utils import libentry, libtuner


def _bench_graph(call, quantiles=None):
    """Choose the local tile with native NPU Graph replay, outside inference."""
    api = runtime.torch_device_fn
    for _ in range(5):
        call()
    api.synchronize()
    stream = api.Stream()
    stream.wait_stream(api.current_stream())
    graph = api.NPUGraph()
    with api.stream(stream):
        graph.capture_begin()
        for _ in range(32):
            call()
        graph.capture_end()
    stream.synchronize()
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(5):
        begin, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / 32)
    if quantiles is None:
        return statistics.median(samples)
    samples.sort()
    return [samples[round(q * (len(samples) - 1))] for q in quantiles]


@libentry()
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=runtime.get_tuned_config("fused_kda_gate_chunk_cumsum_ascend_scan"),
    key=[
        "H",
        "D",
        "BT",
        "IS_VARLEN",
        "AUTO_INDICES",
        "HAS_BIAS",
        "SAFE_GATE",
        "TUNE_T",
        "TUNE_NT",
        "TUNE_B",
        "TUNE_N",
    ],
    do_bench=_bench_graph,
)
@triton.jit(do_not_specialize=["T"])
def _ascend_scan_kernel(
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
    ROWS: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    AUTO_INDICES: tl.constexpr,
    TUNE_T,
    TUNE_NT,
    TUNE_B,
    TUNE_N,
    INT64_INDEX: tl.constexpr,
):
    BLOCK_T: tl.constexpr = ROWS
    i_d = tl.program_id(0)
    i_t, i_bh = tl.program_id(1), tl.program_id(2)
    if INT64_INDEX:
        i_t = i_t.to(tl.int64)
    i_b, i_h = i_bh // H, i_bh % H
    if INT64_INDEX:
        i_b = i_b.to(tl.int64)
    if IS_VARLEN:
        if AUTO_INDICES:
            # Rectangular sequence/chunk grid: no host tensor reads or index
            # construction. Empty/short sequences simply skip excess programs.
            i_n = i_b
        else:
            i_n, i_t = (
                tl.load(chunk_indices + i_t * 2).to(tl.int32),
                tl.load(chunk_indices + i_t * 2 + 1).to(i_t.dtype),
            )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(i_t.dtype),
            tl.load(cu_seqlens + i_n + 1).to(i_t.dtype),
        )
        T = eos - bos
        if AUTO_INDICES:
            if i_t * BT >= T:
                return
    else:
        bos = i_b * T
    if INT64_INDEX:
        bos = bos.to(tl.int64)

    o_t = i_t * BT + tl.arange(0, BLOCK_T)
    o_d = i_d * BD + tl.arange(0, BD)
    m_t = o_t < T
    m_g = m_t[:, None] & (o_d[None, :] < D)
    p_g = g + (bos * H + i_h) * D + o_t[:, None] * (H * D) + o_d[None, :]
    p_y = y + (bos * H + i_h) * D + o_t[:, None] * (H * D) + o_d[None, :]

    # Complete chunks need no vector predicates. Sequence-local tails retain
    # the original masks, including when packed sequences have different lengths.
    full_rows = i_t * BT + BLOCK_T <= T
    if D % BD == 0:
        if full_rows:
            b_g = tl.load(p_g).to(tl.float32)
        else:
            b_g = tl.load(p_g, mask=m_t[:, None], other=0.0).to(tl.float32)
    else:
        b_g = tl.load(p_g, mask=m_g, other=0.0).to(tl.float32)
    if HAS_BIAS:
        b_bias = tl.load(g_bias + i_h * D + o_d, mask=o_d < D, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = tl.exp(b_a) if SAFE_GATE else -tl.exp(b_a)
    if SAFE_GATE:
        b_gate = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_g)))
    else:
        b_g_scaled = b_g * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_g,
            (1.0 / beta) * tl.log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = b_a * b_softplus

    # A triangular dot multiplies future rows by zero. IEEE 0*NaN/Inf is NaN,
    # so preserve that behavior without requiring matrix hardware on a backend.
    # Include padded gate rows: their nonfinite values also affect the dot.
    o_c = tl.arange(0, BLOCK_T)
    nonfinite = (b_gate != b_gate) | (tl.abs(b_gate) == float("inf"))
    last_nonfinite = tl.max(tl.where(nonfinite, o_c[:, None], -1), axis=0)
    b_gate = tl.where(m_g, b_gate, 0.0)
    b_y = tl.cumsum(b_gate, axis=0)
    b_y = tl.where(o_c[:, None] < last_nonfinite[None, :], float("nan"), b_y)
    if BLOCK_T < BT:
        # The short physical tile omits some rows of the logical chunk.
        # A nonfinite padded gate in the original dot still contaminates every
        # visible row through 0*NaN/Inf. Evaluate one representative padded row.
        if HAS_BIAS:
            b_pad = b_bias
        else:
            b_pad = tl.full((BD,), 0.0, tl.float32)
        if SAFE_GATE:
            pad_gate = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_pad)))
        else:
            pad_scaled = b_pad * beta
            pad_softplus = tl.where(
                pad_scaled > threshold,
                b_pad,
                (1.0 / beta) * tl.log(1.0 + tl.exp(pad_scaled)),
            )
            pad_gate = b_a * pad_softplus
        pad_nonfinite = (pad_gate != pad_gate) | (tl.abs(pad_gate) == float("inf"))
        b_y = tl.where(pad_nonfinite[None, :], float("nan"), b_y)
    b_y = tl.where(b_y == 0.0, 0.0, b_y) * cumsum_scale
    if D % BD == 0:
        if full_rows:
            tl.store(p_y, b_y.to(p_y.dtype.element_ty))
        else:
            tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_t[:, None])
    else:
        tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_g)


def fused_kda_gate_chunk_cumsum(
    raw_g: torch.Tensor,
    A_log: torch.Tensor,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_size: int = 64,
    output_dtype: torch.dtype | None = torch.float,
    safe_gate: bool = False,
    lower_bound: float = -5.0,
) -> torch.Tensor:
    """Return gate prefixes in log2 space, with the pinned vLLM semantics.

    Inputs are contiguous BTHD tensors. A_log has H elements and optional
    g_bias has H*D elements. Floating inputs/outputs support FP16/BF16/FP32.
    chunk_size supports 32/64/128. This is an inference-only implementation.

    Packed inputs require B=1 and valid int32/int64 cu_seqlens. Supplied
    chunk_indices are [NT,2] (sequence, sequence-local chunk) pairs. Values
    must describe the input exactly; metadata contents are not copied to the
    host for validation. Omitted indices use a rectangular device grid, which
    can be slower for many short sequences. Empty packed sequences are valid
    on this path. No Torch index-building helper is called.
    """
    if not isinstance(raw_g, torch.Tensor):
        raise TypeError("raw_g must be a Tensor")
    if runtime.device.vendor_name != "ascend":
        raise NotImplementedError("This implementation targets Ascend")
    if raw_g.device.type != runtime.device.name:
        raise NotImplementedError("raw_g must use the active accelerator backend")
    _check_tensor("raw_g", raw_g, raw_g.device, _FLOAT_DTYPES)
    if raw_g.ndim != 4:
        raise ValueError("raw_g must have shape [B,T,H,D]")
    B, T, H, D = raw_g.shape
    if H == 0 or D == 0:
        raise ValueError("H and D must be positive")
    if not isinstance(chunk_size, int) or chunk_size not in (32, 64, 128):
        raise NotImplementedError("chunk_size must be 32, 64 or 128")
    if not isinstance(safe_gate, bool):
        raise TypeError("safe_gate must be a Python bool")
    if not all(
        isinstance(value, numbers.Real) for value in (beta, threshold, lower_bound)
    ):
        raise TypeError("beta, threshold and lower_bound must be real scalars")
    if output_dtype is not None and not isinstance(output_dtype, torch.dtype):
        raise TypeError("output_dtype must be a torch.dtype or None")
    dtype = raw_g.dtype if output_dtype is None else output_dtype
    if dtype not in _FLOAT_DTYPES:
        raise NotImplementedError(f"output dtype {dtype} is unsupported")
    _check_tensor("A_log", A_log, raw_g.device, _FLOAT_DTYPES)
    if A_log.numel() != H:
        raise ValueError("A_log must have H elements")
    if g_bias is not None:
        _check_tensor("g_bias", g_bias, raw_g.device, _FLOAT_DTYPES)
        if g_bias.numel() != H * D:
            raise ValueError("g_bias must have H*D elements")
    if safe_gate:
        if not math.isfinite(lower_bound):
            raise ValueError("lower_bound must be finite")
    elif not math.isfinite(beta) or beta <= 0 or not math.isfinite(threshold):
        raise ValueError("softplus beta must be positive/finite and threshold finite")

    auto_indices = cu_seqlens is not None and chunk_indices is None
    grid_batches = B
    NT = triton.cdiv(T, chunk_size)
    if cu_seqlens is not None:
        if B != 1:
            raise ValueError("Packed inputs require B=1")
        _check_tensor(
            "cu_seqlens", cu_seqlens, raw_g.device, (torch.int32, torch.int64)
        )
        if cu_seqlens.ndim != 1 or cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must have shape [N+1] with N>=1")
        if auto_indices:
            grid_batches = cu_seqlens.numel() - 1
        else:
            _check_tensor(
                "chunk_indices", chunk_indices, raw_g.device, (torch.int32, torch.int64)
            )
            if chunk_indices.ndim != 2 or chunk_indices.shape[1] != 2:
                raise ValueError("chunk_indices must have shape [NT,2]")
            NT = chunk_indices.shape[0]
            if T > 0 and NT == 0:
                raise ValueError("Nonempty packed input requires chunk indices")
    if grid_batches * H > 65535:
        raise NotImplementedError("Sequence/batch times H exceeds the grid z limit")
    if runtime.device.vendor_name == "ascend" and NT * grid_batches * H >= 2**31:
        # Bound total logical programs, including idle automatic-index pairs.
        raise NotImplementedError("Head/chunk grid exceeds the 32-bit program limit")

    y = torch.empty_like(raw_g, dtype=dtype)
    if raw_g.numel() == 0:
        return y
    # Contiguity was checked, so view cannot copy or launch a Torch kernel.
    A_log = A_log.view(-1)
    if g_bias is not None:
        g_bias = g_bias.view(-1)

    def grid(meta):
        return (triton.cdiv(D, meta["BD"]), NT, grid_batches * H)

    with runtime.torch_device_fn.device(raw_g.device):
        kernel = _ascend_scan_kernel
        kernel[grid](
            # Limit UB buffering for this launch only; no compiler-wide switch.
            multibuffer=False,
            g=raw_g,
            A=A_log,
            y=y,
            g_bias=g_bias,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            cumsum_scale=RCP_LN2,
            beta=beta,
            threshold=threshold,
            SAFE_GATE=safe_gate,
            LOWER_BOUND=lower_bound,
            T=T,
            H=H,
            D=D,
            BT=chunk_size,
            ROWS=min(chunk_size, triton.next_power_of_2(T)),
            INT64_INDEX=(T + chunk_size) * H * D * B >= 2**31,
            AUTO_INDICES=auto_indices,
            # Ordinary scalar parameters keep exact values in LibEntry's key
            # and in the tuner key, without one CANN binary per unused constant.
            # N separates equal packed shapes with different sequence splits.
            TUNE_T=T,
            TUNE_NT=NT,
            TUNE_B=B,
            TUNE_N=cu_seqlens.numel() - 1 if cu_seqlens is not None else B,
        )
    return y
