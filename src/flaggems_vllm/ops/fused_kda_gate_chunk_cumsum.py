# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
"""GLM5-Next gate activation and chunk-local prefix sum, forward only.

Ported from vLLM commit 373614f66dc8dc678faaf33223256f814238b7cb,
vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py:763-900.
The upstream file credits flash-linear-attention (MIT), Copyright (c)
2023-2025, Songlin Yang, Yu Zhang.

The original gate formulas and post-sum RCP_LN2 scaling are retained. The
handoff's FP32 inclusive scan replaces triangular matrix multiplication for
large launches. HCU/MACA retain a matrix-prefix path for short launches.
Nonfinite values retain the upstream zero-times-NaN/Inf propagation. Changes:
repository tuner/device integration,
explicit metadata validation, empty handling, and a correctness-first device
path for packed inputs without precomputed indices. No vLLM dependency and no
Torch compute/copy fallback are used.
"""

import math
import numbers
import os

import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:
    gluon = None

RCP_LN2 = 1.4426950216
_FLOAT_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
_MIN_SCAN_WARPS = min(
    config.num_warps
    for config in runtime.get_tuned_config("fused_kda_gate_chunk_cumsum")
)
log = libdevice.fast_logf if os.environ.get("FLA_USE_FAST_OPS", "0") == "1" else tl.log

_WAVE64_SCHEDULE = tl.constexpr(runtime.device.vendor_name in ("hygon", "metax"))
_DOT_EVICTION_POLICY = tl.constexpr(
    "evict_first" if runtime.device.vendor_name == "metax" else ""
)
_SCAN_CONFIG = (
    "fused_kda_gate_chunk_cumsum_scan"
    if runtime.device.vendor_name == "hygon"
    else "fused_kda_gate_chunk_cumsum"
)


@triton.jit
def _program_coordinates(
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    TUNE_T: tl.constexpr,
    TUNE_NT: tl.constexpr,
    TUNE_B: tl.constexpr,
):
    i_t, i_bh = tl.program_id(1), tl.program_id(2)
    if _WAVE64_SCHEDULE:
        # Adjacent programs visit adjacent heads instead of distant token
        # chunks of one head. Keep the same number of blocks and chunk pairs.
        linear = i_t + i_bh * TUNE_NT
        i_t = (linear // H) % TUNE_NT
        i_bh = linear % H + (linear // (H * TUNE_NT)) * H
    # Include a padded chunk in the bound so masked tail offsets cannot wrap.
    if not _WAVE64_SCHEDULE or (TUNE_T + BT) * H * D * TUNE_B >= 2**31:
        i_t = i_t.to(tl.int64)
    return i_t, i_bh


@libentry()
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=runtime.get_tuned_config(_SCAN_CONFIG),
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
    use_cuda_graph=True,
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
    AUTO_INDICES: tl.constexpr,
    TUNE_T: tl.constexpr,
    TUNE_NT: tl.constexpr,
    TUNE_B: tl.constexpr,
    TUNE_N: tl.constexpr,
):
    BLOCK_T: tl.constexpr = (
        triton.next_power_of_2(min(TUNE_T, BT)) if _WAVE64_SCHEDULE else BT
    )
    i_d = tl.program_id(0)
    i_t, i_bh = _program_coordinates(H, D, BT, TUNE_T, TUNE_NT, TUNE_B)
    i_b, i_h = i_bh // H, i_bh % H
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
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        if AUTO_INDICES:
            if i_t * BT >= T:
                return
    else:
        bos = i_b * T

    o_t = i_t * BT + tl.arange(0, BLOCK_T)
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
        b_gate = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_g)))
    else:
        b_g_scaled = b_g * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_g,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
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
                (1.0 / beta) * log(1.0 + tl.exp(pad_scaled)),
            )
            pad_gate = b_a * pad_softplus
        pad_nonfinite = (pad_gate != pad_gate) | (tl.abs(pad_gate) == float("inf"))
        b_y = tl.where(pad_nonfinite[None, :], float("nan"), b_y)
    b_y = tl.where(b_y == 0.0, 0.0, b_y) * cumsum_scale
    tl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_g)


@libentry()
@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["g_bias"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@libtuner(
    configs=runtime.get_tuned_config("fused_kda_gate_chunk_cumsum"),
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
    use_cuda_graph=True,
)
@triton.jit(do_not_specialize=["T"])
def _kda_gate_cumsum_dot_kernel(
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
    AUTO_INDICES: tl.constexpr,
    TUNE_T: tl.constexpr,
    TUNE_NT: tl.constexpr,
    TUNE_B: tl.constexpr,
    TUNE_N: tl.constexpr,
):
    i_d = tl.program_id(0)
    i_t, i_bh = _program_coordinates(H, D, BT, TUNE_T, TUNE_NT, TUNE_B)
    i_b, i_h = i_bh // H, i_bh % H
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
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        if AUTO_INDICES:
            if i_t * BT >= T:
                return
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
        b_gate = LOWER_BOUND / (1.0 + tl.exp(-(b_a * b_g)))
    else:
        b_g_scaled = b_g * beta
        b_softplus = tl.where(
            b_g_scaled > threshold,
            b_g,
            (1.0 / beta) * log(1.0 + tl.exp(b_g_scaled)),
        )
        b_gate = b_a * b_softplus

    # Keep the upstream dot, including its nonfinite propagation behavior.
    # Padded rows contribute only to masked output rows for finite inputs.
    o_c = tl.arange(0, BT)
    m_cumsum = tl.where(o_c[:, None] >= o_c[None, :], 1.0, 0.0)
    b_y = tl.dot(m_cumsum, b_gate, allow_tf32=False) * cumsum_scale
    # Outputs are written once and never read in this kernel. On MACA, lower
    # their cache priority so output addresses do not evict the reused inputs.
    tl.store(
        p_y,
        b_y.to(p_y.dtype.element_ty),
        mask=m_g,
        eviction_policy=_DOT_EVICTION_POLICY,
    )


_wave64_scan_kernel = None
# HCU's installed FlagTree exposes Gluon but cannot infer even a blocked
# tensor layout (HCUMFMALayout is missing). Keep its standard Triton path.
if gluon is not None and runtime.device.vendor_name == "metax":

    @gluon.jit
    def _wave64_sum(a, b):
        return a + b

    @libentry()
    @triton.heuristics(
        {
            "HAS_BIAS": lambda args: args["g_bias"] is not None,
            "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        }
    )
    @libtuner(
        configs=runtime.get_tuned_config("fused_kda_gate_chunk_cumsum_wave64"),
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
        use_cuda_graph=True,
    )
    @gluon.jit(do_not_specialize=["T"])
    def _kda_gate_cumsum_wave64_kernel(
        g,
        A,
        y,
        g_bias,
        cu_seqlens,
        chunk_indices,
        cumsum_scale,
        beta,
        threshold,
        SAFE_GATE: gl.constexpr,
        LOWER_BOUND: gl.constexpr,
        T,
        H: gl.constexpr,
        D: gl.constexpr,
        BT: gl.constexpr,
        BD: gl.constexpr,
        LANE_T: gl.constexpr,
        HAS_BIAS: gl.constexpr,
        IS_VARLEN: gl.constexpr,
        AUTO_INDICES: gl.constexpr,
        TUNE_T: gl.constexpr,
        TUNE_NT: gl.constexpr,
        TUNE_B: gl.constexpr,
        TUNE_N: gl.constexpr,
    ):
        BLOCK_T: gl.constexpr = triton.next_power_of_2(min(TUNE_T, BT))
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 1], [LANE_T, 64 // LANE_T], [1, gl.num_warps()], [1, 0]
        )
        i_d = gl.program_id(0)
        i_t, i_bh = gl.program_id(1), gl.program_id(2)
        linear = i_t + i_bh * TUNE_NT
        i_t = (linear // H) % TUNE_NT
        i_bh = linear % H + (linear // (H * TUNE_NT)) * H
        if (TUNE_T + BT) * H * D * TUNE_B >= 2**31:
            i_t = i_t.to(gl.int64)
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            if AUTO_INDICES:
                # Rectangular sequence/chunk grid: no host tensor reads or index
                # construction. Empty/short sequences simply skip excess programs.
                i_n = i_b
            else:
                i_n, i_t = (
                    gl.load(chunk_indices + i_t * 2).to(gl.int32),
                    gl.load(chunk_indices + i_t * 2 + 1).to(i_t.dtype),
                )
            bos, eos = (
                gl.load(cu_seqlens + i_n).to(gl.int32),
                gl.load(cu_seqlens + i_n + 1).to(gl.int32),
            )
            T = eos - bos
            if AUTO_INDICES:
                if i_t * BT >= T:
                    return
        else:
            bos = i_b * T

        o_t = i_t * BT + gl.arange(0, BLOCK_T, layout=gl.SliceLayout(1, layout))
        o_d = i_d * BD + gl.arange(0, BD, layout=gl.SliceLayout(0, layout))
        m_t = o_t < T
        m_g = m_t[:, None] & (o_d[None, :] < D)
        p_g = g + (bos * H + i_h) * D + o_t[:, None] * (H * D) + o_d[None, :]
        p_y = y + (bos * H + i_h) * D + o_t[:, None] * (H * D) + o_d[None, :]

        b_g = gl.load(p_g, mask=m_g, other=0.0).to(gl.float32)
        if HAS_BIAS:
            b_bias = gl.load(g_bias + i_h * D + o_d, mask=o_d < D, other=0.0).to(
                gl.float32
            )
            b_g = b_g + b_bias[None, :]

        b_a = gl.load(A + i_h).to(gl.float32)
        b_a = gl.exp(b_a) if SAFE_GATE else -gl.exp(b_a)
        if SAFE_GATE:
            b_gate = LOWER_BOUND / (1.0 + gl.exp(-(b_a * b_g)))
        else:
            b_g_scaled = b_g * beta
            b_softplus = gl.where(
                b_g_scaled > threshold,
                b_g,
                (1.0 / beta) * gl.log(1.0 + gl.exp(b_g_scaled)),
            )
            b_gate = b_a * b_softplus

        # A triangular dot multiplies future rows by zero. IEEE 0*NaN/Inf is NaN,
        # so preserve that behavior without requiring matrix hardware on a backend.
        # Include padded gate rows: their nonfinite values also affect the dot.
        o_c = gl.arange(0, BLOCK_T, layout=gl.SliceLayout(1, layout))
        nonfinite = (b_gate != b_gate) | (gl.abs(b_gate) == float("inf"))
        last_nonfinite = gl.max(gl.where(nonfinite, o_c[:, None], -1), axis=0)
        b_gate = gl.where(m_g, b_gate, 0.0)
        b_y = gl.associative_scan(b_gate, 0, _wave64_sum)
        b_y = gl.where(o_c[:, None] < last_nonfinite[None, :], float("nan"), b_y)
        if BLOCK_T < BT:
            # The short physical tile omits some rows of the logical chunk.
            # A nonfinite padded gate in the original dot still contaminates every
            # visible row through 0*NaN/Inf. Evaluate one representative padded row.
            if HAS_BIAS:
                b_pad = b_bias
            else:
                b_pad = gl.full(
                    (BD,), 0.0, gl.float32, layout=gl.SliceLayout(0, layout)
                )
            if SAFE_GATE:
                pad_gate = LOWER_BOUND / (1.0 + gl.exp(-(b_a * b_pad)))
            else:
                pad_scaled = b_pad * beta
                pad_softplus = gl.where(
                    pad_scaled > threshold,
                    b_pad,
                    (1.0 / beta) * gl.log(1.0 + gl.exp(pad_scaled)),
                )
                pad_gate = b_a * pad_softplus
            pad_nonfinite = (pad_gate != pad_gate) | (gl.abs(pad_gate) == float("inf"))
            b_y = gl.where(pad_nonfinite[None, :], float("nan"), b_y)
        b_y = gl.where(b_y == 0.0, 0.0, b_y) * cumsum_scale
        gl.store(p_y, b_y.to(p_y.dtype.element_ty), mask=m_g)

    _wave64_scan_kernel = _kda_gate_cumsum_wave64_kernel


def _select_kernel(
    device,
    heads,
    chunks,
    batches,
    chunk_size,
    auto_indices,
    tokens=None,
    safe_gate=None,
):
    """Choose an algorithm before tuning its tile/warp configuration.

    On HCU/MACA, short launches favor the native matrix-prefix schedule. Use
    scan after a conservative launch-coverage estimate reaches one quarter of
    the device thread capacity, counted in minimum-warp full-head chunk groups.
    This is a coverage heuristic, not an achieved-occupancy or capacity claim.
    """
    if (
        _wave64_scan_kernel is not None
        and safe_gate
        and tokens is not None
        and 0 < tokens <= chunk_size // 2
    ):
        return _wave64_scan_kernel
    if runtime.device.vendor_name not in ("hygon", "metax") or chunk_size > 64:
        return kda_gate_cumsum_fwd_kernel
    if tokens is not None and 0 < tokens <= chunk_size // 2:
        return kda_gate_cumsum_fwd_kernel
    if auto_indices:
        return _kda_gate_cumsum_dot_kernel
    props = runtime.torch_device_fn.get_device_properties(device)
    units = getattr(props, "multi_processor_count", 0)
    threads = getattr(props, "max_threads_per_multi_processor", 0)
    warp_size = getattr(props, "warp_size", 0)
    if not (units and threads and warp_size):
        return kda_gate_cumsum_fwd_kernel
    minimum_groups = triton.cdiv(units * threads, 4 * _MIN_SCAN_WARPS * warp_size)
    if chunks * batches * heads < minimum_groups:
        return _kda_gate_cumsum_dot_kernel
    return kda_gate_cumsum_fwd_kernel


def _check_tensor(name, tensor, device, dtypes):
    """Metadata-only checks; never normalize layout with a copy."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}")
    if tensor.dtype not in dtypes:
        raise NotImplementedError(f"{name} dtype {tensor.dtype} is unsupported")
    if not tensor.is_contiguous():
        raise NotImplementedError(f"{name} must be contiguous")


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
    if runtime.device.vendor_name == "ascend" or runtime.device.name == "cpu":
        raise NotImplementedError("This implementation targets non-Ascend accelerators")
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
    if (
        runtime.device.vendor_name in ("hygon", "metax")
        and NT * grid_batches * H >= 2**31
    ):
        # The head-first program permutation uses a signed 32-bit linear ID.
        # Automatic-index grids include idle sequence/chunk combinations.
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
        kernel = _select_kernel(
            raw_g.device, H, NT, grid_batches, chunk_size, auto_indices, T, safe_gate
        )
        kernel[grid](
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
            AUTO_INDICES=auto_indices,
            # Cache-only metadata: T is do_not_specialize, and LibEntry's hot
            # launch bypasses the tuner. Both caches must distinguish workload.
            # N separates equal packed shapes with different sequence splits.
            TUNE_T=T,
            TUNE_NT=NT,
            TUNE_B=B,
            TUNE_N=cu_seqlens.numel() - 1 if cu_seqlens is not None else B,
        )
    return y
