# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""GLM5Next KDA gate, with FP32 output and optional bounded sigmoid."""

import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    from triton.language.extra.libdevice import fast_logf as log
else:
    log = tl.log


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_kda_gate"),
    key=["T", "H", "D", "HAS_BIAS", "SAFE_GATE"],
    warmup=5,
    rep=15,
)
@triton.jit
def _fused_kda_gate_kernel(
    G,
    A,
    BIAS,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < T * H * D
    columns = offsets % (H * D)
    heads = columns // D
    value = tl.load(G + offsets, mask=mask, other=0).to(tl.float32)
    if HAS_BIAS:
        value += tl.load(BIAS + columns, mask=mask, other=0).to(tl.float32)
    scale = tl.exp(tl.load(A + heads, mask=mask, other=0).to(tl.float32))
    if SAFE_GATE:
        result = LOWER_BOUND / (1.0 + tl.exp(-(scale * value)))
    else:
        scaled = value * BETA
        softplus = tl.where(
            scaled > THRESHOLD, value, (1.0 / BETA) * log(1.0 + tl.exp(scaled))
        )
        result = -scale * softplus
    tl.store(Y + offsets, result, mask=mask)


@libentry()
@libtuner(
    configs=runtime.get_tuned_config("fused_kda_gate_tiled"),
    key=["T", "H", "D", "HAS_BIAS", "SAFE_GATE"],
    warmup=5,
    rep=15,
)
@triton.jit
def _fused_kda_gate_tiled_kernel(
    G,
    A,
    BIAS,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BETA: tl.constexpr,
    THRESHOLD: tl.constexpr,
    SAFE_GATE: tl.constexpr,
    LOWER_BOUND: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
):
    # Share one head's exp(A) and bias across multiple tokens. Neighboring
    # programs visit neighboring heads before moving to the next token tile.
    program = tl.program_id(0)
    head = program % H
    rows = (program // H).to(tl.int64) * BT + tl.arange(0, BT)
    cols = tl.arange(0, BD)
    offsets = rows[:, None] * (H * D) + head * D + cols[None, :]
    mask = (rows[:, None] < T) & (cols[None, :] < D)
    value = tl.load(G + offsets, mask=mask, other=0).to(tl.float32)
    if HAS_BIAS:
        bias = tl.load(BIAS + head * D + cols, mask=cols < D, other=0).to(tl.float32)
        value += bias[None, :]
    scale = tl.exp(tl.load(A + head).to(tl.float32))
    if SAFE_GATE:
        result = LOWER_BOUND / (1.0 + tl.exp(-(scale * value)))
    else:
        scaled = value * BETA
        softplus = tl.where(
            scaled > THRESHOLD, value, (1.0 / BETA) * log(1.0 + tl.exp(scaled))
        )
        result = -scale * softplus
    tl.store(Y + offsets, result, mask=mask)


_launchers = OrderedDict()
# LibEntry passes constexpr arguments to CompiledKernel on Triton 3.3-3.6.
# Keep other versions on the normal LibEntry path.
_cache_launch = (
    (3, 3)
    <= tuple(map(int, triton.__version__.split(".")[:2]))
    <= (
        3,
        6,
    )
)


def _metadata(tensor):
    if tensor is None:
        return None, None
    if tensor.layout != torch.strided:
        raise NotImplementedError("KDA gate requires strided tensors")
    pointer = tensor.data_ptr()
    return (
        tensor.shape,
        tensor.stride(),
        tensor.dtype,
        tensor.device,
        pointer % 16,
    ), pointer


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
    safe_gate: bool = False,
    lower_bound: float | None = -5.0,
) -> torch.Tensor:
    """Return a fresh FP32 gate of shape ``[..., H, head_k_dim]``.

    The interface follows GLM5Next's ``third_party/kda/kernels.py``. With
    ``safe_gate=False``, compute ``-exp(A) * softplus(g + g_bias)`` using
    ``beta`` and ``threshold``. Otherwise compute
    ``lower_bound / (1 + exp(-exp(A) * (g + g_bias)))``. A ``None`` lower bound
    means -5.0, and the lower bound has no effect on the softplus branch.
    Inputs must be contiguous CUDA floating tensors. This is forward only.
    """
    g_meta, gp = _metadata(g)
    a_meta, ap = _metadata(A)
    bias_meta, bp = _metadata(g_bias)
    key = (
        g_meta,
        a_meta,
        bias_meta,
        head_k_dim,
        beta,
        threshold,
        safe_gate,
        lower_bound,
    )
    cached = _launchers.get(key) if _cache_launch else None
    if cached is not None:
        shape, launch, constants = cached
        output = torch.empty(shape, device=g.device, dtype=torch.float32)
        with runtime.torch_device_fn.device(g.device):
            launch(gp, ap, bp, output.data_ptr(), *constants)
        return output

    if g.ndim < 1 or head_k_dim < 1 or A.numel() < 1:
        raise ValueError("KDA gate requires a nonempty head/key dimension")
    heads = A.numel()
    hidden = heads * head_k_dim
    if g.shape[-1] != hidden:
        raise ValueError("g.shape[-1] must equal A.numel() * head_k_dim")
    if g_bias is not None and g_bias.numel() != hidden:
        raise ValueError("g_bias must contain H * head_k_dim elements")
    for tensor in (g, A, g_bias):
        if tensor is None:
            continue
        if tensor.device != g.device:
            raise ValueError("KDA gate inputs must be on the same device")
        if tensor.device.type != "cuda":
            raise NotImplementedError("fused_kda_gate currently supports CUDA")
        if tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise NotImplementedError("KDA gate requires floating point inputs")
        if not tensor.is_contiguous():
            raise NotImplementedError("KDA gate requires contiguous inputs")

    output = torch.empty(
        (*g.shape[:-1], heads, head_k_dim), device=g.device, dtype=torch.float32
    )
    tokens = g.numel() // hidden
    if tokens == 0:
        return output
    constants = (
        tokens,
        heads,
        head_k_dim,
        beta,
        threshold,
        safe_gate,
        -5.0 if lower_bound is None else lower_bound,
        g_bias is not None,
    )
    with runtime.torch_device_fn.device(g.device):
        if g.numel() >= 4 * 1024 * 1024:
            bd = triton.next_power_of_2(head_k_dim)
            kernel, meta = _fused_kda_gate_tiled_kernel[
                lambda meta: (triton.cdiv(tokens, meta["BT"]) * heads,)
            ](g, A, g_bias, output, *constants, BD=bd)
            grid = (triton.cdiv(tokens, meta["BT"]) * heads, 1, 1)
            launch_constants = (*constants, meta["BT"], bd)
        else:
            kernel, meta = _fused_kda_gate_kernel[
                lambda meta: (triton.cdiv(g.numel(), meta["BLOCK"]),)
            ](g, A, g_bias, output, *constants)
            grid = (triton.cdiv(g.numel(), meta["BLOCK"]), 1, 1)
            launch_constants = (*constants, meta["BLOCK"])
        if _cache_launch:
            # Store validated metadata and compiled code, never input/output tensors.
            _launchers[key] = (output.shape, kernel[grid], launch_constants)
            if len(_launchers) > 128:
                _launchers.popitem(last=False)
    return output
