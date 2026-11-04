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

import torch
import triton
import triton.language as tl

from flaggems_vllm import runtime
from flaggems_vllm.utils import libentry, libtuner, tl_extra_shim

_IS_NVIDIA = runtime.device.vendor_name == "nvidia"

if os.environ.get("FLA_USE_FAST_OPS", "0") == "1":
    log = getattr(tl_extra_shim, "fast_logf", tl.log)
else:
    log = tl.log


@libentry()
@libtuner(
    configs=runtime.get_tuned_config(
        "fused_kda_gate_nv" if _IS_NVIDIA else "fused_kda_gate"
    ),
    key=["T", "H", "D", "HAS_BIAS", "SAFE_GATE"],
    warmup=5,
    rep=50,
    use_cuda_graph=_IS_NVIDIA,
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
    configs=runtime.get_tuned_config(
        "fused_kda_gate_tiled_nv" if _IS_NVIDIA else "fused_kda_gate_tiled"
    ),
    key=["T", "H", "D", "HAS_BIAS", "SAFE_GATE"],
    warmup=5,
    rep=50,
    use_cuda_graph=_IS_NVIDIA,
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
    Inputs must be contiguous floating tensors on the runtime accelerator.
    Ascend and CPU are unsupported. This is forward only. NVIDIA configurations
    are tuned with CUDA Graph; other backends use their normal device timer.
    """
    if runtime.device.vendor_name == "ascend" or runtime.device.name == "cpu":
        raise NotImplementedError("fused_kda_gate does not support Ascend or CPU")
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
        if tensor.device.type != runtime.device.name:
            raise NotImplementedError("KDA gate requires the runtime accelerator")
        if tensor.dtype not in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
        ):
            raise NotImplementedError("KDA gate requires floating point inputs")
        if tensor.dtype == torch.float64 and not runtime.device.support_fp64:
            raise NotImplementedError("KDA gate FP64 is unsupported on this backend")
        if tensor.dtype == torch.bfloat16 and not runtime.device.support_bf16:
            raise NotImplementedError("KDA gate BF16 is unsupported on this backend")
        if tensor.layout != torch.strided or not tensor.is_contiguous():
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
            _fused_kda_gate_tiled_kernel[
                lambda meta: (triton.cdiv(tokens, meta["BT"]) * heads,)
            ](g, A, g_bias, output, *constants, BD=bd)
        else:
            _fused_kda_gate_kernel[
                lambda meta: (triton.cdiv(g.numel(), meta["BLOCK"]),)
            ](g, A, g_bias, output, *constants)
    return output
