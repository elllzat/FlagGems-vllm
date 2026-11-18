#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

# fmt: off
# isort: skip_file
# Exact standalone gate snapshot from vllm-project/vllm-ascend.
# Revision: a1df499e0762621c2b9561e2ee66c498a70d0328
# Path: vllm_ascend/ops/triton/kda/kda.py
# Full upstream source SHA256: 1e45592a0675af7dfcc879368f3d1f4ec0065455b2f49109c27a11a586516d5e
# Constants, decorators and function bodies remain unchanged.

import torch
import triton
import triton.language as tl
from triton import cdiv, next_power_of_2

BT_LIST_AUTOTUNE = [32, 64, 128]


NUM_WARPS_AUTOTUNE = [4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({"BT": bt}, num_warps=nw, num_stages=ns)
        for bt in BT_LIST_AUTOTUNE
        for nw in NUM_WARPS_AUTOTUNE
        for ns in [2, 3]
    ],
    key=["H", "D"],
)
@triton.jit
def kda_gate_fwd_kernel(
    g,
    A,
    y,
    g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = -tl.exp(b_a)

    stride_row = H * D
    stride_col = 1

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    # softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
    # When beta * x > threshold, use linear approximation x
    # Use threshold to switch to linear when beta*x > threshold
    g_scaled = b_g * beta
    use_linear = g_scaled > threshold
    sp = tl.where(use_linear, b_g, (1.0 / beta) * tl.log(1.0 + tl.exp(g_scaled)))
    b_y = b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def fused_kda_gate(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """
    Forward pass for KDA gate:
      input g: [..., H*D]
      param A: [H] or [1, 1, H, 1]
      beta: softplus beta parameter
      threshold: softplus threshold parameter
      return  : [..., H, D]
    """
    orig_shape = g.shape[:-1]

    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.numel()
    assert H * head_k_dim == HD

    y = torch.empty_like(g, dtype=torch.float32)

    def grid(meta):
        return (cdiv(T, meta["BT"]), H)

    kda_gate_fwd_kernel[grid](
        g,
        A,
        y,
        g_bias,
        beta,
        threshold,
        T,
        H,
        head_k_dim,
        BD=next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y
