# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Synthetic inputs; pinned designated vLLM baseline, plus a Torch-chain view."""

import pytest
import torch

import flaggems_vllm
from tests.test_fused_kda_gate_chunk_cumsum import (
    PERFORMANCE_CASES,
    make_inputs,
    torch_reference,
    upstream,
)

from . import base


class GateCumsumBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = list(PERFORMANCE_CASES)
        self.shape_desc = "synthetic Case(B,T,H,D,chunk,lengths)"


@pytest.mark.fused_kda_gate_chunk_cumsum
@pytest.mark.parametrize("safe", [False, True])
def test_fused_kda_gate_chunk_cumsum(safe):
    def inputs(case, dtype, device):
        args, kwargs = make_inputs(case, dtype=dtype, safe=safe)
        yield (*args, kwargs)

    bench = GateCumsumBenchmark(
        op_name=f"fused_kda_gate_chunk_cumsum_safe_{safe}",
        input_fn=inputs,
        torch_op=upstream,
        gems_op=flaggems_vllm.fused_kda_gate_chunk_cumsum,
        dtypes=[torch.bfloat16],
    )
    bench.run()


@pytest.mark.fused_kda_gate_chunk_cumsum
def test_fused_kda_gate_chunk_cumsum_torch_chain():
    # Auxiliary baseline only. The upstream-vs-port rows above are authoritative.
    def inputs(case, dtype, device):
        args, kwargs = make_inputs(case, dtype=dtype, safe=True)
        yield (*args, kwargs)

    bench = GateCumsumBenchmark(
        op_name="fused_kda_gate_chunk_cumsum_torch_chain_auxiliary",
        input_fn=inputs,
        torch_op=torch_reference,
        gems_op=flaggems_vllm.fused_kda_gate_chunk_cumsum,
        dtypes=[torch.bfloat16],
    )
    bench.run()


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
    reason="The initial gate+cumsum port targets NVIDIA CUDA",
)
