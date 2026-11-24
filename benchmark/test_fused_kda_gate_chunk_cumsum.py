# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Captured dimensions, synthetic values, and the frozen designated vLLM baseline."""

import pytest
import torch

import flaggems_vllm
from tests.test_fused_kda_gate_chunk_cumsum import (
    CAPTURED_DATA,
    Case,
    make_inputs,
    upstream,
)

from . import base


class GateCumsumBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        self.shapes = CAPTURED_DATA["cases"]
        self.shape_desc = (
            "captured model dimensions (synthetic values), gate mode shown in name"
        )


@pytest.mark.fused_kda_gate_chunk_cumsum
@pytest.mark.parametrize("safe", [False, True])
def test_fused_kda_gate_chunk_cumsum(safe):
    def inputs(spec, dtype, device):
        values = dict(spec["case"])
        values["lengths"] = tuple(values["lengths"])
        args, kwargs = make_inputs(
            Case(**values), dtype=dtype, bias=spec["bias"], safe=safe
        )
        kwargs.update(
            beta=spec["beta"],
            threshold=spec["threshold"],
            lower_bound=spec["lower_bound"],
        )
        for name in ("cu_seqlens", "chunk_indices"):
            if name in kwargs:
                kwargs[name] = kwargs[name].to(getattr(torch, spec["index_dtype"]))
        yield (*args, kwargs)

    bench = GateCumsumBenchmark(
        op_name=f"fused_kda_gate_chunk_cumsum_safe_{safe}",
        input_fn=inputs,
        torch_op=upstream,
        gems_op=flaggems_vllm.fused_kda_gate_chunk_cumsum,
        dtypes=[torch.bfloat16],
    )
    bench.run()


pytestmark = pytest.mark.skipif(
    flaggems_vllm.device == "cpu" or flaggems_vllm.vendor_name == "ascend",
    reason="The portable gate+cumsum path excludes CPU and Ascend",
)
