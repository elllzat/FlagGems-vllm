# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Captured dimensions, synthetic values, and the frozen designated vLLM baseline."""

import os
from pathlib import Path

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

    def get_latency(self, op, *args, **kwargs):
        if (
            flaggems_vllm.vendor_name in ("ascend", "mthreads")
            and base.Config.mode == base.consts.BenchMode.CUDAGRAPH
        ):
            from fused_kda_gate_chunk_cumsum.run_portable import capture, latency

            # Backend-native Graph integration check. The acceptance runner
            # separately records five paired rounds with 256 calls per Graph.
            api = flaggems_vllm.runtime.torch_device_fn
            graph, out = capture(lambda: op(*args, **kwargs), api, 32)
            value = latency(graph, api, 32) / 1000
            del out
            return value
        return super().get_latency(op, *args, **kwargs)


@pytest.mark.fused_kda_gate_chunk_cumsum
@pytest.mark.parametrize("safe", [False, True])
def test_fused_kda_gate_chunk_cumsum(safe):
    is_ascend = flaggems_vllm.vendor_name == "ascend"
    reference = upstream
    if is_ascend:
        if not safe:
            pytest.skip("The pinned native raw-gate reference requires safe_gate=True")
        root = os.environ.get("FUSED_KDA_ASCEND_REFERENCE_ROOT")
        if not root:
            pytest.skip(
                "Set FUSED_KDA_ASCEND_REFERENCE_ROOT to the pinned extension directory"
            )
        from fused_kda_gate_chunk_cumsum.ascend_reference import load_reference

        root = Path(root)
        native, _ = load_reference(
            root / "vllm-ascend-reference",
            root / "ascendc-install",
            root / "binding-build",
        )

        def reference(g, a, **kwargs):
            return native(
                g,
                kwargs["chunk_size"],
                A_log=a.view(-1),
                dt_bias=kwargs["g_bias"],
                cu_seqlens=kwargs["_host_cu"],
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=kwargs["lower_bound"],
                layout="BSND",
            )

    def gems(g, a, **kwargs):
        kwargs.pop("_host_cu", None)
        return flaggems_vllm.fused_kda_gate_chunk_cumsum(g, a, **kwargs)

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
        if is_ascend:
            boundaries = [0]
            for length in values["lengths"]:
                boundaries.append(boundaries[-1] + length)
            kwargs["_host_cu"] = tuple(boundaries) if values["lengths"] else None
        yield (*args, kwargs)

    bench = GateCumsumBenchmark(
        op_name=f"fused_kda_gate_chunk_cumsum_safe_{safe}",
        input_fn=inputs,
        torch_op=reference,
        gems_op=gems,
        dtypes=[torch.bfloat16],
    )
    bench.run()


pytestmark = pytest.mark.skipif(
    flaggems_vllm.device == "cpu",
    reason="The gate+cumsum operator requires an accelerator",
)
