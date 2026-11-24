# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Initial port parity with the designated upstream and independent Torch math."""

import importlib
import sys
from pathlib import Path

import pytest
import torch

import flaggems_vllm

from . import conftest as cfg

# tools is also an unrelated installed package in some NVIDIA images. Expose
# only our standalone fixture package, without importing that global package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
_reference = importlib.import_module("fused_kda_gate_chunk_cumsum.reference")
CORRECTNESS_CASES = _reference.CORRECTNESS_CASES
DTYPES = _reference.DTYPES
PERFORMANCE_CASES = _reference.PERFORMANCE_CASES
Case = _reference.Case
assert_close = _reference.assert_close
make_inputs = _reference.make_inputs
torch_reference = _reference.torch_reference
upstream = importlib.import_module(
    "fused_kda_gate_chunk_cumsum.upstream"
).fused_kda_gate_chunk_cumsum

port = flaggems_vllm.fused_kda_gate_chunk_cumsum
CASES = (
    [CORRECTNESS_CASES[i] for i in (3, 5, 8)] if cfg.QUICK_MODE else CORRECTNESS_CASES
)
INPUT_DTYPES = (
    (torch.bfloat16, torch.float32) if cfg.QUICK_MODE else tuple(DTYPES.values())
)
pytestmark = [
    pytest.mark.fused_kda_gate_chunk_cumsum,
    pytest.mark.skipif(
        not torch.cuda.is_available() or flaggems_vllm.vendor_name != "nvidia",
        reason="The initial gate+cumsum port targets NVIDIA CUDA",
    ),
]


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.name)
@pytest.mark.parametrize("dtype", INPUT_DTYPES)
@pytest.mark.parametrize(
    "bias,safe", [(False, False), (True, False), (False, True), (True, True)]
)
def test_reference_and_upstream(case, dtype, bias, safe):
    args, kwargs = make_inputs(case, dtype, bias, safe)
    before_g, before_a = (x.clone() for x in args)
    before_bias = kwargs["g_bias"].clone() if bias else None
    expected = torch_reference(*args, **kwargs)
    baseline = upstream(*args, **kwargs)
    actual = port(*args, **kwargs)
    assert_close(actual, expected)
    assert_close(actual, baseline)
    assert actual.data_ptr() != args[0].data_ptr()
    torch.testing.assert_close(args[0], before_g, rtol=0, atol=0)
    torch.testing.assert_close(args[1], before_a, rtol=0, atol=0)
    if bias:
        torch.testing.assert_close(kwargs["g_bias"], before_bias, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", INPUT_DTYPES)
@pytest.mark.parametrize("output", [None, torch.float16, torch.bfloat16])
def test_output_dtype(dtype, output):
    args, kwargs = make_inputs(CORRECTNESS_CASES[4], dtype, output=output)
    assert_close(port(*args, **kwargs), upstream(*args, **kwargs))
    assert_close(port(*args, **kwargs), torch_reference(*args, **kwargs))


@pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("safe", [False, True])
@pytest.mark.parametrize("empty_sequences", [False, True])
def test_optional_indices(index_dtype, safe, empty_sequences):
    case = (
        Case("empty_sequences", T=66, lengths=(0, 1, 0, 65))
        if empty_sequences
        else CORRECTNESS_CASES[-1]
    )
    args, kwargs = make_inputs(case, safe=safe)
    kwargs["cu_seqlens"] = kwargs["cu_seqlens"].to(index_dtype)
    kwargs["chunk_indices"] = kwargs["chunk_indices"].to(index_dtype)
    baseline = upstream(*args, **kwargs)
    expected = torch_reference(*args, **kwargs)
    del kwargs["chunk_indices"]
    assert_close(port(*args, **kwargs), expected)
    assert_close(port(*args, **kwargs), baseline)


@pytest.mark.parametrize("dtype", INPUT_DTYPES)
def test_parameter_dtypes(dtype):
    args, kwargs = make_inputs(CORRECTNESS_CASES[3])
    args = (args[0], args[1].to(dtype))
    kwargs["g_bias"] = kwargs["g_bias"].view(4, 128).to(dtype)
    assert_close(port(*args, **kwargs), upstream(*args, **kwargs))
    assert_close(port(*args, **kwargs), torch_reference(*args, **kwargs))


def test_nondefault_parameters():
    args, kwargs = make_inputs(CORRECTNESS_CASES[4], dtype=torch.float32)
    args[0].flatten()[:7] = torch.tensor(
        [-80, -10, 0, 4.999, 5, 5.001, 80], device="cuda"
    )
    kwargs.update(beta=2.0, threshold=10.0)
    assert_close(port(*args, **kwargs), upstream(*args, **kwargs))
    assert_close(port(*args, **kwargs), torch_reference(*args, **kwargs))
    kwargs.update(safe_gate=True, lower_bound=-3.0)
    assert_close(port(*args, **kwargs), upstream(*args, **kwargs))
    assert_close(port(*args, **kwargs), torch_reference(*args, **kwargs))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), -0.0])
@pytest.mark.parametrize("safe", [False, True])
def test_upstream_nonfinite_and_signed_zero(value, safe):
    # Dot and scan need not have the same NaN/Inf propagation. The pinned
    # triangular-dot upstream, rather than a scan, defines compatibility here.
    args, kwargs = make_inputs(CORRECTNESS_CASES[3], dtype=torch.float32, safe=safe)
    args[0][0, 20, 0, 0] = value
    assert_close(port(*args, **kwargs), upstream(*args, **kwargs))


@pytest.mark.parametrize("shape", [(0, 65, 4, 128), (1, 0, 4, 128)])
def test_empty(shape):
    g = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    a = torch.empty(4, device="cuda")
    out = port(g, a)
    assert out.shape == g.shape and out.dtype == torch.float32
    assert out.numel() == 0 and out.device == g.device


def test_empty_packed():
    args, kwargs = make_inputs(Case("empty", T=0, lengths=(0, 0)))
    del kwargs["chunk_indices"]
    assert port(*args, **kwargs).shape == args[0].shape


@pytest.mark.parametrize("kind", ["raw", "A_log", "g_bias", "output", "chunk"])
def test_unsupported(kind):
    args, kwargs = make_inputs(CORRECTNESS_CASES[3])
    if kind == "raw":
        args = (args[0].to(torch.int32), args[1])
    elif kind == "A_log":
        args = (args[0], args[1].to(torch.float64))
    elif kind == "g_bias":
        kwargs["g_bias"] = kwargs["g_bias"].to(torch.int32)
    elif kind == "output":
        kwargs["output_dtype"] = torch.float64
    else:
        kwargs["chunk_size"] = 48
    with pytest.raises(NotImplementedError):
        port(*args, **kwargs)


def test_noncontiguous():
    args, kwargs = make_inputs(CORRECTNESS_CASES[3])
    noncontiguous = args[0].transpose(1, 2)
    with pytest.raises(NotImplementedError, match="contiguous"):
        port(noncontiguous, args[1], **kwargs)
    noncontiguous_bias = torch.empty((512, 2), device="cuda")[:, 0]
    kwargs["g_bias"] = noncontiguous_bias
    with pytest.raises(NotImplementedError, match="contiguous"):
        port(*args, **kwargs)


@pytest.mark.parametrize("kind", ["A_log", "g_bias", "beta", "boundary", "indices"])
def test_bad_arguments(kind):
    args, kwargs = make_inputs(CORRECTNESS_CASES[-1])
    if kind == "A_log":
        args = (args[0], args[1].view(-1)[:1])
    elif kind == "g_bias":
        kwargs["g_bias"] = kwargs["g_bias"][:1]
    elif kind == "beta":
        kwargs["beta"] = 0
    elif kind == "boundary":
        kwargs["cu_seqlens"] = kwargs["cu_seqlens"].view(1, -1)
    else:
        kwargs["chunk_indices"] = kwargs["chunk_indices"].view(-1)
    with pytest.raises(ValueError):
        port(*args, **kwargs)


def test_wrong_device():
    args, kwargs = make_inputs(CORRECTNESS_CASES[3])
    with pytest.raises(ValueError, match="must be on"):
        port(args[0], args[1].cpu(), **kwargs)


@pytest.mark.parametrize(
    "name", ["beta", "threshold", "lower_bound", "safe_gate", "output_dtype"]
)
def test_tensor_scalars_are_rejected(name):
    args, kwargs = make_inputs(CORRECTNESS_CASES[3])
    kwargs[name] = torch.tensor(1.0, device="cuda")
    with pytest.raises(TypeError):
        port(*args, **kwargs)


def test_top_level_integration():
    assert port.__name__ == "fused_kda_gate_chunk_cumsum"
    assert "fused_kda_gate_chunk_cumsum" in flaggems_vllm.ops.__all__
    assert any(
        name == "fused_kda_gate_chunk_cumsum" and function is port
        for name, function in flaggems_vllm._FULL_CONFIG
    )


def test_workload_keys_survive_cached_launches():
    """T, packed NT and sequence count must separate both tuning/cache layers."""
    audit_call = importlib.import_module(
        "fused_kda_gate_chunk_cumsum.run_autotune_perf"
    ).audit_call
    kernel = importlib.import_module(
        "flaggems_vllm.ops.fused_kda_gate_chunk_cumsum"
    ).kda_gate_cumsum_fwd_kernel
    cases = [
        Case("one", T=128, H=2, D=80, lengths=(128,)),
        Case("larger", T=256, H=2, D=80, lengths=(256,)),
        Case("split", T=128, H=2, D=80, lengths=(64, 64)),
        Case("extra_chunk", T=128, H=2, D=80, lengths=(1, 127)),
    ]
    cache = kernel.kernel_cache[torch.cuda.current_device()]
    initial_entries = len(cache)
    prepared, keys = [], set()
    for case in cases:
        args, kwargs = make_inputs(case, safe=True)
        expected = torch_reference(*args, **kwargs)

        def call(args=args, kwargs=kwargs):
            return port(*args, **kwargs)

        output, audit = audit_call(call, kernel)
        assert_close(output, expected)
        keys.add(tuple(audit["tuning_key"]))
        prepared.append((call, expected, audit))
    assert len(keys) == len(cases)
    entries = len(cache)
    assert entries == initial_entries + len(cases)
    for call, expected, cold in reversed(prepared):
        output, hot = audit_call(call, kernel)
        assert_close(output, expected)
        assert hot["tuning_key"] == cold["tuning_key"]
        assert hot["selected_config"] == cold["selected_config"]
    assert len(cache) == entries
