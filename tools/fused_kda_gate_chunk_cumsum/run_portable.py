# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Validate captured contracts and run paired device-Graph measurements."""

import argparse
import copy
import hashlib
import importlib
import importlib.util
import json
import math
import statistics
import subprocess
import sys
import types
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))


def load_operator(isolated=False):
    if not isolated:
        import flaggems_vllm

        return flaggems_vllm, flaggems_vllm.fused_kda_gate_chunk_cumsum
    # Explicit diagnostic path for unrelated package-import incompatibilities.
    # Load the actual runtime, utilities and operator from the checked-out tree.
    source = ROOT / "src/flaggems_vllm"
    package = types.ModuleType("flaggems_vllm")
    package.__path__ = [str(source)]
    sys.modules[package.__name__] = package
    runtime = importlib.import_module("flaggems_vllm.runtime")
    ops = types.ModuleType("flaggems_vllm.ops")
    ops.__path__ = [str(source / "ops")]
    sys.modules[ops.__name__] = ops
    module = importlib.import_module("flaggems_vllm.ops.fused_kda_gate_chunk_cumsum")
    if runtime.device.vendor_name == "ascend":
        path = source / "runtime/backend/_ascend/ops/fused_kda_gate_chunk_cumsum.py"
        name = "flaggems_vllm.ops._ascend_fused_kda_gate_chunk_cumsum"
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    package.runtime = runtime
    package.ops = ops
    package.device = runtime.device.name
    package.vendor_name = runtime.device.vendor_name
    package.fused_kda_gate_chunk_cumsum = module.fused_kda_gate_chunk_cumsum
    ops.fused_kda_gate_chunk_cumsum = module.fused_kda_gate_chunk_cumsum
    ops.__all__ = ["fused_kda_gate_chunk_cumsum"]
    return package, module.fused_kda_gate_chunk_cumsum


def fresh_baseline(index):
    """Keep the frozen source/key/six candidates; avoid inter-shape tuner bias."""
    path = Path(__file__).with_name("upstream.py")
    name = f"_kda_chunk_cumsum_reference_{index}"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.fused_kda_gate_chunk_cumsum, module.kda_gate_cumsum_fwd_kernel


def tuner_of(kernel):
    while not hasattr(kernel, "configs"):
        kernel = kernel.fn
    return kernel


def recorded_launch(call, entry):
    """Read the configuration actually launched, including LibEntry cache hits."""
    original_run = entry.run
    metadata = {}

    def record(*args, **kwargs):
        result = original_run(*args, **kwargs)
        metadata.update(result[1])
        return result

    entry.run = record
    try:
        output = call()
    finally:
        entry.run = original_run
    config = {
        name: metadata[name] for name in ("BD", "num_warps", "num_ctas", "num_stages")
    }
    if "LANE_T" in metadata:
        config["LANE_T"] = metadata["LANE_T"]
    return output, config


def capture(call, api, repeats, retain_previous=False):
    graph_type = next(
        getattr(api, name)
        for name in ("CUDAGraph", "MUSAGraph", "NPUGraph", "PPUGraph", "Graph")
        if hasattr(api, name)
    )
    for _ in range(5):
        call()
    api.synchronize()
    stream = api.Stream()
    stream.wait_stream(api.current_stream())
    graph = graph_type()
    with api.stream(stream):
        graph.capture_begin()
        if retain_previous:
            # Historical diagnostic mode: assignment keeps one previous output
            # alive while the next call allocates, usually alternating buffers.
            for _ in range(repeats):
                output = call()
        else:
            # Match triton.testing.do_bench_cudagraph: discard each return.
            # Keep only the final result so replay correctness can be checked.
            for _ in range(repeats - 1):
                call()
            output = call()
        graph.capture_end()
    stream.synchronize()
    return graph, output


def latency(graph, api, repeats):
    for _ in range(3):
        graph.replay()
    samples = []
    for _ in range(7):
        begin, end = api.Event(enable_timing=True), api.Event(enable_timing=True)
        begin.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) * 1000 / repeats)
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--functional", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=256)
    parser.add_argument(
        "--legacy-retain-output", action="store_true", help="Historical diagnostic only"
    )
    parser.add_argument(
        "--candidate-bd-list", help="Explicit candidate-only tuning trial"
    )
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.rounds < 1 or options.repeats < 1:
        parser.error("rounds and repeats must be positive")
    library, candidate = load_operator(options.isolated)
    candidate_module = importlib.import_module(
        "flaggems_vllm.ops.fused_kda_gate_chunk_cumsum"
    )
    standard_scan = candidate_module.kda_gate_cumsum_fwd_kernel
    candidate_kernel = candidate_module._wave64_scan_kernel or standard_scan
    candidate_tuner = tuner_of(candidate_kernel)
    if options.candidate_bd_list:
        bds = [int(x) for x in options.candidate_bd_list.split(",")]
        if any(x not in (16, 32, 64, 128) for x in bds):
            parser.error("Candidate BD values must be 16, 32, 64 or 128")
        configs = {}
        for bd in bds:
            for original in candidate_tuner.configs:
                config = copy.deepcopy(original)
                config.kwargs["BD"] = bd
                configs[(bd, config.num_warps, config.num_stages)] = config
        # LibTuner's persistent cache identity includes the candidate list.
        # Updating only .configs would silently reuse the old six-config winner.
        candidate_tuner._set_configs_and_strategy(
            list(configs.values()), candidate_tuner.strategy
        )
    if options.functional:
        import pytest

        args = ["-q", "tests/test_fused_kda_gate_chunk_cumsum.py", "-x"]
        if options.quick:
            args.append("--quick")
        if options.isolated:
            args += ["-k", "not top_level_integration"]
        raise SystemExit(pytest.main(args))
    from fused_kda_gate_chunk_cumsum.reference import (
        Case,
        assert_close,
        make_inputs,
        torch_reference,
    )

    dataset = json.loads(Path(__file__).with_name("captured_shapes.json").read_text())
    cases = dataset["cases"]
    if options.quick:
        selected = {1, 65, 129, 257, 1025, 3139, 4096}
        cases = [s for s in cases if s["case"]["T"] in selected]
    api = library.runtime.torch_device_fn
    report = {
        "vendor": library.vendor_name,
        "device": library.device,
        "torch": torch.__version__,
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "operator_sha256": hashlib.sha256(
            (ROOT / "src/flaggems_vllm/ops/fused_kda_gate_chunk_cumsum.py").read_bytes()
        ).hexdigest(),
        "baseline_sha256": hashlib.sha256(
            Path(__file__).with_name("upstream.py").read_bytes()
        ).hexdigest(),
        "candidate_bd_trial": options.candidate_bd_list,
        "isolated_operator_import": options.isolated,
        "public_package_import_verified": not options.isolated,
        "method": "device Graph replay with device events",
        "unit": "us",
        "rounds": options.rounds,
        "graph_repetitions": options.repeats,
        "samples_per_latency": 7,
        "graph_warmup_replays": 3,
        "capture_output_lifetime": (
            "legacy retain previous output"
            if options.legacy_retain_output
            else "discard intermediates, validate final output (standard Triton lifetime)"
        ),
        "baseline": "frozen GLM5Next Triton source, fresh original tuner per case",
        "source_sha256": dataset["cases"][0]["captured_contract"]["source_sha256"],
        "shape_values": "captured model dimensions, synthetic tensor values",
        "rows": [],
    }
    options.output.parent.mkdir(parents=True, exist_ok=True)
    for index, spec in enumerate(cases):
        values = dict(spec["case"])
        values["lengths"] = tuple(values["lengths"])
        args, kwargs = make_inputs(
            Case(**values),
            dtype=getattr(torch, spec["dtype"]),
            bias=spec["bias"],
            safe=spec["safe_gate"],
            device=library.device,
        )
        kwargs.update(
            beta=spec["beta"],
            threshold=spec["threshold"],
            lower_bound=spec["lower_bound"],
        )
        for name in ("cu_seqlens", "chunk_indices"):
            if name in kwargs:
                kwargs[name] = kwargs[name].to(getattr(torch, spec["index_dtype"]))
        upstream, upstream_kernel = fresh_baseline(index)
        calls = {
            "baseline": lambda: upstream(*args, **kwargs),
            "gems": lambda: candidate(*args, **kwargs),
        }
        expected = torch_reference(*args, **kwargs)
        row = {"case": spec["case"], "correctness": "passed", "rounds": []}
        automatic = (
            kwargs.get("cu_seqlens") is not None and kwargs.get("chunk_indices") is None
        )
        batches = len(kwargs["cu_seqlens"]) - 1 if automatic else values["B"]
        chunks = (
            len(kwargs["chunk_indices"])
            if kwargs.get("chunk_indices") is not None
            else (values["T"] + values["chunk"] - 1) // values["chunk"]
        )
        active_kernel = candidate_module._select_kernel(
            args[0].device,
            values["H"],
            chunks,
            batches,
            values["chunk"],
            automatic,
            values["T"],
            kwargs["safe_gate"],
        )
        assert_close(calls["baseline"](), expected)
        actual, launched_config = recorded_launch(calls["gems"], active_kernel)
        assert_close(actual, expected)
        del actual
        row["algorithm"] = (
            "wave64_scan"
            if active_kernel is candidate_module._wave64_scan_kernel
            else "scan" if active_kernel is standard_scan else "dot"
        )
        row["baseline_config"] = tuner_of(upstream_kernel).best_config.all_kwargs()
        row["gems_config"] = launched_config
        if not options.validate_only:
            graphs = {}
            for name, call in calls.items():
                graph, output = capture(
                    call, api, options.repeats, options.legacy_retain_output
                )
                graph.replay()
                api.synchronize()
                assert_close(output, expected)
                del output
                graphs[name] = graph
            for round_id in range(options.rounds):
                order = (
                    ("baseline", "gems") if round_id % 2 == 0 else ("gems", "baseline")
                )
                times = {
                    name: latency(graphs[name], api, options.repeats) for name in order
                }
                times["speedup"] = times["baseline"] / times["gems"]
                row["rounds"].append(times)
            row["speedup"] = statistics.median(r["speedup"] for r in row["rounds"])
            row["baseline_us"] = statistics.median(r["baseline"] for r in row["rounds"])
            row["gems_us"] = statistics.median(r["gems"] for r in row["rounds"])
        report["rows"].append(row)
        ratios = [r["speedup"] for r in report["rows"] if "speedup" in r]
        if ratios:
            report["geomean_speedup"] = math.exp(
                sum(map(math.log, ratios)) / len(ratios)
            )
            report["min_speedup"] = min(ratios)
        options.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
