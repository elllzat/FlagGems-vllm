# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Diagnose allocation-address effects without changing either operator/tuner."""

import argparse
import hashlib
import json
import statistics
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from fused_kda_gate_chunk_cumsum.reference import (  # noqa: E402
    Case,
    assert_close,
    make_inputs,
    torch_reference,
)
from fused_kda_gate_chunk_cumsum.run_portable import (  # noqa: E402
    fresh_baseline,
    latency,
    load_operator,
    recorded_launch,
    tuner_of,
)


def compare_pool(calls, api, expected, shared):
    # Use the same stream in both modes, isolating the pool choice. Calls are
    # stateless and graphs never overlap; no output is consumed by another graph.
    stream = api.Stream()
    pool = api.graph_pool_handle() if shared else None
    graphs, pointers = {}, {}
    for name, call in calls.items():
        for _ in range(5):
            call()
        api.synchronize()
        stream.wait_stream(api.current_stream())
        graph = api.CUDAGraph()
        addresses = set()
        with api.stream(stream):
            graph.capture_begin(pool=pool)
            for _ in range(256):
                output = call()
                addresses.add(output.data_ptr())
            graph.capture_end()
        stream.synchronize()
        graph.replay()
        api.synchronize()
        assert_close(output, expected)
        # Release the Python output before the next capture. The graph keeps
        # its pool alive; shared captures may then reuse exactly these buffers.
        del output
        graphs[name] = graph
        pointers[name] = sorted(addresses)
    rounds = []
    for i in range(5):
        order = ("baseline", "gems") if i % 2 == 0 else ("gems", "baseline")
        times = {name: latency(graphs[name], api, 256) for name in order}
        times["speedup"] = times["baseline"] / times["gems"]
        rounds.append(times)
    return {
        "shared_pool": shared,
        "pointers": pointers,
        "identical_output_addresses": pointers["baseline"] == pointers["gems"],
        "rounds": rounds,
        "speedup": statistics.median(r["speedup"] for r in rounds),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    library, candidate = load_operator()
    import importlib

    module = importlib.import_module("flaggems_vllm.ops.fused_kda_gate_chunk_cumsum")
    dataset = json.loads(Path(__file__).with_name("captured_shapes.json").read_text())
    names = {"runtime_e360a24e5ecf", "runtime_df2b19d422cc", "runtime_c1ba024b2419"}
    report = {
        "scope": "Pool diagnostic only; frozen baseline and original six configs",
        "capture_output_lifetime": "legacy retain previous output",
        "baseline_sha256": hashlib.sha256(
            Path(__file__).with_name("upstream.py").read_bytes()
        ).hexdigest(),
        "cases": [],
    }
    for spec in (s for s in dataset["cases"] if s["case"]["name"] in names):
        values = dict(spec["case"])
        values["lengths"] = tuple(values["lengths"])
        args, kwargs = make_inputs(
            Case(**values), dtype=torch.bfloat16, safe=True, device=library.device
        )
        for key in ("cu_seqlens", "chunk_indices"):
            kwargs[key] = kwargs[key].to(torch.int32)
        expected = torch_reference(*args, **kwargs)
        baseline, baseline_kernel = fresh_baseline(spec["case"]["name"])
        active = module._select_kernel(
            args[0].device,
            values["H"],
            len(kwargs["chunk_indices"]),
            values["B"],
            values["chunk"],
            False,
            values["T"],
            True,
        )
        calls = {
            "baseline": lambda: baseline(*args, **kwargs),
            "gems": lambda: candidate(*args, **kwargs),
        }
        assert_close(calls["baseline"](), expected)
        output, config = recorded_launch(calls["gems"], active)
        assert_close(output, expected)
        del output
        row = {
            "case": spec["case"],
            "baseline_config": tuner_of(baseline_kernel).best_config.all_kwargs(),
            "candidate_config": config,
            "measurements": [],
        }
        for shared in (False, True, False, True):
            row["measurements"].append(
                compare_pool(calls, library.runtime.torch_device_fn, expected, shared)
            )
        report["cases"].append(row)
        options.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
