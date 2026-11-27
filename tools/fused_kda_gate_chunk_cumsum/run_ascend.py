# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 FlagOS Contributors
"""Pair the Ascend implementation with the pinned native fused AscendC op."""

import argparse
import hashlib
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import torch_npu  # noqa: F401

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from fused_kda_gate_chunk_cumsum.ascend_reference import load_reference
from fused_kda_gate_chunk_cumsum.run_portable import (
    capture,
    latency,
    load_operator,
    recorded_launch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--functional", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--source")
    parser.add_argument("--install")
    parser.add_argument("--build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=256)
    opts = parser.parse_args()
    library, candidate = load_operator(opts.isolated)
    if library.vendor_name != "ascend":
        parser.error("This runner requires Ascend")
    if opts.rounds < 1 or opts.repeats < 1:
        parser.error("rounds and repeats must be positive")
    if opts.functional:
        import pytest

        argv = ["-q", "tests/test_fused_kda_gate_chunk_cumsum.py", "-x"]
        if opts.quick:
            argv += ["--quick"]
        if opts.isolated:
            argv += ["-k", "not top_level_integration"]
        raise SystemExit(pytest.main(argv))
    if not all((opts.source, opts.install, opts.build)):
        parser.error("Performance requires --source, --install and --build")
    native, provenance = load_reference(opts.source, opts.install, opts.build)
    from fused_kda_gate_chunk_cumsum.reference import (
        Case,
        assert_close,
        make_inputs,
        torch_reference,
    )

    module = sys.modules[candidate.__module__]
    path = Path(module.__file__)
    specs = json.loads(Path(__file__).with_name("captured_shapes.json").read_text())[
        "cases"
    ]
    if opts.quick:
        specs = [s for s in specs if s["case"]["T"] in (1, 65, 129, 4096)]
    api = library.runtime.torch_device_fn
    report = {
        "vendor": "ascend",
        "revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "candidate_file": str(path.relative_to(ROOT)),
        "candidate_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "baseline": provenance,
        "method": "native NPU Graph replay with device events",
        "unit": "us",
        "rounds": opts.rounds,
        "graph_repetitions": opts.repeats,
        "samples_per_latency": 7,
        "graph_warmup_replays": 3,
        "capture_output_lifetime": "discard intermediates, retain final output for validation",
        "public_package_import_verified": not opts.isolated,
        "shape_values": "all captured model dimensions, deterministic synthetic tensor values",
        "quick_subset": opts.quick,
        "torch": torch.__version__,
        "rows": [],
    }
    opts.output.parent.mkdir(parents=True, exist_ok=True)
    for spec in specs:
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
        if not kwargs["safe_gate"]:
            raise NotImplementedError(
                "The pinned native raw-gate adapter requires safe gate"
            )
        boundaries = [0]
        for length in values["lengths"]:
            boundaries.append(boundaries[-1] + length)
        host_cu = tuple(boundaries) if values["lengths"] else None
        a_flat = args[1].view(-1)
        calls = {
            "baseline": lambda: native(
                args[0],
                kwargs["chunk_size"],
                A_log=a_flat,
                dt_bias=kwargs["g_bias"],
                cu_seqlens=host_cu,
                use_gate_in_kernel=True,
                safe_gate=True,
                lower_bound=kwargs["lower_bound"],
                layout="BSND",
            ),
            "gems": lambda: candidate(*args, **kwargs),
        }
        cpu_args = tuple(x.cpu() for x in args)
        cpu_kwargs = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()
        }
        expected = torch_reference(*cpu_args, **cpu_kwargs).to(library.device)
        assert_close(calls["baseline"](), expected)
        actual, config = recorded_launch(calls["gems"], module._ascend_scan_kernel)
        assert_close(actual, expected)
        del actual
        row = {
            "case": spec["case"],
            "correctness": "passed",
            "gems_config": config,
            "rounds": [],
        }
        graphs = {}
        for name, call in calls.items():
            graph, out = capture(call, api, opts.repeats)
            graph.replay()
            api.synchronize()
            assert_close(out, expected)
            del out
            graphs[name] = graph
        for i in range(opts.rounds):
            order = ("baseline", "gems") if i % 2 == 0 else ("gems", "baseline")
            times = {name: latency(graphs[name], api, opts.repeats) for name in order}
            times["speedup"] = times["baseline"] / times["gems"]
            row["rounds"].append(times)
        row["baseline_us"] = statistics.median(r["baseline"] for r in row["rounds"])
        row["gems_us"] = statistics.median(r["gems"] for r in row["rounds"])
        row["speedup"] = statistics.median(r["speedup"] for r in row["rounds"])
        report["rows"].append(row)
        ratios = [r["speedup"] for r in report["rows"]]
        report["geomean_speedup"] = math.exp(sum(map(math.log, ratios)) / len(ratios))
        report["min_speedup"] = min(ratios)
        opts.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
