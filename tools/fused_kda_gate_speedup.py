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
"""Probe/measure the unchanged pinned vLLM gate on other vendor devices.

Only the original reference definitions and their candidate-list assignments are
loaded from the test source. No reference functions, decorators or configs are
rewritten, filtered or overridden. Unsupported compilation is reported without a
speedup. This avoids importing unrelated pytest/package code on isolated backends.
"""

import argparse
import ast
import hashlib
import json
import os
import statistics
import sys
import traceback
import types
from pathlib import Path

import torch
import triton
import triton.language as tl

from tools.fused_kda_gate_perf import check_graph, check_output, make_inputs, measure
from tools.fused_kda_gate_portable_smoke import load_operator


def load_pinned_reference():
    path = Path(__file__).resolve().parents[1] / "tests/test_fused_kda_gate.py"
    source = path.read_text(encoding="utf-8")
    names = {"BT_LIST_AUTOTUNE", "NUM_WARPS_AUTOTUNE"}
    functions = {"kda_gate_fwd_kernel", "fused_kda_gate"}
    selected = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in names
            for target in node.targets
        ):
            selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in functions:
            selected.append(node)
    assert len(selected) == 4
    tree = ast.Module(body=selected, type_ignores=[])
    digest = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    module = types.ModuleType("_flaggems_kda_pinned_speedup_reference")
    module.__file__ = str(path)
    module.__dict__.update(
        torch=torch,
        triton=triton,
        tl=tl,
        cdiv=triton.cdiv,
        next_power_of_2=triton.next_power_of_2,
        log=tl.log,
    )
    sys.modules[module.__name__] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    configs = module.kda_gate_fwd_kernel.configs
    assert len(configs) == 24
    metadata = dict(
        source="tests/test_fused_kda_gate.py: pinned GLM5Next vLLM reference",
        ast_sha256=digest,
        function_and_decorator_asts_unchanged=True,
        candidates_unchanged=True,
        candidates=[config.all_kwargs() for config in configs],
    )
    return module.fused_kda_gate, module.kda_gate_fwd_kernel, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep", type=int, default=500)
    options = parser.parse_args()
    if os.environ.get("FLA_USE_FAST_OPS", "0") != "0":
        parser.error("this pinned-reference runner requires FLA_USE_FAST_OPS=0")
    if options.rounds < 1 or options.rep < 1:
        parser.error("rounds and rep must be positive")
    runtime, gate = load_operator(options.isolated)
    mode = "npu" if runtime.device.vendor_name == "ascend" else "cudagraph"
    report = dict(
        status="running",
        vendor=runtime.device.vendor_name,
        device=runtime.torch_device_fn.get_device_name(0),
        torch=torch.__version__,
        triton=triton.__version__,
        isolated=options.isolated,
        mode=mode,
        rounds=options.rounds,
        rep_ms=options.rep if mode == "cudagraph" else None,
    )

    def save():
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    stage = "load_reference"
    try:
        baseline, kernel, report["reference"] = load_pinned_reference()
        fixture = (
            Path(__file__).resolve().parents[1]
            / "benchmark/fused_kda_gate_captured_contracts.json"
        )
        payload = json.loads(fixture.read_text(encoding="utf-8"))
        report["contracts_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
        contracts = payload["contracts"][:1] if options.probe else payload["contracts"]
        prepared = []
        for case in contracts:
            inputs = make_inputs(case, runtime.device.name)
            stage = "baseline_compile_and_validate"
            check_output(baseline(*inputs), inputs)
            stage = "gems_compile_and_validate"
            check_output(gate(*inputs), inputs)
            if mode == "cudagraph":
                stage = "graph_replay_validate"
                for function in (baseline, gate):
                    check_graph(
                        function,
                        make_inputs(case, runtime.device.name),
                        runtime.torch_device_fn,
                    )
            g, A, D = inputs[:3]
            prepared.append(
                (
                    inputs,
                    dict(
                        T=g.numel() // (A.numel() * D),
                        H=A.numel(),
                        D=D,
                        baseline_config=kernel.best_config.all_kwargs(),
                        round_baseline_us=[],
                        round_gems_us=[],
                        round_speedup=[],
                    ),
                )
            )
        stage = "measure"
        rounds = 1 if options.probe else options.rounds
        for repeat in range(rounds):
            for index, (inputs, row) in enumerate(prepared):
                pair = {}
                functions = [("baseline", baseline), ("gems", gate)]
                if (repeat + index) % 2:
                    functions.reverse()
                for label, function in functions:
                    pair[label] = measure(
                        lambda: function(*inputs), mode, options.rep, 100
                    )
                row["round_baseline_us"].append(pair["baseline"])
                row["round_gems_us"].append(pair["gems"])
                row["round_speedup"].append(pair["baseline"] / pair["gems"])
                print(
                    f"round={repeat + 1} T={row['T']} speedup={row['round_speedup'][-1]:.6f}",
                    flush=True,
                )
        report["results"] = []
        for _, row in prepared:
            row["baseline_us"] = statistics.median(row["round_baseline_us"])
            row["gems_us"] = statistics.median(row["round_gems_us"])
            row["speedup"] = statistics.median(row["round_speedup"])
            report["results"].append(row)
        report["mean_speedup"] = statistics.mean(
            row["speedup"] for row in report["results"]
        )
        report["status"] = "probe_passed" if options.probe else "passed"
    except Exception as error:
        report.update(
            status="reference_unavailable_or_error", stage=stage, error=repr(error)
        )
        traceback.print_exc()
    save()
    print(f"{report['vendor']}: {report['status']}", flush=True)


if __name__ == "__main__":
    main()
