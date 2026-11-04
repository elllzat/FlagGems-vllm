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
"""Compare both KDA gate paths/configs with CUDA Graph on captured contracts.

This diagnostic bypasses autotune to measure every explicit configuration.
Formal operator acceptance still uses benchmark/test_fused_kda_gate.py.
"""

import argparse
import importlib
import json
import random
import statistics
from pathlib import Path

import torch
import triton
from triton.testing import do_bench_cudagraph

from flaggems_vllm import runtime
from tests.test_fused_kda_gate import (
    captured_gate_cases,
    captured_gate_inputs,
    torch_fused_kda_gate,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shapes", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rep", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    if runtime.device.vendor_name != "nvidia":
        parser.error("this CUDA Graph diagnostic requires NVIDIA")
    module = importlib.import_module("flaggems_vllm.ops.fused_kda_gate")
    cases = captured_gate_cases(json.loads(args.shapes.read_text(encoding="utf-8")))
    result = []
    for case in cases:
        inputs = captured_gate_inputs(case, runtime.device.name)
        g, A, D, bias, beta, threshold, safe, lower = inputs
        H, T = A.numel(), g.numel() // (A.numel() * D)
        output = torch.empty(
            (*g.shape[:-1], H, D), device=g.device, dtype=torch.float32
        )
        reference = torch_fused_kda_gate(*inputs)
        constants = (T, H, D, beta, threshold, safe, lower, bias is not None)
        candidates = []
        for path, name, entry in (
            ("flat", "fused_kda_gate", module._fused_kda_gate_kernel),
            ("tiled", "fused_kda_gate_tiled", module._fused_kda_gate_tiled_kernel),
        ):
            for config in runtime.get_tuned_config(name):
                options = config.all_kwargs()
                if path == "flat":
                    grid = (triton.cdiv(g.numel(), options["BLOCK"]),)
                else:
                    grid = (triton.cdiv(T, options["BT"]) * H,)
                    options = {**options, "BD": triton.next_power_of_2(D)}

                def launch(entry=entry, grid=grid, options=options):
                    entry.jit_function[grid](g, A, bias, output, *constants, **options)

                launch()
                torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
                candidates.append(
                    dict(path=path, config=options, launch=launch, latency_us=[])
                )
        for repeat in range(args.rounds):
            order = list(range(len(candidates)))
            random.Random(42 + repeat).shuffle(order)
            for index in order:
                candidate = candidates[index]
                candidate["latency_us"].append(
                    do_bench_cudagraph(candidate["launch"], rep=args.rep) * 1000
                )
        for candidate in candidates:
            del candidate["launch"]
            candidate["median_us"] = statistics.median(candidate["latency_us"])
        candidates.sort(key=lambda row: row["median_us"])
        result.append(dict(T=T, H=H, D=D, candidates=candidates))
        print(f"T={T}: {candidates[:2]}", flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
