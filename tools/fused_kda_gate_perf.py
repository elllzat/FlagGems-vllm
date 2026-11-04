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
"""Measure the production KDA gate on captured contracts across vendor devices.

This reports implementation latency, without replacing the NVIDIA vLLM baseline.
Timers are the same Triton helpers used by benchmark/base.py. --probe checks
timer support and Graph replay correctness before a full measurement run.
"""

import argparse
import hashlib
import importlib
import json
import os
import random
import statistics
from pathlib import Path

import torch
import triton

from tools.fused_kda_gate_portable_smoke import load_operator


def make_inputs(case, device):
    torch.manual_seed(42)

    def tensor(metadata):
        if metadata is None:
            return None
        dtype = getattr(torch, metadata["dtype"].removeprefix("torch."))
        return torch.empty_strided(
            metadata["shape"], metadata["stride"], device=device, dtype=dtype
        ).normal_(std=0.5)

    return (
        tensor(case["operator_g"]),
        tensor(case["A"]),
        case["D"],
        tensor(case["g_bias"]),
        case["beta"],
        case["threshold"],
        case["safe_gate"],
        case["lower_bound"],
    )


def reference(inputs):
    g, A, D, bias, beta, threshold, safe, lower = inputs
    value = g.float().reshape(-1, A.numel(), D)
    if bias is not None:
        value = value + bias.float().reshape(A.numel(), D)
    scale = A.float().reshape(-1, 1).exp()
    if safe:
        result = (-5.0 if lower is None else lower) / (
            1.0 + torch.exp(-(scale * value))
        )
    else:
        result = -scale * torch.where(
            value * beta > threshold,
            value,
            (1.0 / beta) * torch.log(1.0 + torch.exp(value * beta)),
        )
    return result.reshape(*g.shape[:-1], A.numel(), D)


def check_output(output, inputs):
    assert output.dtype == torch.float32 and output.device == inputs[0].device
    torch.testing.assert_close(output, reference(inputs), rtol=1e-4, atol=1e-4)


def check_graph(gate, inputs, device_fn):
    stream = device_fn.Stream()
    stream.wait_stream(device_fn.current_stream())
    with device_fn.stream(stream):
        gate(*inputs)
    device_fn.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gate(*inputs)
    inputs[0].add_(0.25)
    graph.replay()
    device_fn.synchronize()
    check_output(output, inputs)


def measure(fn, mode, rep, warmup):
    if mode == "cudagraph":
        return (
            triton.testing.do_bench_cudagraph(fn, rep=rep, return_mode="median") * 1000
        )
    return (
        triton.testing.do_bench(fn, warmup=warmup, rep=rep, return_mode="median") * 1000
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shapes",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmark/fused_kda_gate_captured_contracts.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--probe", action="store_true")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("cudagraph", "kernel"),
        default=["cudagraph", "kernel"],
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--rep", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=100)
    options = parser.parse_args()
    if min(options.rounds, options.rep, options.warmup) < 1:
        parser.error("rounds, rep and warmup must be positive")
    runtime, gate = load_operator(options.isolated)
    device_fn = runtime.torch_device_fn
    payload = json.loads(options.shapes.read_text(encoding="utf-8"))
    report = dict(
        status="running",
        vendor=runtime.device.vendor_name,
        torch=torch.__version__,
        triton=triton.__version__,
        device_name=device_fn.get_device_name(0),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        isolated=options.isolated,
        public_package_import_verified=not options.isolated,
        contracts_sha256=hashlib.sha256(options.shapes.read_bytes()).hexdigest(),
        provenance=payload["provenance"],
        original_capture_sha256=payload["original_capture_sha256"],
        rounds=options.rounds,
        rep_ms=options.rep,
        warmup_ms=options.warmup,
        unit="us",
        modes={},
    )

    def save():
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if options.probe:
        for mode in options.modes:
            try:
                inputs = make_inputs(payload["contracts"][0], runtime.device.name)
                check_output(gate(*inputs), inputs)
                if mode == "cudagraph":
                    check_graph(gate, inputs, device_fn)
                elapsed = measure(
                    lambda: gate(*inputs), mode, options.rep, options.warmup
                )
                report["modes"][mode] = dict(status="passed", sample_latency_us=elapsed)
            except Exception as error:
                report["modes"][mode] = dict(status="failed", error=repr(error))
            print(f"{mode}: {report['modes'][mode]}", flush=True)
        report["status"] = "probe_complete"
        save()
        return

    prepared = []
    module = importlib.import_module(gate.__module__)
    for case in payload["contracts"]:
        inputs = make_inputs(case, runtime.device.name)
        check_output(gate(*inputs), inputs)
        for _ in range(5):
            gate(*inputs)
        device_fn.synchronize()
        g, A, D = inputs[:3]
        path = "tiled" if g.numel() >= 4 * 1024 * 1024 else "flat"
        entry = (
            module._fused_kda_gate_tiled_kernel
            if path == "tiled"
            else module._fused_kda_gate_kernel
        )
        config = getattr(entry.fn, "best_config", None)
        prepared.append(
            dict(
                T=g.numel() // (A.numel() * D),
                H=A.numel(),
                D=D,
                contract=case,
                path=path,
                selected_config=config.all_kwargs() if config is not None else None,
                inputs=inputs,
            )
        )
    for mode in options.modes:
        rows = [{k: v for k, v in item.items() if k != "inputs"} for item in prepared]
        for row in rows:
            row["round_latency_us"] = []
        for repeat in range(options.rounds):
            order = list(range(len(prepared)))
            random.Random(42 + repeat).shuffle(order)
            for index in order:
                inputs = prepared[index]["inputs"]
                latency = measure(
                    lambda: gate(*inputs), mode, options.rep, options.warmup
                )
                rows[index]["round_latency_us"].append(latency)
                print(
                    f"{runtime.device.vendor_name} {mode} round={repeat + 1} "
                    f"T={rows[index]['T']}: {latency:.6f} us",
                    flush=True,
                )
        for row in rows:
            row["median_us"] = statistics.median(row["round_latency_us"])
        report["modes"][mode] = dict(status="passed", results=rows)
        save()
    report["status"] = "passed"
    save()


if __name__ == "__main__":
    main()
