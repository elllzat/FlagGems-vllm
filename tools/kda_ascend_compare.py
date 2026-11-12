"""Compare the public host dispatch and kernel revisions with the NPU profiler."""

import argparse
import functools
import hashlib
import importlib.util
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

import torch
import triton
from triton.backends.ascend.testing import do_bench_npu

sys.path.insert(0, str(Path.cwd() / "tools"))
from fused_kda_gate_perf import check_output, load_operator, make_inputs
from fused_kda_gate_speedup import load_pinned_reference


def load_previous(repo):
    source = repo / "src/flaggems_vllm/ops/fused_kda_gate.py"
    spec = importlib.util.spec_from_file_location("_previous_kda_gate", source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.fused_kda_gate


def revision(repo):
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    runtime, candidate = load_operator(True)
    previous = load_previous(args.previous)
    baseline, _, baseline_metadata = load_pinned_reference()
    device = runtime.device.name
    torch.manual_seed(42)
    fixture = Path("benchmark/fused_kda_gate_captured_contracts.json").read_bytes()
    report = dict(
        status="running",
        previous=revision(args.previous),
        candidate=revision(Path.cwd()),
        device=runtime.torch_device_fn.get_device_name(0),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        triton=triton.__version__,
        torch=torch.__version__,
        reference=baseline_metadata,
        contracts_sha256=hashlib.sha256(fixture).hexdigest(),
        rounds=args.rounds,
        method="NPU profiler, mean of 30 active calls per round, 5 warmups",
        results=[],
    )
    jobs = []
    contracts = json.loads(fixture)["contracts"]
    cases = [
        ("model", make_inputs(case, device)) for case in contracts
    ]
    for t, h, d, safe in ((37, 12, 80, True), (257, 12, 80, False), (1025, 12, 80, True), (5003, 12, 80, True)):
        g = torch.randn(t, h * d, device=device, dtype=torch.bfloat16)
        a = torch.randn(h, device=device) * 0.5
        bias = torch.randn(h * d, device=device) * 0.1
        cases.append(("aligned_width", (g, a, d, bias, 1.0, 20.0, safe, -5.0)))
    for case_index, (category, inputs) in enumerate(cases):
        g, a, d = inputs[:3]
        h = a.numel()
        t = g.numel() // (h * d)
        row = dict(category=category, T=t, H=h, D=d, safe=inputs[-2], samples_us={})
        functions = {"previous": previous, "candidate": candidate}
        if category == "model":
            functions = {"vllm": baseline, **functions}
        for label, function in functions.items():
            f = functools.partial(function, *inputs)
            check_output(f(), inputs)
            jobs.append((case_index, label, f))
            row["samples_us"][label] = []
        report["results"].append(row)
        print(f"validated {category} {t}/{h}/{d}", flush=True)
    for rep in range(args.rounds):
        random.Random(521 + rep).shuffle(jobs)
        times = do_bench_npu([f for _, _, f in jobs])
        for (case_index, label, _), time_ms in zip(jobs, times, strict=True):
            report["results"][case_index]["samples_us"][label].append(time_ms * 1000)
        for row in report["results"]:
            samples = row["samples_us"]
            row["median_us"] = {label: statistics.median(values) for label, values in samples.items()}
            row["previous_speedup"] = statistics.median(p / c for p, c in zip(samples["previous"], samples["candidate"]))
            if "vllm" in samples:
                row["vllm_speedup"] = statistics.median(b / c for b, c in zip(samples["vllm"], samples["candidate"]))
        report["completed_rounds"] = rep + 1
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"completed round {rep + 1}", flush=True)
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
