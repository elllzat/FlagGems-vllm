"""Measure identical gate+cumsum math against the native AscendC primitive."""

import argparse
import functools
import importlib.metadata
import json
import os
import random
import statistics
from pathlib import Path

import torch
import triton
from fla_npu.ops import ascendc
from triton.backends.ascend.testing import do_bench_npu

import flaggems_vllm


CHUNK = 64
LOWER_BOUND = -5.0
LOG2_E = 1.4426950408889634


def validate(actual, reference):
    torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-4)


def reference_cumsum(gate):
    result = torch.empty_like(gate)
    for start in range(0, gate.shape[1], CHUNK):
        end = start + CHUNK
        result[:, start:end] = gate[:, start:end].cumsum(1) * LOG2_E
    return result


def native(g, a, bias):
    return ascendc.npu_kda_gate_cumsum(
        g,
        CHUNK,
        A_log=a,
        dt_bias=bias,
        use_gate_in_kernel=True,
        safe_gate=True,
        lower_bound=LOWER_BOUND,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(73)
    fixture = Path("benchmark/fused_kda_gate_captured_contracts.json")
    contracts = json.loads(fixture.read_bytes())["contracts"]
    report = dict(
        status="running",
        package=importlib.metadata.version("flash-linear-attention-npu-a2"),
        triton=triton.__version__,
        torch=torch.__version__,
        device=torch.npu.get_device_name(0),
        physical_device=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        method="NPU profiler device kernel, 5 warmup and 30 active launches per round",
        rounds=args.rounds,
        native_layout="BHTD contiguous, prepacked outside kernel timing",
        triton_layout="BTHD contiguous",
        chunk_size=CHUNK,
        sequences_per_shape=1,
        results=[],
    )
    jobs = []
    for contract in contracts:
        g_info = contract["operator_g"]
        batch, t, hd = g_info["shape"]
        d = contract["D"]
        h = hd // d
        assert batch == 1 and contract["safe_gate"] and contract["lower_bound"] == LOWER_BOUND
        g = torch.randn(batch, t, h, d, device="npu", dtype=torch.bfloat16) * 0.5
        a = torch.randn(h, device="npu", dtype=torch.float32) * 0.5
        bias = torch.randn(h * d, device="npu", dtype=torch.float32) * 0.5
        cu = torch.tensor([0, t], device="npu", dtype=torch.int64)
        idx = torch.tensor([(0, i) for i in range(triton.cdiv(t, CHUNK))], device="npu", dtype=torch.int64)
        g_native = g.transpose(1, 2).contiguous()
        raw_gate = flaggems_vllm.fused_kda_gate(
            g.view(batch, t, hd), a, d, bias, safe_gate=True, lower_bound=LOWER_BOUND
        )
        gate_ref = LOWER_BOUND * torch.sigmoid(a.exp().view(1, 1, h, 1) * (g.float() + bias.view(1, 1, h, d)))
        validate(raw_gate, gate_ref)
        expected = reference_cumsum(gate_ref)
        base = functools.partial(native, g_native, a, bias)
        fused = functools.partial(flaggems_vllm.kda_gate_cumsum_triton, g, a, bias, cu, idx, LOWER_BOUND, CHUNK)
        validate(base().transpose(1, 2), expected)
        validate(fused(), expected)
        row = dict(T=t, H=h, D=d, samples_us={"ascendc": [], "gems_fused": []})
        report["results"].append(row)
        jobs.extend([(row, "ascendc", base), (row, "gems_fused", fused)])
        print(f"validated T={t} H={h} D={d}", flush=True)
    for rep in range(args.rounds):
        random.Random(101 + rep).shuffle(jobs)
        times = do_bench_npu([fn for _, _, fn in jobs])
        for (row, label, _), latency_ms in zip(jobs, times, strict=True):
            row["samples_us"][label].append(latency_ms * 1000)
        for row in report["results"]:
            samples = row["samples_us"]
            row["median_us"] = {name: statistics.median(values) for name, values in samples.items()}
            row["speedup"] = statistics.median(a / b for a, b in zip(samples["ascendc"], samples["gems_fused"], strict=True))
        report["completed_rounds"] = rep + 1
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"round {rep + 1} complete", flush=True)
    report["status"] = "passed"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
