"""Temporary 910B4 profiling sweep, with the previous production path in every group."""

import functools
import json
import os
import random

import torch
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu

from fused_kda_gate_perf import check_output, load_operator, make_inputs


@triton.jit
def gate_tile(
    G, A, BIAS, Y,
    T: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
    BT: tl.constexpr, BG: tl.constexpr, BD: tl.constexpr,
    SAFE: tl.constexpr, BIAS_PRESENT: tl.constexpr,
):
    pid = tl.program_id(0)
    base_head = (pid % (H // BG)) * BG
    rows = pid // (H // BG) * BT + tl.arange(0, BT)
    lanes = tl.arange(0, BG * BD)
    cols = base_head * D + lanes
    offsets = rows[:, None] * (H * D) + cols[None, :]
    mask = rows[:, None] < T
    if BD != D:
        mask = mask & (lanes[None, :] < D)
    value = tl.load(G + offsets, mask, other=0).to(tl.float32)
    if BIAS_PRESENT:
        bias = tl.load(BIAS + cols, lanes < BG * D, other=0).to(tl.float32)
        value += bias[None, :]
    if BG == 1:
        scale = tl.exp(tl.load(A + base_head).to(tl.float32))
    else:
        head_scale = tl.exp(tl.load(A + base_head + tl.arange(0, BG)).to(tl.float32))
        scale = tl.reshape(tl.broadcast_to(head_scale[:, None], (BG, BD)), (BG * BD,))[None, :]
    if SAFE:
        result = -5.0 / (1.0 + tl.exp(-(scale * value)))
    else:
        softplus = tl.where(value > 20.0, value, tl.log(1.0 + tl.exp(value)))
        result = -scale * softplus
    tl.store(Y + offsets, result, mask)


def launch(g, a, bias, output, t, h, d, bt, bg, bd, safe):
    gate_tile[(triton.cdiv(t, bt) * (h // bg),)](
        g, a, bias, output, t, h, d, bt, bg, bd, safe, bias is not None
    )
    return output


def sweep():
    runtime, old = load_operator(True)
    with open("benchmark/fused_kda_gate_captured_contracts.json") as fixture:
        contracts = json.load(fixture)["contracts"]
    matrix = [
        (1, 4), (1, 8), (1, 16), (1, 32), (1, 64), (1, 128),
        (2, 8), (2, 16), (2, 32), (2, 64),
        (4, 4), (4, 8), (4, 16), (4, 32),
        (8, 4), (8, 8), (8, 16), (16, 2), (16, 4), (16, 8),
    ]
    results = {}
    output_path = os.environ["KDA_SWEEP_OUT"]
    test_cases = []
    for case in contracts:
        inputs = make_inputs(case, runtime.device.name)
        g, a, d = inputs[:3]
        h = a.numel()
        t = g.numel() // (h * d)
        if t > 2:
            test_cases.append((t, h, d, case["safe_gate"], inputs))
    for t, h, d, safe in ((37, 12, 80, True), (257, 12, 80, False), (1025, 12, 80, True)):
        g = torch.randn(t, h * d, dtype=torch.bfloat16, device=runtime.device.name)
        a = torch.randn(h, dtype=torch.float32, device=runtime.device.name)
        bias = torch.randn(h * d, dtype=torch.float32, device=runtime.device.name)
        test_cases.append((t, h, d, safe, (g, a, d, bias, 1.0, 20.0, safe, -5.0)))
    for t, h, d, safe, inputs in test_cases:
        print("START", t, h, d, safe, flush=True)
        g, a, _, bias = inputs[:4]
        out = torch.empty((*g.shape[:-1], h, d), device=g.device, dtype=torch.float32)
        old_gate = functools.partial(old, *inputs)
        check_output(old_gate(), inputs)
        variants = [("old", old_gate)]
        bd = triton.next_power_of_2(d)
        for bg, bt in matrix:
            if h % bg or (bg > 1 and bd != d) or bt > triton.next_power_of_2(t):
                continue
            f = functools.partial(launch, g, a, bias, out, t, h, d, bt, bg, bd, safe)
            name = f"BG={bg},BT={bt}"
            try:
                check_output(f(), inputs)
                variants.append((name, f))
            except Exception as error:
                print("INVALID", t, d, name, repr(error)[:450], flush=True)
        samples = {name: [] for name, _ in variants}
        for rep in range(3):
            random.Random(101 + rep).shuffle(variants)
            times = do_bench_npu([fn for _, fn in variants])
            for (name, _), latency in zip(variants, times, strict=True):
                samples[name].append(1000 * latency)
        results[f"{t}/{h}/{d}/{safe}"] = samples
        with open(output_path, "w") as report:
            json.dump(results, report, indent=2)
        print("END", t, d, json.dumps(samples), flush=True)


if __name__ == "__main__":
    sweep()
