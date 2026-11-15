"""Sweep direct kernel launches on the captured GLM5Next safe-gate shapes."""

import functools
import importlib
import json
import random
import sys
from pathlib import Path

import torch
import triton

from tools.fused_kda_gate_perf import check_output, make_inputs, measure
from tools.fused_kda_gate_portable_smoke import load_operator
from tools.fused_kda_gate_speedup import load_pinned_reference


def main():
    runtime, gate = load_operator(False)
    baseline, _, _ = load_pinned_reference()
    module = importlib.import_module(gate.__module__)
    flat = module._fused_kda_gate_kernel.fn.fn
    tiled = module._fused_kda_gate_tiled_kernel.fn.fn
    mode = "musagraph" if runtime.device.name == "musa" else "cudagraph"
    fixture = Path("benchmark/fused_kda_gate_captured_contracts.json")
    rows = {}
    for contract in json.loads(fixture.read_text())["contracts"]:
        inputs = make_inputs(contract, runtime.device.name)
        g, a, d, bias, beta, threshold, safe, bounds = inputs
        h = a.numel()
        t = g.numel() // (h * d)
        y = torch.empty((*g.shape[:-1], h, d), dtype=torch.float32, device=g.device)
        args = (g, a, bias, y, t, h, d, beta, threshold, safe, bounds, bias is not None)
        fs = {"production": functools.partial(gate, *inputs), "vllm": functools.partial(baseline, *inputs)}
        for warps in (1, 2, 4, 8):
            for block in (64, 128, 256, 512, 1024, 2048, 4096):
                if block > g.numel():
                    continue
                name = f"flat:{block}:{warps}"
                launch = flat[(triton.cdiv(g.numel(), block),)]
                fs[name] = functools.partial(launch, *args, BLOCK=block, num_warps=warps, num_stages=1)
            for bt in (1, 2, 4, 8, 16, 32, 64):
                if bt > t and t <= 16:
                    continue
                name = f"tile:{bt}:{warps}"
                launch = tiled[(triton.cdiv(t, bt) * h,)]
                fs[name] = functools.partial(launch, *args, BT=bt, num_warps=warps, num_stages=1)
        order = list(fs)
        random.Random(87).shuffle(order)
        latencies = {}
        for name in order:
            fn = fs[name]
            try:
                out = fn()
                check_output(y if out is None else out if isinstance(out, torch.Tensor) else y, inputs)
                latencies[name] = measure(fn, mode, 3, 5)
            except Exception as error:
                print(t, name, repr(error)[:230], flush=True)
        rows[str(t)] = latencies
        Path(sys.argv[1]).write_text(json.dumps(rows, indent=2))
        best = sorted(latencies.items(), key=lambda kv: kv[1])[:5]
        print(f"{runtime.device.vendor_name} T={t} old={latencies.get('production'):.3f} ref={latencies.get('vllm'):.3f} best={best}", flush=True)


if __name__ == "__main__":
    main()
