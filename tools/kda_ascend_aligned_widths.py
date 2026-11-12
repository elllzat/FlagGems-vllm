"""Check the newly vectorized Ascend head widths at their largest tile sizes."""

import itertools
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd() / "tools"))
from fused_kda_gate_perf import check_output, load_operator


def main():
    runtime, gate = load_operator(True)
    torch.manual_seed(42)
    dtypes = (torch.float16, torch.bfloat16, torch.float32)
    widths = (48, 80, 96, 112, 144, 160, 176, 192, 208, 224, 240)
    cases = itertools.product(widths, dtypes, (False, True))
    checked = 0
    for d, dtype, safe in cases:
        t, h = 513, 3
        device = runtime.device.name
        g = torch.randn(t, h * d, device=device, dtype=dtype)
        a = torch.randn(h, device=device, dtype=dtype) * 0.5
        bias = torch.randn(h * d, device=device, dtype=dtype) * 0.1
        inputs = (g, a, d, bias, 0.5, 10.0, safe, -3.0)
        check_output(gate(*inputs), inputs)
        checked += 1
        print(f"validated D={d} {dtype} safe={safe}", flush=True)
    print(f"passed {checked}", flush=True)


if __name__ == "__main__":
    main()
