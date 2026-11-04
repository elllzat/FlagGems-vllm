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
"""Validate only the portable KDA gate on a vendor device.

--isolated skips the package's all-operator imports when an unrelated optional
dependency blocks collection. It still loads the actual runtime, configuration,
LibEntry and operator source. Such a run does not validate public-package imports.
"""

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

import torch
import triton


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isolated", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.isolated:
        source = Path(__file__).resolve().parents[1] / "src" / "flaggems_vllm"
        for name, path in (
            ("flaggems_vllm", source),
            ("flaggems_vllm.ops", source / "ops"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
        runtime = importlib.import_module("flaggems_vllm.runtime")
        gate = importlib.import_module(
            "flaggems_vllm.ops.fused_kda_gate"
        ).fused_kda_gate
    else:
        from flaggems_vllm import fused_kda_gate as gate
        from flaggems_vllm import runtime

    dtypes = [torch.float16, torch.float32]
    if runtime.device.support_bf16:
        dtypes.append(torch.bfloat16)
    if runtime.device.support_fp64:
        dtypes.append(torch.float64)
    shapes = [(T, 16, 128) for T in (1, 2, 128, 129, 257, 1024, 1025, 2048)]
    shapes += [(0, 2, 8), (37, 12, 80), (5003, 12, 80)]
    rows = []
    torch.manual_seed(42)
    for T, H, D in shapes:
        for dtype in dtypes:
            for safe in (False, True):
                for has_bias in (False, True):
                    g = torch.randn(
                        1, T, H * D, device=runtime.device.name, dtype=dtype
                    )
                    A = torch.randn(1, 1, H, 1, device=g.device) * 0.5
                    bias = (
                        torch.randn(H * D, device=g.device) * 0.1 if has_bias else None
                    )
                    output = gate(g, A, D, bias, safe_gate=safe)
                    value = g.float().reshape(1, T, H, D)
                    if has_bias:
                        value = value + bias.reshape(H, D)
                    scale = A.exp()
                    if safe:
                        reference = -5.0 / (1.0 + torch.exp(-(scale * value)))
                    else:
                        reference = -scale * torch.where(
                            value > 20.0, value, torch.log(1.0 + value.exp())
                        )
                    assert output.shape == (1, T, H, D)
                    assert output.dtype == torch.float32 and output.device == g.device
                    torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
                    rows.append(
                        dict(T=T, H=H, D=D, dtype=str(dtype), safe=safe, bias=has_bias)
                    )
    runtime.torch_device_fn.synchronize()
    report = dict(
        status="passed",
        vendor=runtime.device.vendor_name,
        torch=torch.__version__,
        triton=triton.__version__,
        isolated=args.isolated,
        public_package_import_verified=not args.isolated,
        passed=len(rows),
        cases=rows,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"{report['vendor']}: {len(rows)} cases passed; isolated={args.isolated}")


if __name__ == "__main__":
    main()
