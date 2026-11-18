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
"""Profile standalone Ascend softplus gates against the frozen vLLM-Ascend op."""

import argparse
import ast
import functools
import hashlib
import importlib
import json
import linecache
import math
import os
import random
import statistics
import subprocess
import sys
import types
from pathlib import Path

import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.testing import do_bench_npu

from tools.fused_kda_gate_perf import check_output, load_operator, make_inputs

BASELINE_REV = "a1df499e0762621c2b9561e2ee66c498a70d0328"
BASELINE_AST = "390b4212e9dd4c36470ac6225d474565bbd2c8350fbf64de8f4e726bcb213b08"
ROOT = Path(__file__).resolve().parents[1]


def baseline(tokens):
    path = ROOT / "baseline/fused_kda_gate_ascend.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    nodes = [
        node for node in tree.body if isinstance(node, (ast.Assign, ast.FunctionDef))
    ]
    tree = ast.Module(body=nodes, type_ignores=[])
    digest = hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    assert digest == BASELINE_AST, digest
    filename = str(path)
    linecache.cache[filename] = (
        len(source),
        None,
        source.splitlines(keepends=True),
        filename,
    )
    module = types.ModuleType(f"_ascend_original_gate_{tokens}")
    module.__file__ = filename
    module.__dict__.update(
        torch=torch,
        triton=triton,
        tl=tl,
        cdiv=triton.cdiv,
        next_power_of_2=triton.next_power_of_2,
    )
    sys.modules[module.__name__] = module
    exec(compile(tree, filename, "exec"), module.__dict__)
    assert len(module.kda_gate_fwd_kernel.configs) == 24
    assert module.kda_gate_fwd_kernel.keys == ["H", "D"]
    return module.fused_kda_gate, module.kda_gate_fwd_kernel


@triton.jit
def candidate_kernel(
    G,
    A,
    BIAS,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    BLOCK_PTR: tl.constexpr,
):
    pid = tl.program_id(0)
    head = (pid % (H // BH)) * BH
    token = (pid // (H // BH)) * BT
    cols = tl.arange(0, BH * D)
    if BLOCK_PTR:
        g_ptr = tl.make_block_ptr(
            G + head * D,
            (T, BH * D),
            (H * D, 1),
            (token, 0),
            (BT, BH * D),
            (1, 0),
        )
        y_ptr = tl.make_block_ptr(
            Y + head * D,
            (T, BH * D),
            (H * D, 1),
            (token, 0),
            (BT, BH * D),
            (1, 0),
        )
        value = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)
    else:
        rows = token + tl.arange(0, BT)
        offsets = rows[:, None] * (H * D) + head * D + cols[None, :]
        mask = rows[:, None] < T
        value = tl.load(G + offsets, mask, other=0).to(tl.float32)
    bias = tl.load(BIAS + head * D + cols).to(tl.float32)
    value += bias[None, :]
    if BH == 1:
        scale = tl.exp(tl.load(A + head).to(tl.float32))
    else:
        scale = tl.exp(tl.load(A + head + tl.arange(0, BH)).to(tl.float32))
        scale = tl.reshape(tl.broadcast_to(scale[:, None], (BH, D)), (BH * D,))
        scale = scale[None, :]
    softplus = tl.where(value > 20.0, value, tl.log(1.0 + tl.exp(value)))
    result = -scale * softplus
    if BLOCK_PTR:
        tl.store(y_ptr, result, boundary_check=(0, 1))
    else:
        tl.store(Y + offsets, result, mask)


def candidate(inputs, output, bt, bh, block_ptr, multibuffer=True):
    g, a, d, bias = inputs[:4]
    h = a.numel()
    t = g.numel() // (h * d)
    candidate_kernel[(triton.cdiv(t, bt) * (h // bh),)](
        g, a, bias, output, t, h, d, bt, bh, block_ptr, multibuffer=multibuffer
    )
    return output


@triton.jit
def loop_kernel(
    G,
    A,
    BIAS,
    Y,
    T: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    BT: tl.constexpr,
    BH: tl.constexpr,
    LOOPS: tl.constexpr,
    BLOCK_PTR: tl.constexpr,
):
    pid = tl.program_id(0)
    head = (pid % (H // BH)) * BH
    first = (pid // (H // BH)) * (BT * LOOPS)
    cols = tl.arange(0, BH * D)
    bias = tl.load(BIAS + head * D + cols).to(tl.float32)
    scales = tl.exp(tl.load(A + head + tl.arange(0, BH)).to(tl.float32))
    scales = tl.reshape(tl.broadcast_to(scales[:, None], (BH, D)), (BH * D,))
    for step in range(LOOPS):
        token = first + step * BT
        if BLOCK_PTR:
            gp = tl.make_block_ptr(
                G + head * D, (T, BH * D), (H * D, 1), (token, 0), (BT, BH * D), (1, 0)
            )
            yp = tl.make_block_ptr(
                Y + head * D, (T, BH * D), (H * D, 1), (token, 0), (BT, BH * D), (1, 0)
            )
            value = tl.load(gp, boundary_check=(0, 1)).to(tl.float32)
        else:
            rows = token + tl.arange(0, BT)
            offsets = rows[:, None] * (H * D) + head * D + cols[None, :]
            mask = rows[:, None] < T
            value = tl.load(G + offsets, mask, other=0).to(tl.float32)
        value += bias[None, :]
        result = -scales[None, :] * tl.where(
            value > 20.0, value, tl.log(1.0 + tl.exp(value))
        )
        if BLOCK_PTR:
            tl.store(yp, result, boundary_check=(0, 1))
        else:
            tl.store(Y + offsets, result, mask)


def loop_candidate(inputs, output, bt, bh, loops, pointer):
    g, a, d, bias = inputs[:4]
    h = a.numel()
    t = g.numel() // (h * d)
    loop_kernel[(triton.cdiv(t, bt * loops) * (h // bh),)](
        g, a, bias, output, t, h, d, bt, bh, loops, pointer, multibuffer=False
    )
    return output


def write_report(options, report):
    options.output.parent.mkdir(parents=True, exist_ok=True)
    options.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def validate_isolated(options, runtime, gate):
    """Run the real operator tests when an unrelated TLE import blocks the package."""
    import pytest

    package = sys.modules["flaggems_vllm"]
    package.device = runtime.device.name
    package.vendor_name = runtime.device.vendor_name
    package.runtime = runtime
    package.fused_kda_gate = gate
    package.ops = sys.modules["flaggems_vllm.ops"]
    package.ops.fused_kda_gate = gate
    package.testing = importlib.import_module("flaggems_vllm.testing")
    # Public-export checks need an ordinary package import and are excluded here.
    code = pytest.main(
        [
            "-q",
            "tests/test_fused_kda_gate.py",
            "--ref",
            "cpu",
            "-x",
            "-k",
            "not public_export_and_signature and not captured_model_shapes",
            "--junitxml",
            str(options.output.with_suffix(".xml")),
        ]
    )
    write_report(
        options,
        {
            "status": "passed" if code == 0 else "failed",
            "pytest_exitcode": int(code),
            "isolated_operator_import": True,
            "public_package_import_verified": False,
        },
    )
    return int(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("compare", "sweep", "validate"), default="compare"
    )
    parser.add_argument(
        "--sweep-kind", choices=("layout", "storage", "loops"), default="layout"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--active", type=int, default=30)
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument("--keep-profile", action="store_true")
    options = parser.parse_args()
    runtime, gems = load_operator(isolated=True)
    assert runtime.device.vendor_name == "ascend"
    assert Path("/.dockerenv").exists(), "run inside the test container"
    torch.npu.set_device(0)
    if options.mode == "validate":
        options.output.parent.mkdir(parents=True, exist_ok=True)
        raise SystemExit(validate_isolated(options, runtime, gems))
    contracts_bytes = (
        ROOT / "benchmark/fused_kda_gate_captured_contracts.json"
    ).read_bytes()
    contracts = json.loads(contracts_bytes)["contracts"]
    report = dict(
        status="running",
        revision=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        device=torch.npu.get_device_name(0),
        visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        torch=torch.__version__,
        torch_npu=torch_npu.__version__,
        triton=triton.__version__,
        safe_gate=False,
        method="NPU profiler device kernel duration",
        baseline_revision=BASELINE_REV,
        baseline_ast_sha256=BASELINE_AST,
        baseline_original_configs=24,
        baseline_fresh_tuner_per_shape=True,
        fixture_sha256=hashlib.sha256(contracts_bytes).hexdigest(),
        public_package_import_verified=False,
        mode=options.mode,
        rounds=options.rounds,
        warmup_calls=options.warmup,
        active_calls=options.active,
        results=[],
    )
    jobs = []
    for case in contracts:
        inputs = make_inputs({**case, "safe_gate": False}, "npu")
        g, a, d = inputs[:3]
        h = a.numel()
        t = g.numel() // (h * d)
        if options.tokens and t not in options.tokens:
            continue
        base, original = baseline(t)
        base_call = functools.partial(base, *inputs[:6])
        gems_call = functools.partial(gems, *inputs)
        print(
            f"KDA_PROGRESS T={t} compiling frozen baseline and production gate",
            flush=True,
        )
        base_result = base_call()
        check_output(base_result, inputs)
        out = gems_call()
        check_output(out, inputs)
        torch.testing.assert_close(out, base_result, rtol=1e-4, atol=1e-4)
        row = dict(
            T=t,
            H=h,
            D=d,
            baseline_config=original.best_config.all_kwargs(),
            max_abs_error_vs_baseline=(out - base_result).abs().max().item(),
            timings={"baseline": [], "gems": []},
            errors={},
        )
        report["results"].append(row)
        jobs.extend([(row, "baseline", base_call), (row, "gems", gems_call)])
        if options.mode == "sweep":
            matrix = [
                (1, 64),
                (1, 128),
                (2, 32),
                (4, 16),
                (4, 32),
                (8, 8),
                (8, 16),
                (16, 8),
            ]
            buffers = (True,)
            if options.sweep_kind == "storage":
                matrix = [(1, 128), (4, 32), (8, 16), (16, 4), (16, 8)]
                buffers = (True, False)
            for bh, bt in matrix:
                if bt > triton.next_power_of_2(t):
                    continue
                variants = [
                    (pointer, buffering)
                    for pointer in (False, True)
                    for buffering in buffers
                ]
                for block_ptr, buffering in variants:
                    label = (
                        f"BH={bh},BT={bt},block_ptr={block_ptr},multibuffer={buffering}"
                    )
                    buffer = torch.empty_like(out)
                    call = functools.partial(
                        candidate, inputs, buffer, bt, bh, block_ptr, buffering
                    )
                    try:
                        check_output(call(), inputs)
                        torch.testing.assert_close(
                            buffer, base_result, rtol=1e-4, atol=1e-4
                        )
                    except Exception as error:
                        row["errors"][label] = repr(error)[:1000]
                        print(
                            f"KDA_PROGRESS T={t} rejected {label}: {str(error)[:180]}",
                            flush=True,
                        )
                        continue
                    row["timings"][label] = []
                    jobs.append((row, label, call))
            if options.sweep_kind == "loops":
                # The loop reuses parameters while keeping a bounded 8K live tile.
                for bh, bt in [(4, 16), (16, 4)]:
                    for loops in (1, 2, 4, 8):
                        for pointer in (False, True):
                            label = f"loop BH={bh},BT={bt},steps={loops},ptr={pointer}"
                            buffer = torch.empty_like(out)
                            call = functools.partial(
                                loop_candidate, inputs, buffer, bt, bh, loops, pointer
                            )
                            try:
                                check_output(call(), inputs)
                                torch.testing.assert_close(
                                    buffer, base_result, rtol=1e-4, atol=1e-4
                                )
                            except Exception as error:
                                row["errors"][label] = repr(error)[:1000]
                                print(
                                    f"KDA_PROGRESS T={t} rejected {label}", flush=True
                                )
                                continue
                            row["timings"][label] = []
                            jobs.append((row, label, call))
            print(
                f"KDA_PROGRESS T={t} validated {len(row['timings'])} variants",
                flush=True,
            )
        write_report(options, report)
    for repeat in range(options.rounds):
        order = jobs[:]
        random.Random(521 + repeat).shuffle(order)
        timings = do_bench_npu(
            [fn for _, _, fn in order],
            warmup=options.warmup,
            active=options.active,
            keep_res=options.keep_profile,
        )
        for (row, label, _), latency in zip(order, timings):
            us = float(latency) * 1000
            assert math.isfinite(us) and us > 0, (row["T"], label, us)
            row["timings"][label].append(us)
        for row in report["results"]:
            times = row["timings"]
            row["median_us"] = {
                label: statistics.median(values) for label, values in times.items()
            }
            ratios = [base / new for base, new in zip(times["baseline"], times["gems"])]
            row["paired_speedup"] = ratios
            row["speedup"] = statistics.median(ratios)
            best = min(row["median_us"], key=row["median_us"].get)
            print(
                f"KDA_PROGRESS round={repeat+1} T={row['T']} best={best} "
                f"us={row['median_us'][best]:.3f} speedup={row['speedup']:.3f}",
                flush=True,
            )
        report["completed_rounds"] = repeat + 1
        write_report(options, report)
    report["mean_speedup"] = statistics.mean(
        row["speedup"] for row in report["results"]
    )
    report["at_least_1_10"] = sum(
        int(row["speedup"] >= 1.10) for row in report["results"]
    )
    report["status"] = "passed"
    write_report(options, report)
    print(f"KDA_PROGRESS completed mean={report['mean_speedup']:.3f}", flush=True)


if __name__ == "__main__":
    main()
