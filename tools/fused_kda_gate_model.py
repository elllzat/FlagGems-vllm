#!/usr/bin/env python3
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

"""Real-checkpoint gate validation; see fused_kda_gate_model.md for methodology."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path


def _tensor_metadata(tensor):
    if tensor is None:
        return None
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "dtype": str(tensor.dtype),
    }


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _operator_timings(torch, functions):
    """Repository operator-mode methodology, on tensors captured from the model."""
    rounds = []
    for index in range(3):
        times = {}
        order = list(functions.items())
        if index % 2:
            order.reverse()
        for name, function in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(5):
                function()
            torch.cuda.synchronize()
            estimate_ms = (time.perf_counter() - start) * 1000 / 5
            warmup = max(1, int(100 / estimate_ms))
            repeat = max(1, int(500 / estimate_ms))
            for _ in range(warmup):
                function()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeat):
                function()
            torch.cuda.synchronize()
            times[name] = (time.perf_counter() - start) * 1000 / repeat
        times["speedup"] = times["native_ms"] / times["gems_ms"]
        rounds.append(times)
    return {"mode": "operator", "cuda_graph": False, "paired_rounds": rounds}


class GateObserver:
    """Worker-local test adapter. Never imported by production operators."""

    def __init__(self):
        import torch
        from vllm.models.glm5next.common import kda
        from vllm.models.glm5next.nvidia.ops.third_party.kda import kernels

        from flaggems_vllm import fused_kda_gate

        self.torch = torch
        self.layer = kda
        self.kernels = kernels
        self.candidate = fused_kda_gate
        self.layer_class = kda.Glm5NextLinearAttention
        self.mode = "native"
        self.trace = False
        self.profile = False
        self.rows = {}
        self.original = {
            "decode": kda.fused_recurrent_kda,
            "prefill": kda.chunk_kda_with_fused_gate,
            "flashkda": self.layer_class._flashkda_prefill,
        }
        self.signatures = {
            name: inspect.signature(function)
            for name, function in self.original.items()
        }
        self.standalone = kernels.fused_kda_gate
        self.standalone_calls = 0
        self.candidate_calls = 0

    def gate(self, path, g, A, bias, safe, lower):
        h, d = g.shape[-2:]
        # view fails instead of silently inserting a copy for unsupported layouts.
        flat = g.view(-1, h * d)
        candidate = None
        if self.trace:
            row = {
                "path": path,
                "g": _tensor_metadata(g),
                "A": _tensor_metadata(A),
                "bias": _tensor_metadata(bias),
                "T": flat.shape[0],
                "H": h,
                "D": d,
                "safe_gate": safe,
                "lower_bound": lower,
                "beta": 1.0,
                "threshold": 20.0,
            }
            key = json.dumps(row, sort_keys=True)
            if key not in self.rows:
                candidate = self.candidate(flat, A, d, bias, 1.0, 20.0, safe, lower)
                expected = self.standalone(flat, A, d, bias, 1.0, 20.0, safe, lower)
                self.torch.testing.assert_close(
                    candidate, expected, rtol=3e-6, atol=3e-6
                )
                row.update(calls=0, gate_check="passed")
                if self.profile:
                    arguments = (flat, A, d, bias, 1.0, 20.0, safe, lower)
                    row["standalone_operator_timings"] = _operator_timings(
                        self.torch,
                        {
                            "native_ms": lambda: self.standalone(*arguments),
                            "gems_ms": lambda: self.candidate(*arguments),
                        },
                    )
                self.rows[key] = row
            self.rows[key]["calls"] += 1
        if self.mode == "gems-split":
            if candidate is None:
                candidate = self.candidate(flat, A, d, bias, 1.0, 20.0, safe, lower)
            self.candidate_calls += 1
            return candidate.view(g.shape)
        return None

    def recurrent(self, *args, **kwargs):
        native = self.original["decode"]
        bound = self.signatures["decode"].bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        if values["compute_gate"]:
            gate = self.gate(
                "decode",
                values["g"],
                values["a_log"],
                values["g_bias"],
                True,
                values["lower_bound"],
            )
            if gate is not None:
                values["g"] = gate
                values["compute_gate"] = False
        return native(*bound.args, **bound.kwargs)

    def chunk(self, *args, **kwargs):
        native = self.original["prefill"]
        bound = self.signatures["prefill"].bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        gate = self.gate(
            "prefill",
            values["raw_g"],
            values["A_log"],
            values["g_bias"],
            values["safe_gate"],
            values["lower_bound"],
        )
        if gate is None:
            return native(*bound.args, **bound.kwargs)
        forwarded = {
            key: value
            for key, value in values.items()
            if key not in ("raw_g", "A_log", "g_bias", "safe_gate", "lower_bound")
        }
        forwarded.update(forwarded.pop("kwargs", {}))
        return self.kernels.chunk_kda(g=gate, **forwarded)

    def flash(self, layer, *args, **kwargs):
        native = self.original["flashkda"]
        bound = self.signatures["flashkda"].bind(layer, *args, **kwargs)
        bound.apply_defaults()
        self.gate(
            "prefill-flashkda",
            bound.arguments["g"],
            layer.A_log,
            layer.dt_bias,
            True,
            layer.kda_lower_bound,
        )
        return native(*bound.args, **bound.kwargs)

    def standalone_wrapper(self, *args, **kwargs):
        self.standalone_calls += 1
        return self.standalone(*args, **kwargs)

    def configure(self, mode, trace, profile=False):
        self.mode, self.trace = mode, trace
        self.profile = profile
        if mode == "native" and not trace:
            self.layer.fused_recurrent_kda = self.original["decode"]
            self.layer.chunk_kda_with_fused_gate = self.original["prefill"]
            self.layer_class._flashkda_prefill = self.original["flashkda"]
            self.kernels.fused_kda_gate = self.standalone
        else:
            self.layer.fused_recurrent_kda = self.recurrent
            self.layer.chunk_kda_with_fused_gate = self.chunk
            observer = self

            def flash(layer, *args, **kwargs):
                return observer.flash(layer, *args, **kwargs)

            self.layer_class._flashkda_prefill = flash
            self.kernels.fused_kda_gate = self.standalone_wrapper


class KdaGateWorkerExtension:
    def kda_gate_configure(self, mode, trace):
        if not hasattr(self, "_gate_observer"):
            self._gate_observer = GateObserver()
        self._gate_observer.configure(mode, trace, profile=trace)

    def kda_gate_report(self):
        observer = self._gate_observer
        torch = observer.torch
        return {
            "rank": self.rank,
            "gpu": torch.cuda.get_device_name(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": importlib.metadata.version("triton"),
            "kda_source_sha256": _digest(observer.layer.__file__),
            "kernels_sha256": _digest(observer.kernels.__file__),
            "standalone_model_calls": observer.standalone_calls,
            "candidate_integrated_calls": observer.candidate_calls,
            "shapes": list(observer.rows.values()),
        }


def _write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    shapes = {
        (row["T"], row["H"], row["D"], row["safe_gate"])
        for case in result.get("cases", [])
        for worker in case.get("workers", [])
        for row in worker["shapes"]
    }
    if shapes:
        # JSON is valid YAML and is accepted by the existing --shape_file reader.
        path.with_suffix(".shapes.yaml").write_text(
            json.dumps({"fused_kda_gate": {"shapes": sorted(shapes)}}, indent=2),
            encoding="utf-8",
        )


def _prompts(tokenizer, length, batch):
    text = (
        "This is a reproducible attention-kernel evaluation using public text. "
        "A gate controls how previous information is retained in a sequence. "
        "Describe this process clearly. "
    )
    tokens = tokenizer.encode(text, add_special_tokens=False)
    prompt = list(itertools.islice(itertools.cycle(tokens), length))
    return [{"prompt_token_ids": prompt.copy()} for _ in range(batch)]


def _request_metrics(output):
    metrics = output.metrics
    if metrics is None:
        return {"ttft_s": None, "tpot_s": None}
    first = getattr(metrics, "first_token_time", None)
    arrival = getattr(metrics, "arrival_time", None)
    finished = getattr(metrics, "finished_time", None)
    count = len(output.outputs[0].token_ids)
    return {
        "ttft_s": (
            first - arrival if first is not None and arrival is not None else None
        ),
        "tpot_s": (
            (finished - first) / (count - 1)
            if finished is not None and first is not None and count > 1
            else None
        ),
    }


def _weights_ready(model):
    config = json.loads((model / "config.json").read_text())
    if config.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        raise ValueError("Expected the complete GLM-5.3-Flash checkpoint")
    index = json.loads((model / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    missing = [name for name in files if not (model / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint incomplete: {len(missing)} shards missing")
    return {
        "config_sha256": _digest(model / "config.json"),
        "index_sha256": _digest(model / "model.safetensors.index.json"),
        "shards": len(files),
        "weight_bytes": sum((model / name).stat().st_size for name in files),
    }


def _idle_devices(tp):
    import pynvml

    pynvml.nvmlInit()
    try:
        if pynvml.nvmlDeviceGetCount() != tp:
            raise RuntimeError("Expose exactly TP GPUs in the isolated model container")
        devices = []
        for index in range(tp):
            handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
            if processes:
                raise BlockingIOError(f"GPU {index} has existing compute processes")
            devices.append(
                {
                    "uuid": str(pynvml.nvmlDeviceGetUUID(handle)),
                    "memory_used": pynvml.nvmlDeviceGetMemoryInfo(handle).used,
                    "temperature_c": pynvml.nvmlDeviceGetTemperature(handle, 0),
                }
            )
        return devices
    finally:
        pynvml.nvmlShutdown()


def _adapter_smoke(output):
    """Check KDA routing on small tensors before allocating model weights."""
    import torch

    observer = GateObserver()
    observer.configure("gems-split", True)
    torch.manual_seed(42)
    results = []
    for dtype in (torch.bfloat16, torch.float16):
        for path, length in (("decode", 1), ("prefill", 37)):
            shape = (1, length, 8, 128)
            q, k, v = [torch.randn(shape, device="cuda", dtype=dtype) for _ in range(3)]
            raw_g = torch.randn(shape, device="cuda", dtype=dtype) * 0.1
            A = torch.randn(8, device="cuda", dtype=torch.float32) * 0.1
            bias = torch.randn(8, 128, device="cuda", dtype=torch.float32) * 0.1
            beta = torch.rand(1, length, 8, device="cuda", dtype=torch.float32)
            state = (
                torch.randn(
                    2 if path == "decode" else 1,
                    8,
                    128,
                    128,
                    device="cuda",
                    dtype=torch.float32,
                )
                * 0.1
            )
            cu = torch.tensor([0, length], device="cuda", dtype=torch.int32)
            common = dict(
                q=q,
                k=k,
                v=v,
                beta=beta,
                use_qk_l2norm_in_kernel=True,
                cu_seqlens=cu,
            )
            if path == "decode":
                call = dict(
                    **common,
                    g=raw_g,
                    a_log=A,
                    g_bias=bias,
                    compute_gate=True,
                    lower_bound=-5.0,
                    # Cache slot zero is vLLM's NULL_BLOCK_ID and skips the kernel.
                    ssm_state_indices=torch.ones(1, device="cuda", dtype=torch.int32),
                )
                candidate_fn = observer.recurrent
            else:
                call = dict(
                    **common,
                    raw_g=raw_g,
                    A_log=A,
                    g_bias=bias,
                    safe_gate=True,
                    lower_bound=-5.0,
                    output_final_state=True,
                )
                candidate_fn = observer.chunk
            # Native chunk KDA reuses v as its output buffer. Independent inputs
            # keep the reference result and the candidate input from aliasing it.
            reference_call = {
                key: value.clone() if key in ("q", "k", "v") else value
                for key, value in call.items()
            }
            expected = observer.original[path](
                initial_state=state.clone(), **reference_call
            )
            candidate = candidate_fn(initial_state=state.clone(), **call)
            for actual, reference in zip(candidate, expected):
                torch.testing.assert_close(actual, reference, rtol=1e-2, atol=1e-2)
            results.append({"path": path, "dtype": str(dtype), "status": "passed"})
    observer.configure("native", False)
    _write(output, {"status": "adapter_smoke_passed", "cases": results})
    print(json.dumps(results))


def main():
    # CUDA JIT helpers must find this environment's ninja and Python executables.
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["native", "gems-split"], default="native")
    parser.add_argument("--prefill-backend", choices=["auto", "triton"], default="auto")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--output-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.97)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--adapter-smoke-only", action="store_true")
    parser.add_argument("--wait-for-checkpoint", type=int, default=0, metavar="SECONDS")
    args = parser.parse_args()
    if args.adapter_smoke_only:
        _adapter_smoke(args.output)
        return
    if args.model is None:
        parser.error("--model is required for real-checkpoint inference")
    if args.mode == "gems-split" and args.prefill_backend != "triton":
        parser.error("gems-split requires the explicitly labelled Triton experiment")
    if min(args.input_lengths + args.batches + [args.output_length, args.repeats]) < 1:
        parser.error("workload sizes and repeats must be positive")
    deadline = time.monotonic() + args.wait_for_checkpoint
    while True:
        try:
            weights = _weights_ready(args.model)
            devices = _idle_devices(args.tp)
            break
        except (FileNotFoundError, json.JSONDecodeError, BlockingIOError) as error:
            if time.monotonic() >= deadline:
                raise
            _write(
                args.output,
                {
                    "status": (
                        "waiting_for_gpus"
                        if isinstance(error, BlockingIOError)
                        else "waiting_for_checkpoint"
                    ),
                    "detail": str(error),
                },
            )
            time.sleep(30)
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    from vllm import LLM, SamplingParams

    root = Path(__file__).resolve().parents[1]
    config = {
        "model": str(args.model.resolve()),
        "weights": weights,
        "tp": args.tp,
        "input_lengths": args.input_lengths,
        "batches": args.batches,
        "output_length": args.output_length,
        "repeats": args.repeats,
        "seed": 42,
        "prefill_backend": args.prefill_backend,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True,
        "prefix_caching": False,
        "speculation": False,
        "modality": "text",
    }
    reference = None
    if args.reference:
        reference = json.loads(args.reference.read_text())
        if reference["config"] != config or reference["mode"] != "native":
            raise ValueError(
                "Reference must be native with identical checkpoint/workload"
            )
        if reference["vllm"] != importlib.metadata.version("vllm"):
            raise ValueError("Reference and candidate must use the same vLLM version")
    result = {
        "status": "initializing",
        "mode": args.mode,
        "config": config,
        "python": platform.python_version(),
        "devices_before_run": devices,
        "vllm": importlib.metadata.version("vllm"),
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip(),
        "operator_sha256": _digest(root / "src/flaggems_vllm/ops/fused_kda_gate.py"),
        "cases": [],
    }
    _write(args.output, result)
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=args.tp,
        enforce_eager=True,
        enable_prefix_caching=False,
        limit_mm_per_prompt={"image": 0, "video": 0},
        seed=42,
        max_model_len=max(args.input_lengths) + args.output_length + 16,
        max_num_seqs=max(args.batches),
        max_num_batched_tokens=max(2048, max(args.batches)),
        gpu_memory_utilization=args.gpu_memory_utilization,
        additional_config={"kda_prefill_backend": args.prefill_backend},
        worker_extension_cls="tools.fused_kda_gate_model.KdaGateWorkerExtension",
    )
    params = SamplingParams(
        temperature=0,
        max_tokens=args.output_length,
        min_tokens=args.output_length,
        ignore_eos=True,
        seed=42,
    )
    result["status"] = "running"
    for length, batch in itertools.product(args.input_lengths, args.batches):
        prompts = _prompts(llm.get_tokenizer(), length, batch)
        llm.collective_rpc("kda_gate_configure", args=(args.mode, True))
        trace_outputs = llm.generate(prompts, params, use_tqdm=False)
        trace_ids = [list(o.outputs[0].token_ids) for o in trace_outputs]
        workers = llm.collective_rpc("kda_gate_report")
        llm.collective_rpc("kda_gate_configure", args=(args.mode, False))
        # Compile and tune before timed calls. Native timed calls restore originals.
        llm.generate(prompts, params, use_tqdm=False)
        runs = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            outputs = llm.generate(prompts, params, use_tqdm=False)
            elapsed = time.perf_counter() - start
            token_ids = [list(o.outputs[0].token_ids) for o in outputs]
            if token_ids != trace_ids:
                raise AssertionError("Trace and timed greedy generation differ")
            runs.append(
                {
                    "elapsed_s": elapsed,
                    "output_tokens_per_s": sum(map(len, token_ids)) / elapsed,
                    "requests": [_request_metrics(o) for o in outputs],
                }
            )
        case = {
            "input_length": length,
            "batch": batch,
            "token_ids": trace_ids,
            "workers": workers,
            "runs": runs,
            "median_elapsed_s": statistics.median(r["elapsed_s"] for r in runs),
        }
        if args.mode == "gems-split" and not all(
            worker["candidate_integrated_calls"] > 0 for worker in workers
        ):
            raise AssertionError("Candidate was never executed on at least one worker")
        if reference:
            base = next(
                c
                for c in reference["cases"]
                if c["input_length"] == length and c["batch"] == batch
            )
            case["generated_tokens_match"] = trace_ids == base["token_ids"]
            case["speedup"] = base["median_elapsed_s"] / case["median_elapsed_s"]
            if not case["generated_tokens_match"]:
                result["status"] = "validation_failed"
                result["cases"].append(case)
                _write(args.output, result)
                raise AssertionError("Generated token IDs differ from native model")
        result["cases"].append(case)
        _write(args.output, result)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in case.items()
                    if k not in ("workers", "token_ids", "runs")
                }
            )
        )
    result["status"] = "measured"
    _write(args.output, result)


if __name__ == "__main__":
    main()
