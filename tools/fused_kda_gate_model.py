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
"""Collect native GLM KDA gate shapes without checkpoint weights."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path


def _metadata(tensor):
    if tensor is None:
        return None
    return dict(
        shape=list(tensor.shape), stride=list(tensor.stride()), dtype=str(tensor.dtype)
    )


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_revision(root):
    # Host worktree gitdir paths need not exist inside the mounted container.
    revision = os.environ.get("KDA_SHAPE_SOURCE_COMMIT")
    if revision:
        return revision
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


class ShapeObserver:
    """Read native gate metadata and delegate with the original arguments."""

    def __init__(self):
        from vllm.models.glm5next.common import kda
        from vllm.models.glm5next.nvidia.ops.third_party.kda import kernels

        self.layer, self.kernels = kda, kernels
        self.rows = {}
        self.original = {
            "decode": kda.fused_recurrent_kda,
            "prefill": kda.chunk_kda_with_fused_gate,
            "flashkda": kda.Glm5NextLinearAttention._flashkda_prefill,
            "standalone": kernels.fused_kda_gate,
        }
        self.signatures = {
            name: inspect.signature(fn) for name, fn in self.original.items()
        }

    def record(self, phase, source, g, A, bias, safe, lower, direct=False):
        h = A.numel()
        d = g.shape[-1] // h if direct else g.shape[-1]
        # A metadata-only view: no allocation, copy, or arithmetic kernel.
        api_g = g if direct else g.view(*g.shape[:-2], h * d)
        row = {
            "phase": phase,
            "observed_at": source,
            "source_kind": "direct_call" if direct else "gate_inside_fused_kda",
            "raw_g": _metadata(g),
            "operator_g": _metadata(api_g),
            "A": _metadata(A),
            "g_bias": _metadata(bias),
            "T": g.numel() // (h * d),
            "H": h,
            "D": d,
            "safe_gate": safe,
            "lower_bound": lower,
            "beta": 1.0,
            "threshold": 20.0,
        }
        key = json.dumps(row, sort_keys=True)
        if key not in self.rows:
            self.rows[key] = dict(row, calls=0)
        self.rows[key]["calls"] += 1

    def bound(self, name, args, kwargs):
        bound = self.signatures[name].bind(*args, **kwargs)
        bound.apply_defaults()
        return bound.arguments

    def recurrent(self, *args, **kwargs):
        values = self.bound("decode", args, kwargs)
        if values["compute_gate"]:
            self.record(
                "decode",
                "fused_recurrent_kda",
                values["g"],
                values["a_log"],
                values["g_bias"],
                True,
                values["lower_bound"],
            )
        return self.original["decode"](*args, **kwargs)

    def chunk(self, *args, **kwargs):
        values = self.bound("prefill", args, kwargs)
        self.record(
            "prefill",
            "chunk_kda_with_fused_gate",
            values["raw_g"],
            values["A_log"],
            values["g_bias"],
            values["safe_gate"],
            values["lower_bound"],
        )
        return self.original["prefill"](*args, **kwargs)

    def flash(self, layer, *args, **kwargs):
        values = self.bound("flashkda", (layer, *args), kwargs)
        self.record(
            "prefill",
            "flashkda",
            values["g"],
            layer.A_log,
            layer.dt_bias,
            True,
            layer.kda_lower_bound,
        )
        return self.original["flashkda"](layer, *args, **kwargs)

    def standalone(self, *args, **kwargs):
        values = self.bound("standalone", args, kwargs)
        self.record(
            "standalone",
            "fused_kda_gate",
            values["g"],
            values["A"],
            values["g_bias"],
            values["safe_gate"],
            values["lower_bound"],
            direct=True,
        )
        return self.original["standalone"](*args, **kwargs)

    def install(self):
        self.layer.fused_recurrent_kda = self.recurrent
        self.layer.chunk_kda_with_fused_gate = self.chunk
        observer = self

        def flash(layer, *args, **kwargs):
            return observer.flash(layer, *args, **kwargs)

        self.layer.Glm5NextLinearAttention._flashkda_prefill = flash
        self.kernels.fused_kda_gate = self.standalone


class KdaShapeWorkerExtension:
    def kda_shape_begin(self):
        if not hasattr(self, "_kda_shape_observer"):
            self._kda_shape_observer = ShapeObserver()
            self._kda_shape_observer.install()
        self._kda_shape_observer.rows.clear()

    def kda_shape_report(self):
        observer = self._kda_shape_observer
        return {
            "rank": self.rank,
            "kda_source_sha256": _digest(observer.layer.__file__),
            "kernels_source_sha256": _digest(observer.kernels.__file__),
            "shapes": list(observer.rows.values()),
        }


def _write(path, result):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(path)
    shapes = {
        (row["T"], row["H"], row["D"], row["safe_gate"])
        for case in result["cases"]
        for worker in case["workers"]
        for row in worker["shapes"]
    }
    if shapes:
        # JSON is valid YAML; the existing benchmark --shape_file accepts it.
        path.with_suffix(".shapes.yaml").write_text(
            json.dumps({"fused_kda_gate": {"shapes": sorted(shapes)}}, indent=2),
            encoding="utf-8",
        )


def _prompts(tokenizer, length, batch):
    tokens = tokenizer.encode(
        "A gate controls how previous information is retained in a sequence. ",
        add_special_tokens=False,
    )
    prompt = list(itertools.islice(itertools.cycle(tokens), length))
    return [{"prompt_token_ids": prompt.copy()} for _ in range(batch)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--input-lengths", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--output-length", type=int, default=2)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.95)
    parser.add_argument("--cpu-offload-gb", type=float, default=8)
    parser.add_argument("--prefill-backend", choices=["auto", "triton"], default="auto")
    args = parser.parse_args()
    if min(args.input_lengths + args.batches + [args.tp]) < 1:
        parser.error("dimensions must be positive")
    if args.output_length < 2:
        parser.error("two generated tokens are needed to observe decode")
    config_path = args.model / "config.json"
    config = json.loads(config_path.read_text())
    if config.get("architectures") != ["Glm5NextForConditionalGeneration"]:
        parser.error("expected the original GLM-5.3-Flash configuration")
    for name in ("tokenizer.json", "tokenizer_config.json", "processor_config.json"):
        if not (args.model / name).is_file():
            raise FileNotFoundError(args.model / name)
    os.environ.update(
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        VLLM_PLUGINS="",
        VLLM_NO_USAGE_STATS="1",
        DO_NOT_TRACK="1",
    )
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    from vllm import LLM, SamplingParams

    root = Path(__file__).resolve().parents[1]
    result = {
        "status": "initializing",
        "purpose": "operator_shape_capture_only",
        "provenance": "native_full_architecture_execution_with_dummy_weights",
        "trained_weights_loaded": False,
        "architecture_overrides": None,
        "model_config_sha256": _digest(config_path),
        "commit": _source_revision(root),
        "collector_sha256": _digest(__file__),
        "environment": {
            name: importlib.metadata.version(name)
            for name in ("vllm", "torch", "triton")
        },
        "workload": {
            "tp": args.tp,
            "input_lengths": args.input_lengths,
            "batches": args.batches,
            "output_length": args.output_length,
            "max_num_batched_tokens": 2048,
            "prefill_backend": args.prefill_backend,
            "cpu_offload_gb": args.cpu_offload_gb,
            "enforce_eager": True,
        },
        "cases": [],
    }
    _write(args.output, result)
    try:
        llm = LLM(
            model=str(args.model),
            load_format="dummy",
            tensor_parallel_size=args.tp,
            enforce_eager=True,
            enable_prefix_caching=False,
            limit_mm_per_prompt={"image": 0, "video": 0},
            seed=42,
            max_model_len=max(args.input_lengths) + args.output_length + 16,
            max_num_seqs=max(args.batches),
            max_num_batched_tokens=2048,
            gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=args.cpu_offload_gb,
            additional_config={"kda_prefill_backend": args.prefill_backend},
            worker_extension_cls="tools.fused_kda_gate_model.KdaShapeWorkerExtension",
        )
        params = SamplingParams(
            temperature=0,
            max_tokens=args.output_length,
            min_tokens=args.output_length,
            ignore_eos=True,
            seed=42,
        )
        for length, batch in itertools.product(args.input_lengths, args.batches):
            llm.collective_rpc("kda_shape_begin")
            llm.generate(
                _prompts(llm.get_tokenizer(), length, batch), params, use_tqdm=False
            )
            workers = llm.collective_rpc("kda_shape_report")
            if not all(worker["shapes"] for worker in workers):
                raise RuntimeError("at least one rank did not observe any gate shapes")
            result["cases"].append(
                dict(input_length=length, batch=batch, workers=workers)
            )
            _write(args.output, result)
        phases = {
            row["phase"]
            for case in result["cases"]
            for worker in case["workers"]
            for row in worker["shapes"]
        }
        if not {"prefill", "decode"} <= phases:
            raise RuntimeError(f"prefill/decode capture incomplete: {phases}")
        result["status"] = "shapes_collected"
    except Exception as error:
        result.update(status="capture_failed", error=str(error))
        raise
    finally:
        _write(args.output, result)


if __name__ == "__main__":
    main()
