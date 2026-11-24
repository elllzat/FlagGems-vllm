# Portable KDA gate and chunk-local cumsum

## Source and scope

This development branch imports the handoff from H800 container `vllm_nyp`,
`/workspace/FlagGems-vllm`, local branch
`q4/fused-kda-gate-chunk-cumsum` (base `601514f66857e33848acfc3dacfafef403cac675`).
The handoff contains uncommitted production files and experimental
`native_scan.py` / `native_scan_cg.py`; these two experimental sources are kept
as received, and are not production dependencies.

`source.json` identifies the actual GLM5Next baseline, vLLM commit
`373614f66dc8dc678faaf33223256f814238b7cb`, source SHA256
`6f5865164682e2ea475995fbdeb1df6942baecfb06abeee3a4468c7c6344775e`.
`upstream.py` is the unchanged extraction (SHA256 in the manifest).
The public vLLM v0.32.0 tag mentioned in the handoff has a different signature
(`lower_bound` selects the gate); it is not substituted for the captured
GLM5Next `safe_gate` baseline. Only the test dependency binding in `support.py`
uses backend-neutral libdevice imports. All measurements keep the original
kernel, signature, tuning key and six BD/warp candidates.

`captured_shapes.json` retains all 38 captured contracts from
`tools/kda_shape_capture/results/coverage-20261106T055716Z-37e253cb/dataset/benchmark_cases.json`.
The original core_shapes.yaml contains no entry for this operator.
These are full-model dummy-weight runtime dimensions, TP=2, H=32, D=128,
BF16 input / FP32 output and safe gate, including ragged sequence boundaries.
Benchmark tensor values are synthetic; this is not a model accuracy test.
The JSON retains the original dataset hash and source metadata, excluding
unnecessary per-request identifiers and historical aggregation tables.

## Implementation contract

Inputs: contiguous `[B,T,H,D]`, A_log containing H elements, optional H*D bias.
FP16/BF16/FP32 inputs and output, FP32 intermediate computation, FP32 default
output; `output_dtype=None` follows the input. Gate formulas and parameters
match the frozen GLM5Next API. Inclusive sums reset at each sequence/chunk
boundary and are multiplied by RCP_LN2 after the sum. Chunk sizes: 32/64/128.
Forward inference only. Invalid metadata shapes/layouts/dtypes raise explicitly.
Metadata values are the caller's responsibility, with no host readback.

Dense and indexed packed inputs use the handoff grid. Packed inputs without
indices use a rectangular device grid, skipping excess programs, avoiding the
upstream Torch-based index-building helper. Empty inputs return without launch.
CPU and Ascend are explicitly excluded from this implementation. Other vendors
use the active runtime device and backend-neutral Triton operations.
The original CUDA-only libdevice dependency is removed from production.

The scan preserves the original dot's future NaN/Inf contamination by detecting
the last nonfinite gate row per channel, including padded rows. This closes the
nonfinite limitation documented in the handoff experimental scan.
Production permits only Torch allocation and metadata/no-copy views, with no
Torch compute fallback. Optional automatic-index processing remains on device.

## Frozen validation and measurement

Base: upstream main `248d413c`. Operator, exports, metadata, NVIDIA tuning YAML,
tests and benchmark use the standard repository drop points. The YAML supplies
BD=32/64 x num_warps=2/4/8, num_stages=3, reused through the configuration
loader's normal default-backend mechanism on other vendors. `libtuner` uses
device Graph timing and the handoff workload keys, including shape and index
metadata. Unsupported backend configuration failures must be reported, not
replaced silently with another baseline or an eager timing method.

Run all commands in the user's own vendor container after a Git pull:

```bash
PYTHONPATH=src:tools python -m pytest -q tests/test_fused_kda_gate_chunk_cumsum.py --quick
PYTHONPATH=src:tools python tools/fused_kda_gate_chunk_cumsum/run_portable.py --output /workspace/chunk-cumsum-results/performance.json
```

The runner checks candidate and reference against independent Torch math before
capturing each Graph, and checks captured output again after replay. Compiling
and tuning finish outside capture. Five paired rounds alternate order, share
inputs and prebuilt packed indices, and measure device events per Graph replay.
Each shape gets a fresh original baseline tuner to avoid an earlier shape
choosing its configuration. Report median paired speedup per case, geometric
mean and minimum across the fixed active set. Never exclude slow shapes.
Acceptance follows workflow.md (no key shape below 0.9x); the handoff's 1.1x
optimization goal is reported separately. No measured result is asserted here.

`--isolated` is an explicit diagnostic option for unrelated whole-package import
failures. It loads the actual runtime/utilities/operator from the same checkout,
marks public-package integration unverified and deselects its integration test.
It must not be reported as a full public-package pass.
