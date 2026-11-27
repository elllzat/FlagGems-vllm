# Portable KDA gate and chunk-local cumsum

## Source and scope

This development branch imports the handoff from H800 container `vllm_nyp`,
`/workspace/FlagGems-vllm`, local branch
`q4/fused-kda-gate-chunk-cumsum` (base `601514f66857e33848acfc3dacfafef403cac675`).
The handoff contains uncommitted production files and experimental
`native_scan.py` / `native_scan_cg.py`; these two experimental sources are kept
byte-for-byte as `.py.txt` snapshots, and are not production dependencies.

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

HCU and MetaX keep an additional Triton matrix-prefix kernel for intermediate launches.
This is a candidate implementation, not a change to the frozen baseline.
The host selects the algorithm first; each path separately tunes BD/warps.
The dispatch compares logical head/chunk groups to a conservative coverage
estimate: CU count * maximum resident threads / (4 * reference group warps *
warp size). The factor four is a calibrated 25% coverage heuristic, not a
physical limit or proof of achieved occupancy. It uses actual device metadata
instead of a model-specific T threshold. Measured properties are 80 CUs,
2560 threads/CU and 64-thread waves on BW1000, and 104 CUs, 2048 threads/CU and
64-thread waves on C550. The original configuration family's two-warp group
is the reference unit, so the estimates remain 400 and 416 full-head chunk
groups even when a vendor tuner adds one-warp scan candidates. All 38 contracts
are used to validate the crossover; neither slow cases nor baseline candidates
are removed.
Other backends keep the scan path. Automatic-index rectangular grids use the
matrix path on HCU/MetaX for small chunks because idle programs would otherwise
inflate that coverage estimate.

Chunk=128 always uses scan. The original matrix prefix can exceed device
shared-memory limits: C550 reports 81920 bytes required versus 65536 available.
The full test checks our result against independent Torch math before marking
only the original-reference comparison as expected-resource-failure. No such
exception applies to the 38 actual contracts, which all use chunk=64.

## Selected Ascend reference, 2026-11-27

For the future Ascend implementation, the selected reference is the handoff's
native vLLM-Ascend AscendC `torch.ops._C_ascend.kda_gate_cumsum_vllm`, pinned to
`4f869f9a8d3257e243fdd8fd9d998231f069ffe8`. Source paths and Git blob identities
are recorded in [`ascend_source.json`](ascend_source.json). The installed
extension's provenance must be checked before claiming measurements against
that exact revision; the operator name alone does not identify a binary build.

Use the existing 38 captured safe-gate contracts, not the old standalone
softplus workload. Both sides must compute the gate, chunk-local inclusive
cumsum and 1/ln(2) scaling. The reference accepts our token-major input as
`layout="BSND"` and returns FP32 in the same shape. A_log is flattened to [H]
before capture, and sequence boundaries are supplied as a prebuilt host tuple.
No device-to-host metadata conversion belongs in the captured/timed call.

```python
output = torch.ops._C_ascend.kda_gate_cumsum_vllm(
    g, 64, A_log=A_log_flat, dt_bias=g_bias,
    cu_seqlens=cu_seqlens_host, use_gate_in_kernel=True,
    safe_gate=True, lower_bound=-5.0, layout="BSND",
)
```

This selects the baseline only. The portable production implementation above
still excludes Ascend, and no new Ascend performance result is claimed here.
The 2026-11-27 preflight ran inside the personal `flagtree-dev-elzat` container
on 910B4: NPU and NPUGraph are available, but the probed Python environment has
no vLLM-Ascend package and does not register the selected operator. Its native
extension and binary provenance must therefore be prepared before benchmarking.

## Frozen validation and measurement

Base: upstream main `248d413c`. Operator, exports, metadata, NVIDIA tuning YAML,
tests and benchmark use the standard repository drop points. The default YAML
supplies BD=32/64 x num_warps=2/4/8, num_stages=3. HCU scan and MACA short
safe-gate paths have separate eight-candidate vendor entries; their matrix
paths retain the default six candidates. Other backends use the configuration
loader's normal default-backend mechanism. `libtuner` uses
device Graph timing and the handoff workload keys, including shape and index
metadata. Unsupported backend configuration failures must be reported, not
replaced silently with another baseline or an eager timing method.

Run all commands in the user's own vendor container after a Git pull:

```bash
PYTHONPATH=src:tools:${PYTHONPATH:-} python -m pytest -q tests/test_fused_kda_gate_chunk_cumsum.py --quick
PYTHONPATH=src:tools:${PYTHONPATH:-} python tools/fused_kda_gate_chunk_cumsum/run_portable.py --output /workspace/chunk-cumsum-results/performance.json
PYTHONPATH=src:tools:${PYTHONPATH:-} python -m pytest -q benchmark/test_fused_kda_gate_chunk_cumsum.py -k True --mode cudagraph --level core --iter 1 --warmup 1
```

The runner checks candidate and reference against independent Torch math before
capturing each Graph, and checks captured output again after replay. Compiling
and tuning finish outside capture. Five paired rounds alternate order, share
inputs and prebuilt packed indices, and measure device events per Graph replay.
Each shape gets a fresh original baseline tuner to avoid an earlier shape
choosing its configuration. Report median paired speedup per case, geometric
mean and minimum across the fixed active set. Never exclude slow shapes.
Each final Graph contains 256 calls, with three replay warmups and seven event
samples per latency. Earlier 64-call trials are retained separately. Candidate
configuration reporting records the actual LibEntry launch, including cache
hits; reading the tuner's last best_config alone can describe a previous case.
The default capture loop discards each intermediate return, as the installed
`triton.testing.do_bench_cudagraph` does, and retains only the final output for
replay validation. Both implementations use this same lifetime and separate
private Graph pools. The earlier loop kept the previous output alive during
the next allocation, usually alternating two buffers. A pool diagnostic showed
that output addresses affected the MACA T=255/256 timings. This establishes
address sensitivity, not a specific cache/bank cause. The corrected loop is a
measurement fix, not a kernel optimization. `--legacy-retain-output` reproduces
that historical diagnostic; do not mix its numbers with the standard results.
Acceptance follows workflow.md (no key shape below 0.9x); the handoff's 1.1x
optimization goal is reported separately in the results below.

`--isolated` is an explicit diagnostic option for unrelated whole-package import
failures. It loads the actual runtime/utilities/operator from the same checkout,
marks public-package integration unverified and deselects its integration test.
It must not be reported as a full public-package pass.

## Follow-up schedules

The follow-up keeps the original 38 cases, pinned reference and five-round
256-call Graph method with the output-lifetime correction above. All arithmetic
remains FP32 before the requested output
conversion. It adds these operator-local schedules:

1. **Head-first program order.** The launch dimensions are unchanged. A bijective
   permutation makes adjacent programs visit adjacent heads before moving to
   another token chunk. This brings neighboring programs closer in the BTHD
   input layout; it does not combine prefix sums across heads or sequences.
2. **Bounded 32-bit token offsets on HCU/MACA.** Common shapes use 32-bit token
   arithmetic when the tensor plus a padded chunk fits the signed range. Larger
   token offsets keep the original 64-bit route. The head/chunk permutation has
   a separate 2^31-program guard, including idle automatic-index combinations.
   No host pointer-specialization cache or compiler-wide switch is added.
3. **Short physical token tiles.** For T <= chunk_size/2, the physical row count
   becomes next_power_of_2(T); the logical chunk size, sequence offsets and
   reset boundaries stay unchanged. For captured T=1/2/4 this avoids computing
   64 gate rows and a full triangular matrix. One representative omitted padded
   gate row preserves the original dot's 0*NaN/Inf behavior. Six new tests cover
   this distinction, including infinite A and zero bias.
4. **HCU scan configurations.** The scan has its own vendor entry: BD=32/64/128
   with 1/2/4 waves where applicable (eight candidates), one pipeline stage.
   A native wave contains 64 threads. For the captured chunk=64 cases, the
   largest candidate has 64*128 FP32 elements (32 KiB of values) spread across
   four waves, or 32 values per
   thread before other live data. This constrains the register working set;
   one-wave alternatives remove inter-wave scan communication on small tiles.
   There is no staged loop requiring a multi-stage memory pipeline. The matrix
   path keeps its separate original six candidates.
5. **MACA short safe-gate layouts.** When Gluon is available, the short safe
   path explicitly assigns [LANE_T, 64/LANE_T] lanes to token/channel axes and
   [1, num_warps] waves to those axes. LANE_T=16/32 keeps each channel's scan
   inside one wave plus that thread's registers; independent waves own channel
   groups. Eight BD/lane/wave candidates balance contiguous channel access,
   register work and independent groups. Sample compiled kernels report zero
   shared memory and zero CTA-barrier calls. Longer inputs keep standard
   Triton matrix/scan paths because this layout regressed in the initial probe.

The installed HCU compiler exposes Gluon but its layout inference raises a
missing-HCUMFMALayout error. HCU therefore uses standard Triton; this is a
compiler-package limitation, not a hardware statement. No shared compiler
package is patched. NVIDIA/PPU retain their standard scan route and tuning
family; Gluon decoration and the new vendor-only configuration entries are
not used on those backends.

`probe_graph_pool.py` is a separate allocation-address diagnostic. It uses
unchanged public operators and their normal tuners, records output addresses,
and compares separate/shared pools with a common capture stream. Its results
must not replace the fixed acceptance method or be mixed into the aggregate.

## Optimized HCU/MACA validation, 2026-11-24

[`validation/optimized_2026-11-24.json`](validation/optimized_2026-11-24.json)
preserves all 38 cases and five paired rounds from both final reports, plus
configuration/source identities, full functional summaries and the separate
pool diagnostic. The tested production source SHA256 is
`6d8208de00a3f0be7a778b99f1c54fd352ee5c9fa3e39029244c88c54b863bd2`.
The current operator, vendor YAMLs and tests have the same Git blobs as the
full-functional snapshot `adedc64`. HCU performance was measured at `342fd5c`
and the final MACA report at `14d05be`; the HCU configuration is unchanged
between those revisions.

| Platform | Geometric mean | Minimum | Cases >= 1.1x | Full functional validation |
| --- | ---: | ---: | ---: | --- |
| HCU BW1000 | 1.180x | 1.001x | 20/38 | 216 passed |
| MetaX C550 | 1.178x | 0.996x | 22/38 | 204 passed, 12 baseline resource xfails |

Both meet the **1.1x geometric-mean** goal and the 0.9x per-case workflow floor.
This does not claim every individual case reaches 1.1x. Both use their public
package exports in the user's own containers. Each also passed the standard
repository CUDA Graph benchmark smoke (one test covering all 38 safe-gate
contracts; the other gate-mode parameter is deselected). Baseline source, its
six candidates and original key are unchanged. No captured case is omitted.
NVIDIA/PPU have no new performance run in this follow-up; their old results
remain historical below. The initial reports used the previous output lifetime,
so their timings are not a controlled before/after comparison to this table.

Two extra MACA experiments were removed. An output `evict_first` hint produced
identical normalized LLVM instruction streams for three inspected kernel pairs
with the installed backend; it is not credited with a performance gain.
Adding BD=128 to the base matrix/scan family selected no new configuration in
38 cases, so those two candidates were removed as well. HCU scan and MACA
short Gluon entries still retain their own BD=128 candidates. Applying the
short Gluon layout to longer inputs also regressed and was not retained.

## Initial portable validation, 2026-11-24 (13d176f)

The initial five-round, 256-call Graph reports are preserved in
[`validation/graph_2026-11-24.json`](validation/graph_2026-11-24.json), including
every case, round, actual launch configuration, revision and source hashes.
All four reports measure the same production source SHA256
`4a7c0d94b50c441d96ab69dd20e6538add5e52d38aab1c6ee751bb72ec29ddd0`.
The designated baseline is the frozen vLLM Triton source compiled on each
backend; these numbers do not compare against a separate vendor-native kernel.

| Platform | Geometric mean | Minimum | Cases >= 1.1x | Functional validation |
| --- | ---: | ---: | ---: | --- |
| NVIDIA H800 | 1.436x | 1.133x | 38/38 | 209 passed |
| HCU BW1000 | 1.084x | 0.999x | 16/38 | 209 passed |
| MetaX C550 | 1.106x | 0.962x | 17/38 | 197 passed, 12 baseline resource xfails |
| PPU (isolated operator) | 1.613x | 1.305x | 38/38 | 208 passed, public integration deselected; final quick 120 passed |

All 38 actual cases on each measured backend meet the workflow's 0.9x minimum.
At that snapshot HCU had not reached the 1.1x geometric-mean goal. PPU numbers
retain the isolated-operator qualification described below. Functional runs
cover both gate modes, multiple float/output/index types, packed/dense inputs,
tails, empty inputs, unsupported metadata and nonfinite g/A/bias. The C550
chunk128 xfails concern the unchanged reference after our output matches
independent Torch math; they do not exclude any actual case.

An initial C550 64-call trial reported 0.856x for T=256 with one sequence.
Fixed-configuration checks put both kernels around 14 us; the final uniform
256-call five-round run records 1.208x for that case. This was not reproduced
as a stable algorithm regression. The specific hardware cause remains
unconfirmed; the earlier result is not treated as disproven or silently omitted.
Do not use the expanded-BD trials or the earlier stale configuration fields as
the final result. Final performance revisions are recorded in the JSON;
subsequent documentation/benchmark-input fixes do not change production code.
The standard repository benchmark also passed its safe-gate integration smoke
on H800 (one parametrized test covering all 38 contracts). Its short framework
timing is a separate integration check; the table above uses only the recorded
five-round paired runner, not a mixture of benchmark methods.

## Container environment notes

- H800: `vllm_elzat`, `/workspace/FlagGems-vllm-chunk-cumsum`, image Python,
  CUDA_VISIBLE_DEVICES=3 for the recorded runs.
- BW1000: `vllm_elzat`, same repository path, `/flagos/bin/python`.
  A login shell drops `/flagos/bin` from PATH, so prepend it before the DTK
  environment setup. Preserve the image's PYTHONPATH=/opt/flagtree when
  prepending this checkout. Set TRITON_HIP_CLANG_PATH as in the login document.
- C550: `flagtree-dev-elzat`, same repository path,
  `/opt/conda/envs/flagtree36/bin/python`. The default conda interpreter has
  old Triton 3.0, which fails in the unchanged reference autotuner. The
  existing flagtree36 environment supplies Triton 3.6. MACA CPU-to-device
  conversion defaults to channels-last; the test factory explicitly calls
  contiguous so both implementations receive the captured packed layout.
- PPU: the login document's PAI DSW container and personal `/root/elzat/venv`,
  repository `/root/elzat/FlagGems-vllm-chunk-cumsum`. Full-package import hits
  an unrelated new w8a8 TLE dependency unavailable in the installed compiler.
  The explicit isolated path is used and public-package integration remains
  unverified.
- MThreads: JumpServer currently returns no asset for 10.121.38.73; no run is
  claimed for that platform. Other unlisted hardware has no new measurement.

Two initial candidate-only BD=128 experiments were not the default config.
The first is invalid for assessing the expanded search because it reused the
six-config persistent winner. The second resets the candidate cache identity
and uses a dedicated cache directory: larger scan tiles help some large HCU
cases but do not fix the short-launch regressions. The host algorithm split
is therefore a separate subsequent change. The six baseline candidates and
their original tuning key remain unchanged throughout.
