# fused_kda_gate shape collection

The model run is only a source of operator shapes, as clarified by the user.
There is no model A/B test, model speedup acceptance, gate replacement, or
splitting of native fused KDA. This follows the shape collection section of the
[testing document](https://jwolpxeehx.feishu.cn/docx/Ox7LdmmtGoUUxmxuYNCcx5ynnJd).

## Project facts

| Item | Contract |
| --- | --- |
| Editable | Collector, this document, captured-contract tests, and host launch-cache removal |
| Read-only | GPU kernel math, native vLLM code, reference implementations, benchmark timing, autotune configs |
| Model | Original GLM-5.3-Flash configuration; TP and scheduler settings recorded |
| Weights | Native vLLM dummy loader; no trained checkpoint weights or downloads |
| Runtime | vLLM 0.32.0+cu129 / torch 2.13.0+cu129 / Triton 3.7.1 |
| Version exception | The testing document's vLLM 0.24.0 has no glm5next implementation |
| Validation | Successful native forward; all ranks observe prefill and decode gate metadata |
| Performance | None during collection; later standalone operator benchmark uses captured shapes |
| Source policy | Local edits, Git push, then server git pull |

## Shape contract

| Field | Recorded information |
| --- | --- |
| Gate tensors | Raw gate logits, corresponding standalone API view, A, bias |
| Metadata | Shape, stride, dtype, T/H/D, safe_gate, lower_bound, beta, threshold |
| Context | Prefill/decode, native fused path, TP rank, input length, batch, source hashes |
| Provenance | Actual full-architecture execution using dummy weights; not trained-model inference |
| Outputs | Detailed JSON and deduplicated benchmark-compatible .shapes.yaml |
| Coverage | Only the recorded workloads; no claim of all possible sequence/batch/TP sizes |

## Collection paths

| Native path | Observation |
| --- | --- |
| FlashKDA prefill | Record the gate logits and layer gate parameters |
| Triton gate+cumsum prefill | Record raw_g, A_log and bias |
| Recurrent decode | Record g, a_log and bias when compute_gate is enabled |
| Direct standalone gate, if called | Record its actual input contract separately |

In vLLM 0.32.0 the gate is normally embedded in fused KDA. These rows are labelled
gate_inside_fused_kda, not direct standalone calls. The collector only reads
metadata, then calls the original function with the original arguments once.
It does not import Gems, compute an extra gate, copy tensors, or change fusion.

The full architecture is instantiated without hf_overrides and uses the native
dummy loader. This avoids checkpoint downloads while obtaining executed tensor
dimensions and scheduler shapes. Numeric outputs, accuracy, and model performance
are not evaluated. The result explicitly records trained_weights_loaded=false.

Run from the repository root inside the prepared container:

~~~bash
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/opt/kda-model-env/bin:$CUDA_HOME/bin:$PATH
export PYTHONPATH=src:.
export MAX_JOBS=4
python tools/fused_kda_gate_model.py --model /models/GLM-5.3-Flash \
  --tp 4 --input-lengths 128 1024 --batches 1 4 --output-length 2 \
  --output /workspace/fused_kda_gate-work/gate-shapes.json
~~~

Collection is offline, disables CUDA Graph and prefix caching, and uses text
inputs. Only the small configuration/tokenizer files are needed on disk. Dummy
weights still occupy GPU memory; the collector does not shrink the model.
The default CPU offload allowance is 8 GiB per GPU to fit the full architecture
on four H800s. It is recorded in the workload and does not change gate dimensions.

Use the detailed JSON for standalone validation and benchmarking to preserve
observed tensor ranks, strides, parameter dtypes and scalar values. Duplicate
contracts across layers and ranks are tested once. Inputs are newly generated
at the captured contracts, not activation values saved from the model.

~~~bash
export FUSED_KDA_GATE_SHAPES=/workspace/fused_kda_gate-work/gate-shapes.json
python -m pytest -q tests/test_fused_kda_gate.py -k captured_model_shapes --ref cuda
python -m pytest -q -s benchmark/test_fused_kda_gate.py \
  --shape_file "$FUSED_KDA_GATE_SHAPES" --dtypes bfloat16 --mode cudagraph \
  --iter 500 --record json \
  --output /workspace/fused_kda_gate-work/cudagraph-20261102/round1.json
~~~

Select the dtype actually recorded in the JSON. The normalized
gate-shapes.shapes.yaml is also available for the benchmark's original T/H/D
matrix input mode. The captured-contract accuracy test checks both eager and
CUDA Graph replay, including changed input values after capture.

The custom metadata/compiled-launch cache has been removed at the user's request.
The portable Triton kernels use the runtime device guard and math shim.
FP64/BF16 follow backend capabilities. Ascend uses fixed vector-core tiles
selected from measurements on the captured contracts, with a flat path for
unaligned widths. CPU is unsupported.
Other vendors inherit the portable `fused_kda_gate` / `fused_kda_gate_tiled`
candidates through ConfigLoader unless their own YAML overrides those names.
NVIDIA uses the wider `fused_kda_gate_nv` / `fused_kda_gate_tiled_nv` candidates
with `use_cuda_graph=True`, so tuning and acceptance both time Graph replay.
The retained NVIDIA space has 14 flat and 10 tiled candidates, with 4/8 warps;
the wider exploratory sweep found no winning 1/2-warp candidates on the eight
captured contracts. Flat tiles larger than the entire input (rounded to a power
of two, with a 128-element minimum) are pruned before compilation and timing.
Autotune uses 20 ms Graphs; the diagnostic defaults to 50 ms and acceptance uses
the existing benchmark's 500 ms setting.
Other vendors use normal device timing during tuning; CUDA Graph support is not
assumed. Changing the candidate set invalidates the old eager-tuned config hash.
For reproducible experiments, also use a fresh `FLAGGEMS_DB_URL` database.

The pinned vLLM baseline retains its NVIDIA-specific autotune space, including
32-warps candidates that some vendor compilers reject. Its source is unchanged.
Native-reference comparisons and this performance benchmark therefore run only
on NVIDIA. Other vendors validate against the independent Torch formula, including
both gate modes, optional bias, scalar parameters and layout/dtype boundaries.
No cross-vendor speedup claim is made against an incompatible baseline.

All paths use the standard LibEntry/Triton launcher. The experiment uses the existing
benchmark runner's cudagraph mode for both the native reference and Gems.
It measures replay latency, with compilation and tuning completed before capture.
Keep these results separate from the historical eager full-call measurements
(--mode operator). No model latency or speedup is measured, and shape collection
itself continues to use eager execution so its Python observation hooks can run.

For a fixed-configuration CUDA Graph comparison of both paths on NVIDIA:

~~~bash
PYTHONPATH=src:. python tools/fused_kda_gate_tune.py \
  --shapes "$FUSED_KDA_GATE_SHAPES" --output sweep.json --rounds 3 --rep 50
~~~

The diagnostic checks every candidate against the Torch formula, then measures
each already compiled kernel in randomized order. It does not change the host
dispatch or replace the formal operator benchmark.

### Portable latency measurements

`benchmark/fused_kda_gate_captured_contracts.json` contains the eight deduplicated
contracts from the original shape capture, including tensor ranks, strides,
dtypes and scalar arguments. Its provenance is full-architecture execution with
dummy weights, and it records the original capture SHA256.

Run all GPU commands inside the platform's container, with `PYTHONPATH=src:.`:

~~~bash
python tools/fused_kda_gate_perf.py --probe --rep 20 --warmup 20 \
  --modes cudagraph kernel --output probe.json
python tools/fused_kda_gate_perf.py --rounds 3 --rep 500 --warmup 100 \
  --modes cudagraph kernel --output latency.json
~~~

Only request modes that pass the probe. The Graph probe also verifies replay
after changing the input. Both timers call the same Triton helpers as the common
benchmark runner. Graph timing is invoked from an explicit side stream on all
platforms because some vendor helpers require a caller-owned non-default stream.
All inputs are checked against an independent Torch formula
and compiled/tuned before timing. Per-shape results retain each raw round,
selected configuration and environment. Kernels and their host dispatch are not
modified by this utility.

Thead may require `--isolated` while its unrelated all-operator imports depend on
an unavailable TLE module. This loads the actual runtime, LibEntry, configuration
and gate source, but does not verify the normal public-package import.

This utility reports implementation latency only. NVIDIA speedup remains measured
with the unchanged dedicated vLLM benchmark. Do not substitute a Torch baseline
or change NVIDIA's reference configs to manufacture a cross-vendor speedup.

### Strict pinned-reference comparison on a vendor device

The installed vendor vLLM may be older than the pinned GLM5Next reference and lack
the safe-gate interface. To check the fixed reference itself without unrelated
test imports, use:

~~~bash
python tools/fused_kda_gate_speedup.py --probe --rep 20 --output strict-probe.json
python tools/fused_kda_gate_speedup.py --rounds 3 --rep 500 --output speedup.json
~~~

Thead may require `--isolated`. The utility loads the two original reference
definitions and their two config-list assignments as unchanged AST nodes from
`tests/test_fused_kda_gate.py`, retaining the source filename and line numbers.
It preserves all 24 original configs (including 32 warps), decorators, equations
and autotune keys. It does not filter failing configs or replace the reference.
Compilation failures are saved with their stage and traceback, without claiming
a speedup. Only run the full comparison after a successful strict probe.

On Ascend, the comparison uses the same `do_bench_npu` profiler as the
repository benchmark instead of CUDA Graph. To measure the Ascend implementation
alone, use `fused_kda_gate_perf.py --modes npu`.
