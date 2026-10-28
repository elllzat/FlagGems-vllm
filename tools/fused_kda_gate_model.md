# fused_kda_gate: GLM-5.3-Flash model validation

This supplements the operator benchmark with real checkpoint execution. It follows
the shape-dumping and no-CUDA-Graph requirements in the
[testing document](https://jwolpxeehx.feishu.cn/docx/Ox7LdmmtGoUUxmxuYNCcx5ynnJd).
It does not turn the standalone operator speedup into a model speedup.

## Project facts (frozen before implementation)

| Item | Value |
| --- | --- |
| Editable | This document and `tools/fused_kda_gate_model.py` |
| Read-only | Operator, tests, benchmark baseline, installed vLLM sources, checkpoint |
| Model | Official `zai-org/GLM-5.3-Flash` FP8 checkpoint |
| Model revision | `46a050bec13002ad29ee795c16e61cf4fa3c4a9c` |
| Runtime | vLLM 0.32.0+cu129, torch 2.13.0+cu129, Triton 3.7.1; isolated environment |
| Version exception | Document's vLLM 0.24.0 has no `glm5next` implementation |
| Build | No production kernel change; Python compile and pre-commit |
| Validation | Actual gate inputs versus native gate; greedy generated token IDs versus native model |
| Workloads | Input lengths 128/1024, batches 1/4, output length 32; prefill and decode |
| Timing | Whole `LLM.generate` call, seconds; throughput tokens/s; request TTFT/TPOT when exposed |
| Aggregation | Three separate runs per workload; median within workload; no arithmetic mean of model speedups |
| Timeouts | Download resumable; model initialization/run bounded by external job timeout |
| Profiler | Separate diagnostic run only; excluded from timings |
| Autotune | Existing operator configs, unchanged; trace/warmup before timings |
| Fallback | No production torch fallback; test adapters use existing native KDA routines |

## Contract

| Item | Value |
| --- | --- |
| Gate contract | Existing fused_kda_gate API; FP32 output, bounded sigmoid for this model |
| Actual dimensions | Checkpoint has 64 KDA heads of dimension 128; TP4 has 16 local heads |
| Inputs | Read actual projection/parameter tensors from real model inference, no random-weight substitute |
| Layout | A no-copy view flattens the head dimensions; unsupported layout fails |
| Correctness | Gate output rtol/atol 3e-6; deterministic generation compared exactly |
| Autograd | Inference only |
| Torch use | Model execution and correctness checks only; operator production path unchanged |

The isolated environment uses FlagGems 5.0.0 and NumPy 2.3.5. FlagGems 5.0.2
pins NumPy 1.26.4, conflicting with vLLM 0.32's OpenCV/ML-dtypes requirements.
FlagGems 5.0.0 permits the compatible NumPy version; `uv pip check` passes.
The existing operator tests pass 131/131 on this environment, including the
two-device guard. Adapter smoke passes FP16/BF16 prefill and decode with nonzero
state. Those checks do not replace real-checkpoint model acceptance.

## Execution paths

| Path | Behavior | Meaning |
| --- | --- | --- |
| `native --prefill-backend auto` | Original FlashKDA/Triton prefill and recurrent decode | Default native model baseline |
| `native --prefill-backend triton` | Original gate+cumsum prefill and recurrent decode | Controlled Triton baseline |
| `gems-split --prefill-backend triton` | Gems gate then existing chunk/recurrent KDA with gate already computed | Explicit integration experiment; removes native gate fusion and may regress |
| Trace pass | Counts actual calls, dumps shapes/scalars, validates gates | Excluded from performance timings |
| Timed pass | Original functions for native; adapters for gems; tracing/checks disabled | Complete model call measurement |

In vLLM 0.32.0 the standalone native `fused_kda_gate` has **zero direct model
call sites**. Prefill fuses gate with cumsum (or FlashKDA); decode computes it
inside the recurrent kernel. A simple monkeypatch of the standalone gate does
not integrate it. The `gems-split` experiment explicitly changes that topology;
it must be compared with the native fusion and cannot be described as an
equivalent replacement of an existing standalone call.

Run the following from the repository root after pulling the locally committed
code. Do not edit installed vLLM or the baseline sources.

```bash
export PYTHONPATH=src:.
export CUDA_HOME=/usr/local/cuda-12.9
export PATH=/opt/kda-model-env/bin:$CUDA_HOME/bin:$PATH
export MAX_JOBS=4
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
python tools/fused_kda_gate_model.py --model /models/GLM-5.3-Flash \
  --mode native --prefill-backend auto --output /results/native-auto.json
python tools/fused_kda_gate_model.py --model /models/GLM-5.3-Flash \
  --mode native --prefill-backend triton --output /results/native-triton.json
python tools/fused_kda_gate_model.py --model /models/GLM-5.3-Flash \
  --mode gems-split --prefill-backend triton --output /results/gems-triton.json \
  --reference /results/native-triton.json
```

All three use `enforce_eager=True`, disable prefix caching and speculation, and
use text prompts with image/video limits set to zero, loading the complete
checkpoint. The default native baseline must be reported
alongside the controlled Triton comparison. Results contain environment/source
hashes, worker traces, generated token IDs, and per-workload timings. A nonzero
candidate call count and successful real-weight inference are required before
claiming model integration. Missing weights, zero candidate calls, or a failed
comparison leave model acceptance incomplete.

Each real-model result also produces `<output>.shapes.yaml`, accepted by the
existing operator benchmark's `--shape_file` option. This collects **observed**
T/H/D/safe_gate combinations from all ranks and both phases; it does not infer
model shapes from the standalone benchmark matrix. Run the operator benchmark
with the observed dtype(s) and `--mode operator`, separately from model timings.
The trace pass also measures the actual native standalone interface and Gems on
the captured real tensors: 100 ms warmup, 500 ms measurement, three paired rounds,
alternating order, wall time plus synchronization, no CUDA Graph. These fields
are explicitly standalone timings; they do not measure native fused KDA latency.
They run concurrently on TP workers and must be distinguished from the previous
single-GPU standalone matrix.

`--wait-for-checkpoint 21600` allows a currently running test job to wait for the
resumable weight download. It only waits on missing checkpoint files, writes
`waiting_for_checkpoint`, and cannot report a model result until real inference
completes.

Before loading the model, validate both adapters on the installed native KDA
implementation (small synthetic tensors, **not** a real-model acceptance result):

```bash
PYTHONPATH=src:. python tools/fused_kda_gate_model.py --adapter-smoke-only \
  --output /results/adapter-smoke.json
```
