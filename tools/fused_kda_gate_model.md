# fused_kda_gate shape collection

The model run is only a source of operator shapes, as clarified by the user.
There is no model A/B test, model speedup acceptance, gate replacement, or
splitting of native fused KDA. This follows the shape collection section of the
[testing document](https://jwolpxeehx.feishu.cn/docx/Ox7LdmmtGoUUxmxuYNCcx5ynnJd).

## Project facts

| Item | Contract |
| --- | --- |
| Editable | This document and tools/fused_kda_gate_model.py |
| Read-only | Operator math/dispatch, native vLLM code, references, tests, autotune configs |
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

Use the exported gate-shapes.shapes.yaml with the existing benchmark's
--shape_file option, selecting observed dtypes and --mode operator. Operator
accuracy and speedup are separate from this metadata collection.

The previously tested Triton 3.7 compiled-launch compatibility improvement remains
in the operator: both Triton 3.6.0 and 3.7.1 passed 131 accuracy tests. Its synthetic
prechecks are not presented as model-derived measurements.
