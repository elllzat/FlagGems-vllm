// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 FlagOS Contributors
// Registration only: the adapter, ACLNN implementation and kernel are upstream.
#include <torch/extension.h>
#include <torch/library.h>
#include <acl/acl.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include <torch_npu/csrc/framework/utils/OpPreparation.h>
#include "aclnn_torch_adapter/op_api_common.h"
#include "attention/kda_gate_cumsum_vllm/kda_gate_cumsum_vllm_torch_adpt.h"

TORCH_LIBRARY_FRAGMENT(_C_ascend, m) {
    m.def("kda_gate_cumsum_vllm(Tensor g, int chunk_size, *, Tensor? A_log=None, "
          "Tensor? dt_bias=None, int[]? cu_seqlens=None, bool? use_gate_in_kernel=False, "
          "bool? safe_gate=False, float? lower_bound=-5.0, str layout=\"BSND\") -> Tensor");
}

TORCH_LIBRARY_IMPL(_C_ascend, PrivateUse1, m) {
    m.impl("kda_gate_cumsum_vllm", &vllm_ascend::kda_gate_cumsum_vllm);
}
