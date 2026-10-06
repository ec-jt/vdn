// SPDX-License-Identifier: Apache-2.0
// PyTorch binding for the modified SGLang VDN factor kernel.
// See licenses/SGLang-NOTICE.txt for the pinned source and modifications.
#include <torch/extension.h>

#include <vector>

std::vector<at::Tensor> vdn_delta_factors_cuda(
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& alpha);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("delta_factors", &vdn_delta_factors_cuda,
        "VDN FP32 factors: diag(alpha) inv(I+A), B inv(I+A)");
}
