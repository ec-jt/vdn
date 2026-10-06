// SPDX-License-Identifier: Apache-2.0
// Modified SGLang PR #37903 host interface, 2026-10-05: replace TVM FFI with
// PyTorch allocation, validation, device/stream guards and checked CUDA launch.
// The device math lives in vdn_delta_factors.cuh. See licenses/SGLang-NOTICE.txt.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <vector>

#include "vdn_delta_factors.cuh"

namespace {

void check_tensor(const at::Tensor& t, const char* name) {
  TORCH_CHECK(t.is_cuda(), name, " must be CUDA");
  TORCH_CHECK(t.scalar_type() == at::kFloat, name, " must be float32");
  TORCH_CHECK(t.layout() == at::kStrided && t.is_contiguous(),
              name, " must be contiguous strided storage");
  TORCH_CHECK(!t.requires_grad(), name, " must not require gradients");
  TORCH_CHECK(!t.is_conj() && !t.is_neg(), name, " must not have lazy view bits");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr()) % 16 == 0,
              name, " must be 16-byte aligned");
}

void configure_kernel(int device, const cudaDeviceProp& props) {
  using namespace vdn_paper::vdn_delta_factors;
  // Unlike upstream's fixed bool[64], this is thread-safe and has no device-id cap.
  static std::mutex mutex;
  static std::unordered_set<int> configured;
  std::lock_guard<std::mutex> lock(mutex);
  if (configured.count(device)) return;
  cudaFuncAttributes attrs{};
  C10_CUDA_CHECK(cudaFuncGetAttributes(&attrs, vdn_delta_factors_kernel));
  TORCH_CHECK(props.sharedMemPerBlockOptin >= kXsBytes + attrs.sharedSizeBytes,
              "paper_delta_factors requires 64 KiB dynamic plus kernel static shared memory");
  TORCH_CHECK(props.maxThreadsPerBlock >= kBlockSize && attrs.maxThreadsPerBlock >= kBlockSize,
              "paper_delta_factors requires a 256-thread CUDA block");
  C10_CUDA_CHECK(cudaFuncSetAttribute(
      vdn_delta_factors_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kXsBytes));
  configured.insert(device);
}

}  // namespace

std::vector<at::Tensor> vdn_delta_factors_cuda(
    const at::Tensor& A, const at::Tensor& B, const at::Tensor& alpha) {
  using namespace vdn_paper::vdn_delta_factors;
  check_tensor(A, "A");
  check_tensor(B, "B");
  check_tensor(alpha, "alpha");
  TORCH_CHECK(A.dim() == 3 && A.size(1) == kDim && A.size(2) == kDim,
              "A must have flattened shape [N, 128, 128]");
  TORCH_CHECK(B.sizes() == A.sizes(), "B shape must equal A shape");
  TORCH_CHECK(alpha.dim() == 2 && alpha.size(0) == A.size(0) && alpha.size(1) == kDim,
              "alpha must have flattened shape [N, 128]");
  TORCH_CHECK(A.device() == B.device() && A.device() == alpha.device(),
              "all inputs must be on the same CUDA device");

  const c10::cuda::CUDAGuard guard(A.device());
  const auto* props = at::cuda::getDeviceProperties(A.get_device());
  TORCH_CHECK(props->major >= 8, "paper_delta_factors requires SM80 or newer");
  const int64_t num = A.size(0);
  TORCH_CHECK(num <= props->maxGridSize[0], "paper_delta_factors batch exceeds CUDA grid limit");
  auto transition = at::empty_like(A);
  auto injection = at::empty_like(B);
  if (num == 0) return {transition, injection};
  configure_kernel(A.get_device(), *props);
  // Allocation guarantees alignment/nonaliasing. Never launch on an implicit
  // default stream: callers may produce/consume statistics on VDA side streams.
  const auto stream = c10::cuda::getCurrentCUDAStream(A.get_device());
  vdn_delta_factors_kernel<<<static_cast<unsigned int>(num), kBlockSize, kXsBytes, stream.stream()>>>(
      A.data_ptr<float>(), B.data_ptr<float>(), alpha.data_ptr<float>(),
      transition.data_ptr<float>(), injection.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {transition, injection};
}
