// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <aten/cuda_tensor_view.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/empty_strided.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <c10/core/DefaultDtype.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/SmallVector.h>
#include <flagos.h>
#include <runtime/allocator/caching_device_allocator.h>
#include <torch/library.h>

#include <optional>

namespace flagos::nvidia {
namespace {

using torch::jit::Stack;
using at::native::flagos::FlagosToCudaView;

constexpr auto forward_name = "flash_attn_3::_flash_attn_forward";
constexpr auto raw_forward_name = "flash_attn_3::fwd";

bool MatchesVendorSchema(const c10::FunctionSchema& forward,
                         const c10::FunctionSchema& raw) {
  const auto& args = forward.arguments();
  const auto& vendor_args = raw.arguments();
  if (args.size() != 34 || args.size() != vendor_args.size() ||
      args[30].name() != "scheduler_metadata" ||
      forward.returns().size() != raw.returns().size())
    return false;

  for (size_t i = 0; i < args.size(); ++i) {
    auto name = args[i].name();
    if (name == "qv") name = "q_v";
    if (name == "out_") name = "out";
    if (name == "causal") name = "is_causal";
    if (name == "rotary_interleaved") name = "is_rotary_interleaved";
    if (name != vendor_args[i].name() || *args[i].type() != *vendor_args[i].type())
      return false;
  }
  for (size_t i = 0; i < forward.returns().size(); ++i) {
    if (*forward.returns()[i].type() != *raw.returns()[i].type())
      return false;
  }
  return true;
}

void Forward(const c10::OperatorHandle& forward, Stack* stack) {
  static const auto raw = [&] {
    const auto op = c10::Dispatcher::singleton().findSchema({raw_forward_name, ""});
    TORCH_CHECK(op && MatchesVendorSchema(forward.schema(), op->schema()),
                "FlashAttention 3 on FlagOS requires matching registered forward and CUDA operator schemas");
    return *op;
  }();
  const auto& query = (*stack)[0].toTensor();
  const auto device = query.device();
  TORCH_CHECK(device.is_privateuseone(), "FlashAttention 3 expects a FlagOS query");
  // The vendor API writes through these arguments despite the forward schema.
  TORCH_CHECK((*stack)[3].isNone() && (*stack)[4].isNone() &&
              (*stack)[6].isNone() && (*stack)[30].isNone(),
              "FlashAttention 3 on FlagOS supports read-only forward without an "
              "output buffer, KV-cache updates, or external scheduler metadata");

  static const auto device_count = [] {
    const auto count = c10::cuda::device_count();
    c10::cuda::CUDACachingAllocator::init(count);
    return count;
  }();
  const auto stream = GetCurrentStreamForDevice(device.index());
  const auto queue = reinterpret_cast<cudaStream_t>(stream);
  const auto cuda_stream = c10::cuda::getStreamFromExternal(queue, device.index());
  std::optional<c10::cuda::CUDAStreamGuard> guard;
  if ((device_count > 1 && c10::cuda::current_device() != device.index()) ||
      c10::cuda::getCurrentCUDAStream(device.index()).stream() != queue)
    guard.emplace(cuda_stream);
  cudaStreamCaptureStatus capture_status;
  C10_CUDA_CHECK(cudaStreamIsCapturing(queue, &capture_status));
  TORCH_CHECK(capture_status == cudaStreamCaptureStatusNone,
              "FlashAttention 3 on FlagOS requires an uncaptured stream");

  Stack native;
  native.reserve(stack->size());
  c10::SmallVector<at::Tensor, 8> borrowed;
  for (size_t i = 0; i < stack->size(); ++i) {
    const auto& argument = (*stack)[i];
    if (!argument.isTensor()) {
      native.push_back(argument);
      continue;
    }
    auto tensor = argument.toTensor();
    TORCH_CHECK(tensor.device() == device,
                "FlashAttention 3 expects all tensors on the query device");
    if (tensor.is_conj()) tensor = tensor.resolve_conj();
    if (tensor.is_neg()) tensor = tensor.resolve_neg();
    if (tensor.dim() && tensor.stride(-1) != 1) {
      // Match the vendor's public forward wrapper. It also accepts a
      // transposed value when its sequence dimension is contiguous.
      const bool require_contiguous = i < 2 || (i >= 7 && i <= 19);
      const bool strided_value = i == 2 && tensor.dim() >= 3 && tensor.stride(-3) != 1;
      if (require_contiguous || strided_value) tensor = tensor.contiguous();
    }
    native.emplace_back(FlagosToCudaView(tensor));
    borrowed.push_back(std::move(tensor));
  }

  const auto finish_borrows = [&] {
    auto* allocator = c10::flagos::GetCachingAllocator();
    bool ordered = allocator && c10::flagos::CachingDeviceAllocator::is_enabled();
    if (ordered) {
      for (const auto& tensor : borrowed) {
        ordered &= allocator->record_stream(tensor.storage().data_ptr(), stream);
      }
    }
    if (!ordered) C10_CUDA_CHECK(cudaStreamSynchronize(queue));
  };

  Stack outputs;
  try {
    raw.callBoxed(&native);
    outputs.reserve(native.size());
    for (const auto& value : native) {
      if (value.isNone() || !value.toTensor().defined()) {
        outputs.emplace_back(at::empty(
            {0}, query.options().dtype(c10::get_default_dtype_as_scalartype())));
        continue;
      }
      const auto& data = value.toTensor();
      auto output = at::empty_strided(
          data.sizes(), data.strides(), data.options().device(device));
      if (data.numel()) FlagosToCudaView(output).copy_(data);
      outputs.emplace_back(std::move(output));
    }
  } catch (...) {
    finish_borrows();
    throw;
  }
  finish_borrows();
  *stack = std::move(outputs);
}

} // namespace

TORCH_LIBRARY_IMPL(flash_attn_3, PrivateUse1, m) {
  const auto forward = c10::Dispatcher::singleton().findSchema({forward_name, ""});
  if (!forward || !forward->hasKernelForDispatchKey(c10::DispatchKey::PrivateUse1)) {
    m.impl("_flash_attn_forward", torch::CppFunction::makeFromBoxedFunction<&Forward>());
  }
}

} // namespace flagos::nvidia
