// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include "cudnn_attention.h"

#include <ATen/ops/empty.h>
#include <ATen/ops/empty_like.h>
#include <c10/core/DeviceGuard.h>
#include <c10/util/hash.h>
#include <cudnn_frontend.h>
#include <flagos.h>

#include <array>
#include <cmath>
#include <memory>
#include <unordered_map>
#include <vector>

namespace flagos::nvidia {
namespace {

namespace fe = cudnn_frontend;
using PlanKey = std::array<int64_t, 4 * 4 * 2 + 1>;
constexpr size_t kPlanLimit = 128;

enum Operand : int64_t {
  kQuery = 1,
  kKey,
  kValue,
  kOutput,
  kScale,
};

bool SupportsTensor(const at::Tensor& tensor, c10::Device device) {
  return tensor.device() == device && tensor.layout() == at::kStrided &&
      tensor.scalar_type() == at::kBFloat16 && tensor.dim() == 4 &&
      tensor.numel() && tensor.stride(3) == 1 && !tensor.is_neg() &&
      !tensor.has_names();
}

std::vector<int64_t> DescriptorStrides(const at::Tensor& tensor) {
  auto strides = tensor.strides().vec();
  for (int dim = 2; dim >= 0; --dim) {
    if (tensor.size(dim) == 1 && strides[dim] == 0)
      strides[dim] = strides[dim + 1] * tensor.size(dim + 1);
  }
  return strides;
}

struct PlanKeyHash {
  size_t operator()(const PlanKey& key) const {
    size_t hash = 0;
    for (const auto value : key) hash = c10::hash_combine(hash, std::hash<int64_t>{}(value));
    return hash;
  }
};

struct AttentionPlan {
  fe::graph::Graph graph;
  int64_t workspace_size = 0;
  std::unordered_map<int64_t, void*> pointers{
      {kQuery, nullptr}, {kKey, nullptr}, {kValue, nullptr},
      {kOutput, nullptr}, {kScale, nullptr}};
};

struct DevicePlans {
  DevicePlans() {
    if (cudnnCreate(&handle) != CUDNN_STATUS_SUCCESS) handle = nullptr;
  }
  ~DevicePlans() {
    plans.clear();
    if (handle) cudnnDestroy(handle);
  }

  cudnnHandle_t handle = nullptr;
  cudaStream_t stream = nullptr;
  std::unordered_map<PlanKey, std::unique_ptr<AttentionPlan>, PlanKeyHash> plans;
};

std::unique_ptr<AttentionPlan> BuildPlan(
    cudnnHandle_t handle, const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& output, bool causal) {
  auto plan = std::make_unique<AttentionPlan>();
  auto& graph = plan->graph;
  graph.set_io_data_type(fe::DataType_t::BFLOAT16)
       .set_intermediate_data_type(fe::DataType_t::FLOAT)
       .set_compute_data_type(fe::DataType_t::FLOAT);
  auto tensor = [&](const at::Tensor& t, int64_t uid) {
    return graph.tensor(fe::graph::Tensor_attributes()
        .set_uid(uid).set_dim(t.sizes().vec()).set_stride(DescriptorStrides(t)));
  };
  const auto scale = graph.tensor(fe::graph::Tensor_attributes()
      .set_uid(kScale).set_dim({1, 1, 1, 1}).set_stride({1, 1, 1, 1})
      .set_data_type(fe::DataType_t::FLOAT).set_is_pass_by_value(true));
  auto options = fe::graph::SDPA_attributes().set_generate_stats(false)
      .set_causal_mask(causal).set_attn_scale(scale);
  const auto [result, stats] = graph.sdpa(
      tensor(query, kQuery), tensor(key, kKey), tensor(value, kValue), options);
  result->set_uid(kOutput).set_output(true).set_data_type(fe::DataType_t::BFLOAT16)
      .set_dim(output.sizes().vec()).set_stride(DescriptorStrides(output));
  if (!graph.validate().is_good() ||
      !graph.build_operation_graph(handle).is_good() ||
      !graph.create_execution_plans({fe::HeurMode_t::A}).is_good()) {
    return nullptr;
  }
  graph.deselect_numeric_notes({fe::NumericalNote_t::DOWN_CONVERT_INPUTS,
                                fe::NumericalNote_t::NONDETERMINISTIC});
  if (!graph.check_support(handle).is_good() || !graph.build_plans(handle).is_good() ||
      !graph.get_workspace_size(plan->workspace_size).is_good()) {
    return nullptr;
  }
  return plan;
}

}  // namespace

at::Tensor TryCuDnnAttention(const at::Tensor& query, const at::Tensor& key,
                             const at::Tensor& value, bool causal,
                             std::optional<double> scale, bool enable_gqa) {
  const auto device = query.device();
  if (!device.is_privateuseone() || !SupportsTensor(query, device) ||
      !SupportsTensor(key, device) || !SupportsTensor(value, device) ||
      !key.sizes().equals(value.sizes()) || key.size(0) != query.size(0) ||
      query.size(3) != key.size(3) || key.size(3) < 8 || key.size(3) > 128 ||
      key.size(3) % 8 || query.size(1) % key.size(1) ||
      (!enable_gqa && query.size(1) != key.size(1))) {
    return {};
  }
  float factor = scale.value_or(1.0 / std::sqrt(query.size(3)));
  if (!std::isfinite(factor)) return {};
  const c10::OptionalDeviceGuard guard(device);
  const auto stream = reinterpret_cast<cudaStream_t>(GetCurrentStreamForDevice(device.index()));
  static thread_local std::unordered_map<c10::DeviceIndex, DevicePlans> devices;
  auto& state = devices.try_emplace(device.index()).first->second;
  if (!state.handle) return {};
  if (state.stream != stream) {
    if (cudnnSetStream(state.handle, stream) != CUDNN_STATUS_SUCCESS) return {};
    state.stream = stream;
  }

  auto output = at::empty_like(query);
  PlanKey signature{};
  auto item = signature.begin();
  for (const auto* tensor : {&query, &key, &value, static_cast<const at::Tensor*>(&output)}) {
    for (const auto size : tensor->sizes()) *item++ = size;
    for (const auto stride : tensor->strides()) *item++ = stride;
  }
  *item = causal;
  auto plan = state.plans.find(signature);
  if (plan == state.plans.end()) {
    if (state.plans.size() == kPlanLimit) state.plans.clear();
    plan = state.plans.emplace(signature,
        BuildPlan(state.handle, query, key, value, output, causal)).first;
  }
  if (!plan->second) return {};
  auto& compiled = *plan->second;
  at::Tensor workspace;
  if (compiled.workspace_size)
    workspace = at::empty({compiled.workspace_size}, query.options().dtype(at::kByte));
  auto& pointers = compiled.pointers;
  pointers[kQuery] = const_cast<void*>(query.const_data_ptr());
  pointers[kKey] = const_cast<void*>(key.const_data_ptr());
  pointers[kValue] = const_cast<void*>(value.const_data_ptr());
  pointers[kOutput] = output.mutable_data_ptr();
  pointers[kScale] = &factor;
  auto status = compiled.graph.execute(
      state.handle, pointers, workspace.defined() ? workspace.mutable_data_ptr() : nullptr);
  TORCH_CHECK(status.is_good(), "cuDNN attention failed: ", status.get_message());
  return output;
}

}  // namespace flagos::nvidia
