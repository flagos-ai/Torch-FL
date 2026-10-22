// Copyright (c) 2026, BAAI. All rights reserved.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include "op_preparation.h"
#include "op_api_common.h"

namespace at::native::flagos {

namespace {

// aclnnLe was introduced in CANN 8.0. Older versions may expose the
// kernel under aclnnLessEqual or aclnnLeTensor. We probe at runtime
// so the plugin works across versions.
enum class LeApi { kUnknown, kLe, kLessEqual, kLeTensor };

struct LeApiAddrs {
  void* workspace_fn = nullptr;
  void* exec_fn = nullptr;
};

LeApiAddrs ResolveLeApi(LeApi& cached) {
  void* handle = ascend::GetOpApiLibHandle();
  // Try aclnnLe (CANN 8.0+)
  void* ws = dlsym(handle, "aclnnLeGetWorkspaceSize");
  void* exec = dlsym(handle, "aclnnLe");
  if (ws && exec) {
    cached = LeApi::kLe;
    return {ws, exec};
  }
  // Try aclnnLeTensor
  ws = dlsym(handle, "aclnnLeTensorGetWorkspaceSize");
  exec = dlsym(handle, "aclnnLeTensor");
  if (ws && exec) {
    cached = LeApi::kLeTensor;
    return {ws, exec};
  }
  // Try aclnnLessEqual
  ws = dlsym(handle, "aclnnLessEqualGetWorkspaceSize");
  exec = dlsym(handle, "aclnnLessEqual");
  if (ws && exec) {
    cached = LeApi::kLessEqual;
    return {ws, exec};
  }
  TORCH_CHECK(false,
      "No le/less-equal operator found in libopapi.so "
      "(tried aclnnLe, aclnnLeTensor, aclnnLessEqual). "
      "Please check your CANN version.");
  return {};
}

LeApiAddrs GetLeApiAddrs() {
  static LeApi api = LeApi::kUnknown;
  static LeApiAddrs addrs = ResolveLeApi(api);
  return addrs;
}

} // namespace

at::Tensor LeTensorKernelAscend(const at::Tensor& self, const at::Tensor& other) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  namespace ascend = at::native::flagos::ascend;

  auto out_shape = at::infer_size(self.sizes(), other.sizes());
  auto self_expanded = self.expand(out_shape).contiguous();
  auto other_expanded = other.expand(out_shape).contiguous();

  auto out = ascend::OpPreparation::apply_tensor_without_format(
      out_shape, self.options().dtype(at::kBool));

  ascend::AclTensorWrapper acl_self(self_expanded);
  ascend::AclTensorWrapper acl_other(other_expanded);
  ascend::AclTensorWrapper acl_out(out);

  auto addrs = GetLeApiAddrs();
  auto acl_stream = ascend::GetCurrentAclStream();

  uint64_t workspace_size = 0;
  aclOpExecutor* executor = nullptr;

  typedef int (*GetWorkspaceSizeFunc)(...);
  auto getWorkspaceSize =
      reinterpret_cast<GetWorkspaceSizeFunc>(addrs.workspace_fn);
  int ws_ret = getWorkspaceSize(
      acl_self.get(), acl_other.get(), acl_out.get(),
      &workspace_size, &executor);
  TORCH_CHECK(ws_ret == 0, "aclnn Le/LessEqual GetWorkspaceSize failed, ret=", ws_ret);

  void* workspace_addr = nullptr;
  if (workspace_size > 0) {
    auto malloc_ret = ::Malloc(&workspace_addr, workspace_size);
    TORCH_CHECK(malloc_ret == Success, "Workspace allocation failed for Le");
  }

  typedef int (*ExecFunc)(void*, uint64_t, aclOpExecutor*, aclrtStream);
  auto executeFunc = reinterpret_cast<ExecFunc>(addrs.exec_fn);
  int exec_ret = executeFunc(workspace_addr, workspace_size, executor, acl_stream);
  TORCH_CHECK(exec_ret == 0, "aclnn Le/LessEqual execution failed, ret=", exec_ret);

  if (workspace_addr) {
    ::Free(workspace_addr);
  }

  return out;
}

REGISTER_IMPL_TO_DISPATCHER(LeTensorFn, le_tensor_dispatcher, Backend::kAscend, LeTensorKernelAscend)

} // namespace at::native::flagos
