#!/bin/bash
# Create minimal torch_npu header stubs for triton-ascend JIT compilation.
#
# triton-ascend's generated npu_utils.cpp / launcher unconditionally includes
# torch_npu headers:
#
#   #include <torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h>
#   #include <torch_npu/csrc/framework/OpCommand.h>
#
# The second include is guarded by `if {enable_taskqueue}` inside an f-string,
# which evaluates a *set literal* `{False}` -- always truthy -- so the include is
# emitted even when TRITON_ENABLE_TASKQUEUE is off. With taskqueue disabled no
# at_npu:: symbol is actually referenced by the generated code, so empty stubs
# are sufficient to compile.
#
# torch_fl does not depend on torch_npu, so the stubs are installed into
# torch/include (which triton-ascend puts on the compiler include path) as well
# as site-packages for good measure.

set -e

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
TORCH_INCLUDE=$(python3 -c "import os, torch; print(os.path.join(os.path.dirname(os.path.realpath(torch.__file__)), 'include'))")

write_stubs() {
  local root="$1"
  mkdir -p "${root}/torch_npu/csrc/core/npu" "${root}/torch_npu/csrc/framework"

  cat > "${root}/torch_npu/csrc/core/npu/NPUWorkspaceAllocator.h" << 'EOF'
#pragma once
// Stub header for triton-ascend JIT compilation compatibility.
// torch_fl does not use torch_npu's workspace allocator; workspace memory is
// allocated through at::empty() on the PrivateUse1 device instead.
EOF

  cat > "${root}/torch_npu/csrc/framework/OpCommand.h" << 'EOF'
#pragma once
// Stub header for triton-ascend JIT compilation compatibility.
// triton-ascend emits this include unconditionally (its enable_taskqueue guard
// is a truthy set literal). With TRITON_ENABLE_TASKQUEUE disabled no
// at_npu::native::OpCommand symbol is referenced, so an empty stub suffices.
EOF

  echo "torch_npu header stubs written under ${root}"
}

write_stubs "${TORCH_INCLUDE}"
write_stubs "${SITE_PACKAGES}"

echo "torch_npu header stubs created successfully"
