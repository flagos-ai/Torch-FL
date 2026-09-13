#!/bin/bash
# Create minimal torch_npu header stubs for triton-ascend JIT compilation
# triton-ascend's npu_utils.cpp includes torch_npu headers that aren't needed
# for torch_fl operation, so we provide empty stubs to satisfy the compiler.

set -e

SITE_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")
TORCH_NPU_DIR="${SITE_PACKAGES}/torch_npu"
HEADERS_DIR="${TORCH_NPU_DIR}/csrc/core/npu"

echo "Creating torch_npu header stubs in ${HEADERS_DIR}"
mkdir -p "${HEADERS_DIR}"

# NPUWorkspaceAllocator.h - empty stub
cat > "${HEADERS_DIR}/NPUWorkspaceAllocator.h" << 'EOF'
#pragma once
// Stub header for triton-ascend JIT compilation compatibility
// torch_fl does not use torch_npu's workspace allocator
EOF

echo "torch_npu header stubs created successfully"
