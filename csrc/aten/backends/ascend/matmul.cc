// Copyright (c) 2026, BAAI. All rights reserved.
//
// Direct aten::matmul interception for the Ascend backend via aclnnMatmul.
//
// PyTorch's aten::matmul is CompositeImplicitAutograd and normally decomposes
// into mm + bmm + view operations before reaching PrivateUse1. torch_npu
// intercepts it at the aten::matmul level (254 aten.matmul.default/step vs
// torch_fl's mm 197 + bmm 57 + view churn 423). By registering here we
// eliminate the ~5ms/step view churn and collapse 254 ops to one aclnnMatmul
// call each, matching torch_npu's operator path exactly.
//
// Registration: register.cc TORCH_LIBRARY_IMPL(aten, PrivateUse1) adds
//   m.impl("matmul", WrapperMatmul)
// which routes to this kernel when GetBackendForOp("matmul") == kAscend;
// non-Ascend backends keep PyTorch's composite decomposition by calling
// at::native::matmul directly, and register on AutogradPrivateUse1 instead so
// the autograd key never binds matmul_backward for them.
//
// Owning aten::matmul means autograd can no longer record the decomposed
// sub-ops, so it binds the op's real derivative, aten::matmul_backward. That is
// implemented below (MatmulBackwardKernelAscend) and reached via the generated
// AutogradPrivateUse1 kernel in csrc/aten/generated/variable_type.cc. Without
// it, training would decay to CPU.

#include "../../generated/ops.h"
#include <ATen/core/Tensor.h>
#include <algorithm>
#include <vector>
#include "op_preparation.h"
#include "op_api_common.h"
#include "dtype_support.h"
#include <ATen/ops/matmul.h>

namespace at::native::flagos {

// Compute the output shape for aten::matmul following NumPy/PyTorch semantics.
// aclnnMatmul handles all input dimensionalities ("for any shape mat multiply").
static std::vector<int64_t> matmul_output_shape(
    const at::Tensor& a, const at::Tensor& b) {
  int64_t da = a.dim(), db = b.dim();
  // 1-D cases
  if (da == 1 && db == 1) return {};                     // dot -> scalar
  if (da == 1 && db == 2) return {b.size(1)};            // (K,)x(K,N)->(N,)
  if (da == 2 && db == 1) return {a.size(0)};            // (M,K)x(K,)->(M,)
  if (da == 2 && db == 2) return {a.size(0), b.size(1)}; // mm
  // N-D batched: treat 1-D inputs as row/col vector, broadcast batch dims,
  // output M×N from last two dims of each input.
  auto a_sz = a.sizes().vec();
  auto b_sz = b.sizes().vec();
  bool a_1d = (da == 1), b_1d = (db == 1);
  if (a_1d) a_sz.insert(a_sz.begin(), 1);
  if (b_1d) b_sz.push_back(1);
  int64_t na = static_cast<int64_t>(a_sz.size());
  int64_t nb = static_cast<int64_t>(b_sz.size());
  int64_t n  = std::max(na, nb);
  std::vector<int64_t> out;
  for (int64_t i = 0; i < n - 2; ++i) {
    int64_t ai = na - n + i, bi = nb - n + i;
    int64_t sa = (ai >= 0) ? a_sz[ai] : 1;
    int64_t sb = (bi >= 0) ? b_sz[bi] : 1;
    out.push_back(sa == 1 ? sb : sa);
  }
  out.push_back(a_sz[na - 2]);  // M
  out.push_back(b_sz[nb - 1]);  // N
  if (a_1d) out.erase(out.end() - 2);
  if (b_1d) out.pop_back();
  return out;
}

at::Tensor MatmulKernelAscend(const at::Tensor& self,
                               const at::Tensor& other) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  namespace ascend = at::native::flagos::ascend;
  if (!ascend::IsMatmulDtypeSupported(self.scalar_type()) ||
      self.scalar_type() != other.scalar_type()) {
    auto cpu_result = at::matmul(self.cpu(), other.cpu());
    return ascend::MoveCpuResultToDevice(std::move(cpu_result), self);
  }
  int8_t cube_math_type = ascend::OpPreparation::get_cube_math_type(true);
  auto out = ascend::OpPreparation::apply_tensor_without_format(
      matmul_output_shape(self, other), self.options());

  static void* opApiFuncAddr  = nullptr;
  static void* getWsFuncAddr  = nullptr;
  ascend::SigHasher hsh;
  hsh.tensor(self);
  hsh.tensor(other);
  hsh.val(cube_math_type);
  ascend::ExecAscendCached(
      "aclnnMatmul", "aclnnMatmulGetWorkspaceSize",
      opApiFuncAddr, getWsFuncAddr, hsh.h,
      {&self, &other}, {&out},
      [&](ascend::GwsFunc gws,
          std::vector<ascend::AclTensorWrapper>& in,
          std::vector<ascend::AclTensorWrapper>& out_t,
          uint64_t* pws, aclOpExecutor** pex) {
        return gws(in[0].acl_tensor, in[1].acl_tensor,
                   out_t[0].acl_tensor, cube_math_type, pws, pex);
      });
  return out;
}

// --- aten::matmul_backward ---
//
// d/dself = grad @ other^T and d/dother = self^T @ grad, but only after undoing
// the shape normalization aten::matmul applied in the forward pass, which is
// where all the subtlety is. Three things have to be undone:
//
//  * 1-D operands were promoted to a row/column vector, so grad must be
//    unsqueezed on the matching side to line the contraction back up;
//  * batch dims were broadcast, so the raw gradient can be larger than the
//    operand and has to be summed back down to the operand's shape;
//  * when one side is 2-D and the other batched, matmul folded the batch dims
//    into the contraction. Reproducing that fold turns the gradient into one
//    2-D matmul instead of a batched one -- cheaper, and it performs the sum
//    over the batch dims implicitly.
//
// op-plugin's MatmulBackwardKernelNpuOpApi.cpp was the starting reference (it
// is what aclnnMatmul is known to be driven with), but two of its shape rules
// are wrong and are deliberately not reproduced here; see the comments at the
// fold branch and at squeeze_broadcast_batch_dims.
//
// Each branch issues one aclnnMatmul through the cached executor path, so
// backward costs two fused calls instead of the mm/bmm/view chain the composite
// decomposition produced.

// Sum a raw gradient back down to `shape`.
//
// matmul broadcasts batch dims, so d(out)/d(operand) is shaped like the
// *broadcast* operand and every dim the forward pass expanded has to be summed
// away. op-plugin instead squeezes leading size-1 dims off the operand before
// the matmul, which only coincidentally works when the broadcasting is a pure
// prefix: for an interior singleton such as (2,1,3,4) x (2,5,4,6) it leaves the
// dim in place and returns a (2,5,3,4) gradient for a (2,1,3,4) input. Summing
// after the fact is both correct in general and cheap (a no-op when no dim was
// broadcast, which is the common case, so the transformer path pays nothing).
static at::Tensor sum_to_shape(at::Tensor grad, at::IntArrayRef shape) {
  if (grad.sizes() == shape) {
    return grad;
  }
  return at::sum_to(std::move(grad), shape);
}

static at::Tensor matmul_mat1_backward(const at::Tensor& self,
                                       const at::Tensor& other,
                                       const at::Tensor& grad_output) {
  at::Tensor mat1 = self;
  at::Tensor mat2 = other;
  at::Tensor grad = grad_output;

  // 1-D operands were promoted to vectors by matmul; match that on grad.
  if (mat2.dim() == 1) {
    mat2 = mat2.unsqueeze(-1);
    grad = grad.unsqueeze(-1);
  }
  if (mat1.dim() == 1) {
    mat1 = mat1.unsqueeze(0);
    grad = grad.unsqueeze(-2);
  }
  // Target the *promoted* shape, not self's: a 1-D self is still a row vector
  // at this point and only gets flattened back at the end of the kernel.
  const auto target = mat1.sizes().vec();

  if (mat1.dim() == 2 && mat2.dim() > 2) {
    // self is 2-D against a batched other, so grad is [B..., M, N] and the sum
    // over B that the gradient needs can be folded into a single 2-D matmul:
    // mat2^T is [B..., N, K] flattened to (B*N, K), i.e. its row index is the
    // pair (b, n). grad must carry the *same* pair as its column index, so M
    // has to be permuted to the front before flattening. op-plugin reshapes
    // grad to {M, -1} directly, which pairs (m, n) columns against (b, n) rows
    // and silently mixes batches; that is a bug, not a convention.
    std::vector<int64_t> perm;
    perm.reserve(grad.dim());
    perm.push_back(grad.dim() - 2);  // M
    for (int64_t i = 0; i < grad.dim() - 2; ++i) {
      perm.push_back(i);  // B...
    }
    perm.push_back(grad.dim() - 1);  // N
    const int64_t m = grad.size(-2);
    mat2 = mat2.transpose(-2, -1);
    mat2 = mat2.reshape({-1, mat2.size(-1)});
    grad = grad.permute(perm).contiguous().reshape({m, -1});
    // The flattened contraction already summed over B, so this lands on
    // target directly and needs no further reduction.
    return MatmulKernelAscend(grad, mat2).reshape(target);
  }
  return sum_to_shape(MatmulKernelAscend(grad, mat2.transpose(-2, -1)), target);
}

static at::Tensor matmul_mat2_backward(const at::Tensor& self,
                                       const at::Tensor& other,
                                       const at::Tensor& grad_output) {
  at::Tensor mat1 = self;
  at::Tensor mat2 = other;
  at::Tensor grad = grad_output;

  if (mat2.dim() == 1) {
    mat2 = mat2.unsqueeze(-1);
    grad = grad.unsqueeze(-1);
  }
  if (mat1.dim() == 1) {
    mat1 = mat1.unsqueeze(0);
    grad = grad.unsqueeze(-2);
  }
  const auto target = mat2.sizes().vec();

  if (mat2.dim() == 2 && mat1.dim() > 2) {
    // Mirror of the fold above. Here matmul flattened self's batch dims into
    // the M axis, so both operands flatten to 2-D with the same (b, m) row
    // index and no permute is needed -- B and M are already adjacent and in
    // order on both sides.
    at::Tensor lhs = mat1.reshape({-1, mat1.size(-1)});
    at::Tensor rhs = grad.reshape({-1, grad.size(-1)});
    return MatmulKernelAscend(lhs.transpose(-2, -1), rhs).reshape(target);
  }
  return sum_to_shape(MatmulKernelAscend(mat1.transpose(-2, -1), grad), target);
}

std::tuple<at::Tensor, at::Tensor> MatmulBackwardKernelAscend(
    const at::Tensor& grad,
    const at::Tensor& self,
    const at::Tensor& other,
    ::std::array<bool, 2> mask) {
  // Issue #326: aclnn reads the ambient device, so make the one
  // this kernel actually operates on current.
  ::at::native::flagos::ascend::OpDeviceGuard device_guard_(
      ::at::native::flagos::ascend::DeviceOf(self));
  if (!grad.defined()) {
    return std::make_tuple(at::Tensor(), at::Tensor());
  }

  at::Tensor self_grad, other_grad;
  if (mask[1]) {
    other_grad = matmul_mat2_backward(self, other, grad);
  }
  if (mask[0]) {
    self_grad = matmul_mat1_backward(self, other, grad);
  }

  // The 1-D promotions above leave a stray dim on the gradient of a 1-D
  // operand; strip it so each gradient matches its operand's shape.
  if (self.dim() == 1 && self_grad.defined() && self_grad.dim() != 1) {
    self_grad = self_grad.reshape(self.sizes());
  }
  if (other.dim() == 1 && other_grad.defined() && other_grad.dim() != 1) {
    other_grad = other_grad.reshape(other.sizes());
  }
  return std::make_tuple(self_grad, other_grad);
}

} // namespace at::native::flagos
