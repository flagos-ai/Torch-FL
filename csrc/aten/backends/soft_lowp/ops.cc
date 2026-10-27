// Copyright 2026 FlagOS Contributors
//
// Device-side software emulation for low-precision matrix operations.

#include "ops.h"

#include "../../generated/ops.h"
#include "convert.h"
#include "format.h"

#include <ATen/ATen.h>
#include <ATen/Dispatch.h>
#include <ATen/ops/bitwise_and.h>
#include <ATen/ops/bitwise_right_shift.h>
#include <ATen/ops/add.h>
#include <ATen/ops/bmm.h>
#include <ATen/ops/empty.h>
#include <ATen/ops/eq.h>
#include <ATen/ops/mul.h>
#include <ATen/ops/exp2.h>
#include <ATen/ops/mm.h>
#include <ATen/ops/ne.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/where.h>
#include <ATen/native/Resize.h>
#include <c10/util/Exception.h>
#include <cstring>
#include <vector>

#include "../../device_boxing.h"
#include "../../../include/flagos.h"

namespace at::native::flagos::soft_lowp {

namespace {
thread_local bool dispatch_suppressed = false;
}

bool IsDispatchSuppressed() {
  return dispatch_suppressed;
}

DispatchSuppressionGuard::DispatchSuppressionGuard()
    : previous_(dispatch_suppressed) {
  dispatch_suppressed = true;
}

DispatchSuppressionGuard::~DispatchSuppressionGuard() {
  dispatch_suppressed = previous_;
}

namespace {

at::Tensor DecodeFp4Nibbles(const at::Tensor& raw) {
  // Decode FP4 nibbles using arithmetic operations.
  // FP4 format: seem (sign, 2-bit exponent, 1-bit mantissa)
  auto decode = [](const at::Tensor& nibble) {
    // Extract fields as bfloat16 for computation
    auto exponent = at::bitwise_and(
                        at::bitwise_right_shift(nibble, at::Scalar(1)),
                        at::Scalar(3))
                        .to(c10::kBFloat16);
    auto mantissa = at::bitwise_and(nibble, at::Scalar(1)).to(c10::kBFloat16);

    // Normal values: 2^(e-1) * (1 + m*0.5)
    auto normal = at::exp2(exponent - 1) *
                  (1 + mantissa * static_cast<double>(0.5));
    // Subnormal (e==0): m*0.5
    auto subnormal = mantissa * static_cast<double>(0.5);
    auto magnitude = at::where(
        at::eq(exponent, static_cast<double>(0)), subnormal, normal);

    // Apply sign bit (bit 3)
    auto negative = at::ne(at::bitwise_and(nibble, at::Scalar(8)),
                           static_cast<int64_t>(0));
    return at::where(negative, -magnitude, magnitude);
  };

  // Extract low and high nibbles
  auto low = at::bitwise_and(raw, at::Scalar(0x0F));
  auto high = at::bitwise_right_shift(raw, at::Scalar(4));

  // Decode both and stack along last dimension
  auto decoded_low = decode(low);
  auto decoded_high = decode(high);
  return at::stack({decoded_low, decoded_high}, -1);
}

at::Tensor DecodeMatrix(const at::Tensor& input, bool packed_dim0) {
  if (input.scalar_type() == c10::kFloat4_e2m1fn_x2) {
    auto raw = input.view(c10::kByte);
    auto unpacked = DecodeFp4Nibbles(raw);
    if (packed_dim0) {
      // [K / 2, N] -> [K, N]
      auto shape = input.sizes().vec();
      shape[0] *= 2;
      return unpacked.reshape(shape);
    }
    // [M, K / 2] -> [M, K]
    auto shape = input.sizes().vec();
    shape.back() *= 2;
    return unpacked.reshape(shape);
  }

  // FP8 handled via boxing. This is the one boxing site outside the guards, so
  // it binds the device itself: the boxed copy goes through c10's CUDA
  // machinery, and the `at::empty` below relies on the runtime's current device
  // to place the result. See BoxingDeviceScope.
  BoxingDeviceScope device;
  if (input.device().is_privateuseone()) {
    device.bind(input.device().index());
  }
  auto result = at::empty(input.sizes(), input.options().dtype(c10::kBFloat16));
  if (IsFp8(input.scalar_type())) {
    at::Tensor boxed = input;
    at::Tensor boxed_result = result;
    SetTensorDevice(boxed, c10::DeviceType::CUDA);
    SetTensorDevice(boxed_result, c10::DeviceType::CUDA);
    auto decoded = at::native::copy_(boxed_result, boxed, false);
    UnboxToFlagos(decoded);
    UnboxToFlagos(boxed);
    return decoded;
  }

  return result.copy_(input);
}

at::Tensor DecodeBatchMatrix(const at::Tensor& input, bool packed_dim1) {
  if (input.scalar_type() == c10::kFloat4_e2m1fn_x2) {
    auto raw = input.view(c10::kByte);
    auto unpacked = DecodeFp4Nibbles(raw);
    auto shape = input.sizes().vec();
    if (packed_dim1) {
      // [B, K / 2, N] -> [B, K, N]
      shape[1] *= 2;
      return unpacked.reshape(shape);
    }
    // [B, M, K / 2] -> [B, M, K]
    shape.back() *= 2;
    return unpacked.reshape(shape);
  }

  // FP8 handled via boxing; see the note in DecodeMatrix.
  BoxingDeviceScope device;
  if (input.device().is_privateuseone()) {
    device.bind(input.device().index());
  }
  auto result = at::empty(input.sizes(), input.options().dtype(c10::kBFloat16));
  if (IsFp8(input.scalar_type())) {
    at::Tensor boxed = input;
    at::Tensor boxed_result = result;
    SetTensorDevice(boxed, c10::DeviceType::CUDA);
    SetTensorDevice(boxed_result, c10::DeviceType::CUDA);
    auto decoded = at::native::copy_(boxed_result, boxed, false);
    UnboxToFlagos(decoded);
    UnboxToFlagos(boxed);
    return decoded;
  }

  return result.copy_(input);
}

bool HasLowpInput(const at::Tensor& tensor) {
  return tensor.defined() && IsLowPrecision(tensor.scalar_type());
}

void CheckDevice(const at::Tensor& tensor, const char* name) {
  TORCH_CHECK(
      tensor.device().is_privateuseone(),
      "soft-lowp ",
      name,
      " requires PrivateUse1 tensors, got ",
      tensor.device());
}

// Whether both operands are ordinary dtypes -- no FP8/FP4 on either side. Such
// a pair is not this path's to compute: it leaves the way it came, on the route
// the conf names.
//
// The wrapper cannot make that call itself. The generated gates live in a shared
// .inc that runs with no idea whether the build compiles the software path at
// all, so they read `IsLowPrecision(dtype)` and every build that compiles this
// path takes the branch. Once here, a pair that stayed would be computed by
// `at::mm`/`at::bmm` on PrivateUse1 tensors, which on MetaX lands in FlagGems'
// vendored bmm and, for the shapes it cannot compile, back on this same
// function: the two ping-pong until the stack runs out (C550 host, float32
// (2, 4, 4); CI run 40202397139). Vendors are not expected to take the
// low-precision types, but their ordinary GEMMs are their own business.
bool IsOrdinaryMatrixPair(const at::Tensor& self, const at::Tensor& mat2) {
  return !HasLowpInput(self) && !HasLowpInput(mat2);
}

// Whether a decoded pair can go back to the conf's route at all.
//
// The decode is the software part of this path, not the GEMM: an FP8 tensor
// becomes a bf16 one, and a bf16 GEMM on PrivateUse1 operands is the route's,
// not this file's. So the decoded pair follows the conf exactly as an
// ordinary-dtype pair does -- the vendor slot where a build has one, boxing
// elsewhere -- instead of being computed in place.
//
// A packed FP4 tensor cannot go: it is stored as nibbles and decoded
// arithmetically, and there is no FP4 dtype to hand a kernel. It is computed
// here, with the ordinary operands around it.
bool DecodedPairCanRoute(const at::Tensor& self, const at::Tensor& mat2) {
  return !IsPackedFp4(self.scalar_type()) && !IsPackedFp4(mat2.scalar_type());
}

void CheckMatrixInputs(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(HasLowpInput(self) || HasLowpInput(mat2),
              "soft-lowp matrix kernel requires a low-precision input");
  CheckDevice(self, "matrix operation");
  CheckDevice(mat2, "matrix operation");
  TORCH_CHECK(self.is_contiguous() && mat2.is_contiguous(),
              "soft-lowp matrix kernel requires contiguous inputs");
}

at::Tensor DecodeIfNeeded(const at::Tensor& input, bool packed_dim0 = false) {
  if (!IsLowPrecision(input.scalar_type())) {
    return input;
  }
  TORCH_CHECK(input.dim() == 2,
              "soft-lowp matrix decode supports only rank-2 tensors");
  if (input.numel() == 0) {
    auto shape = input.sizes().vec();
    if (IsPackedFp4(input.scalar_type())) {
      (packed_dim0 ? shape[0] : shape.back()) *= 2;
    }
    return at::empty(shape, input.options().dtype(c10::kBFloat16));
  }
  return DecodeMatrix(input, packed_dim0);
}

at::Tensor DecodeBatchIfNeeded(const at::Tensor& input, bool packed_dim1 = false) {
  if (!IsLowPrecision(input.scalar_type())) {
    return input;
  }
  TORCH_CHECK(input.dim() == 3,
              "soft-lowp batch matrix decode supports only rank-3 tensors");
  if (input.numel() == 0) {
    auto shape = input.sizes().vec();
    if (IsPackedFp4(input.scalar_type())) {
      (packed_dim1 ? shape[1] : shape.back()) *= 2;
    }
    return at::empty(shape, input.options().dtype(c10::kBFloat16));
  }
  return DecodeBatchMatrix(input, packed_dim1);
}

c10::ScalarType DefaultOutputDtype(const at::Tensor& self, c10::ScalarType requested) {
  if (requested != c10::ScalarType::Undefined) {
    return requested;
  }
  return IsLowPrecision(self.scalar_type()) ? c10::kBFloat16 : self.scalar_type();
}

} // namespace

// The pair computed with the conf out of the loop: decode, then ATen's own
// mm/bmm on the PrivateUse1 operands. dispatch.cc binds these to the
// mm_fallback/bmm_fallback dispatchers, which is how
// Dispatcher::RouteAcceptsDecoded can tell a FlagGems C++ slot this file
// filled from a real FlagGems C++ kernel.
at::Tensor OrdinaryMm(const at::Tensor& self, const at::Tensor& mat2) {
  auto lhs = DecodeIfNeeded(self, false);
  auto rhs = DecodeIfNeeded(mat2, true);
  return at::mm(lhs, rhs);
}

at::Tensor OrdinaryBmm(const at::Tensor& self, const at::Tensor& mat2) {
  auto lhs = DecodeBatchIfNeeded(self, false);
  auto rhs = DecodeBatchIfNeeded(mat2, true);
  return at::bmm(lhs, rhs);
}

namespace {

// The decoded operands of one call and the conf key they are routed by.
struct TensorPair {
  const char* key = nullptr;  // the conf key this call routes by
  int64_t rank = 2;
  at::Tensor lhs;  // decoded, ordinary dtype
  at::Tensor rhs;  // decoded, ordinary dtype
  at::Tensor bias;  // decoded (addmm); undefined otherwise
};

// What the decoded pair is handed to, keyed the way the conf keys it.
//
// One op name routes several schemas and the conf routes each by its own name,
// so this table is the whole mapping from "which soft-lowp entry point was
// called" to "which kernel signature answers that key":
//   * mm/bmm and their .dtype forms    -> the dtype-taking dispatcher, with the
//                                         answer's dtype always spelled out
//   * mm.out, bmm.out, the .dtype.out forms
//                                      -> that form's own out dispatcher
//   * addmm                           -> the addmm dispatcher, bias and all
//   * addmm.out                      -> the addmm out dispatcher,
//                                         writing the caller's tensor
//
// `addmm.dtype` and `addmm.dtype_out` are not here: no conf names a kernel for
// them, so a decoded pair is computed here rather than routed (AddmmDtype), and
// those two entry points do not go through RunMatrix at all. Routing one by its
// own key would be "backend not registered".
//
// A schema added without an entry fails here, by name, rather than routing with
// the wrong signature.
enum class DecodedShape {
  kDtype,           // Dispatcher<MmDtypeFn>(lhs, rhs, out_dtype)
  kVoid,            // the decoder's own at::mm/at::bmm
  kMmOut,           // Dispatcher<MmOutFn>(lhs, rhs, out)
  kBmmOut,          // Dispatcher<BmmOutFn>(lhs, rhs, out)
  kMmDtypeOut,      // Dispatcher<MmDtypeOutFn>(lhs, rhs, out_dtype, out)
  kBmmDtypeOut,     // Dispatcher<BmmDtypeOutFn>(lhs, rhs, out_dtype, out)
  kAddmm,           // Dispatcher<AddmmFn>(bias, lhs, rhs, beta, alpha)
  kAddmmOut,        // Dispatcher<AddmmOutFn>(bias, lhs, rhs, beta, alpha, out)
};

struct DecodedVariant {
  const char* key;
  DecodedShape shape;
  bool batch;  // rank-3 (bmm) rather than rank-2 (mm/addmm)
};

const DecodedVariant* LookupVariant(const char* variant) {
  static const DecodedVariant variants[] = {
      {"mm", DecodedShape::kDtype, false},
      {"mm.dtype", DecodedShape::kDtype, false},
      {"mm.out", DecodedShape::kMmOut, false},
      {"mm.dtype_out", DecodedShape::kMmDtypeOut, false},
      {"bmm", DecodedShape::kDtype, true},
      {"bmm.dtype", DecodedShape::kDtype, true},
      {"bmm.out", DecodedShape::kBmmOut, true},
      {"bmm.dtype_out", DecodedShape::kBmmDtypeOut, true},
      {"addmm", DecodedShape::kAddmm, false},
      {"addmm.out", DecodedShape::kAddmmOut, false},
  };
  for (const auto& entry : variants) {
    if (std::strcmp(entry.key, variant) == 0) {
      return &entry;
    }
  }
  TORCH_CHECK(false, "soft-lowp has no decoded variant ", variant);
}

// The dtype the call is made under, spelled out.
//
// The decoded operands are bf16, and a kernel must not take the answer's dtype
// from them -- and a pair that already was bf16 (an FP8 operand next to a bf16
// one) still has to say bf16 rather than let a route pick its default. An
// answer the caller asked for in a low-precision dtype cannot be asked of a
// route: no route on this path takes one, and the software path will not spell
// one for a Python dispatcher. Asking here reports that, by dtype, at the call
// that asked for it.
c10::ScalarType RouteAnswerDtype(c10::ScalarType requested) {
  TORCH_CHECK(!IsLowPrecision(requested),
              "soft-lowp cannot ask an accelerated GEMM for a ",
              c10::toString(requested),
              " result");
  return requested;
}

// How a route's kernel is called here.
//
// The dispatcher answers with the kernel (DecodedKernel) and this file calls
// it. Every call binds the kernel to a parameter of the kernel's own type in a
// member of a struct built for that signature, then calls it from there; the
// struct's name says which signature. Binding the kernel to `auto`, or calling
// through a member of a class template, does not build on GCC 11 once a
// translation unit holds more than one kernel signature: it checks each such
// call against the first one it saw, and the calls that write into a caller's
// tensor then get the wrong signature's diagnostics. A call whose callee is a
// typed pointer parameter of a non-template member is not that call, on any
// compiler, and it is what these call sites are. The dispatcher's own call
// (operator()) is the first of the shape GCC 11 checks against, so a decoded
// call cannot go through it either: it passes the op name as the kernel's
// first operand.

// One route's kernel, selected for one key and called on one signature.
//
// Select takes the key the conf routes the call by and runs once, before the
// kernel is used, so a refusal is raised where the call would have been made.
// The call site names the dispatcher once (`const MmDtypeCall call{...}`) and
// passes its Select() result to its Call().
struct MmDtypeCall {
  const Dispatcher<MmDtypeFn>& dispatcher;
  MmDtypeFn Select(const char* key) const {
    return static_cast<MmDtypeFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(MmDtypeFn kernel, const at::Tensor& self,
                       const at::Tensor& mat2,
                       c10::ScalarType out_dtype) const {
    return kernel(self, mat2, out_dtype);
  }
};

struct BmmDtypeCall {
  const Dispatcher<BmmDtypeFn>& dispatcher;
  BmmDtypeFn Select(const char* key) const {
    return static_cast<BmmDtypeFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(BmmDtypeFn kernel, const at::Tensor& self,
                       const at::Tensor& mat2,
                       c10::ScalarType out_dtype) const {
    return kernel(self, mat2, out_dtype);
  }
};

struct AddmmCall {
  const Dispatcher<AddmmFn>& dispatcher;
  AddmmFn Select(const char* key) const {
    return static_cast<AddmmFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(AddmmFn kernel, const at::Tensor& bias,
                       const at::Tensor& self,
                       const at::Tensor& mat2,
                       const c10::Scalar& beta,
                       const c10::Scalar& alpha) const {
    return kernel(bias, self, mat2, beta, alpha);
  }
};

struct MmOutCall {
  const Dispatcher<MmOutFn>& dispatcher;
  MmOutFn Select(const char* key) const {
    return static_cast<MmOutFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(MmOutFn kernel, const at::Tensor& self,
                       const at::Tensor& mat2,
                       at::Tensor& out) const {
    return kernel(self, mat2, out);
  }
};

struct BmmOutCall {
  const Dispatcher<BmmOutFn>& dispatcher;
  BmmOutFn Select(const char* key) const {
    return static_cast<BmmOutFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(BmmOutFn kernel, const at::Tensor& self,
                       const at::Tensor& mat2,
                       at::Tensor& out) const {
    return kernel(self, mat2, out);
  }
};

struct MmDtypeOutCall {
  const Dispatcher<MmDtypeOutFn>& dispatcher;
  MmDtypeOutFn Select(const char* key) const {
    return static_cast<MmDtypeOutFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(MmDtypeOutFn kernel, const at::Tensor& self,
                       const at::Tensor& mat2,
                       c10::ScalarType out_dtype,
                       at::Tensor& out) const {
    return kernel(self, mat2, out_dtype, out);
  }
};

struct BmmDtypeOutCall {
  const Dispatcher<BmmDtypeOutFn>& dispatcher;
  BmmDtypeOutFn Select(const char* key) const {
    return static_cast<BmmDtypeOutFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(BmmDtypeOutFn kernel, const at::Tensor& self,
                       const at::Tensor& mat2,
                       c10::ScalarType out_dtype,
                       at::Tensor& out) const {
    return kernel(self, mat2, out_dtype, out);
  }
};

struct AddmmOutCall {
  const Dispatcher<AddmmOutFn>& dispatcher;
  AddmmOutFn Select(const char* key) const {
    return static_cast<AddmmOutFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(AddmmOutFn kernel, const at::Tensor& bias,
                       const at::Tensor& self,
                       const at::Tensor& mat2,
                       const c10::Scalar& beta,
                       const c10::Scalar& alpha,
                       at::Tensor& out) const {
    return kernel(bias, self, mat2, beta, alpha, out);
  }
};

// The decoded operands, ready for the route the conf names for `variant`.
TensorPair DecodeForRoute(
    const char* variant,
    const at::Tensor& self,
    const at::Tensor& mat2,
    int64_t rank,
    const at::Tensor* bias) {
  const auto& entry = *LookupVariant(variant);
  TORCH_CHECK(entry.batch == (rank == 3),
              "soft-lowp ",
              variant,
              " is ",
              entry.batch ? "batched" : "unbatched");
  // Decoded, the pair is handed back to the conf's route for this call's own
  // key: the vendor slot where a build has one, boxing elsewhere. FlagGems is
  // the route it cannot take (RouteAcceptsDecoded).
  auto lhs = DecodeIfNeeded(self, false);
  auto rhs = rank == 3 ? DecodeBatchIfNeeded(mat2, true)
                       : DecodeIfNeeded(mat2, true);
  auto pair = TensorPair{variant, rank, lhs, rhs, at::Tensor()};
  if (bias != nullptr) {
    pair.bias = IsLowPrecision(bias->scalar_type()) ? DecodeIfNeeded(*bias)
                                                    : *bias;
  }
  return pair;
}

// The GEMM the decoded pair is handed to, on the conf's route for it.
at::Tensor DispatchDecoded(
    const TensorPair& pair,
    c10::ScalarType output_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  const auto shape = LookupVariant(pair.key)->shape;
  output_dtype = RouteAnswerDtype(output_dtype);
  const char* const key = pair.key;
  const c10::OptionalDeviceGuard guard(pair.lhs.device());
  switch (shape) {
    case DecodedShape::kDtype:
      // mm and bmm are both kDtype, told apart by their entry's rank.
      if (LookupVariant(pair.key)->batch) {
        const BmmDtypeCall call{bmm_dtype_dispatcher};
        return call.Call(call.Select(key), pair.lhs, pair.rhs, output_dtype);
      }
      {
        const MmDtypeCall call{mm_dtype_dispatcher};
        return call.Call(call.Select(key), pair.lhs, pair.rhs, output_dtype);
      }
    case DecodedShape::kAddmm: {
      TORCH_CHECK(pair.bias.defined(), "soft-lowp decoded addmm needs its bias");
      const AddmmCall call{addmm_dispatcher};
      return call.Call(call.Select(key), pair.bias, pair.lhs, pair.rhs,
                       beta, alpha);
    }
    case DecodedShape::kVoid:
    case DecodedShape::kMmOut:
    case DecodedShape::kBmmOut:
    case DecodedShape::kMmDtypeOut:
    case DecodedShape::kBmmDtypeOut:
    case DecodedShape::kAddmmOut:
      break;
  }
  TORCH_CHECK(false, "soft-lowp ", pair.key, " has no call that answers a tensor");
}

// The same, for the forms that write the caller's tensor. The caller's tensor
// has to be the decoded answer's own dtype -- bf16 or whatever ordinary dtype
// it was -- and the route writes it; a route that cannot is asked nothing.
at::Tensor& DispatchDecodedOut(
    const TensorPair& pair,
    c10::ScalarType output_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  const auto shape = LookupVariant(pair.key)->shape;
  output_dtype = RouteAnswerDtype(output_dtype);
  const char* const key = pair.key;
  const c10::OptionalDeviceGuard guard(pair.lhs.device());
  switch (shape) {
    case DecodedShape::kMmOut: {
      const MmOutCall call{mm_out_dispatcher};
      return call.Call(call.Select(key), pair.lhs, pair.rhs, out);
    }
    case DecodedShape::kBmmOut: {
      const BmmOutCall call{bmm_out_dispatcher};
      return call.Call(call.Select(key), pair.lhs, pair.rhs, out);
    }
    case DecodedShape::kMmDtypeOut: {
      const MmDtypeOutCall call{mm_dtype_out_dispatcher};
      return call.Call(call.Select(key), pair.lhs, pair.rhs, output_dtype, out);
    }
    case DecodedShape::kBmmDtypeOut: {
      const BmmDtypeOutCall call{bmm_dtype_out_dispatcher};
      return call.Call(call.Select(key), pair.lhs, pair.rhs, output_dtype, out);
    }
    case DecodedShape::kAddmmOut: {
      TORCH_CHECK(pair.bias.defined(),
                  "soft-lowp decoded addmm needs its bias");
      const AddmmOutCall call{addmm_out_dispatcher};
      return call.Call(call.Select(key), pair.bias, pair.lhs, pair.rhs, beta,
                       alpha, out);
    }
    case DecodedShape::kDtype:
    case DecodedShape::kVoid:
    case DecodedShape::kAddmm:
      break;
  }
  TORCH_CHECK(false,
              "soft-lowp ",
              pair.key,
              " does not write the caller's tensor");
}

// The pair computed here, because no route will take it: a packed FP4 pair is
// decoded nibble-wise and there is no FP4 dtype to hand a kernel. A bias keeps
// its fused call through the boxing slot, where the bias and the scalars
// already have a kernel.
at::Tensor ComputeDecoded(
    const TensorPair& pair,
    c10::ScalarType output_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  const c10::OptionalDeviceGuard guard(pair.lhs.device());
  at::Tensor result;
  if (pair.bias.defined()) {
    // Called through AddmmCall, not the dispatcher: see the note above it.
    const AddmmFn kernel = static_cast<AddmmFn>(
        addmm_dispatcher.KernelFor(Backend::kBoxing));
    const AddmmCall call{addmm_dispatcher};
    result = call.Call(kernel, pair.bias, pair.lhs, pair.rhs, beta, alpha);
  } else if (pair.rank == 2) {
    result = at::mm(pair.lhs, pair.rhs);
  } else {
    result = at::bmm(pair.lhs, pair.rhs);
  }
  return output_dtype == result.scalar_type() ? result : result.to(output_dtype);
}

// Every .out form ends here: the answer must have the caller's tensor's shape,
// since only an identical size can be copied into it.
void CheckOutShape(
    const at::Tensor& out,
    const at::Tensor& result,
    const char* schema) {
  TORCH_CHECK(
      out.sizes() == result.sizes(), "soft-lowp ", schema, " shape mismatch");
}

// Whether the decoded pair has a route to go to. The pair is bf16 by the time
// this is asked, so it is the op's own route that decides -- not the dtype --
// and RouteAcceptsDecoded is where that question is answered.
bool DecodedPairHasRoute(const char* variant) {
  const auto& entry = *LookupVariant(variant);
  switch (entry.shape) {
    case DecodedShape::kMmOut:
    case DecodedShape::kMmDtypeOut:
      return mm_dispatcher.RouteAcceptsDecoded(variant);
    case DecodedShape::kBmmOut:
    case DecodedShape::kBmmDtypeOut:
      return bmm_dispatcher.RouteAcceptsDecoded(variant);
    case DecodedShape::kDtype:
      return (entry.batch ? bmm_dtype_dispatcher : mm_dtype_dispatcher)
          .RouteAcceptsDecoded(variant);
    case DecodedShape::kVoid:
      return false;
    case DecodedShape::kAddmm:
      return addmm_dispatcher.RouteAcceptsDecoded(variant);
    case DecodedShape::kAddmmOut:
      return addmm_out_dispatcher.RouteAcceptsDecoded(variant);
  }
  return false;
}

// One soft-lowp matrix call that answers a new tensor.
//
// An ordinary pair is the route's, not this path's: it leaves the way it came,
// on the conf's route for `variant`. A pair with an FP8 side is decoded here
// first -- that is the software part -- and then goes the same way. When no
// route can run the call, it is computed here.
//
// `out_dtype` is what the caller asked for, Undefined when it asked nothing
// (the answer then has the decoded operands' own ordinary dtype).
at::Tensor RunMatrix(
    const char* variant,
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    int64_t rank,
    const at::Tensor& bias,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  const bool has_bias = bias.defined();
  const char* const key = variant;
  const auto& entry = *LookupVariant(variant);
  if (IsOrdinaryMatrixPair(self, mat2)) {
    // Some composite callers (notably linear_out) reach addmm with ordinary
    // operands after their own boxing/decomposition; those keep the fused
    // addmm kernel rather than the low-precision validation below.
    if (entry.shape == DecodedShape::kDtype && !entry.batch) {
      const MmDtypeCall call{mm_dtype_dispatcher};
      return call.Call(call.Select(key), self, mat2, out_dtype);
    }
    if (entry.shape == DecodedShape::kDtype) {
      const BmmDtypeCall call{bmm_dtype_dispatcher};
      return call.Call(call.Select(key), self, mat2, out_dtype);
    }
    const auto output_dtype = DefaultOutputDtype(self, out_dtype);
    const AddmmCall call{addmm_dispatcher};
    const at::Tensor result =
        call.Call(call.Select(key), bias, self, mat2, beta, alpha);
    return output_dtype == result.scalar_type() ? result
                                                : result.to(output_dtype);
  }
  TORCH_CHECK(
      entry.shape == DecodedShape::kDtype || entry.shape == DecodedShape::kAddmm,
      "soft-lowp ",
      variant,
      " cannot answer a new tensor");
  CheckMatrixInputs(self, mat2);
  if (has_bias) {
    CheckDevice(bias, "addmm");
  }
  auto pair =
      DecodeForRoute(variant, self, mat2, rank, has_bias ? &bias : nullptr);
  const auto output_dtype = DefaultOutputDtype(self, out_dtype);
  // A packed FP4 tensor has no dtype to hand a route; an FP8 pair goes to the
  // route the conf names, if that route can run it at all and the answer asked
  // for is one a route can give.
  if (DecodedPairCanRoute(self, mat2) && DecodedPairHasRoute(variant) &&
      !IsLowPrecision(output_dtype)) {
    return DispatchDecoded(pair, output_dtype, beta, alpha);
  }
  return ComputeDecoded(pair, output_dtype, beta, alpha);
}

// The same for the forms that write `out`: computed here means computed and
// copied, routed means the route writes `out` itself.
//
// An ordinary pair never reaches this path at all -- the generated gate in
// front of every entry point here only fires on a low-precision operand -- but
// it leaves the way RunMatrix leaves it, by the conf, when one does.
// Declared before the entry points that call it: each of those passes the
// operands its own schema names, and a call resolved against a signature that
// is not yet in scope is a call the compiler cannot check -- addmm.out handed
// `mat1` where the bias goes and was never told.
at::Tensor& RunMatrixOut(
    const char* variant,
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    int64_t rank,
    const at::Tensor& bias,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  const bool has_bias = bias.defined();
  const bool ordinary = IsOrdinaryMatrixPair(self, mat2);
  // The answer the caller asked of this form, before the pair is looked at:
  // `out`'s own dtype when the form does not take one, the .dtype form's when
  // it does, and the decoded operands' default when neither.
  const auto requested = DefaultOutputDtype(
      self, out_dtype == c10::ScalarType::Undefined ? out.scalar_type()
                                                    : out_dtype);
  // A routed write asks the route to write `out` itself, which needs the
  // route to have a kernel and to be able to take a decoded pair, and `out` to
  // be in the answer's dtype. Anything else is computed and copied.
  if (!ordinary && DecodedPairCanRoute(self, mat2) &&
      DecodedPairHasRoute(variant) && !IsLowPrecision(requested) &&
      requested == out.scalar_type()) {
    // The route takes the caller's tensor as an ordinary dtype's, so it has to
    // be one: a rank-3 addmm does not exist, and a non-contiguous `out` is
    // computed and copied below (`out.copy_` takes any layout).
    if (out.dim() == rank && out.is_contiguous()) {
      CheckMatrixInputs(self, mat2);
      if (has_bias) {
        CheckDevice(bias, "addmm");
      }
      CheckDevice(out, "matrix operation");
      auto pair =
          DecodeForRoute(variant, self, mat2, rank, has_bias ? &bias : nullptr);
      return DispatchDecodedOut(pair, requested, beta, alpha, out);
    }
  }
  auto result =
      RunMatrix(variant, self, mat2, requested, rank, bias, beta, alpha);
  CheckOutShape(out, result, variant);
  out.copy_(result);
  return out;
}

} // namespace

at::Tensor Mm(const at::Tensor& self, const at::Tensor& mat2) {
  return MmDtype(self, mat2, c10::ScalarType::Undefined);
}

at::Tensor MmDtype(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype) {
  return RunMatrix("mm", self, mat2, out_dtype, 2, at::Tensor(),
                   at::Scalar(1), at::Scalar(1));
}

at::Tensor& MmDtypeOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    at::Tensor& out) {
  return RunMatrixOut("mm.dtype", self, mat2, out_dtype, 2, at::Tensor(),
                      at::Scalar(1), at::Scalar(1), out);
}

at::Tensor& MmOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  return RunMatrixOut("mm.out", self, mat2, c10::ScalarType::Undefined, 2,
                      at::Tensor(), at::Scalar(1), at::Scalar(1), out);
}

at::Tensor Bmm(const at::Tensor& self, const at::Tensor& mat2) {
  return BmmDtype(self, mat2, c10::ScalarType::Undefined);
}

at::Tensor BmmDtype(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype) {
  return RunMatrix("bmm", self, mat2, out_dtype, 3, at::Tensor(),
                   at::Scalar(1), at::Scalar(1));
}

at::Tensor& BmmDtypeOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    at::Tensor& out) {
  return RunMatrixOut("bmm.dtype", self, mat2, out_dtype, 3, at::Tensor(),
                      at::Scalar(1), at::Scalar(1), out);
}

at::Tensor& BmmOut(
    const at::Tensor& self,
    const at::Tensor& mat2,
    at::Tensor& out) {
  return RunMatrixOut("bmm.out", self, mat2, c10::ScalarType::Undefined, 3,
                      at::Tensor(), at::Scalar(1), at::Scalar(1), out);
}

at::Tensor Addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  return AddmmDtype(self, mat1, mat2, c10::ScalarType::Undefined, beta, alpha);
}

at::Tensor AddmmDtype(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  // No conf names a kernel for `addmm.dtype`, so a low-precision pair is
  // computed here rather than routed (the variant table has no entry for that
  // key, and dispatching by it would be "backend not registered"). An ordinary
  // pair keeps its fused call, the way it always has.
  if (IsOrdinaryMatrixPair(mat1, mat2)) {
    // `addmm.dtype` has no variant entry, so its route is the key the
    // dispatcher answers for, asked of it by name.
    const auto output_dtype = DefaultOutputDtype(mat1, out_dtype);
    const char* const key = addmm_dispatcher.OpKey();
    const AddmmCall call{addmm_dispatcher};
    const auto result =
        call.Call(call.Select(key), self, mat1, mat2, beta, alpha);
    return output_dtype == result.scalar_type() ? result
                                                : result.to(output_dtype);
  }
  return RunMatrix("addmm", mat1, mat2, out_dtype, 2, self, beta, alpha);
}

at::Tensor& AddmmDtypeOut(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    c10::ScalarType out_dtype,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  auto result = AddmmDtype(self, mat1, mat2, out_dtype, beta, alpha);
  CheckOutShape(out, result, "addmm.dtype_out");
  out.copy_(result);
  return out;
}

at::Tensor& AddmmOut(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha,
    at::Tensor& out) {
  return RunMatrixOut("addmm.out", mat1, mat2, out.scalar_type(), 2, self, beta,
                      alpha, out);
}

at::Tensor& AddmmInplace(
    at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {
  return RunMatrixOut("addmm_", mat1, mat2, self.scalar_type(), 2, self, beta,
                      alpha, self);
}

} // namespace at::native::flagos::soft_lowp
