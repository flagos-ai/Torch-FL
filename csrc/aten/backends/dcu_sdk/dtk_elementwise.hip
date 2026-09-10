// Copyright (c) 2026, BAAI. All rights reserved.

// HIP device kernels for the DCU SDK-native elementwise operators, compiled by
// DTK's hipcc.
//
// This translation unit deliberately includes no ATen header. It is the reason
// the long tail of elementwise operators can scale at all: the DTK math and DNN
// libraries cover matmul, convolution, norms, FFT, RNG and linalg, but nothing
// there implements `gcd` or `nextafter`. Rather than leaving those on the vendor
// libtorch_hip, the generator emits a row in dtk_launch.h's op enum and hipcc
// compiles the arithmetic here -- so coverage is not capped by what the vendor
// libraries happen to export.
//
// Only plain data crosses in from the host side (see dtk_launch.h), so nothing
// in this file can pull in a dependency on the vendor's PyTorch build.

#include "dtk_launch.h"

#include <c10/util/Exception.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <type_traits>

namespace at::native::flagos::dtk_ops {

namespace {

constexpr int kBlockSize = 256;

inline int64_t GridFor(int64_t numel) {
  return (numel + kBlockSize - 1) / kBlockSize;
}

// Integral gcd, used by gcd.out. Euclid on the absolute values; gcd(0, 0) == 0,
// matching torch. Written iteratively because recursion in device code costs
// stack for no benefit here.
template <typename T>
__device__ __forceinline__ T GcdDevice(T a, T b) {
  // Careful with the most negative value: negating it overflows, but the modulo
  // chain below is correct on the raw two's-complement value anyway, so only
  // normalize after the loop.
  while (b != T(0)) {
    T t = a % b;
    a = b;
    b = t;
  }
  return a < T(0) ? -a : a;
}

// --- unary -----------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T ApplyUnaryFloat(T a, UnaryOp op) {
  switch (op) {
    case UnaryOp::ABS: return a < T(0) ? -a : a;
    case UnaryOp::ACOS: return ::acos(a);
    case UnaryOp::ACOSH: return ::acosh(a);
    case UnaryOp::ASIN: return ::asin(a);
    case UnaryOp::ATAN: return ::atan(a);
    case UnaryOp::ATANH: return ::atanh(a);
    case UnaryOp::COS: return ::cos(a);
    case UnaryOp::ERF: return ::erf(a);
    case UnaryOp::ERFC: return ::erfc(a);
    case UnaryOp::EXP2: return ::exp2(a);
    // torch's frac truncates toward zero: frac(-1.5) == -0.5.
    case UnaryOp::FRAC: return a - ::trunc(a);
    case UnaryOp::LOG: return ::log(a);
    case UnaryOp::LOG2: return ::log2(a);
  }
  return a;
}

template <typename T>
__global__ void UnaryFloatKernel(
    const T* __restrict__ in, T* __restrict__ out, int64_t numel, UnaryOp op) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < numel) {
    out[i] = ApplyUnaryFloat<T>(in[i], op);
  }
}

// abs is the only unary in the current set that is meaningful on integers.
template <typename T>
__global__ void AbsIntKernel(
    const T* __restrict__ in, T* __restrict__ out, int64_t numel) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < numel) {
    T a = in[i];
    out[i] = a < T(0) ? -a : a;
  }
}

// --- binary ----------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T ApplyBinaryInt(T a, T b, BinaryOp op) {
  switch (op) {
    case BinaryOp::BITWISE_AND_TENSOR: return a & b;
    case BinaryOp::BITWISE_OR_TENSOR: return a | b;
    case BinaryOp::BITWISE_XOR_TENSOR: return a ^ b;
    case BinaryOp::GCD: return GcdDevice<T>(a, b);
    default: return a;
  }
}

template <typename T>
__global__ void BinaryIntKernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    int64_t numel,
    BinaryOp op) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < numel) {
    out[i] = ApplyBinaryInt<T>(a[i], b[i], op);
  }
}

template <typename T>
__global__ void BinaryFloatKernel(
    const T* __restrict__ a,
    const T* __restrict__ b,
    T* __restrict__ out,
    int64_t numel,
    BinaryOp op) {
  int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < numel) {
    T x = a[i], y = b[i];
    switch (op) {
      case BinaryOp::HYPOT: out[i] = ::hypot(x, y); break;
      case BinaryOp::NEXTAFTER: out[i] = ::nextafter(x, y); break;
      default: out[i] = x; break;
    }
  }
}

bool IsBitwiseOrGcd(BinaryOp op) {
  return op == BinaryOp::BITWISE_AND_TENSOR ||
      op == BinaryOp::BITWISE_OR_TENSOR ||
      op == BinaryOp::BITWISE_XOR_TENSOR || op == BinaryOp::GCD;
}

} // namespace

void LaunchUnary(
    const void* in,
    void* out,
    int64_t numel,
    c10::ScalarType dtype,
    UnaryOp op,
    Stream_t stream) {
  if (numel == 0) return;
  const int64_t grid = GridFor(numel);
  hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);

  switch (dtype) {
    case c10::ScalarType::Float:
      hipLaunchKernelGGL(
          (UnaryFloatKernel<float>), dim3(grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const float*>(in), static_cast<float*>(out), numel, op);
      return;
    case c10::ScalarType::Double:
      hipLaunchKernelGGL(
          (UnaryFloatKernel<double>), dim3(grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const double*>(in), static_cast<double*>(out), numel, op);
      return;
    case c10::ScalarType::Int:
      TORCH_CHECK(
          op == UnaryOp::ABS,
          "dcu_sdk: this unary operator is not implemented for integral dtypes");
      hipLaunchKernelGGL(
          (AbsIntKernel<int32_t>), dim3(grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const int32_t*>(in), static_cast<int32_t*>(out), numel);
      return;
    case c10::ScalarType::Long:
      TORCH_CHECK(
          op == UnaryOp::ABS,
          "dcu_sdk: this unary operator is not implemented for integral dtypes");
      hipLaunchKernelGGL(
          (AbsIntKernel<int64_t>), dim3(grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const int64_t*>(in), static_cast<int64_t*>(out), numel);
      return;
    default:
      // Half/BFloat16/complex are not in this cohort yet. Fail loudly: a silent
      // fallthrough would write an unwritten output buffer.
      TORCH_CHECK(
          false,
          "dcu_sdk: unary kernel has no implementation for dtype ",
          dtype);
  }
}

void LaunchBinary(
    const void* a,
    const void* b,
    void* out,
    int64_t numel,
    c10::ScalarType dtype,
    BinaryOp op,
    Stream_t stream) {
  if (numel == 0) return;
  const int64_t grid = GridFor(numel);
  hipStream_t hip_stream = reinterpret_cast<hipStream_t>(stream);
  const bool int_op = IsBitwiseOrGcd(op);

  switch (dtype) {
    case c10::ScalarType::Int:
      TORCH_CHECK(
          int_op,
          "dcu_sdk: this binary operator is not implemented for integral dtypes");
      hipLaunchKernelGGL(
          (BinaryIntKernel<int32_t>), dim3(grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const int32_t*>(a), static_cast<const int32_t*>(b),
          static_cast<int32_t*>(out), numel, op);
      return;
    case c10::ScalarType::Long:
      TORCH_CHECK(
          int_op,
          "dcu_sdk: this binary operator is not implemented for integral dtypes");
      hipLaunchKernelGGL(
          (BinaryIntKernel<int64_t>), dim3(grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const int64_t*>(a), static_cast<const int64_t*>(b),
          static_cast<int64_t*>(out), numel, op);
      return;
    case c10::ScalarType::Bool:
      TORCH_CHECK(
          op != BinaryOp::GCD,
          "dcu_sdk: gcd is not defined for bool");
      TORCH_CHECK(
          int_op,
          "dcu_sdk: this binary operator is not implemented for bool");
      hipLaunchKernelGGL(
          (BinaryIntKernel<int8_t>), dim3(grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const int8_t*>(a), static_cast<const int8_t*>(b),
          static_cast<int8_t*>(out), numel, op);
      return;
    case c10::ScalarType::Float:
      TORCH_CHECK(
          !int_op,
          "dcu_sdk: bitwise/gcd operators require an integral dtype");
      hipLaunchKernelGGL(
          (BinaryFloatKernel<float>), dim3(grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const float*>(a), static_cast<const float*>(b),
          static_cast<float*>(out), numel, op);
      return;
    case c10::ScalarType::Double:
      TORCH_CHECK(
          !int_op,
          "dcu_sdk: bitwise/gcd operators require an integral dtype");
      hipLaunchKernelGGL(
          (BinaryFloatKernel<double>), dim3(grid), dim3(kBlockSize), 0, hip_stream,
          static_cast<const double*>(a), static_cast<const double*>(b),
          static_cast<double*>(out), numel, op);
      return;
    default:
      TORCH_CHECK(
          false,
          "dcu_sdk: binary kernel has no implementation for dtype ",
          dtype);
  }
}

} // namespace at::native::flagos::dtk_ops
