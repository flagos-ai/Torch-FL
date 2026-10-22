// Copyright 2026 FlagOS Contributors
//
// Native Moore Threads MUSA RNG kernels backed by muRAND and mudnn Dropout.

#include "../../generated/ops.h"
#include "mudnn_common.h"

#include <ATen/core/Tensor.h>
#include <c10/core/Device.h>
#include <murand.h>

#include <limits>
#include <mutex>
#include <optional>
#include <unordered_map>

#include "runtime/generator.h"

namespace at::native::flagos {
namespace {

using musa_ops::MusaDeviceGuard;

uint64_t next_seed(const at::Tensor& tensor,
                   const std::optional<at::Generator>& generator) {
  return c10::flagos::ReserveSeed(
      generator, static_cast<c10::DeviceIndex>(tensor.device().index()));
}

std::mutex& murand_operation_mutex() {
  static std::mutex mutex;
  return mutex;
}

const char* murand_status_name(murandStatus_t status) {
  switch (status) {
    case MURAND_STATUS_SUCCESS: return "SUCCESS";
    case MURAND_STATUS_NOT_INITIALIZED: return "NOT_INITIALIZED";
    case MURAND_STATUS_ALLOCATION_FAILED: return "ALLOCATION_FAILED";
    case MURAND_STATUS_TYPE_ERROR: return "TYPE_ERROR";
    case MURAND_STATUS_OUT_OF_RANGE: return "OUT_OF_RANGE";
    case MURAND_STATUS_LENGTH_NOT_MULTIPLE: return "LENGTH_NOT_MULTIPLE";
    case MURAND_STATUS_LAUNCH_FAILURE: return "LAUNCH_FAILURE";
    case MURAND_STATUS_PREEXISTING_FAILURE: return "PREEXISTING_FAILURE";
    case MURAND_STATUS_ARCH_MISMATCH: return "ARCH_MISMATCH";
    case MURAND_STATUS_NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    default: return "UNKNOWN";
  }
}

void check_murand(murandStatus_t status, const char* op) {
  TORCH_CHECK(
      status == MURAND_STATUS_SUCCESS, op, " failed: ", murand_status_name(status));
}

murandGenerator_t get_murand_generator(const at::Tensor& tensor, uint64_t seed) {
  static std::mutex mutex;
  static std::unordered_map<int, murandGenerator_t> generators;

  MusaDeviceGuard guard(tensor);
  int device = static_cast<int>(tensor.device().index());
  if (device < 0) {
    device = static_cast<int>(c10::flagos::CurrentDevice());
  }
  std::lock_guard<std::mutex> lock(mutex);
  auto it = generators.find(device);
  if (it == generators.end()) {
    murandGenerator_t generator = nullptr;
    check_murand(
        murandCreateGenerator(&generator, MURAND_RNG_PSEUDO_PHILOX4_32_10),
        "murandCreateGenerator");
    it = generators.emplace(device, generator).first;
  }
  auto generator = it->second;
  check_murand(
      murandSetStream(
          generator, musa::GetDefaultMusaStream()),
      "murandSetStream");
  check_murand(
      murandSetPseudoRandomGeneratorSeed(generator, seed),
      "murandSetPseudoRandomGeneratorSeed");
  check_murand(murandSetGeneratorOffset(generator, 0), "murandSetGeneratorOffset");
  return generator;
}

at::Tensor make_empty(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                      std::optional<at::Layout> layout,
                      std::optional<at::Device> device,
                      std::optional<bool> pin_memory) {
  auto target = device.value_or(at::Device(
      c10::DeviceType::PrivateUse1, c10::flagos::CurrentDevice()));
  TORCH_CHECK(target.is_privateuseone(), "MUSA RNG requires a flagos device, got ", target);
  return at::empty(
      size,
      at::TensorOptions()
          .dtype(dtype.value_or(at::kFloat))
          .layout(layout.value_or(at::kStrided))
          .device(target)
          .pinned_memory(pin_memory.value_or(false)));
}

void generate_uniform_raw(const at::Tensor& out, uint64_t seed) {
  MusaDeviceGuard guard(out);
  TORCH_CHECK(out.is_contiguous(), "MUSA RNG output must be contiguous");
  TORCH_CHECK(
      out.scalar_type() == at::kFloat || out.scalar_type() == at::kDouble,
      "internal uniform output must be float or double");
  if (out.numel() == 0) return;
  std::lock_guard<std::mutex> lock(murand_operation_mutex());
  auto generator = get_murand_generator(out, seed);
  murandStatus_t status;
  if (out.scalar_type() == at::kDouble) {
    status = murandGenerateUniformDouble(
        generator, out.data_ptr<double>(), static_cast<size_t>(out.numel()));
  } else {
    status = murandGenerateUniform(
        generator, out.data_ptr<float>(), static_cast<size_t>(out.numel()));
  }
  check_murand(status, "murandGenerateUniform");
  TORCH_CHECK(musaStreamSynchronize(musa::GetDefaultMusaStream()) == musaSuccess,
              "murand uniform stream sync failed");
}

void generate_uniform(const at::Tensor& out, double from, double to,
                      uint64_t seed) {
  MusaDeviceGuard guard(out);
  TORCH_CHECK(out.is_floating_point(), "uniform_ expects a floating-point tensor");
  if (out.numel() == 0) return;
  auto work_dtype = out.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  auto work = out.scalar_type() == work_dtype && out.is_contiguous()
      ? out
      : at::empty(out.sizes(), out.options().dtype(work_dtype));
  generate_uniform_raw(work, seed);
  // muRAND returns values in (0, 1], while PyTorch requires [0, 1).
  // Remainder preserves every interior sample and maps only 1 to 0.
  work.remainder_(1.0);
  if (from != 0.0 || to != 1.0) {
    work.mul_(to - from).add_(from);
  }
  if (work.data_ptr() != out.data_ptr()) out.copy_(work);
}

void generate_normal_raw(const at::Tensor& out, double mean, double std,
                         uint64_t seed) {
  MusaDeviceGuard guard(out);
  TORCH_CHECK(out.is_floating_point(), "normal_ expects a floating-point tensor");
  if (out.numel() == 0) return;
  auto work_dtype = out.scalar_type() == at::kDouble ? at::kDouble : at::kFloat;
  auto work = out.scalar_type() == work_dtype && out.is_contiguous() &&
          out.numel() % 2 == 0
      ? out
      : at::empty({out.numel() + (out.numel() % 2)},
                  out.options().dtype(work_dtype));
  std::lock_guard<std::mutex> lock(murand_operation_mutex());
  auto generator = get_murand_generator(work, seed);
  murandStatus_t status;
  if (work_dtype == at::kDouble) {
    status = murandGenerateNormalDouble(
        generator, work.data_ptr<double>(), static_cast<size_t>(work.numel()),
        mean, std);
  } else {
    status = murandGenerateNormal(
        generator, work.data_ptr<float>(), static_cast<size_t>(work.numel()),
        static_cast<float>(mean), static_cast<float>(std));
  }
  check_murand(status, "murandGenerateNormal");
  TORCH_CHECK(musaStreamSynchronize(musa::GetDefaultMusaStream()) == musaSuccess,
              "murand normal stream sync failed");
  if (work.data_ptr() != out.data_ptr()) {
    out.copy_(work.narrow(0, 0, out.numel()).view(out.sizes()));
  }
}

void generate_integer(const at::Tensor& out, int64_t base, uint64_t span,
                      uint64_t seed) {
  MusaDeviceGuard guard(out);
  if (out.numel() == 0) return;

  // S5000's Philox muRAND generator does not implement the native 64-bit
  // output entry point (it returns TYPE_ERROR). Generate two uint32 words and
  // combine them into a signed int64 tensor instead.
  auto raw = at::empty({static_cast<int64_t>(out.numel()) * 2},
                       out.options().dtype(at::kInt));
  {
    std::lock_guard<std::mutex> lock(murand_operation_mutex());
    auto generator = get_murand_generator(raw, seed);
    check_murand(
        murandGenerate(
            generator, reinterpret_cast<unsigned int*>(raw.data_ptr<int32_t>()),
            static_cast<size_t>(raw.numel())),
        "murandGenerate");
    TORCH_CHECK(musaStreamSynchronize(musa::GetDefaultMusaStream()) == musaSuccess,
                "murand integer stream sync failed");
  }

  constexpr int64_t kTwoTo32 = int64_t{1} << 32;
  auto low_word = raw.narrow(0, 0, out.numel()).to(at::kLong);
  low_word.remainder_(kTwoTo32);
  auto high_word = raw.narrow(0, out.numel(), out.numel()).to(at::kLong);

  at::Tensor work;
  if (span == 0) {
    // A zero uint64 span denotes the complete 2^64 int64 domain.
    work = high_word.mul_(kTwoTo32).add_(low_word);
  } else if (
      span <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
    // Use a non-negative 63-bit source so signed remainder has exactly the
    // unsigned modulo semantics required by random_[from, to).
    high_word.remainder_(int64_t{1} << 31);
    work = high_word.mul_(kTwoTo32).add_(low_word);
    work.remainder_(static_cast<int64_t>(span)).add_(base);
  } else {
    // With span > 2^63, a uint64 value needs at most one subtraction. In the
    // signed representation, values below span are non-negative or smaller
    // than span reinterpreted as int64.
    work = high_word.mul_(kTwoTo32).add_(low_word);
    const auto span_signed = static_cast<int64_t>(span);
    auto in_range = work.ge(0).logical_or(work.lt(span_signed));
    work = at::where(in_range, work, work - span_signed).add_(base);
  }
  out.copy_(work.view(out.sizes()));
}

std::pair<int64_t, int64_t> integer_bounds(at::ScalarType dtype) {
  switch (dtype) {
    case at::kByte: return {0, std::numeric_limits<uint8_t>::max()};
    case at::kChar:
      return {std::numeric_limits<int8_t>::lowest(),
              std::numeric_limits<int8_t>::max()};
    case at::kShort:
      return {std::numeric_limits<int16_t>::lowest(),
              std::numeric_limits<int16_t>::max()};
    case at::kInt:
      return {std::numeric_limits<int32_t>::lowest(),
              std::numeric_limits<int32_t>::max()};
    case at::kLong:
      return {std::numeric_limits<int64_t>::lowest(),
              std::numeric_limits<int64_t>::max()};
    default: TORCH_CHECK(false, "unsupported random_ dtype: ", dtype);
  }
}

uint64_t inclusive_span(int64_t low, int64_t high) {
  return static_cast<uint64_t>(high) - static_cast<uint64_t>(low) + 1;
}

std::pair<int64_t, uint64_t> default_integer_range(at::ScalarType dtype) {
  auto bounds = integer_bounds(dtype);
  return {0, static_cast<uint64_t>(bounds.second) + 1};
}

at::Tensor rand_kernel(at::IntArrayRef size, std::optional<at::Generator> generator,
                       std::optional<at::ScalarType> dtype,
                       std::optional<at::Layout> layout,
                       std::optional<at::Device> device,
                       std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  generate_uniform(out, 0.0, 1.0, next_seed(out, generator));
  return out;
}

at::Tensor randn_kernel(at::IntArrayRef size, std::optional<at::Generator> generator,
                        std::optional<at::ScalarType> dtype,
                        std::optional<at::Layout> layout,
                        std::optional<at::Device> device,
                        std::optional<bool> pin_memory) {
  auto out = make_empty(size, dtype, layout, device, pin_memory);
  generate_normal_raw(out, 0.0, 1.0, next_seed(out, generator));
  return out;
}

at::Tensor& random_inplace(at::Tensor& self, int64_t low, int64_t high,
                           const std::optional<at::Generator>& generator) {
  TORCH_CHECK(
      self.scalar_type() == at::kByte || self.scalar_type() == at::kChar ||
          self.scalar_type() == at::kShort || self.scalar_type() == at::kInt ||
          self.scalar_type() == at::kLong,
      "random_ expects an integral output");
  TORCH_CHECK(high > low, "random_ expects high > low, got ", low, " and ", high);
  auto bounds = integer_bounds(self.scalar_type());
  TORCH_CHECK(
      low >= bounds.first && high - 1 <= bounds.second,
      "random_ bounds [",
      low,
      ", ",
      high,
      ") are out of range for ",
      self.scalar_type());
  auto span = static_cast<uint64_t>(high) - static_cast<uint64_t>(low);
  generate_integer(self, low, span, next_seed(self, generator));
  return self;
}

at::Tensor& normal_inplace(at::Tensor& self, double mean, double std,
                           const std::optional<at::Generator>& generator) {
  TORCH_CHECK(std >= 0, "normal_ expects std >= 0, got ", std);
  generate_normal_raw(self, mean, std, next_seed(self, generator));
  return self;
}

at::Tensor& uniform_inplace(at::Tensor& self, double from, double to,
                            const std::optional<at::Generator>& generator) {
  TORCH_CHECK(from <= to, "uniform_ expects from <= to, got ", from, " and ", to);
  generate_uniform(self, from, to, next_seed(self, generator));
  return self;
}

at::Tensor rand_like_kernel(const at::Tensor& self, std::optional<at::Generator> generator,
                            std::optional<at::ScalarType> dtype,
                            std::optional<at::Layout> layout,
                            std::optional<at::Device> device,
                            std::optional<bool> pin_memory,
                            std::optional<at::MemoryFormat> memory_format) {
  auto out = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  generate_uniform(out, 0.0, 1.0, next_seed(out, generator));
  return out;
}

at::Tensor randn_like_kernel(const at::Tensor& self, std::optional<at::Generator> generator,
                             std::optional<at::ScalarType> dtype,
                             std::optional<at::Layout> layout,
                             std::optional<at::Device> device,
                             std::optional<bool> pin_memory,
                             std::optional<at::MemoryFormat> memory_format) {
  auto out = at::empty_like(self, dtype, layout, device, pin_memory, memory_format);
  generate_normal_raw(out, 0.0, 1.0, next_seed(out, generator));
  return out;
}

std::tuple<at::Tensor, at::Tensor> native_dropout_kernel(
    const at::Tensor& input, double p, std::optional<bool> train) {
  TORCH_CHECK(p >= 0.0 && p <= 1.0, "native_dropout p must be in [0, 1]");
  if (!train.value_or(true) || p == 0.0) {
    return {input.clone(), at::ones(input.sizes(), input.options().dtype(at::kBool))};
  }
  if (p == 1.0) {
    return {at::zeros_like(input), at::zeros(input.sizes(), input.options().dtype(at::kBool))};
  }

  auto out = at::empty_like(input);
  auto mask = at::empty(input.sizes(), input.options().dtype(at::kBool));
  MusaDeviceGuard guard(input);
  auto& handle = musa_ops::GetMudnnHandle();
  musa_ops::MudnnTensorWrapper t_input(input), t_out(out), t_mask(mask);
  musa_ops::mudnn::Dropout op;
  auto seed = next_seed(input, std::nullopt);
  TORCH_CHECK(op.SetP(p) == musa_ops::mudnn::Status::SUCCESS, "mudnn Dropout SetP failed");
  TORCH_CHECK(op.SetScale(1.0 / (1.0 - p)) == musa_ops::mudnn::Status::SUCCESS,
              "mudnn Dropout SetScale failed");
  TORCH_CHECK(op.SetSeed(seed) == musa_ops::mudnn::Status::SUCCESS,
              "mudnn Dropout SetSeed failed");
  TORCH_CHECK(op.SetOffset(0) == musa_ops::mudnn::Status::SUCCESS,
              "mudnn Dropout SetOffset failed");
  auto status = op.RunDropout(handle, t_out.get(), t_input.get(), t_mask.get());
  TORCH_CHECK(status == musa_ops::mudnn::Status::SUCCESS,
              "mudnn Dropout failed: ", musa_ops::MudnnStatusName(status));
  TORCH_CHECK(
      musaStreamSynchronize(handle.GetStream()) == musaSuccess,
      "mudnn Dropout stream sync failed");
  return {out, mask};
}

} // namespace

at::Tensor RandKernelMusa(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                          std::optional<at::Layout> layout, std::optional<at::Device> device,
                          std::optional<bool> pin_memory) {
  return rand_kernel(size, std::nullopt, dtype, layout, device, pin_memory);
}
at::Tensor RandGeneratorKernelMusa(at::IntArrayRef size, std::optional<at::Generator> generator,
                                   std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                                   std::optional<at::Device> device, std::optional<bool> pin_memory) {
  return rand_kernel(size, generator, dtype, layout, device, pin_memory);
}
at::Tensor RandnKernelMusa(at::IntArrayRef size, std::optional<at::ScalarType> dtype,
                           std::optional<at::Layout> layout, std::optional<at::Device> device,
                           std::optional<bool> pin_memory) {
  return randn_kernel(size, std::nullopt, dtype, layout, device, pin_memory);
}
at::Tensor RandnGeneratorKernelMusa(at::IntArrayRef size, std::optional<at::Generator> generator,
                                    std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                                    std::optional<at::Device> device, std::optional<bool> pin_memory) {
  return randn_kernel(size, generator, dtype, layout, device, pin_memory);
}

at::Tensor& NormalInplaceKernelMusa(at::Tensor& self, double mean, double std,
                                    std::optional<at::Generator> generator) {
  return normal_inplace(self, mean, std, generator);
}
at::Tensor& UniformInplaceKernelMusa(at::Tensor& self, double from, double to,
                                     std::optional<at::Generator> generator) {
  return uniform_inplace(self, from, to, generator);
}
at::Tensor& RandomInplaceKernelMusa(at::Tensor& self, std::optional<at::Generator> generator) {
  auto range = default_integer_range(self.scalar_type());
  generate_integer(self, range.first, range.second, next_seed(self, generator));
  return self;
}
at::Tensor& RandomInplaceToKernelMusa(at::Tensor& self, int64_t to,
                                      std::optional<at::Generator> generator) {
  return random_inplace(self, 0, to, generator);
}
at::Tensor& RandomInplaceFromKernelMusa(at::Tensor& self, int64_t from,
                                        std::optional<int64_t> to,
                                        std::optional<at::Generator> generator) {
  if (to.has_value()) {
    return random_inplace(self, from, *to, generator);
  }
  auto bounds = integer_bounds(self.scalar_type());
  TORCH_CHECK(
      from >= bounds.first && from <= bounds.second,
      "random_ lower bound ",
      from,
      " is out of range for ",
      self.scalar_type());
  generate_integer(
      self, from, inclusive_span(from, bounds.second), next_seed(self, generator));
  return self;
}

at::Tensor RandintLowGeneratorKernelMusa(
    int64_t low, int64_t high, at::IntArrayRef size,
    std::optional<at::Generator> generator,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  TORCH_CHECK(high > low, "randint expects high > low");
  auto out = make_empty(size, dtype.value_or(at::kLong), layout, device, pin_memory);
  return random_inplace(out, low, high, generator);
}

at::Tensor RandintKernelMusa(
    int64_t high, at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return RandintLowGeneratorKernelMusa(
      0, high, size, std::nullopt, dtype, layout, device, pin_memory);
}

at::Tensor RandintGeneratorKernelMusa(
    int64_t high, at::IntArrayRef size,
    std::optional<at::Generator> generator,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return RandintLowGeneratorKernelMusa(
      0, high, size, generator, dtype, layout, device, pin_memory);
}

at::Tensor RandintLowKernelMusa(
    int64_t low, int64_t high, at::IntArrayRef size,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory) {
  return RandintLowGeneratorKernelMusa(
      low, high, size, std::nullopt, dtype, layout, device, pin_memory);
}

at::Tensor RandLikeKernelMusa(const at::Tensor& self,
                               std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> layout,
                               std::optional<at::Device> device,
                               std::optional<bool> pin_memory,
                               std::optional<at::MemoryFormat> memory_format) {
  return rand_like_kernel(self, std::nullopt, dtype, layout, device, pin_memory, memory_format);
}
at::Tensor RandLikeGeneratorKernelMusa(const at::Tensor& self, std::optional<at::Generator> generator,
                                       std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                                       std::optional<at::Device> device, std::optional<bool> pin_memory,
                                       std::optional<at::MemoryFormat> memory_format) {
  return rand_like_kernel(self, generator, dtype, layout, device, pin_memory, memory_format);
}
at::Tensor RandnLikeKernelMusa(const at::Tensor& self,
                               std::optional<at::ScalarType> dtype,
                               std::optional<at::Layout> layout,
                               std::optional<at::Device> device,
                               std::optional<bool> pin_memory,
                               std::optional<at::MemoryFormat> memory_format) {
  return randn_like_kernel(self, std::nullopt, dtype, layout, device, pin_memory, memory_format);
}
at::Tensor RandnLikeGeneratorKernelMusa(const at::Tensor& self, std::optional<at::Generator> generator,
                                        std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout,
                                        std::optional<at::Device> device, std::optional<bool> pin_memory,
                                        std::optional<at::MemoryFormat> memory_format) {
  return randn_like_kernel(self, generator, dtype, layout, device, pin_memory, memory_format);
}
at::Tensor& RandOutKernelMusa(at::IntArrayRef size, at::Tensor& out) {
  out.resize_(size);
  return uniform_inplace(out, 0.0, 1.0, std::nullopt);
}
at::Tensor& RandnNamesOutKernelMusa(at::IntArrayRef size, std::optional<at::DimnameList>, at::Tensor& out) {
  out.resize_(size);
  return normal_inplace(out, 0.0, 1.0, std::nullopt);
}
at::Tensor& RandNamesOutKernelMusa(at::IntArrayRef size, std::optional<at::DimnameList>, at::Tensor& out) {
  return RandOutKernelMusa(size, out);
}
at::Tensor& RandLikeOutKernelMusa(const at::Tensor& self, std::optional<at::MemoryFormat>, at::Tensor& out) {
  out.resize_(self.sizes());
  return uniform_inplace(out, 0.0, 1.0, std::nullopt);
}
at::Tensor& RandnLikeOutKernelMusa(const at::Tensor& self, std::optional<at::MemoryFormat>, at::Tensor& out) {
  out.resize_(self.sizes());
  return normal_inplace(out, 0.0, 1.0, std::nullopt);
}
at::Tensor& RandintLowOutKernelMusa(int64_t low, int64_t high, at::IntArrayRef size, at::Tensor& out) {
  out.resize_(size);
  return random_inplace(out, low, high, std::nullopt);
}
at::Tensor& RandintOutKernelMusa(int64_t high, at::IntArrayRef size, at::Tensor& out) {
  return RandintLowOutKernelMusa(0, high, size, out);
}
at::Tensor NativeDropoutBackwardKernelMusa(const at::Tensor& grad_output, const at::Tensor& mask, double scale) {
  auto out = at::empty_like(grad_output);
  MusaDeviceGuard guard(grad_output);
  auto& handle = musa_ops::GetMudnnHandle();
  musa_ops::MudnnTensorWrapper t_out(out), t_grad(grad_output), t_mask(mask);
  musa_ops::mudnn::Dropout op;
  TORCH_CHECK(op.SetScale(scale) == musa_ops::mudnn::Status::SUCCESS,
              "mudnn Dropout SetScale failed");
  auto status = op.RunDropoutBwd(handle, t_out.get(), t_grad.get(), t_mask.get());
  TORCH_CHECK(status == musa_ops::mudnn::Status::SUCCESS,
              "mudnn Dropout backward failed: ", musa_ops::MudnnStatusName(status));
  TORCH_CHECK(
      musaStreamSynchronize(handle.GetStream()) == musaSuccess,
      "mudnn Dropout backward stream sync failed");
  return out;
}

REGISTER_IMPL_TO_DISPATCHER(RandFn, rand_dispatcher, Backend::kMusa, RandKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandGeneratorFn, rand_generator_dispatcher, Backend::kMusa, RandGeneratorKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandnFn, randn_dispatcher, Backend::kMusa, RandnKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandnGeneratorFn, randn_generator_dispatcher, Backend::kMusa, RandnGeneratorKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandOutFn, rand_out_dispatcher, Backend::kMusa, RandOutKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandNamesOutFn, rand_names_out_dispatcher, Backend::kMusa, RandNamesOutKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandnNamesOutFn, randn_names_out_dispatcher, Backend::kMusa, RandnNamesOutKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(NormalInplaceFn, normal_inplace_dispatcher, Backend::kMusa, NormalInplaceKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(UniformInplaceFn, uniform_inplace_dispatcher, Backend::kMusa, UniformInplaceKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceFn, random_inplace_dispatcher, Backend::kMusa, RandomInplaceKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceToFn, random_inplace_to_dispatcher, Backend::kMusa, RandomInplaceToKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandomInplaceFromFn, random_inplace_from_dispatcher, Backend::kMusa, RandomInplaceFromKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandintFn, randint_dispatcher, Backend::kMusa, RandintKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandintGeneratorFn, randint_generator_dispatcher, Backend::kMusa, RandintGeneratorKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandintLowFn, randint_low_dispatcher, Backend::kMusa, RandintLowKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandintLowGeneratorFn, randint_low_generator_dispatcher, Backend::kMusa, RandintLowGeneratorKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandLikeFn, rand_like_dispatcher, Backend::kMusa, RandLikeKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandLikeGeneratorFn, rand_like_generator_dispatcher, Backend::kMusa, RandLikeGeneratorKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeFn, randn_like_dispatcher, Backend::kMusa, RandnLikeKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeGeneratorFn, randn_like_generator_dispatcher, Backend::kMusa, RandnLikeGeneratorKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandLikeOutFn, rand_like_out_dispatcher, Backend::kMusa, RandLikeOutKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandnLikeOutFn, randn_like_out_dispatcher, Backend::kMusa, RandnLikeOutKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandintLowOutFn, randint_low_out_dispatcher, Backend::kMusa, RandintLowOutKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(RandintOutFn, randint_out_dispatcher, Backend::kMusa, RandintOutKernelMusa)
REGISTER_IMPL_TO_DISPATCHER(NativeDropoutFn, native_dropout_dispatcher, Backend::kMusa, native_dropout_kernel)
REGISTER_IMPL_TO_DISPATCHER(NativeDropoutBackwardFn, native_dropout_backward_dispatcher, Backend::kMusa, NativeDropoutBackwardKernelMusa)

} // namespace at::native::flagos
