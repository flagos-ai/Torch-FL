"""Generate CANN categorical sampling with the caller's flagos RNG state."""

TEMPLATES = {
    "multinomial": """\
at::Tensor {kernel}(const at::Tensor& self, int64_t num_samples,
    bool replacement, std::optional<at::Generator> generator) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(self.dim() == 1 || self.dim() == 2, "multinomial: expected one or two dimensions");
  TORCH_CHECK(self.is_floating_point(), "multinomial: expected floating probabilities");
  TORCH_CHECK(num_samples > 0, "multinomial: num_samples must be positive");
  TORCH_CHECK(self.size(-1) > 0 && self.size(-1) <= (1 << 24), "multinomial: invalid number of categories");
  TORCH_CHECK(replacement || num_samples <= self.size(-1), "multinomial: too many samples without replacement");
  auto valid = (self >= 0) & (self < std::numeric_limits<double>::infinity());
  TORCH_CHECK(at::all(valid).item<bool>(), "multinomial: probabilities must be finite and non-negative");
  TORCH_CHECK(at::all(at::sum(self, -1) > 0).item<bool>(), "multinomial: each distribution must have positive sum");
  auto shape = self.sizes().vec(); shape.back() = num_samples;
  auto out = at::empty(shape, self.options().dtype(at::kLong));
  auto seed = static_cast<int64_t>(c10::flagos::ReserveSeed(generator, self.device().index()));
  if (out.numel() == 0) return out;
  ascend::AclTensorWrapper input(self), result(out);
  EXEC_ASCEND_CMD({aclnn}, input.get(), num_samples, replacement, seed, int64_t(0), result.get());
  return out;
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
""",
    "multinomial_out": """\
at::Tensor& {kernel}(const at::Tensor& self, int64_t num_samples,
    bool replacement, std::optional<at::Generator> generator, at::Tensor& out) {{
  TORCH_CHECK(out.scalar_type() == at::kLong, "multinomial: out dtype must be int64");
  TORCH_CHECK(out.device() == self.device(), "multinomial: out device must match input");
  auto result = at::multinomial(self, num_samples, replacement, generator);
  out.resize_(result.sizes()); out.copy_(result);
  return out;
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
""",
}
