"""CANN matrix-factorization templates with ATen shape and dtype contracts."""

HELPERS = r"""
namespace ascend {
static void FillMatrixIdentity(at::Tensor& matrix) {
  matrix.zero_();
  auto diagonal = matrix.diagonal(0, -2, -1);
  diagonal.copy_(at::ones(diagonal.sizes(), matrix.options()));
}

// Scaling and squaring with a degree-18 Taylor polynomial. All matrix data
// stays on the device in its original precision; only the norm bound is read
// for the scaling decision. The entrywise 1-norm bounds the matrix 1-norm.
static at::Tensor MatrixExp(const at::Tensor& input) {
  TORCH_CHECK(input.dim() >= 2 && input.size(-1) == input.size(-2),
              "linalg.matrix_exp: expected batches of square matrices");
  TORCH_CHECK(input.is_floating_point() || input.is_complex(),
              "linalg.matrix_exp: expected floating or complex input");
  if (input.numel() == 0) return input.clone();
  const auto original_dtype = input.scalar_type();
  auto source = (original_dtype == at::kHalf || original_dtype == at::kBFloat16)
      ? input.to(at::kFloat) : input;
  const int64_t n = input.size(-1);
  auto matrices = source.reshape({-1, n, n});
  std::vector<at::Tensor> outputs;
  outputs.reserve(matrices.size(0));
  for (int64_t index = 0; index < matrices.size(0); ++index) {
    auto matrix = matrices[index];
    const double bound = at::abs(matrix).sum().item<double>();
    if (!std::isfinite(bound)) {
      outputs.push_back(at::full_like(matrix, std::numeric_limits<double>::quiet_NaN()));
      continue;
    }
    const int scaling = bound > 0.5
        ? static_cast<int>(std::ceil(std::log2(bound) + 1.0)) : 0;
    auto scaled = matrix * std::ldexp(1.0, -scaling);
    auto identity = at::empty_like(matrix);
    FillMatrixIdentity(identity);
    auto result = identity.clone();
    auto term = identity;
    for (int degree = 1; degree <= 18; ++degree) {
      term = at::mm(term, scaled) / degree;
      result = result + term;
    }
    for (int step = 0; step < scaling; ++step) result = at::mm(result, result);
    outputs.push_back(result);
  }
  return at::stack(outputs).reshape(input.sizes()).to(original_dtype);
}
} // namespace ascend

"""

TEMPLATES = {
    "matrix_exp": """\
at::Tensor {kernel}(const at::Tensor& self) {{
  return ascend::MatrixExp(self);
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
""",
    "matrix_exp_out": """\
at::Tensor& {kernel}(const at::Tensor& self, at::Tensor& out) {{
  TORCH_CHECK(out.device() == self.device(), "matrix_exp.out: expected the same device");
  TORCH_CHECK(at::canCast(self.scalar_type(), out.scalar_type()),
              "matrix_exp.out: result cannot be cast to the output dtype");
  auto result = ascend::MatrixExp(self);
  out.resize_(result.sizes());
  out.copy_(result);
  return out;
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
""",
    "linalg_qr": """\
std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& A, c10::string_view mode) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(A.dim() >= 2, "linalg.qr: input must have at least two dimensions");
  TORCH_CHECK(mode == "reduced" || mode == "complete" || mode == "r",
              "linalg.qr: expected reduced, complete, or r mode");
  TORCH_CHECK(A.is_floating_point() || A.is_complex(), "linalg.qr: expected floating or complex input");
  const int64_t m = A.size(-2), n = A.size(-1), k = std::min(m, n);
  int64_t mode_value = mode == "complete" ? 1 : (mode == "r" ? 2 : 0);
  auto q_shape = A.sizes().vec();
  q_shape.back() = mode == "complete" ? m : k;
  if (mode == "r") q_shape = {{0}};
  auto r_shape = A.sizes().vec();
  r_shape[r_shape.size() - 2] = mode == "complete" ? m : k;
  auto q = at::empty(q_shape, A.options());
  auto r = at::empty(r_shape, A.options());
  if (A.numel() == 0) {{
    if (mode == "complete") ascend::FillMatrixIdentity(q);
    r.zero_();
    return std::make_tuple(q, r);
  }}
  ascend::AclTensorWrapper input(A), q_out(q), r_out(r);
  EXEC_ASCEND_CMD({aclnn}, input.get(), mode_value, q_out.get(), r_out.get());
  return std::make_tuple(q, r);
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
""",
    "linalg_svd": """\
std::tuple<at::Tensor, at::Tensor, at::Tensor> {kernel}(
    const at::Tensor& A, bool full_matrices, bool compute_uv,
    std::optional<c10::string_view> driver) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(A.dim() >= 2, "linalg.svd: input must have at least two dimensions");
  TORCH_CHECK(A.is_floating_point() || A.is_complex(), "linalg.svd: expected floating or complex input");
  TORCH_CHECK(!driver.has_value(), "linalg.svd: CUDA driver selection is unavailable on Ascend");
  const int64_t m = A.size(-2), n = A.size(-1), k = std::min(m, n);
  auto s_shape = A.sizes().vec();
  s_shape.pop_back();
  s_shape.back() = k;
  auto u_shape = A.sizes().vec();
  u_shape.back() = full_matrices ? m : k;
  auto v_shape = A.sizes().vec();
  v_shape[v_shape.size() - 2] = n;
  v_shape.back() = full_matrices ? n : k;
  if (!compute_uv) {{ u_shape = {{0}}; v_shape = {{0}}; }}
  auto sigma = at::empty(s_shape, A.options());
  auto u = at::empty(u_shape, A.options());
  auto v = at::empty(v_shape, A.options());
  if (A.numel() == 0) {{
    if (compute_uv && full_matrices) {{
      ascend::FillMatrixIdentity(u);
      ascend::FillMatrixIdentity(v);
    }}
  }} else {{
    ascend::AclTensorWrapper input(A), s_out(sigma), u_out(u), v_out(v);
    EXEC_ASCEND_CMD({aclnn}, input.get(), full_matrices, compute_uv,
                    s_out.get(), u_out.get(), v_out.get());
  }}
  auto singular_values = A.is_complex() ? at::real(sigma).contiguous() : sigma;
  auto vh = compute_uv ? v.mH() : v;
  return std::make_tuple(u, singular_values, vh);
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
""",
}

# Identity factories also serve MPS canonicalization. Construct the diagonal
# directly on the device, retaining complex/FP64 storage and strided out tensors.
for _category, _rectangular, _out in [
    ("eye", False, False),
    ("eye_m", True, False),
    ("eye_out", False, True),
    ("eye_m_out", True, True),
]:
    _parameters = "int64_t n, " + ("int64_t m, " if _rectangular else "")
    _parameters += (
        "at::Tensor& out"
        if _out
        else (
            "std::optional<at::ScalarType> dtype, std::optional<at::Layout> layout, "
            "std::optional<at::Device> device, std::optional<bool> pin_memory"
        )
    )
    _body = "" if _rectangular else "  const int64_t m = n;\n"
    _body += (
        '  TORCH_CHECK(n >= 0 && m >= 0, "eye: dimensions must be non-negative");\n'
    )
    if _out:
        _body += "  out.resize_({{n, m}});\n"
    else:
        _body += """  auto options = at::TensorOptions()
    .dtype(dtype.value_or(c10::get_default_dtype_as_scalartype()))
    .layout(layout.value_or(at::kStrided))
    .device(device.value_or(at::Device(at::kPrivateUse1, 0)))
    .pinned_memory(pin_memory.value_or(false));
  auto out = at::empty({{n, m}}, options);
"""
    _body += "  ascend::FillMatrixIdentity(out);\n  return out;\n"
    TEMPLATES[_category] = (
        ("at::Tensor& " if _out else "at::Tensor ")
        + "{kernel}("
        + _parameters
        + ") {{\n"
        "  namespace ascend = at::native::flagos::ascend;\n" + _body + "}}\n"
        "REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})\n"
    )

TEMPLATES["solve_triangular"] = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& B,
    bool upper, bool left, bool unitriangular) {{
  namespace ascend = at::native::flagos::ascend;
  TORCH_CHECK(self.dim() >= 2 && B.dim() >= 2, "linalg.solve_triangular: expected matrices");
  TORCH_CHECK(self.size(-1) == self.size(-2), "linalg.solve_triangular: A must be square");
  TORCH_CHECK(self.scalar_type() == B.scalar_type(), "linalg.solve_triangular: dtypes must match");
  TORCH_CHECK(self.device() == B.device(), "linalg.solve_triangular: devices must match");
  TORCH_CHECK(self.size(-1) == B.size(left ? -2 : -1),
              "linalg.solve_triangular: incompatible matrix dimensions");
  auto batch = at::infer_size(self.sizes().slice(0, self.dim() - 2),
                              B.sizes().slice(0, B.dim() - 2));
  auto a_shape = batch; a_shape.push_back(self.size(-2)); a_shape.push_back(self.size(-1));
  auto b_shape = batch; b_shape.push_back(B.size(-2)); b_shape.push_back(B.size(-1));
  auto a = self.resolve_conj().resolve_neg().expand(a_shape).contiguous();
  auto b = B.resolve_conj().resolve_neg().expand(b_shape).contiguous();
  if (!left) {{ a = a.transpose(-2, -1).contiguous(); b = b.transpose(-2, -1).contiguous(); }}
  // CANN's unitriangular variant rejects Double/complex on this SDK.
  // Materialize the implicit diagonal without modifying caller-owned A.
  if (unitriangular) {{
    a = a.clone();
    auto diagonal = a.diagonal(0, -2, -1);
    diagonal.copy_(at::ones(diagonal.sizes(), a.options()));
  }}
  auto out = at::empty(b.sizes(), b.options());
  if (out.numel() == 0) return left ? out : out.transpose(-2, -1);
  auto a_copy = at::empty(a.sizes(), a.options());
  ascend::AclTensorWrapper a_in(a), b_in(b), result(out), a_out(a_copy);
  EXEC_ASCEND_CMD({aclnn}, b_in.get(), a_in.get(), left ? upper : !upper,
                  false, false, result.get(), a_out.get());
  return left ? out : out.transpose(-2, -1);
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

TEMPLATES["solve_triangular_out"] = """\
at::Tensor& {kernel}(const at::Tensor& self, const at::Tensor& B,
    bool upper, bool left, bool unitriangular, at::Tensor& out) {{
  TORCH_CHECK(out.scalar_type() == self.scalar_type(), "linalg.solve_triangular: out dtype must match");
  TORCH_CHECK(out.device() == self.device(), "linalg.solve_triangular: out device must match");
  auto result = at::linalg_solve_triangular(self, B, upper, left, unitriangular);
  out.resize_(result.sizes());
  out.copy_(result);
  return out;
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# Shift a normalized Hermitian matrix into the positive-definite cone. Its SVD
# then supplies its orthonormal eigenbasis, with eigenvalues shifted back. The
# row-sum bound makes the shifted spectrum lie in [1, 3] for either input sign.
# All matrix work remains on-device and retains the original component dtype.
TEMPLATES["linalg_eigh"] = """\
std::tuple<at::Tensor, at::Tensor> {kernel}(const at::Tensor& A,
    c10::string_view UPLO, bool compute_v) {{
  TORCH_CHECK(A.dim() >= 2 && A.size(-1) == A.size(-2), "linalg.eigh: expected square matrices");
  TORCH_CHECK(A.scalar_type() == at::kFloat || A.scalar_type() == at::kDouble ||
              A.scalar_type() == at::kComplexFloat || A.scalar_type() == at::kComplexDouble,
              "linalg.eigh: expected float32, float64, complex64 or complex128");
  const bool lower = UPLO == "L" || UPLO == "l";
  TORCH_CHECK(lower || UPLO == "U" || UPLO == "u", "linalg.eigh: UPLO must be L or U");
  auto value_shape = A.sizes().vec(); value_shape.pop_back();
  auto values = at::empty(value_shape, A.options().dtype(c10::toRealValueType(A.scalar_type())));
  auto vectors = compute_v ? at::empty(A.sizes(), A.options()) : at::empty({{0}}, A.options());
  if (A.numel() == 0) return std::make_tuple(values, vectors);
  auto triangle = lower ? at::tril(A, -1) : at::triu(A, 1);
  auto hermitian = triangle + triangle.mH();
  auto diagonal = hermitian.diagonal(0, -2, -1);
  diagonal.copy_(at::real(A.diagonal(0, -2, -1)));
  auto scale = at::amax(at::sum(at::abs(hermitian), {{-1}}, true), {{-2}}, true);
  TORCH_CHECK(at::all(scale < std::numeric_limits<double>::infinity()).item<bool>(),
              "linalg.eigh: input must contain finite values");
  scale = at::where(scale == 0, at::ones_like(scale), scale);
  auto shifted = hermitian / scale + 2 * at::eye(A.size(-1), A.options());
  auto [u, singular, vh] = at::_linalg_svd(shifted, false, true, std::nullopt);
  values = (at::flip(singular, {{-1}}) - 2) * scale.squeeze(-1);
  if (compute_v) vectors = at::flip(u, {{-1}});
  return std::make_tuple(values, vectors);
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

# Device reductions needed by matrix-free dynamics, with strided/lazy views
# handled by the existing complex arithmetic and metadata kernels.
for _name in ("vdot", "trace"):
    _args = "const at::Tensor& self" + (
        ", const at::Tensor& other" if _name == "vdot" else ""
    )
    _check = (
        '  TORCH_CHECK(self.dim() == 1 && other.dim() == 1 && self.numel() == other.numel(), "vdot: expected equal-sized vectors");\n'
        '  TORCH_CHECK(self.device() == other.device() && self.scalar_type() == other.scalar_type(), "vdot: expected matching devices and dtypes");\n'
        if _name == "vdot"
        else '  TORCH_CHECK(self.dim() == 2, "trace: expected a matrix");\n'
    )
    _check += '  TORCH_CHECK(self.scalar_type() != at::kBool, "reduction: bool input is unsupported");\n'
    _dtype = (
        "self.scalar_type()"
        if _name == "vdot"
        else "(self.is_floating_point() || self.is_complex() ? self.scalar_type() : at::kLong)"
    )
    _value = (
        "at::sum(at::conj(self) * other, self.scalar_type())"
        if _name == "vdot"
        else "at::sum(self.diagonal(), " + _dtype + ")"
    )
    TEMPLATES["device_" + _name] = (
        "at::Tensor {kernel}("
        + _args
        + ") {{\n"
        + _check
        + "  return "
        + _value
        + ";\n}}\n"
        + "REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})\n"
    )
    TEMPLATES["device_" + _name + "_out"] = (
        "at::Tensor& {kernel}("
        + _args
        + ", at::Tensor& out) {{\n"
        + _check
        + "  TORCH_CHECK(out.device() == self.device() && out.scalar_type() == "
        + _dtype
        + ', "reduction.out: expected matching device and dtype");\n'
        + "  auto value = "
        + _value
        + ";\n  out.resize_({{}});\n  out.copy_(value);\n  return out;\n}}\n"
        + "REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})\n"
    )

# Logit and its backward retain the ATen clamp/domain semantics in device ops.
_HELPER_LOGIT = r"""
namespace ascend {
static at::Tensor DeviceLogit(const at::Tensor& self, std::optional<double> eps) {
  TORCH_CHECK(!self.is_complex(), "logit: complex input is unsupported");
  auto x = self.is_floating_point() ? self : self.to(at::kFloat);
  if (eps.has_value() && eps.value() >= 0) {
    auto low = at::full_like(x, eps.value());
    auto high = at::full_like(x, 1 - eps.value());
    // PyTorch 2.10's aarch64 CPU reference uses vectorized FP64 clipping
    // and scalar FP32 clipping; keep its inverted-bound behavior observable.
    x = x.scalar_type() == at::kDouble
        ? at::minimum(at::maximum(x, low), high)
        : at::where(x < eps.value(), low, at::where(x > 1 - eps.value(), high, x));
  }
  return at::log(x / (1 - x));
}
static at::Tensor DeviceLogitBackward(const at::Tensor& grad, const at::Tensor& self,
                                     std::optional<double> eps) {
  TORCH_CHECK(self.is_floating_point(), "logit_backward: expected floating input");
  auto value = grad / (self * (1 - self));
  const double bound = eps.value_or(-1);
  if (bound < 0) {
    return at::where((self < 0) | (self > 1),
        at::full_like(value, std::numeric_limits<double>::quiet_NaN()), value);
  }
  // The same pinned reference returns zero for a clamped FP64 NaN and
  // propagates the NaN in FP32. Neither choice changes finite-input gradients.
  if (self.scalar_type() == at::kDouble) {
    return at::where((self >= bound) & (self <= 1 - bound), value, at::zeros_like(value));
  }
  return at::where((self < bound) | (self > 1 - bound), at::zeros_like(value), value);
}
} // namespace ascend
"""
HELPERS += _HELPER_LOGIT
for _name, _backward, _out, _inplace in [
    ("logit_device", False, False, False),
    ("logit_device_out", False, True, False),
    ("logit_device_inplace", False, False, True),
    ("logit_device_backward", True, False, False),
    ("logit_device_backward_out", True, True, False),
]:
    _args = "const at::Tensor& grad, " if _backward else ""
    _args += (
        "at::Tensor& self" if _inplace else "const at::Tensor& self"
    ) + ", std::optional<double> eps"
    if _out:
        _args += ", at::Tensor& out"
    _call = (
        "ascend::DeviceLogitBackward(grad, self, eps)"
        if _backward
        else "ascend::DeviceLogit(self, eps)"
    )
    _body = "  auto value = " + _call + ";\n"
    if _inplace:
        _body = (
            '  TORCH_CHECK(self.is_floating_point(), "logit_: expected floating input");\n'
            + _body
        )
        _body += "  self.copy_(value);\n  return self;\n"
    elif _out:
        _body += '  TORCH_CHECK(out.device() == value.device() && at::canCast(value.scalar_type(), out.scalar_type()), "logit.out: incompatible output");\n'
        _body += "  out.resize_(value.sizes());\n  out.copy_(value);\n  return out;\n"
    else:
        _body += "  return value;\n"
    TEMPLATES[_name] = (
        ("at::Tensor& " if _out or _inplace else "at::Tensor ")
        + "{kernel}("
        + _args
        + ") {{\n"
        + _body
        + "}}\n"
        + "REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})\n"
    )

TEMPLATES["remainder_scalar"] = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Scalar& other) {{
  auto dtype = at::result_type(self, other);
  auto compute_dtype = c10::isIntegralType(dtype, false) && c10::elementSize(dtype) < 4 ? at::kInt : dtype;
  auto source = self.to(compute_dtype);
  auto out = at::empty(self.sizes(), self.options().dtype(dtype));
  ascend::AclTensorWrapper input(source), output(out);
  ascend::AclScalarWrapper scalar(other, compute_dtype);
  EXEC_ASCEND_CMD({aclnn}, input.get(), scalar.get(), output.get());
  return out;
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""

TEMPLATES["remainder_tensor"] = """\
at::Tensor {kernel}(const at::Tensor& self, const at::Tensor& other) {{
  TORCH_CHECK(self.device() == other.device(), "remainder: expected the same device");
  auto dtype = at::result_type(self, other);
  auto compute_dtype = c10::isIntegralType(dtype, false) && c10::elementSize(dtype) < 4 ? at::kInt : dtype;
  auto shape = at::infer_size(self.sizes(), other.sizes());
  auto left = self.to(compute_dtype).expand(shape).contiguous();
  auto right = other.to(compute_dtype).expand(shape).contiguous();
  auto out = at::empty(shape, self.options().dtype(dtype));
  ascend::AclTensorWrapper a(left), b(right), result(out);
  EXEC_ASCEND_CMD({aclnn}, a.get(), b.get(), result.get());
  return out;
}}
REGISTER_IMPL_TO_DISPATCHER({fn}, {disp}, Backend::kAscend, {kernel})
"""
