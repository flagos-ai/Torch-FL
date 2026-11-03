"""Generate complex arithmetic from device-resident real operations."""

HELPERS = r"""
namespace ascend {

static std::pair<at::Tensor, at::Tensor> ComplexParts(const at::Tensor& tensor) {
  if (!tensor.is_complex()) return {tensor, at::zeros_like(tensor)};
  auto storage = tensor;
  if (tensor.is_conj()) storage = at::native::_conj(storage);
  if (tensor.is_neg()) storage = at::native::_neg_view(storage);
  auto parts = at::native::view_as_real(storage);
  // Real kernels may be selected from either ACLNN or FlagGems. Normalize
  // interleaved component strides before arithmetic, including offset views.
  auto real = parts.select(-1, 0).clone(at::MemoryFormat::Contiguous);
  auto imag = parts.select(-1, 1).clone(at::MemoryFormat::Contiguous);
  if (tensor.is_neg()) { real = at::neg(real); imag = at::neg(imag); }
  if (tensor.is_conj()) imag = at::neg(imag);
  return {real, imag};
}

static at::Tensor PackComplex(const at::Tensor& real, const at::Tensor& imag) {
  TORCH_CHECK(real.scalar_type() == imag.scalar_type(),
              "complex: real and imaginary dtypes must match");
  TORCH_CHECK(real.device() == imag.device(),
              "complex: real and imaginary devices must match");
  TORCH_CHECK(real.scalar_type() == at::kFloat ||
              real.scalar_type() == at::kDouble || real.scalar_type() == at::kHalf,
              "complex: expected float32, float64 or float16 components");
  auto shape = at::infer_size(real.sizes(), imag.sizes());
  auto output = at::empty(shape, real.options().dtype(c10::toComplexType(real.scalar_type())));
  auto parts = at::native::view_as_real(output);
  parts.select(-1, 0).copy_(real.expand(shape));
  parts.select(-1, 1).copy_(imag.expand(shape));
  return output;
}

static at::Tensor ComplexMagnitude(const at::Tensor& real, const at::Tensor& imag) {
  auto high = at::maximum(at::abs(real), at::abs(imag));
  auto low = at::minimum(at::abs(real), at::abs(imag));
  auto ratio = low / high;
  auto result = high * at::sqrt(1 + ratio * ratio);
  result = at::where(high == 0, at::zeros_like(result), result);
  return at::where(high == std::numeric_limits<double>::infinity(), high, result);
}

// ACLNN GEMM has no FP64 instantiation. Retain FP64 arithmetic and storage
// through real device operations instead of demoting or returning to the host.
// This correctness path bounds intermediate storage independently of K.
at::Tensor RealMatmulDouble(const at::Tensor& left, const at::Tensor& right) {
  TORCH_CHECK(left.dim() > 0 && right.dim() > 0,
              "matmul: both arguments must be at least one-dimensional");
  const bool left_vector = left.dim() == 1;
  const bool right_vector = right.dim() == 1;
  auto a = left_vector ? left.unsqueeze(0) : left;
  auto b = right_vector ? right.unsqueeze(-1) : right;
  TORCH_CHECK(a.size(-1) == b.size(-2), "matmul: incompatible contraction sizes");
  auto shape = at::infer_size(a.sizes().slice(0, a.dim() - 2),
                              b.sizes().slice(0, b.dim() - 2));
  shape.push_back(a.size(-2));
  shape.push_back(b.size(-1));
  auto output = at::zeros(shape, a.options());
  for (int64_t k = 0; k < a.size(-1); ++k) {
    output = output + a.select(-1, k).unsqueeze(-1) *
                      b.select(-2, k).unsqueeze(-2);
  }
  if (left_vector) output = output.squeeze(-2);
  if (right_vector) output = output.squeeze(-1);
  return output;
}

at::Tensor ComplexIndex(const at::Tensor& self,
    const c10::List<std::optional<at::Tensor>>& indices) {
  auto [real, imag] = ComplexParts(self);
  return PackComplex(at::index(real, indices), at::index(imag, indices));
}

at::Tensor ComplexMatmul(const at::Tensor& left, const at::Tensor& right) {
  TORCH_CHECK(left.dim() > 0 && right.dim() > 0,
              "matmul: both arguments must be at least one-dimensional");
  TORCH_CHECK(left.scalar_type() == right.scalar_type(),
              "matmul: input dtypes must match");
  TORCH_CHECK(left.device() == right.device(), "matmul: input devices must match");
  auto [ar, ai] = ComplexParts(left);
  auto [br, bi] = ComplexParts(right);
  auto multiply = [](const at::Tensor& a, const at::Tensor& b) {
    return a.scalar_type() == at::kDouble ? RealMatmulDouble(a, b) : at::matmul(a, b);
  };
  return PackComplex(multiply(ar, br) - multiply(ai, bi),
                     multiply(ar, bi) + multiply(ai, br));
}

static at::ScalarType VectorNormDtype(
    const at::Tensor& self, std::optional<at::ScalarType> dtype) {
  auto target = dtype.value_or(self.scalar_type());
  TORCH_CHECK(self.is_floating_point() || self.is_complex(),
              "linalg.vector_norm: expected floating or complex input");
  TORCH_CHECK(c10::isComplexType(target) == self.is_complex(),
              "linalg.vector_norm: dtype must preserve real/complex kind");
  TORCH_CHECK(c10::promoteTypes(self.scalar_type(), target) == target,
              "linalg.vector_norm: dtype must not narrow the input");
  return target;
}

static at::Tensor PreciseVectorNorm(
    const at::Tensor& self, const at::Scalar& order, at::OptionalIntArrayRef dim,
    bool keepdim, at::ScalarType target) {
  TORCH_CHECK(!order.isComplex(), "linalg.vector_norm: order must be real");
  const double p = order.toDouble();
  const auto real_dtype = c10::toRealValueType(target);
  auto input = self.to(target);
  auto magnitude = at::abs(input);
  std::vector<int64_t> axes;
  if (dim.has_value() && !dim->empty()) {
    for (int64_t axis : *dim) axes.push_back(at::maybe_wrap_dim(axis, self.dim()));
  } else {
    for (int64_t axis = 0; axis < self.dim(); ++axis) axes.push_back(axis);
  }
  std::sort(axes.begin(), axes.end());
  TORCH_CHECK(std::adjacent_find(axes.begin(), axes.end()) == axes.end(),
              "linalg.vector_norm: repeated reduction dimension");
  if (self.dim() == 0) {
    return p == 0 ? (magnitude != 0).to(real_dtype) : magnitude;
  }
  if (magnitude.numel() == 0) {
    int64_t volume = 1;
    for (int64_t axis : axes) volume *= magnitude.size(axis);
    TORCH_CHECK(volume != 0 || (!std::isinf(p) && p >= 0),
                "linalg.vector_norm: this order is undefined on an empty reduction");
    std::vector<int64_t> shape;
    for (int64_t axis = 0; axis < magnitude.dim(); ++axis) {
      if (std::find(axes.begin(), axes.end(), axis) == axes.end()) {
        shape.push_back(magnitude.size(axis));
      } else if (keepdim) {
        shape.push_back(1);
      }
    }
    return at::zeros(shape, magnitude.options());
  }
  at::Tensor result;
  if (p == 0) {
    result = at::sum((magnitude != 0).to(real_dtype), axes, true, real_dtype);
  } else if (std::isinf(p)) {
    result = p > 0 ? at::amax(magnitude, axes, true) : at::amin(magnitude, axes, true);
  } else if (p == 1) {
    result = at::sum(magnitude, axes, true, real_dtype);
  } else {
    // Scaling avoids overflow in positive orders and loss of small values in
    // negative orders. All reductions and coefficients stay on the device.
    auto scale = p > 0 ? at::amax(magnitude, axes, true) : at::amin(magnitude, axes, true);
    auto scaled = magnitude / scale;
    result = scale * at::pow(at::sum(at::pow(scaled, p), axes, true, real_dtype), 1 / p);
    result = at::where(scale == 0, at::zeros_like(result), result);
    result = at::where(scale == std::numeric_limits<double>::infinity(), scale, result);
  }
  if (!keepdim) {
    for (auto axis = axes.rbegin(); axis != axes.rend(); ++axis) result = result.squeeze(*axis);
  }
  return result;
}

} // namespace ascend

"""


def branch(op: str) -> str:
    """Return a dtype-specific prologue before a generated ACLNN kernel."""
    pack = "ascend::PackComplex"
    parts = "auto [real, imag] = ascend::ComplexParts(self);\n"
    unary = {
        "neg": f"return {pack}(-real, -imag);",
        "abs": "return ascend::ComplexMagnitude(real, imag);",
        "exp": f"return {pack}(at::exp(real) * at::cos(imag), at::exp(real) * at::sin(imag));",
        "cos": f"return {pack}(at::cos(real) * at::cosh(imag), -at::sin(real) * at::sinh(imag));",
        "sin": f"return {pack}(at::sin(real) * at::cosh(imag), at::cos(real) * at::sinh(imag));",
        "sgn": "auto norm = ascend::ComplexMagnitude(real, imag);\n"
        f"return {pack}(at::where(norm == 0, at::zeros_like(real), real / norm), "
        "at::where(norm == 0, at::zeros_like(imag), imag / norm));",
    }
    prefix = ""
    condition = "self.is_complex()"
    if op == "pow.Tensor_Scalar":
        return "  if (self.dim() > 8 && !self.is_complex()) return at::pow(self.reshape({-1}), s).reshape(self.sizes());\n"
    if op == "linalg_vector_norm":
        return (
            "  auto norm_dtype = ascend::VectorNormDtype(self, dtype);\n"
            "  if (self.is_complex() || self.scalar_type() == at::kDouble ||\n"
            "      norm_dtype == at::kDouble || c10::isComplexType(norm_dtype)) {\n"
            "    return ascend::PreciseVectorNorm(self, ord, dim, keepdim, norm_dtype);\n  }\n"
        )
    if op in unary:
        body = (
            f"if (self.dim() > 8) return at::{op}(self.reshape({{-1}})).reshape(self.sizes());\n"
            + parts
            + unary[op]
        )
    elif op == "isinf":
        body = parts + "return at::logical_or(at::isinf(real), at::isinf(imag));"
        return (
            f"  if ({condition}) {{\n    "
            + body.replace("\n", "\n    ")
            + "\n  }\n  if (self.scalar_type() == at::kDouble) {\n"
            "    return at::logical_or(self == std::numeric_limits<double>::infinity(),\n"
            "                          self == -std::numeric_limits<double>::infinity());\n  }\n"
        )
    elif op in {"mul.Tensor", "div.Tensor", "add.Tensor", "sub.Tensor"}:
        condition += " || other.is_complex()"
        operation = op.split(".")[0]
        arguments = ", alpha" if operation in {"add", "sub"} else ""
        prefix = "  if (self.dim() > 8 || other.dim() > 8) {\n"
        prefix += "  auto shape = at::infer_size(self.sizes(), other.sizes());\n"
        prefix += "  auto target = at::result_type(self, other);\n"
        prefix += "  auto a = self.to(target).expand(shape).reshape({-1});\n"
        prefix += "  auto b = other.to(self.options().dtype(target)).expand(shape).reshape({-1});\n"
        prefix += f"  return at::{operation}(a, b{arguments}).reshape(shape);\n}}\n"
        body = "auto dtype = at::result_type(self, other);\n"
        body += "auto a = self.to(self.options().dtype(dtype));\n"
        body += "auto b = other.to(self.options().dtype(dtype));\n"
        body += "auto [ar, ai] = ascend::ComplexParts(a);\n"
        body += "auto [br, bi] = ascend::ComplexParts(b);\n"
        if op == "mul.Tensor":
            body += f"return {pack}(ar * br - ai * bi, ar * bi + ai * br);"
        elif op == "div.Tensor":
            body += "auto scale = at::maximum(at::abs(br), at::abs(bi));\n"
            body += "auto cr = br / scale; auto ci = bi / scale;\n"
            body += "auto denominator = cr * cr + ci * ci;\n"
            body += (
                f"return {pack}(((ar / scale) * cr + (ai / scale) * ci) / denominator, "
            )
            body += "((ai / scale) * cr - (ar / scale) * ci) / denominator);"
        else:
            sign = "+" if op == "add.Tensor" else "-"
            body += "const auto value = alpha.toComplexDouble();\n"
            body += f"return {pack}(ar {sign} (br * value.real() - bi * value.imag()), "
            body += f"ai {sign} (br * value.imag() + bi * value.real()));"
    elif op in {"mul.Scalar", "div.Scalar", "add.Scalar", "sub.Scalar"}:
        operation = op.split(".")[0]
        arguments = ", alpha" if operation in {"add", "sub"} else ""
        prefix = f"  if (self.dim() > 8) return at::{operation}(self.reshape({{-1}}), other{arguments}).reshape(self.sizes());\n"
        condition += " || other.isComplex()"
        body = "auto target = at::result_type(self, other);\n"
        body += "auto [real, imag] = ascend::ComplexParts(self.to(target));\n"
        body += "const auto value = other.toComplexDouble();\n"
        if op == "mul.Scalar":
            body += f"return {pack}(real * value.real() - imag * value.imag(), "
            body += "real * value.imag() + imag * value.real());"
        elif op == "div.Scalar":
            body += "auto denominator = value.real() * value.real() + value.imag() * value.imag();\n"
            body += f"return {pack}((real * value.real() + imag * value.imag()) / denominator, "
            body += "(imag * value.real() - real * value.imag()) / denominator);"
        else:
            sign = "+" if op == "add.Scalar" else "-"
            body += "const auto scale = alpha.toComplexDouble();\n"
            body += "const auto offset = value * scale;\n"
            body += (
                f"return {pack}(real {sign} offset.real(), imag {sign} offset.imag());"
            )
    elif op in {
        "add_.Tensor",
        "sub_.Tensor",
        "mul_.Tensor",
        "div_.Tensor",
        "add_.Scalar",
        "sub_.Scalar",
        "mul_.Scalar",
        "div_.Scalar",
    }:
        base = op.split("_")[0]
        arguments = ", alpha" if base in {"add", "sub"} else ""
        body = f"self.copy_(at::{base}(self, other{arguments}));\nreturn self;"
    elif op in {"mm", "bmm", "mm.out", "bmm.out"}:
        ndim = 2 if op.startswith("mm") else 3
        checks = (
            f"TORCH_CHECK(self.dim() == {ndim} && mat2.dim() == {ndim}, "
            f'"{op}: incorrect input rank");\n'
            "TORCH_CHECK(self.size(-1) == mat2.size(-2), "
            f'"{op}: incompatible contraction sizes");\n'
        )
        if op.startswith("bmm"):
            checks += 'TORCH_CHECK(self.size(0) == mat2.size(0), "bmm: batch sizes must match");\n'
        complex_result = "ascend::ComplexMatmul(self, mat2)"
        double_result = "ascend::RealMatmulDouble(self, mat2)"
        if op.endswith(".out"):
            complex_result = f"out.copy_({complex_result}); return out"
            double_result = f"out.copy_({double_result}); return out"
        else:
            complex_result = f"return {complex_result}"
            double_result = f"return {double_result}"
        body = complex_result + ";"
        return (
            f"  if ({condition}) {{\n{checks}    {body}\n  }}\n"
            "  if (self.scalar_type() == at::kDouble && mat2.scalar_type() == at::kDouble &&\n"
            "      !ascend::IsFp16OnlyMatmulSoc(ascend::GetSocName())) {\n"
            + checks
            + f"    {double_result};\n  }}\n"
        )
    elif op in {"sum", "sum.dim_IntList"}:
        if op == "sum":
            prefix = "  if (self.dim() > 8 && !self.is_complex()) return at::sum(self.reshape({-1}), dtype);\n"
        else:
            prefix = ""
        args = "dim, keepdim, " if op == "sum.dim_IntList" else ""
        condition += " || (dtype.has_value() && c10::isComplexType(*dtype))"
        body = parts + "auto target = dtype.value_or(self.scalar_type());\n"
        body += "auto real_dtype = c10::toRealValueType(target);\n"
        body += f"auto r = at::sum(real, {args}real_dtype);\n"
        body += "if (!c10::isComplexType(target)) return r;\n"
        body += f"return {pack}(r, at::sum(imag, {args}real_dtype));"
    elif op == "flip":
        body = parts + f"return {pack}(at::flip(real, dims), at::flip(imag, dims));"
    elif op in {"triu", "tril", "triu_", "tril_"}:
        operation = op.rstrip("_")
        offset = "n" if op.endswith("_") else "i"
        body = (
            parts
            + f"auto result = {pack}(at::{operation}(real, {offset}), at::{operation}(imag, {offset}));\n"
        )
        body += (
            "self.copy_(result);\nreturn self;"
            if op.endswith("_")
            else "return result;"
        )
    elif op in {"zero_", "fill_.Scalar"}:
        factory = (
            "at::zeros({self.numel()}, self.options())"
            if op == "zero_"
            else "at::full({self.numel()}, value, self.options())"
        )
        prefix = "  if (self.dim() > 8) {\n"
        prefix += f"    auto dense = {factory}.reshape(self.sizes());\n"
        prefix += "    self.copy_(dense); return self;\n  }\n"
        body = "auto parts = at::native::view_as_real(self);\n"
        if op == "zero_":
            body += "parts.copy_(at::zeros(parts.sizes(), parts.options()));\n"
        else:
            body += "auto scalar = value.toComplexDouble();\n"
            body += (
                "auto real = parts.select(-1, 0); auto imag = parts.select(-1, 1);\n"
            )
            body += (
                "real.copy_(at::full(real.sizes(), scalar.real(), real.options()));\n"
            )
            body += (
                "imag.copy_(at::full(imag.sizes(), scalar.imag(), imag.options()));\n"
            )
        body += "return self;"
    elif op in {"masked_fill.Scalar", "masked_fill_.Scalar"}:
        condition += " || self.scalar_type() == at::kDouble"
        body = "auto result = at::where(mask, at::full_like(self, value), self);\n"
        body += (
            "self.copy_(result);\nreturn self;"
            if op.startswith("masked_fill_")
            else "return result;"
        )
    elif op == "where.self":
        condition += " || other.is_complex()"
        body = "auto target = at::result_type(self, other);\n"
        body += "auto options = condition.options().dtype(target);\n"
        body += "auto [ar, ai] = ascend::ComplexParts(self.to(options));\n"
        body += "auto [br, bi] = ascend::ComplexParts(other.to(options));\n"
        body += f"return {pack}(at::where(condition, ar, br), at::where(condition, ai, bi));"
    elif op == "stack":
        condition = "std::any_of(tensors.begin(), tensors.end(), [](const at::Tensor& t) { return t.is_complex(); })"
        body = "auto first = *tensors.begin();\n"
        body += "auto dtype = first.scalar_type();\n"
        body += "for (const auto& tensor : tensors) {\n"
        body += '  TORCH_CHECK(tensor.sizes() == first.sizes(), "stack: tensor shapes must match");\n'
        body += '  TORCH_CHECK(tensor.device() == first.device(), "stack: tensor devices must match");\n'
        body += "  dtype = c10::promoteTypes(dtype, tensor.scalar_type());\n}\n"
        body += "auto axis = at::maybe_wrap_dim(dim, first.dim() + 1);\n"
        body += "auto shape = first.sizes().vec();\n"
        body += "shape.insert(shape.begin() + axis, tensors.size());\n"
        body += "auto result = at::empty(shape, first.options().dtype(dtype));\n"
        body += "int64_t index = 0;\n"
        body += "for (const auto& tensor : tensors) result.select(axis, index++).copy_(tensor);\n"
        body += "return result;"
    elif op == "cat":
        condition = "std::any_of(tensors.begin(), tensors.end(), [](const at::Tensor& t) { return t.is_complex(); })"
        body = "auto dtype = (*tensors.begin()).scalar_type();\n"
        body += "for (const auto& tensor : tensors) dtype = c10::promoteTypes(dtype, tensor.scalar_type());\n"
        body += "auto component_dtype = c10::toRealValueType(dtype);\n"
        body += "std::vector<at::Tensor> real, imag;\n"
        body += "for (const auto& tensor : tensors) {\n"
        body += "  auto pair = ascend::ComplexParts(tensor);\n"
        body += "  real.push_back(pair.first.to(component_dtype));\n"
        body += "  imag.push_back(pair.second.to(component_dtype));\n}\n"
        body += f"return {pack}(at::{op}(real, dim), at::{op}(imag, dim));"
    else:
        return ""
    result = f"  if ({condition}) {{\n    " + body.replace("\n", "\n    ") + "\n  }\n"
    return prefix + result
