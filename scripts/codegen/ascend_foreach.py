"""Lower FP64 optimizer tensor lists to precision-preserving device operations."""


def branch(op: str) -> str:
    bodies = {
        "_foreach_mul_.Scalar": "self[i].copy_(self[i] * scalar);",
        "_foreach_add_.Scalar": "self[i].copy_(self[i] + scalar);",
        "_foreach_add_.List": "self[i].copy_(self[i] + alpha * other[i]);",
        "_foreach_lerp_.Scalar": "self[i].copy_(self[i] + weight * (tensors1[i] - self[i]));",
        "_foreach_addcmul_.Scalar": "self[i].copy_(self[i] + value * tensor1[i] * tensor2[i]);",
        "_foreach_div_.ScalarList": "self[i].copy_(self[i] / scalars[i]);",
        "_foreach_addcdiv_.ScalarList": "self[i].copy_(self[i] + scalars[i] * (tensor1[i] / tensor2[i]));",
        "_foreach_mul_.Tensor": "self[i].copy_(self[i] * other);",
        "_foreach_clamp_min_.Scalar": "self[i].copy_(at::where(self[i] < scalar, at::full_like(self[i], scalar), self[i]));",
        "_foreach_clamp_max_.Scalar": "self[i].copy_(at::where(self[i] > scalar, at::full_like(self[i], scalar), self[i]));",
        "_foreach_sqrt": "outs.push_back(at::sqrt(self[i]));",
        "_foreach_norm.Scalar": "outs.push_back(at::linalg_vector_norm(self[i], ord, std::nullopt, false, dtype));",
    }
    if op not in bodies:
        return ""
    lists = {
        "_foreach_add_.List": ["other"],
        "_foreach_lerp_.Scalar": ["tensors1"],
        "_foreach_addcmul_.Scalar": ["tensor1", "tensor2"],
        "_foreach_div_.ScalarList": ["scalars"],
        "_foreach_addcdiv_.ScalarList": ["tensor1", "tensor2", "scalars"],
    }
    result = "  if (std::any_of(self.begin(), self.end(), [](const at::Tensor& t) { return t.scalar_type() == at::kDouble; })) {\n"
    for name in lists.get(op, []):
        result += f'    TORCH_CHECK(self.size() == {name}.size(), "{op}: list sizes must match");\n'
    returns_list = op in {"_foreach_sqrt", "_foreach_norm.Scalar"}
    if returns_list:
        result += "    std::vector<at::Tensor> outs; outs.reserve(self.size());\n"
    result += f"    for (size_t i = 0; i < self.size(); ++i) {{ {bodies[op]} }}\n"
    result += "    return outs;\n" if returns_list else "    return;\n"
    return result + "  }\n"
