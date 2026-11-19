// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/ArrayRef.h>

#include <string_view>

namespace flagos::native_host {

class Descriptor {
 public:
  explicit Descriptor(c10::IValue value) : value_(std::move(value)) {}
  static bool Is(const c10::IValue& value) {
    if (value.isList()) {
      const auto list = value.toListRef();
      if (list.size() != 6) return false;
      const auto& tag = list[0];
      return tag.isString() && tag.toStringRef() == "__flagos_descriptor_v2__";
    }
    if (!value.isGenericDict()) return false;
    const auto data = value.toGenericDict();
    const c10::IValue tag("__flagos_tensor_descriptor_v1__");
    return data.contains(tag) && data.at(tag).isBool() && data.at(tag).toBool();
  }
  static c10::IValue Create(const at::Tensor& base, const c10::List<int64_t>& shape,
                            const c10::List<int64_t>& strides, const c10::List<int64_t>& block,
                            const std::string& padding) {
    static const c10::IValue tag("__flagos_descriptor_v2__");
    c10::List<c10::IValue> values(c10::AnyType::get());
    values.reserve(6);
    for (const auto& value : {tag, c10::IValue(base), c10::IValue(shape), c10::IValue(strides),
                              c10::IValue(block), c10::IValue(padding)})
      values.push_back(value);
    return values;
  }
  c10::IValue at(std::string_view field) const {
    if (value_.isGenericDict()) return value_.toGenericDict().at(std::string(field));
    return value_.toList().get(Index(field));
  }
  void insert_or_assign(std::string_view field, c10::IValue value) const {
    if (value_.isGenericDict())
      value_.toGenericDict().insert_or_assign(std::string(field), std::move(value));
    else
      value_.toList().set(Index(field), std::move(value));
  }

 private:
  static size_t Index(std::string_view field) {
    if (field == "base") return 1;
    if (field == "shape") return 2;
    if (field == "strides") return 3;
    if (field == "block_shape") return 4;
    TORCH_CHECK(field == "padding", "unknown tensor descriptor field: ", field);
    return 5;
  }
  c10::IValue value_;
};

// Borrowed current-call values. The generated caller owns the stack storage;
// neither a specialization nor a kernel launch retains these views.
struct Arguments {
  c10::ArrayRef<c10::IValue> values;
  size_t size() const { return values.size(); }
  const c10::IValue& get(size_t index) const { return values.at(index); }
  auto begin() const { return values.begin(); }
  auto end() const { return values.end(); }
};

struct TensorValues {
  c10::ArrayRef<at::Tensor> values;
  size_t size() const { return values.size(); }
  const at::Tensor& get(size_t index) const { return values.at(index); }
  auto begin() const { return values.begin(); }
  auto end() const { return values.end(); }
};

struct Keyword {
  std::string_view name;
  c10::IValue argument;
  std::string_view key() const { return name; }
  const c10::IValue& value() const { return argument; }
};

struct Keywords {
  c10::ArrayRef<Keyword> values;
  size_t size() const { return values.size(); }
  auto begin() const { return values.begin(); }
  auto end() const { return values.end(); }
  const c10::IValue* find(std::string_view name) const {
    for (const auto& item : values)
      if (item.name == name) return &item.argument;
    return nullptr;
  }
  bool contains(std::string_view name) const { return find(name) != nullptr; }
  const c10::IValue& at(std::string_view name) const {
    const auto* value = find(name);
    TORCH_CHECK(value, "missing current host keyword: ", name);
    return *value;
  }
};

struct GridMetadata {
  const c10::Dict<std::string, int64_t>* constants;
  const std::vector<std::string>* names;
  Arguments args;
  Keywords kwargs;
};

struct NamedArguments {
  const std::vector<std::string>* names;
  Arguments args;
  Keywords kwargs;
  Keywords constants;
  const c10::IValue& at(std::string_view name) const {
    if (const auto* value = constants.find(name)) return *value;
    if (const auto* value = kwargs.find(name)) return *value;
    for (size_t i = 0; i < args.size(); ++i)
      if (names->at(i) == name) return args.get(i);
    TORCH_CHECK(false, "missing descriptor-hook argument: ", name);
  }
};

using MetadataHook = void (*)(const NamedArguments&);
using Foreign = c10::IValue (*)(int64_t, Arguments, Keywords);

using Launch = void (*)(int64_t, Arguments, Keywords, const std::vector<int64_t>&, int64_t);
using Pointwise = std::vector<at::Tensor> (*)(int64_t, Arguments, Keywords);
using PointwiseAt = at::Tensor (*)(int64_t, Arguments, Keywords, int64_t);
using PointwiseLaunch = void (*)(int64_t, int64_t, Arguments, TensorValues);

}  // namespace flagos::native_host
