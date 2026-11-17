// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once

// Field layout of the packed argument struct emitted by the published
// FlagTree Ascend driver. No device/runtime headers or numerical code.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace flagos::native_host::ascend {

class Layout {
 public:
  Layout(const std::vector<std::string>& types, bool ffts, bool simt_only)
      : ffts_(ffts), workspace_(!simt_only) {
    if (ffts_) ffts_offset_ = Add(8, 8);
    if (workspace_) {
      lock_offset_ = Add(8, 8);
      workspace_offset_ = Add(8, 8);
    }
    for (const auto& type : types) {
      size_t size;
      const bool boolean = type == "i1" || type == "u1";
      const bool pointer = !type.empty() && type[0] == '*';
      if (pointer || type == "i64" || type == "u64" || type == "fp64") size = 8;
      else if (type == "i8" || type == "u8") size = 1;
      else if (type == "i16" || type == "u16") size = 2;
      else if (boolean || type == "i32" || type == "u32" || type == "fp32" || type == "f32") size = 4;
      else throw std::invalid_argument("unsupported Ascend wire parameter: " + type);
      arguments_.push_back({Add(size, size == 8 ? 8 : 4), size, boolean});
    }
    for (auto& offset : grid_) offset = Add(4, 4);
    bytes_ = Align(bytes_, alignment_);
  }

  std::vector<unsigned char> Pack(void** values, const std::array<uint64_t, 3>& grid,
                                  uint64_t ffts, uint64_t lock, uint64_t workspace) const {
    std::vector<unsigned char> bytes(bytes_, 0);
    if (ffts_) Write(bytes, ffts_offset_, ffts);
    if (workspace_) {
      Write(bytes, lock_offset_, lock);
      Write(bytes, workspace_offset_, workspace);
    }
    for (size_t i = 0; i < arguments_.size(); ++i) {
      const auto& field = arguments_[i];
      if (field.boolean) {
        uint8_t value;
        std::memcpy(&value, values[i], 1);
        Write(bytes, field.offset, static_cast<uint32_t>(value));
      } else {
        std::memcpy(bytes.data() + field.offset, values[i], field.size);
      }
    }
    for (size_t i = 0; i < grid.size(); ++i) {
      if (grid[i] > INT32_MAX) throw std::overflow_error("Ascend logical grid exceeds int32");
      Write(bytes, grid_[i], static_cast<int32_t>(grid[i]));
    }
    return bytes;
  }

 private:
  struct Field { size_t offset, size; bool boolean; };
  static size_t Align(size_t value, size_t alignment) {
    if (value > UINT32_MAX - (alignment - 1))
      throw std::overflow_error("Ascend argument block exceeds uint32");
    return (value + alignment - 1) & ~(alignment - 1);
  }
  size_t Add(size_t size, size_t alignment) {
    const auto offset = Align(bytes_, alignment);
    bytes_ = offset + size;
    if (bytes_ > UINT32_MAX) throw std::overflow_error("Ascend argument block exceeds uint32");
    if (alignment > alignment_) alignment_ = alignment;
    return offset;
  }
  template <class T>
  static void Write(std::vector<unsigned char>& bytes, size_t offset, T value) {
    std::memcpy(bytes.data() + offset, &value, sizeof(value));
  }
  bool ffts_, workspace_;
  size_t ffts_offset_ = 0, lock_offset_ = 0, workspace_offset_ = 0;
  size_t bytes_ = 0, alignment_ = 1;
  std::vector<Field> arguments_;
  std::array<size_t, 3> grid_;
};

}  // namespace flagos::native_host::ascend
