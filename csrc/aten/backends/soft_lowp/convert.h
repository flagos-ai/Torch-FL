// Copyright (c) 2026, BAAI. All rights reserved.
//
// Host/device low-precision conversion helpers. The bit-level FP8 behavior is
// delegated to PyTorch's C10 headers so the software path has the same special
// values and rounding semantics as native PyTorch.

#pragma once

#include "format.h"

#include <c10/util/irange.h>
#include <torch/headeronly/util/Float8_e4m3fn.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>
#include <torch/headeronly/util/Float8_e5m2.h>
#include <torch/headeronly/util/Float8_e5m2fnuz.h>
#include <torch/headeronly/util/Float8_e8m0fnu.h>

#include <cstdint>

namespace at::native::flagos::soft_lowp {

inline float DecodeFp8(uint8_t bits, FormatId format) {
  switch (format) {
    case FormatId::kFloat8E4M3FN:
      return c10::detail::fp8e4m3fn_to_fp32_value(bits);
    case FormatId::kFloat8E5M2:
      return c10::detail::fp8e5m2_to_fp32_value(bits);
    case FormatId::kFloat8E4M3FNUZ:
      return c10::detail::fp8_fnuz_to_fp32_value<4, 3>(bits);
    case FormatId::kFloat8E5M2FNUZ:
      return c10::detail::fp8_fnuz_to_fp32_value<5, 2>(bits);
    case FormatId::kFloat8E8M0FNU:
      return static_cast<float>(c10::Float8_e8m0fnu(
          bits, c10::Float8_e8m0fnu::from_bits()));
    default:
      return 0.0F;
  }
}

inline uint8_t EncodeFp8(float value, FormatId format) {
  switch (format) {
    case FormatId::kFloat8E4M3FN:
      return c10::detail::fp8e4m3fn_from_fp32_value(value);
    case FormatId::kFloat8E5M2:
      return c10::detail::fp8e5m2_from_fp32_value(value);
    case FormatId::kFloat8E4M3FNUZ:
      return c10::detail::fp8e4m3fnuz_from_fp32_value(value);
    case FormatId::kFloat8E5M2FNUZ:
      return c10::detail::fp8e5m2fnuz_from_fp32_value(value);
    case FormatId::kFloat8E8M0FNU:
      return c10::detail::fp8e8m0fnu_from_fp32_value(value);
    default:
      return 0;
  }
}

inline float DecodeFp4(uint8_t nibble) {
  static constexpr float values[16] = {
      0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F,
      -0.0F, -0.5F, -1.0F, -1.5F, -2.0F, -3.0F, -4.0F, -6.0F};
  return values[nibble & 0x0F];
}

inline float DecodePackedFp4(uint8_t byte, int index_in_byte) {
  return DecodeFp4(index_in_byte == 0 ? byte & 0x0F : byte >> 4);
}

inline uint8_t EncodeFp4(float value) {
  constexpr float values[8] = {0.0F, 0.5F, 1.0F, 1.5F, 2.0F, 3.0F, 4.0F, 6.0F};
  float abs_value = value < 0.0F ? -value : value;
  int best = 0;
  float best_distance = abs_value;
  for (int i = 1; i < 8; ++i) {
    float distance = abs_value - values[i];
    distance = distance < 0.0F ? -distance : distance;
    if (distance < best_distance) {
      best = i;
      best_distance = distance;
    }
  }
  return static_cast<uint8_t>(best | (value < 0.0F ? 0x8 : 0));
}

} // namespace at::native::flagos::soft_lowp
