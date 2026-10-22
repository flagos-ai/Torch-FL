// Copyright (c) 2026, BAAI. All rights reserved.
//
// Low-precision format metadata shared by software-emulated device kernels.

#pragma once

#include <c10/core/ScalarType.h>
#include <cstdint>
#include <optional>
#include <string_view>

namespace at::native::flagos::soft_lowp {

enum class FormatId : uint8_t {
  kFloat8E4M3FN,
  kFloat8E5M2,
  kFloat8E4M3FNUZ,
  kFloat8E5M2FNUZ,
  kFloat8E8M0FNU,
  kFloat4E2M1FNx2,
  kMxfp4,
  kNvfp4,
  kBlockFp4,
};

struct FormatSpec {
  FormatId id;
  int bits_per_value;
  int values_per_byte;
  std::optional<c10::ScalarType> torch_dtype;
  int block_rows;
  int block_cols;
  std::optional<c10::ScalarType> scale_dtype;
  float max_finite;
  bool signed_values;
  bool has_infinity;
  bool has_nan;
  bool fnuz;
};

inline bool IsFp8(c10::ScalarType dtype) {
  return dtype == c10::kFloat8_e4m3fn || dtype == c10::kFloat8_e5m2 ||
      dtype == c10::kFloat8_e4m3fnuz || dtype == c10::kFloat8_e5m2fnuz ||
      dtype == c10::kFloat8_e8m0fnu;
}

inline bool IsPackedFp4(c10::ScalarType dtype) {
  return dtype == c10::kFloat4_e2m1fn_x2;
}

inline bool IsLowPrecision(c10::ScalarType dtype) {
  return IsFp8(dtype) || IsPackedFp4(dtype);
}

inline std::optional<FormatId> FormatForDtype(c10::ScalarType dtype) {
  if (dtype == c10::kFloat8_e4m3fn) {
    return FormatId::kFloat8E4M3FN;
  }
  if (dtype == c10::kFloat8_e5m2) {
    return FormatId::kFloat8E5M2;
  }
  if (dtype == c10::kFloat8_e4m3fnuz) {
    return FormatId::kFloat8E4M3FNUZ;
  }
  if (dtype == c10::kFloat8_e5m2fnuz) {
    return FormatId::kFloat8E5M2FNUZ;
  }
  if (dtype == c10::kFloat8_e8m0fnu) {
    return FormatId::kFloat8E8M0FNU;
  }
  if (dtype == c10::kFloat4_e2m1fn_x2) {
    return FormatId::kFloat4E2M1FNx2;
  }
  return std::nullopt;
}

inline constexpr FormatSpec GetFormatSpec(FormatId id) {
  switch (id) {
    case FormatId::kFloat8E4M3FN:
      return {id, 8, 1, c10::kFloat8_e4m3fn, 0, 0, std::nullopt,
              448.0F, true, false, true, false};
    case FormatId::kFloat8E5M2:
      return {id, 8, 1, c10::kFloat8_e5m2, 0, 0, std::nullopt,
              57344.0F, true, true, true, false};
    case FormatId::kFloat8E4M3FNUZ:
      return {id, 8, 1, c10::kFloat8_e4m3fnuz, 0, 0, std::nullopt,
              240.0F, true, false, true, true};
    case FormatId::kFloat8E5M2FNUZ:
      return {id, 8, 1, c10::kFloat8_e5m2fnuz, 0, 0, std::nullopt,
              57344.0F, true, false, true, true};
    case FormatId::kFloat8E8M0FNU:
      return {id, 8, 1, c10::kFloat8_e8m0fnu, 0, 0, std::nullopt,
              1.7014118e38F, false, false, true, true};
    case FormatId::kFloat4E2M1FNx2:
      return {id, 4, 2, c10::kFloat4_e2m1fn_x2, 0, 0, std::nullopt,
              6.0F, true, false, false, false};
    case FormatId::kMxfp4:
      return {id, 4, 2, std::nullopt, 1, 32, c10::kFloat8_e8m0fnu,
              6.0F, true, false, false, false};
    case FormatId::kNvfp4:
      return {id, 4, 2, std::nullopt, 1, 16, c10::kFloat8_e4m3fn,
              6.0F, true, false, false, false};
    case FormatId::kBlockFp4:
      return {id, 4, 2, std::nullopt, 1, 32, c10::kFloat,
              6.0F, true, false, false, false};
  }
  return {FormatId::kFloat8E4M3FN, 0, 0, std::nullopt, 0, 0, std::nullopt,
          0.0F, false, false, false, false};
}

inline std::string_view FormatName(FormatId id) {
  switch (id) {
    case FormatId::kFloat8E4M3FN:
      return "float8_e4m3fn";
    case FormatId::kFloat8E5M2:
      return "float8_e5m2";
    case FormatId::kFloat8E4M3FNUZ:
      return "float8_e4m3fnuz";
    case FormatId::kFloat8E5M2FNUZ:
      return "float8_e5m2fnuz";
    case FormatId::kFloat8E8M0FNU:
      return "float8_e8m0fnu";
    case FormatId::kFloat4E2M1FNx2:
      return "float4_e2m1fn_x2";
    case FormatId::kMxfp4:
      return "mxfp4";
    case FormatId::kNvfp4:
      return "nvfp4";
    case FormatId::kBlockFp4:
      return "block_fp4";
  }
  return "unknown";
}

} // namespace at::native::flagos::soft_lowp
