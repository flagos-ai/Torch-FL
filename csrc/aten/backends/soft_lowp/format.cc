// Copyright (c) 2026, BAAI. All rights reserved.

#include "format.h"

namespace at::native::flagos::soft_lowp {

static_assert(c10::kFloat8_e4m3fn != c10::kFloat8_e5m2);
static_assert(c10::kFloat8_e4m3fnuz != c10::kFloat8_e5m2fnuz);
static_assert(c10::kFloat8_e8m0fnu != c10::kFloat4_e2m1fn_x2);

} // namespace at::native::flagos::soft_lowp
