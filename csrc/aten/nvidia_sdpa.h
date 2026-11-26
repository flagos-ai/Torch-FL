// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dispatcher.h"

namespace at::native::flagos {

using NvidiaSdpaFn = at::Tensor (*)(
    const at::Tensor&, const at::Tensor&, const at::Tensor&,
    const std::optional<at::Tensor>&, double, bool, std::optional<double>, bool);
DECLARE_DISPATCHER(NvidiaSdpaFn, nvidia_sdpa_dispatcher)

} // namespace at::native::flagos
