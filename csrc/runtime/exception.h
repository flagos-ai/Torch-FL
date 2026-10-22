// Copyright (c) 2026, BAAI. All rights reserved.
//
// Adopted from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#include <flagos.h>

#include <c10/util/Exception.h>

#define FLAGOS_CHECK(EXPR)                                      \
  do {                                                          \
    const Error_t __err = EXPR;                               \
    TORCH_CHECK(__err == Success,                             \
        "FlagOS error: ", __err,                                \
        " when calling " #EXPR);                                \
  } while (0)
