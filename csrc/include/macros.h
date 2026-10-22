// Copyright (c) 2026, BAAI. All rights reserved.
//
// Copied from https://github.com/pytorch/pytorch/tree/main/test/cpp_extensions/open_registration_extension/torch_openreg/include/Macros.h
// with OPENREG_EXPORT renamed to FLAGOS_EXPORT.
// Below is the original copyright:
// Copyright (c) Meta Platforms, Inc. and affiliates.

#pragma once

#ifdef _WIN32
#define FLAGOS_EXPORT __declspec(dllexport)
#else
#define FLAGOS_EXPORT __attribute__((visibility("default")))
#endif
