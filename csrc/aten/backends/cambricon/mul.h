// Copyright (c) 2026, BAAI. All rights reserved.
//
// Entry points of the Cambricon CNNL operator layer.
//
// register.cc includes this from its USE_CAMBRICON arm and then includes the
// generated register file, which names these functions in `m.impl` calls. The
// declarations live apart from the definitions so that the register file stays
// a pure list of names -- the same split the other native backends use.

#pragma once

#ifdef USE_CAMBRICON

#include <ATen/ATen.h>

namespace at::native::flagos::cambricon {

// `mul.Tensor` through cnnlOpTensor. See mul.cc for the measurement that put it
// here and the operand shapes it accepts.
at::Tensor MulTensor(const at::Tensor& self, const at::Tensor& other);

} // namespace at::native::flagos::cambricon

#endif // USE_CAMBRICON