# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``csrc/aten/dispatcher.h`` with its member templates instantiated.

A dispatch-helper signature is not checked until something instantiates it,
and the callers that do live in the software low-precision path, which only a
``FLAGOS_SOFT_LOWP`` build compiles. The helper that sent a decoded pair to a
route called the route's kernel from inside the dispatcher, through the node
the route resolved to; GCC 11 accepts such a call only where the callee is a
typed pointer parameter of a non-template member, so a translation unit with
more than one kernel signature in it was rejected on the DCU hosts (GCC 11.4)
while GCC 12 on the boxing hosts took it, and nothing saw it until a DCU build.

The header now answers with the kernel (``DecodedKernel``) and leaves the call
to the call site, which spells the kernel's own type as ``Call``'s first
parameter -- the shape ``soft_lowp/ops.cc`` uses. This file instantiates that
shape with the two signatures the low-precision path sends decoded pairs to
(an ``mm``-shaped one and an ``addmm``-shaped one), so a change back to calling
through the dispatcher fails here on any compiler, not only on GCC 11.

The translation unit here includes only headers -- ATen's and this file's --
and instantiates each helper with a kernel type it can be given, so it builds
on any host with a compiler and runs at human speed (< 5 s warm). It is skipped
where there is none, or where the configured compiler rejects the
environment's own ATen headers, since neither says anything about this file.
"""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

_TRANSLATION_UNIT = r"""
#include "aten/dispatcher.h"

using MmFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&,
                            c10::ScalarType);
using BmmFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&);
using AddmmFn = at::Tensor (*)(const at::Tensor&, const at::Tensor&,
                               const at::Tensor&, const c10::Scalar&,
                               const c10::Scalar&);

namespace {

// A kernel of each shape the helpers are handed a dispatcher for.
at::Tensor MmLike(const at::Tensor& self, const at::Tensor& mat2,
                  c10::ScalarType dtype) {
  return self;
}

at::Tensor BmmBrand(const at::Tensor& self, const at::Tensor& mat2) {
  return self;
}

at::Tensor AddmmLike(const at::Tensor& bias, const at::Tensor& self,
                     const at::Tensor& mat2, const c10::Scalar& beta,
                     const c10::Scalar& alpha) {
  return self;
}

} // namespace

namespace at::native::flagos {

DECLARE_DISPATCHER(MmFn, test_mm_dispatcher)
DECLARE_DISPATCHER(BmmFn, test_bmm_dispatcher)
DECLARE_DISPATCHER(AddmmFn, test_addmm_dispatcher)

namespace {

const char* RouteKey(const char* key) {
  return key;
}

// DispatchConfigured is operator() verbatim: same kernel type, same arguments.
at::Tensor ConfiguredMm(const at::Tensor& self, const at::Tensor& mat2,
                        c10::ScalarType dtype) {
  return test_mm_dispatcher.DispatchConfigured(self, mat2, dtype);
}

// The shape a decoded pair takes: the dispatcher is asked for the kernel by
// the pair's own conf key, and the kernel is called from a member whose
// signature is the kernel's. Two of them, with different signatures, in one
// translation unit -- the case GCC 11 could not build.
struct MmDtypeCall {
  const Dispatcher<MmFn>& dispatcher;
  MmFn Select(const char* key) const {
    return static_cast<MmFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(MmFn kernel, const at::Tensor& self,
                      const at::Tensor& mat2, c10::ScalarType out_dtype) const {
    return kernel(self, mat2, out_dtype);
  }
};

struct AddmmCall {
  const Dispatcher<AddmmFn>& dispatcher;
  AddmmFn Select(const char* key) const {
    return static_cast<AddmmFn>(dispatcher.DecodedKernel(key));
  }
  decltype(auto) Call(AddmmFn kernel, const at::Tensor& bias,
                      const at::Tensor& self, const at::Tensor& mat2,
                      const c10::Scalar& beta, const c10::Scalar& alpha) const {
    return kernel(bias, self, mat2, beta, alpha);
  }
};

at::Tensor DecodedMmDtype(const at::Tensor& self, const at::Tensor& mat2,
                          c10::ScalarType dtype) {
  const MmDtypeCall call{test_mm_dispatcher};
  return call.Call(call.Select(RouteKey("mm.dtype")), self, mat2, dtype);
}

at::Tensor DecodedAddmm(const at::Tensor& self, const at::Tensor& mat1,
                        const at::Tensor& mat2, const c10::Scalar& beta,
                        const c10::Scalar& alpha) {
  const AddmmCall call{test_addmm_dispatcher};
  return call.Call(call.Select(RouteKey("addmm")), self, mat1, mat2, beta,
                   alpha);
}

// OpKey names the key a dispatcher routes by, for the caller that has to
// resolve by it before it can call (addmm.dtype has no variant entry).
const char* OpKeyOfMm() { return test_mm_dispatcher.OpKey(); }

bool MmAcceptsDecoded() {
  return test_mm_dispatcher.RouteAcceptsDecoded(RouteKey("mm.dtype"));
}

// DispatchAs is the generated wrappers' own call: it resolves by a conf key.
at::Tensor DispatchesAsBmm(const at::Tensor& self, const at::Tensor& mat2) {
  return test_bmm_dispatcher.DispatchAs(RouteKey("bmm"), self, mat2);
}

at::Tensor DispatchesAsMmDtype(const at::Tensor& self, const at::Tensor& mat2,
                               c10::ScalarType dtype) {
  return test_mm_dispatcher.DispatchAs(RouteKey("mm.dtype"), self, mat2, dtype);
}

} // namespace

} // namespace at::native::flagos

int main() { return 0; }
"""


def _torch_include_dirs() -> list[str]:
    torch = pytest.importorskip("torch")
    include = Path(torch.__file__).resolve().parent / "include"
    return [
        str(include),
        str(include / "torch" / "csrc" / "api" / "include"),
    ]


def _compile(workdir: Path) -> subprocess.CompletedProcess[str]:
    source = workdir / "dispatcher_instantiated.cc"
    source.write_text(_TRANSLATION_UNIT, encoding="utf-8")
    includes = [
        "-I",
        str(REPO_ROOT / "csrc"),
        "-I",
        str(REPO_ROOT / "csrc" / "include"),
    ]
    for directory in _torch_include_dirs():
        includes += ["-isystem", directory]
    return subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            "-std=c++17",
            "-fsyntax-only",
            *includes,
            str(source),
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=300,
    )


def test_dispatcher_header_instantiates_every_helper():
    compiler = os.environ.get("CXX", "c++")
    if shutil.which(compiler) is None:
        pytest.skip(f"no C++ compiler named {compiler!r} to check the header with")
    with tempfile.TemporaryDirectory() as tmp:
        try:
            result = _compile(Path(tmp))
        except subprocess.TimeoutExpired:
            pytest.skip("the compiler did not finish within 300 s")
    if "ATen/ATen.h" in result.stderr and "No such file" in result.stderr:
        pytest.skip("this environment has no ATen headers to compile against")
    assert result.returncode == 0, (
        "csrc/aten/dispatcher.h does not compile with its helpers "
        "instantiated:\n" + result.stderr[-4000:]
    )
