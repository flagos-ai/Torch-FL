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

"""The kunlunxin vendor library the loader must find, and who names it.

`libtorch_fl.so` links `libxdnn_pytorch.so` by `DT_NEEDED`, and that library is
not self-contained. Every one of these checks is a source-level guard on a fact
that only shows up as `ImportError: libxpu_blas.so: cannot open shared object
file` at run time, on hardware and a vendor package this repository's unit
environment does not have. They pin the contract:

  * `FLAGOS_KUNLUNXIN_XDNN_LIB_DIR` names the same component libraries in the C++
    reader and the documentation
  * the CUDA-compat runtime is loaded before them, because `libxpu_blas.so`
    resolves `cudaGetDriverEntryPoint` out of the global scope
  * the documented failure mode is the one the loader actually produces
  * the pre-load hook is compiled into `libtorch_fl.so` (under
    `aten/backends/`), not into the separately-built `libflagos.so`, whose
    initializers run before the op library is needed

Compiling here is cheap; running is what is not possible off-device.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
KUNLUNXIN = REPO_ROOT / "csrc" / "aten" / "backends" / "kunlunxin"
HEADER = KUNLUNXIN / "xdnn_library.h"
SOURCE = KUNLUNXIN / "xdnn_library.cc"
INSTALL_DOC = REPO_ROOT / "docs" / "vendors" / "kunlunxin" / "installation.md"
ENV_DOC = REPO_ROOT / "docs" / "reference" / "environment-variables.md"
ENV_PY = REPO_ROOT / "torch_fl" / "_env.py"
CMAKE = REPO_ROOT / "csrc" / "CMakeLists.txt"

VARIABLE = "FLAGOS_KUNLUNXIN_XDNN_LIB_DIR"

# What libxdnn_pytorch.so names in DT_NEEDED that the vendor torch package, not
# the XPU SDK, supplies. Measured with `readelf -d` on the P800 install.
COMPONENT_LIBRARIES = [
    "libxpu_blas.so",
    "libxpu_dnn.so",
    "libxpu_flash_attention.so",
]
# Part of the XPU SDK, not the vendor torch package: the install RPATH covers
# it, so it must not be in the "this directory is missing it" list.
SDK_LIBRARY = "libxpuml.so.1"

# Loaded first, by SONAME, so `libxpu_blas.so`'s unversioned
# `cudaGetDriverEntryPoint` resolves. Measured: standalone dlopen of blas fails
# with `undefined symbol: cudaGetDriverEntryPoint`; it succeeds once these three
# are global.
CUDA_RUNTIME_LIBRARIES = [
    "libxpucuda.so",
    "libcudart.so.12",
    "libxpurt.so.2",
]


def _read(path: Path) -> str:
    assert path.is_file(), f"{path} is missing"
    return path.read_text(encoding="utf-8")


def test_header_names_every_component_library():
    """The reader's list is the DT_NEEDED list, not a sample of it."""
    text = _read(HEADER)
    body = text[text.index("XdnnComponentLibraries") :]
    named = re.findall(r'"(libxpu[a-z_]*\.so[^"]*)"', body)
    assert sorted(named) == sorted(COMPONENT_LIBRARIES), named
    # libxpuml.so.1 may only appear in the comment explaining its exclusion.
    code = body[body.index("XdnnComponentLibraries()") :]
    code = code[: code.index("return libs;")]
    assert SDK_LIBRARY not in code


def test_cuda_runtime_is_named_and_loaded_first():
    """Without it, blas fails to relocate; the order is the whole point.

    The header's list is checked, and so is the call order in
    PreloadXdnnComponentsFromEnv: the CUDA runtime loop has to appear before the
    component loop, otherwise blas is dlopen'd into a process that cannot
    resolve it and reports a missing library that is actually present.
    """
    text = _read(HEADER)
    body = text[text.index("XpuCudaRuntimeLibraries") :]
    named = re.findall(r'"(libxpu[a-z_]*\.so[^"]*|libcudart\.so[^"]*)"', body)
    assert sorted(named[: len(CUDA_RUNTIME_LIBRARIES)]) == sorted(
        CUDA_RUNTIME_LIBRARIES
    ), named

    preload = text[text.index("PreloadXdnnComponentsFromEnv()") :]
    runtime_at = preload.index("XpuCudaRuntimeLibraries()")
    components_at = preload.index("XdnnComponentLibraries()")
    assert runtime_at < components_at, (
        "the CUDA-compat runtime must be loaded before the vendor components; "
        "libxpu_blas.so resolves cudaGetDriverEntryPoint from the global scope"
    )
    # Loaded by SONAME off the loader's own path (the SDK), not out of the
    # vendor directory the variable names.
    assert "LoadBySoname(name)" in preload[runtime_at:components_at]


def test_header_reads_the_documented_variable():
    text = _read(HEADER)
    assert f'EnvRaw("{VARIABLE}"' in text


def test_sources_live_where_libtorch_fl_compiles_them():
    """A static initializer outside aten/backends/ runs too early to help.

    csrc/CMakeLists.txt builds runtime/accelerator/<platform>/ into the separate
    libflagos.so and keeps only aten/backends/<vendor>/ in libtorch_fl.so. The
    hook has to be in the library whose NEEDED entry is libxdnn_pytorch.so.
    """
    assert SOURCE.is_file()
    assert HEADER.is_file()
    assert SOURCE.parent == KUNLUNXIN

    cmake = _read(CMAKE)
    assert "runtime/accelerator/.*" in cmake or "runtime/accelerator/" in cmake
    aten_line = next(
        line for line in cmake.splitlines() if "aten/backends/${_vendor_dir}" in line
    )
    assert "EXCLUDE" in aten_line, (
        "aten/backends/<vendor> must be kept for the current accelerator; the "
        "hook relies on being compiled into libtorch_fl.so"
    )


def test_initializer_is_namespace_scope():
    """A function-local static would never run: nothing calls the loader."""
    text = _read(SOURCE)
    assert "const int kXdnnPreloaded" in text
    # The assignment must not sit inside a function body. A namespace-scope
    # definition starts at column 0.
    line = next(row for row in text.splitlines() if "kXdnnPreloaded" in row)
    assert line == line.lstrip(), line


def test_install_doc_documents_the_variable_and_the_failure():
    doc = _read(INSTALL_DOC)
    assert VARIABLE in doc
    # The message a user actually sees, which the doc has to lead them from.
    assert "libxpu_blas.so: cannot open shared object file" in doc
    for name in COMPONENT_LIBRARIES:
        assert name in doc, f"{name} is not documented"
    # The trap this whole change exists to defuse.
    assert "torch_xmlir" in doc
    # The load-order constraint, with the symbol that causes it.
    assert "cudaGetDriverEntryPoint" in doc
    for name in CUDA_RUNTIME_LIBRARIES:
        assert name in doc, f"{name} is not documented"
    assert re.search(
        r"importing the\s+`torch_xmlir` Python package is \*not\* required", doc
    ) or ("Importing the\n`torch_xmlir` Python package is *not* required" in doc), (
        "the doc must say the package is not imported"
    )


def test_env_reference_documents_the_variable():
    assert f"`{VARIABLE}`" in _read(ENV_DOC)
    assert VARIABLE in _read(ENV_PY)


def test_documented_run_order_is_before_the_interpreter():
    """The variable cannot fix a first import that is already failing."""
    doc = _read(INSTALL_DOC)
    assert (
        "before the interpreter starts" in doc
        or "before starting the interpreter" in doc
    )
    # Both the header and the initializer's own comment state the ordering
    # constraint that makes the variable a "set it up front" knob.
    assert "before any initializer runs" in _read(SOURCE)
    assert "before any of this code runs" in _read(HEADER)
