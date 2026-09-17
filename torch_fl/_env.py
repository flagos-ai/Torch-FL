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

"""The single access point for torch_fl's own ``FLAGOS_*`` environment variables.

Everything torch_fl reads from the environment goes through this module, so the
truth table is stated once instead of being re-invented at each call site. Before
this module existed there were a dozen parsers that disagreed: some accepted
anything that was not ``0`` (so ``off`` meant *on*), the profiler shims enabled
their logging on the *existence* of the variable (so ``=0`` turned it on), and
``ALL_USE_FLAGGEMS=off`` entered strict dispatch mode in C++ while leaving the
routing table alone in ``common.cc`` -- a pair of booleans that collided. That
pair is ``FLAGOS_FORCE_BACKEND`` now, one enum, so the disagreement is not
expressible rather than merely fixed.

Importing this module has no side effects beyond the one-shot scan described
under `check_environment`. It imports nothing from ``torch_fl`` and nothing from
``torch``, so it loads standalone — ``tests/unit/test_env_registry.py`` pulls it
in by path without a torch install.

The C++ mirror is ``csrc/include/flagos_env.h``; the two must stay in step.
"""

from __future__ import annotations

import functools
import importlib.machinery
import importlib.util
import os
import sys
from typing import Any

__all__ = [
    "FOREIGN",
    "RETIRED",
    "SCOPE_BUILD",
    "SCOPE_BUILD_RUNTIME",
    "SCOPE_CODEGEN",
    "SCOPE_RUNTIME",
    "SCOPE_TEST",
    "VARIABLES",
    "build_accelerator",
    "build_kernels",
    "choice",
    "flag",
    "listed",
    "path",
    "set_foreign",
    "value",
    "warn",
]

# ---------------------------------------------------------------------------
# Truth table
# ---------------------------------------------------------------------------
#
# Stated once, here, and mirrored in csrc/include/flagos_env.h. A value outside
# both sets is not silently truthy: it warns and falls back to the default. That
# single rule is what removes the profiler shims' existence-based bug, where
# FLAGOS_CUPTI_SHIM_DEBUG=0 enabled logging.
# The empty string is absent from both sets on purpose: _raw() maps it to
# "unset", so it never reaches the table and always yields the default. A switch
# that defaults on (FLAGOS_ALIAS_CUDA) is therefore on when set to "", which is
# what the shell idiom `FLAGOS_ALIAS_CUDA=$MAYBE_UNSET` needs.
_TRUTHY = frozenset({"1", "true", "on", "yes"})
_FALSY = frozenset({"0", "false", "off", "no"})

# (name, value) pairs already warned about, so a hot read path does not spam.
_warned: set[tuple[str, str]] = set()


def warn(message: str) -> None:
    """Emit one ``[flagos]`` line to stderr.

    Warnings go to stderr rather than through ``warnings.warn`` so they behave
    the same as the C++ side (``csrc/aten/common.cc``), and so importing torch_fl
    with ``-W error`` cannot turn a bad env var into a hard failure.
    """
    print(f"[flagos] {message}", file=sys.stderr)


def _raw(name: str) -> str | None:
    """The variable's value, or ``None`` when unset or empty.

    An empty string is treated as unset throughout: ``FLAGOS_LOG=`` and an
    unexported ``FLAGOS_LOG`` mean the same thing, which is what a shell idiom
    like ``FLAGOS_LOG=${EXTRA_LOG}`` produces.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def _warn_once(name: str, raw: str, message: str) -> None:
    key = (name, raw)
    if key in _warned:
        return
    _warned.add(key)
    warn(f"{message} ({name}={raw!r})")


def flag(name: str, default: bool = False) -> bool:
    """Read a boolean switch.

    ``1/true/on/yes`` are true, ``0/false/off/no`` and an unset or empty value
    are false, case-insensitively. Anything else warns and returns ``default``
    rather than being treated as truthy.
    """
    raw = _raw(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    _warn_once(name, raw, "not a boolean; using the default")
    return default


def value(name: str, default: str | None = None) -> str | None:
    """Read a string value, with an unset or empty variable meaning ``default``."""
    raw = _raw(name)
    return default if raw is None else raw


def choice(name: str, allowed: tuple[str, ...], default: str) -> str:
    """Read one of ``allowed`` (case-insensitively), or ``default``.

    Used where a switch names a mode rather than a boolean, so that a typo names
    the alternatives instead of quietly picking one.
    """
    raw = _raw(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in allowed:
        return lowered
    _warn_once(name, raw, f"expected one of {', '.join(allowed)}; using {default!r}")
    return default


def path(name: str, default: str | None = None) -> str | None:
    """Read a filesystem path. No existence check — the caller reports its own."""
    raw = value(name, default)
    return None if raw is None else os.path.expanduser(raw)


def listed(name: str) -> frozenset[str]:
    """Read a comma-separated set of keywords, lowercased.

    ``FLAGOS_LOG=dispatch,fallback`` reads as ``{"dispatch", "fallback"}``. Blank
    and whitespace-only entries are dropped, so a trailing comma is harmless.
    """
    raw = _raw(name)
    if raw is None:
        return frozenset()
    return frozenset(part.strip().lower() for part in raw.split(",") if part.strip())


# ---------------------------------------------------------------------------
# Variables owned by other projects
# ---------------------------------------------------------------------------
#
# torch_fl both reads and writes these, but none of them is ours: each is a
# contract with another package, and renaming one here would be renaming it
# there. They are declared together so the set of foreign namespaces torch_fl
# reaches into is visible at a glance -- and so a write is a named call rather
# than a bare assignment, of which there were twelve across six modules in
# three spellings of "only if the user has not set it" (setdefault; a
# ``not in os.environ`` guard; an ``os.environ.get`` truthiness guard).
#
# The write must stay an environment write rather than a value passed down: the
# consumer is another library that reads os.environ itself, and some of these
# (TORCH_DEVICE_BACKEND_AUTOLOAD, GEMS_VENDOR) have to be in place before that
# library is imported.

FOREIGN: dict[str, str] = {
    "GEMS_VENDOR": "vendor autodetection in FlagGems",
    "TORCH_DEVICE_BACKEND_AUTOLOAD": "torch's device-backend entry-point autoload",
    "FLAGCX_TORCH_BACKEND": "FlagCX's torch plugin selector",
    "TILELANG_DISABLE_CACHE": "tilelang's kernel cache",
    "TRITON_ENABLE_TASKQUEUE": "the FlagTree Ascend Triton launch queue",
    "COMPILE_ARCH": "Enflame's tops compiler architecture",
    "HB_DNN_USER_DEFINED_L2M_SIZES": "the BPU hbdk runtime's L2M sizing",
}


def set_foreign(name: str, value: str) -> bool:
    """Set one of the ``FOREIGN`` variables, and report whether it was written.

    Always ``setdefault`` semantics, with an empty value counting as unset
    (``FOO=$UNSET`` in a shell means the same as not exporting it): every one of
    these is a hint torch_fl offers the other library, and an explicit export by
    the user -- including an empty one, which is how a user says "not this
    vendor" -- is the better answer. No site needs to win instead; a caller that
    has to overwrite a value the user set is a caller that should be telling the
    user, not silently outvoting them.

    An undeclared name is refused: the point of the table is that the complete
    set of foreign variables is knowable by reading one dict.
    """
    if name not in FOREIGN:
        raise KeyError(f"{name} is not a declared foreign variable; add it to FOREIGN")
    if _raw(name) is not None:
        return False
    os.environ[name] = value
    return True


# ---------------------------------------------------------------------------
# The build record
# ---------------------------------------------------------------------------
#
# setup.py writes torch_fl/_build_config.py next to this file at build time, and
# it is the authoritative answer to "what is this wheel?". The environment
# describes what a user is asking for; the record describes what was compiled.
# Run-time code that would otherwise infer the build from an environment
# variable reads these instead, so there is nothing for a stale export to
# contradict.
#
# Loaded by path rather than by import: torch_fl/__init__.py is still executing
# when these are read, and a package import here would recurse into it.


@functools.lru_cache(maxsize=1)
def _build_record() -> dict[str, Any]:
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_build_config.py")
    try:
        loader = importlib.machinery.SourceFileLoader("torch_fl._build_config", path)
        spec = importlib.util.spec_from_loader(loader.name, loader)
        module = importlib.util.module_from_spec(spec)
        loader.exec_module(module)
    except (OSError, ImportError, SyntaxError):
        # A source checkout with no build yet. Callers treat "" / () as
        # "unknown", never as a platform.
        return {"accelerator": "", "kernels": ()}
    return {
        "accelerator": str(getattr(module, "ACCELERATOR", "")).strip().lower(),
        "kernels": tuple(
            str(k).strip().lower() for k in getattr(module, "KERNELS", ())
        ),
    }


def build_accelerator() -> str:
    """The accelerator this wheel was built for, lowercased ("" if unknown)."""
    return str(_build_record()["accelerator"])


def build_kernels() -> frozenset[str]:
    """The kernel sets compiled into this wheel, lowercased.

    Names are the ones setup.py::KERNEL_SET_NAME declares: "vendor", "flaggems",
    "boxing", "flaggems_cpp", "tileops". Empty for an unbuilt source checkout.
    """
    return frozenset(_build_record()["kernels"])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
#
# Every name torch_fl owns, with the scope and default that
# docs/reference/environment-variables.md states. tests/unit/test_env_registry.py
# cross-checks the two, which is what keeps the doc from rotting again.
#
# "torch_fl owns" means the name is ours to define. Names owned by torch
# (TORCH_DEVICE_BACKEND_AUTOLOAD), FlagGems (GEMS_VENDOR, FLAGGEMS_SOURCE_DIR),
# tilelang (TILELANG_DISABLE_CACHE), FlagCX (FLAGCX_TORCH_BACKEND) or a vendor
# SDK (CUDA_HOME, MACA_PATH, ...) are not listed here: we read or set them, but
# renaming them is not ours to do. They are documented in the interoperability
# section of the reference instead.
#
# Fields: name -> (scope, default, purpose)

SCOPE_BUILD = "Build"
SCOPE_RUNTIME = "Runtime"
SCOPE_BUILD_RUNTIME = "Build & Runtime"
SCOPE_CODEGEN = "Build (codegen)"
SCOPE_TEST = "Test"

VARIABLES: dict[str, tuple[str, str, str]] = {
    # --- Build selection -------------------------------------------------
    "FLAGOS_ACCELERATOR": (
        SCOPE_BUILD,
        "cuda",
        "Hardware platform the wheel is built for: cuda, ppu, metax, ascend, "
        "tsingmicro, dcu, gcu, musa, or bpu. Read by setup.py alone -- the wheel "
        "records it in _build_config.py, and that record, not a re-export, is "
        "what every run-time reader consults",
    ),
    "FLAGOS_BUILD_VENDOR": (
        SCOPE_BUILD,
        "ON, OFF on metax",
        "Compile the accelerator vendor's native kernels (a no-op where the "
        "vendor ships none: cuda, dcu, ppu, tsingmicro, bpu). MetaX defaults OFF "
        "because its native path is retired -- the generated CUDA boxing kernels "
        "are what accelerate that platform",
    ),
    "FLAGOS_BUILD_FLAGGEMS": (
        SCOPE_BUILD,
        "ON, OFF on bpu",
        "Compile the FlagGems Python kernel wrappers (calls into Python, no C++ "
        "linking). Set OFF for a slim pure-boxing build",
    ),
    "FLAGOS_BUILD_BOXING": (
        SCOPE_BUILD,
        "ON, OFF on ascend, gcu and musa",
        "Compile the generated CUDA boxing kernels. OFF is a default for "
        "ascend, gcu and musa, which have no CUDA runtime to box onto",
    ),
    "FLAGOS_BUILD_FLAGGEMS_CPP": (
        SCOPE_BUILD,
        "ON on cuda and tsingmicro, off elsewhere",
        "Compile the FlagGems C++ wrapper, which links liboperators.so. Defaults "
        "OFF outside cuda/tsingmicro because that library has to be built for "
        "the vendor's own toolkit and pointed at with FLAGGEMS_DIR; a build with "
        "one may turn this ON explicitly (MetaX's MACA build is the case). "
        "Pinned OFF for dcu, musa and bpu, where no such library exists at all",
    ),
    "FLAGOS_BUILD_TILEOPS": (
        SCOPE_BUILD,
        "ON on cuda, OFF elsewhere",
        "Compile the TileOps kernel wrappers, which are TileLang on SM90 "
        "NVIDIA parts only",
    ),
    "FLAGOS_BUILD_JOBS": (
        SCOPE_BUILD,
        "System CPU count",
        "Parallel jobs for the CMake build. MAX_JOBS and "
        "CMAKE_BUILD_PARALLEL_LEVEL are honoured as lower-priority fallbacks",
    ),
    "FLAGOS_WHEEL_LOCAL": (
        SCOPE_BUILD,
        "SDK-derived",
        "Local version label for the wheel (e.g. metax3.8.1), for dev builds "
        "that must pin the exact SDK",
    ),
    "FLAGOS_SKIP_CUDA_ASSETS": (
        SCOPE_BUILD,
        "0 (off)",
        "Do not bundle an external libtorch_cuda.so into the wheel, for in-tree "
        "builds. The build-time counterpart of FLAGOS_DISABLE_CUDA_ASSETS",
    ),
    "FLAGOS_CUDA_ASSETS_DIR": (
        SCOPE_BUILD,
        ".libtorch_cuda_assets",
        "Directory the external libtorch_cuda.so is copied from when bundling. A "
        "missing directory downgrades to a warning: the wheel then needs a "
        "runtime-supplied libtorch_cuda.so",
    ),
    "FLAGOS_DCU_VENDOR_CORE": (
        SCOPE_BUILD_RUNTIME,
        "0 (off)",
        "Use DTK's forked core libraries instead of the official PyTorch core. "
        "Must match at build and import time",
    ),
    "FLAGOS_PPU_MKL_DIR": (
        SCOPE_BUILD,
        "/usr/local/lib",
        "Directory the PPU libtorch bundling script takes MKL from",
    ),
    # --- Operator routing ------------------------------------------------
    "FLAGOS_BACKEND_CONFIG": (
        SCOPE_RUNTIME,
        "No default",
        "Absolute path to a backends_*.conf file; overrides the conf torch_fl "
        "selects from the build record. For testing and debugging only -- the "
        "wheel's own selection is not written here, and is reported by "
        "torch_fl.backend_config_path()",
    ),
    "FLAGOS_OP_<name>": (
        SCOPE_RUNTIME,
        "No default",
        "Per-operator backend override (e.g. FLAGOS_OP_add__Tensor=cuda); "
        "replace . with __ in op names",
    ),
    "FLAGOS_FORCE_BACKEND": (
        SCOPE_RUNTIME,
        "No default (off)",
        "Repin every op onto one backend family for A/B measurement: flaggems, "
        "vendor, or tileops. An op the target does not implement is reported "
        "on stderr and left on its configured backend; an op it does implement "
        "but this wheel did not compile raises rather than falling back. The "
        "tileops mode repins the ops the conf annotates `# tileops` and "
        "additionally needs the tileops package, an SM90 device and a "
        "FLAGOS_BUILD_TILEOPS=ON build",
    ),
    "FLAGOS_DISABLE_FLAGGEMS_PY": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Leave the FlagGems Python layer unregistered (C++ stub-only mode)",
    ),
    # --- Runtime diagnostics ---------------------------------------------
    "FLAGOS_LOG": (
        SCOPE_RUNTIME,
        "No default (all off)",
        "Comma-separated stderr diagnostics: dispatch (backend chosen per "
        "operator), fallback (each cpu_fallback dispatch), op_cache (Ascend "
        "operator-cache hit/miss statistics). None changes routing",
    ),
    "FLAGOS_TRACE": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Verbose logging in the device profiler shim compiled into this build",
    ),
    "FLAGOS_TRACER_LIBRARY": (
        SCOPE_RUNTIME,
        "Auto-discovered",
        "Override the tracer library the profiler shim dlopens, when the default "
        "path does not match the installed driver",
    ),
    # --- Vendor compatibility --------------------------------------------
    "FLAGOS_ALIAS_CUDA": (
        SCOPE_RUNTIME,
        "1 (on)",
        "Alias the cuda device string to flagos for drop-in compatibility. Set "
        "0 to opt out",
    ),
    "FLAGOS_DISABLE_CUDA_SHIM": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Skip registering the torch.cuda compatibility shim for generic GPU operations",
    ),
    "FLAGOS_METAX_CUDART_SHIM": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Preload the libcudart version-tag shim before import torch. Required "
        "for MetaX with generic PyTorch wheels",
    ),
    "FLAGOS_METAX_COMPAT": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Patch FlagGems torch.cuda device queries for MetaX compatibility",
    ),
    "FLAGOS_DCU_HIP_VERSION": (
        SCOPE_RUNTIME,
        "No default",
        "Override HIP version detection for the DCU runtime",
    ),
    "FLAGOS_DCU_SKIP_RUNTIME_CHECK": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Skip the DCU post-import checks (torch/DTK version alignment and "
        "CUDA-key kernel presence), for deliberately testing a non-matching "
        "wheel pair",
    ),
    "FLAGOS_DCU_SDPA_FLASH": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Keep the fused SDPA backends enabled on DCU. The default disables "
        "flash/mem-efficient SDPA so scaled_dot_product_attention's own choice "
        "agrees with the only kernel this stack can execute",
    ),
    "FLAGOS_DISABLE_APEX_COMPAT": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Disable the optional Apex multi-tensor compatibility layer",
    ),
    "FLAGOS_DIST_FORCE_NCCL": (
        SCOPE_TEST,
        "0 (off)",
        "In the manual MetaX distributed tests, skip FlagCX and use NCCL",
    ),
    "FLAGOS_DCU_SKIP_LEGACY_SMOKE": (
        SCOPE_TEST,
        "0 (off)",
        "In .github/scripts/set_env_dcu.sh, skip the legacy-mode smoke path "
        "(FLAGOS_DCU_VENDOR_CORE=1) after the decoupled gates have run",
    ),
    # --- Assets and libraries --------------------------------------------
    "FLAGOS_DISABLE_CUDA_ASSETS": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Skip preloading the bundled libtorch_cuda.so and CUDA libraries, for "
        "builds that use system libtorch",
    ),
    "FLAGOS_VENDOR_TORCH_LIB": (
        SCOPE_BUILD_RUNTIME,
        "Auto-discovered",
        "Path to the vendor torch's lib directory, used when no bundled "
        "lib_maca/lib_dcu/lib_ppu is present. Only the active accelerator's "
        "build reads it",
    ),
    "FLAGOS_USE_CACHING_ALLOCATOR": (
        SCOPE_RUNTIME,
        "1 (on)",
        "Caching device allocator. Set 0 to hand every allocation straight to "
        "the vendor runtime",
    ),
    # --- Compiler and feature backends -----------------------------------
    "FLAGOS_USE_FLAGTREE": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Assert that a FlagTree build is the active Triton. Required on Ascend "
        "when the compiler is FlagTree: the check fails loudly if the installed "
        "Triton is not FlagTree",
    ),
    "FLAGOS_COMPILE_FALLBACK_EAGER": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Fall back to eager mode when torch.compile encounters unsupported operations",
    ),
    "FLAGOS_TILEOPS_USE_L2": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Use the TileOps L2-cache tier",
    ),
    "FLAGOS_TILEOPS_CACHE_MAX": (
        SCOPE_RUNTIME,
        "512",
        "TileOps instance-cache capacity. Past the cap, results are rebuilt per "
        "call: slower but bounded",
    ),
    "FLAGOS_TILEOPS_DISABLE_ALL_CACHE": (
        SCOPE_RUNTIME,
        "0 (off)",
        "Neutralize every TileLang cache. Correct but slow; must be set before "
        "tileops is imported. Sets TILELANG_DISABLE_CACHE=1",
    ),
    "FLAGOS_TILEOPS_FULL": (
        SCOPE_TEST,
        "0 (off)",
        "In the TileOps codegen tests, run the full manifest workload shape "
        "instead of the small one",
    ),
    # --- Code generation --------------------------------------------------
    "FLAGOS_EXEC_CACHE": (
        SCOPE_CODEGEN,
        "1 (on)",
        "Cache Ascend operator-codegen execution results; 0 forces regeneration",
    ),
    "FLAGOS_CODEGEN_ALL": (
        SCOPE_CODEGEN,
        "0 (off)",
        "Generate routes for the full leaf-CUDA operator set rather than the "
        "supported subset",
    ),
    # --- BPU compiler -----------------------------------------------------
    "FLAGOS_BPU_MARCH": (
        SCOPE_RUNTIME,
        "nash-p",
        "BPU micro-architecture. nash-p is the BPU, nash-e the S100 and nash-m "
        "the S100P",
    ),
    "FLAGOS_BPU_CACHE": (
        SCOPE_RUNTIME,
        "~/.cache/torch_fl_bpu",
        "Directory holding the BPU compiler cache",
    ),
    "FLAGOS_BPU_QUANTIZE": (
        SCOPE_RUNTIME,
        "1 (on)",
        "Quantize BPU kernels. Without it hbdk4 keeps conv in float and lowers "
        "it to the CPU, so the BPU never runs the heavy work",
    ),
    "FLAGOS_BPU_ACT_SCALE": (
        SCOPE_RUNTIME,
        "0.05",
        "Fallback activation scale for tensors with no calibration entry",
    ),
    "FLAGOS_BPU_MLIR_LIBS": (
        SCOPE_RUNTIME,
        "Unset",
        "Directory of the BPU MLIR plugin libraries (libhbtl.so), preloaded by "
        "the x86 compile driver",
    ),
    "FLAGOS_BPU_X86_PYTHON": (
        SCOPE_RUNTIME,
        "Unset",
        "An x86_64 CPython with hbdk4 installed, run under an emulator: hbdk4 "
        "ships x86_64-only wheels",
    ),
    "FLAGOS_BPU_X86_EMULATOR": (
        SCOPE_RUNTIME,
        "Unset",
        "BPU x86_64 emulator binary. Useful because the distro box64 is usually "
        "too old for hbdk4, or the user has one that is not in PATH",
    ),
    "FLAGOS_BPU_X86_STUBS": (
        SCOPE_RUNTIME,
        "<x86 python prefix>/../stubs",
        "Directory of import-only stand-ins for numba and torch, which hbdk4's "
        "ONNX entry point imports unconditionally",
    ),
}

# Names torch_fl used to read and no longer does. An exported value is inert:
# there is no alias and no deprecation window. Listed here so
# tests/unit/test_env_registry.py can assert the whole set has no reader
# anywhere in the tree, and so check_environment stays quiet about them.
RETIRED: frozenset[str] = frozenset(
    {
        # Renamed to FLAGOS_ACCELERATOR.
        "ACCELERATOR",
        # Renamed to FLAGOS_BUILD_*. These are the pre-rename spellings, so a
        # bare VENDOR_KERNEL=1 export is inert rather than silently enabling a
        # kernel set.
        "VENDOR_KERNEL",
        "FLAGGEMS_KERNEL",
        "BOXING_KERNEL",
        "FLAGGEMS_CPP",
        "TILEOPS_KERNEL",
        # Collapsed into FLAGOS_FORCE_BACKEND.
        "ALL_USE_FLAGGEMS",
        "ALL_USE_VENDOR",
        "FLAGOS_USE_TILEOPS",
        # Collapsed into FLAGOS_LOG.
        "FLAGOS_LOG_DISPATCH",
        "FLAGOS_LOG_FALLBACK",
        "FLAGOS_CACHE_STATS",
        # Collapsed into FLAGOS_TRACE.
        "FLAGOS_CUPTI_SHIM_DEBUG",
        "FLAGOS_MUPTI_DEBUG",
        "FLAGOS_MSPTI_DEBUG",
        "FLAGOS_TOPSPTI_DEBUG",
        "FLAGOS_ROCTRACER_DEBUG",
        "FLAGOS_KINETO_SHIM_DEBUG",
        # Collapsed into FLAGOS_TRACER_LIBRARY.
        "FLAGOS_CUPTI_LIBRARY",
        "FLAGOS_MUPTI_LIBRARY",
        "FLAGOS_TOPSPTI_LIBRARY",
        # Collapsed into FLAGOS_VENDOR_TORCH_LIB.
        "FLAGOS_MACA_TORCH_LIB",
        "FLAGOS_DCU_TORCH_LIB",
        "FLAGOS_PPU_TORCH_LIB",
        # Never read; kept here so a stray export stays silent rather than
        # tripping the unknown-name warning.
        "FLAGOS_USE_FLAGGEMS",
        "FLAGOS_USE_FLAGGEMS_CPP",
        "FLAGOS_USE_VENDOR_OPS",
    }
)

# FLAGOS_OP_<op> is the one dynamic family: ~2000 keys generated from the conf
# files, so it is matched by prefix rather than enumerated.
_DYNAMIC_PREFIXES = ("FLAGOS_OP_",)


def check_environment() -> None:
    """Warn about ``FLAGOS_*`` variables torch_fl does not recognise.

    Catches ``FLAGOS_LOG_DISPACH``-class typos, which are otherwise invisible:
    the misspelled variable is simply never read and the setting silently does
    nothing. Retired names are deliberately excluded — a stale export of one is
    inert by design and must stay quiet.

    Called once at ``import torch_fl``. Idempotent, so a test may call it again
    after monkeypatching the environment.
    """
    for name in os.environ:
        if not name.startswith("FLAGOS_"):
            continue
        if name in VARIABLES or name in RETIRED:
            continue
        if name.startswith(_DYNAMIC_PREFIXES):
            continue
        _warn_once(name, os.environ[name], "not a torch_fl variable; ignored")


check_environment()
