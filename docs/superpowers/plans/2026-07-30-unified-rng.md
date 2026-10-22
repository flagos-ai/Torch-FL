# Unified RNG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every native-CUDA RNG kernel draw from the same per-device generator FlagGems reads (`torch.cuda.default_generators[device]`), so `torch.manual_seed` unifies both RNG worlds and all RNG becomes reproducible.

**Architecture:** Add one C++ helper (`GetFlagosDefaultCudaGenerator(int64_t)`) that fetches the shim's per-device CUDA generator and returns it as `at::Generator`. Teach `scripts/codegen/codegen_ops.py` to emit, in every native RNG kernel body carrying a `Generator?` arg, a one-line "inject shared generator when caller passed none" before the `at::<op>` call. Regenerate `cuda_kernels.cc` and rebuild `_C.so`. No routing/conf changes.

**Tech Stack:** C++17, pybind11, PyTorch 2.10 (CPU wheel + external `libtorch_cuda.so`), Python codegen (torchgen), pytest.

## Global Constraints

- Work in a dedicated worktree on branch `rng-completeness-check`; run every command from that worktree's repository root.
- Env: initialize conda through `source "$(conda info --base)/etc/profile.d/conda.sh"`, then `conda activate torch-fl-210` (torch `2.10.0+cpu`). Use the locally configured proxy, if needed, for network access.
- Build: `FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation` (g++, links torch_cpu; CUDA symbols resolve at runtime).
- Single-wheel auto-preload build: run tests with plain `python` / `pytest`. Do NOT use `scripts/vendor/with_cuda_libtorch.sh` — it double-loads `libc10_cuda.so` → "Duplicated key 'graph_capture_record_stream_reuse'" core dump.
- Standalone repro scripts MUST `import torch_fl` BEFORE `import torch` (CUDAHooks / auto-preload order). The lazy CUDA generator needs ATen_cuda live — trigger it with a first RNG/CUDA op.
- Test env for HF models: `HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1`; set `HF_HOME` to a local cache directory.
- RNG reproducibility tests only hold with `FLAGOS_USE_FLAGGEMS=1` for the flaggems_python ops, but native RNG unification is independent of that switch — test both.
- Generated files are committed on-branch. `csrc/aten/generated/cuda_kernels.cc` header says "AUTO-GENERATED - DO NOT EDIT": change the codegen, never hand-edit the generated file.
- Pre-existing unrelated crash to exclude from suites: `tests/integration/ops/test_conv1d_dispatch.py::test_conv1d_with_bias` (segfaults on clean tree).

---

### Task 1: C++ helper `GetFlagosDefaultCudaGenerator`

**Files:**
- Modify: `csrc/aten/backends/flagos/python_op_caller.h` (add declaration in `namespace at::native::flagos`, after `CallPythonOp_RandomInplace` decl ~line 125)
- Modify: `csrc/aten/backends/flagos/python_op_caller.cc` (add definition; file already has pybind + `py::module_::import("torch")` pattern at ~line 124)
- Test: exercised via Task 4 integration tests (no standalone C++ test harness in this repo; the repo tests kernels through Python).

**Interfaces:**
- Consumes: pybind11 (`py::gil_scoped_acquire`, `py::module_::import`), `torch/csrc/utils/pybind.h` (provides `py::cast`/`THPGenerator` unpacking for `at::Generator`).
- Produces: `at::Generator at::native::flagos::GetFlagosDefaultCudaGenerator(int64_t device_index);` — returns the shim's per-device CUDA generator (`torch.cuda.default_generators[device_index]`) as an `at::Generator`. Used by generated kernels in Task 2/3.

- [ ] **Step 1: Add the declaration to the header**

In `python_op_caller.h`, inside `namespace at::native::flagos {`, add after the `CallPythonOp_RandomInplace` declaration:

```cpp
// Fetch the vendor compat shim's per-device CUDA generator
// (torch.cuda.default_generators[device_index]) as an at::Generator. This is
// the SAME object FlagGems reads via philox_backend_seed_offset, so injecting
// it into native RNG kernels unifies both RNG worlds under torch.manual_seed.
// Cached per device index (the shim object is stable; torch.cuda.manual_seed
// mutates it in place, so a cached handle stays valid).
at::Generator GetFlagosDefaultCudaGenerator(int64_t device_index);
```

- [ ] **Step 2: Add the include for Generator pybind support**

In `python_op_caller.cc`, near the existing includes (after `#include <torch/csrc/utils/pybind.h>` at line 8), add:

```cpp
#include <torch/csrc/Generator.h>
#include <ATen/core/Generator.h>
```

- [ ] **Step 3: Implement the helper**

In `python_op_caller.cc`, inside `namespace at::native::flagos {` (append near the other `CallPythonOp_*` definitions, before the closing brace), add:

```cpp
at::Generator GetFlagosDefaultCudaGenerator(int64_t device_index) {
  static std::mutex cache_mu;
  static std::unordered_map<int64_t, at::Generator> gen_cache;
  {
    std::lock_guard<std::mutex> lk(cache_mu);
    auto it = gen_cache.find(device_index);
    if (it != gen_cache.end()) {
      return it->second;
    }
  }
  py::gil_scoped_acquire gil;
  py::module_ torch_cuda = py::module_::import("torch.cuda");
  py::object gens = torch_cuda.attr("default_generators");
  py::object py_gen = gens[py::cast(device_index)];
  // torch.Generator -> at::Generator via THPGenerator unpack.
  at::Generator gen = py_gen.cast<at::Generator>();
  {
    std::lock_guard<std::mutex> lk(cache_mu);
    gen_cache[device_index] = gen;
  }
  return gen;
}
```

- [ ] **Step 4: Verify it compiles**

Run:
```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate torch-fl-210
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation 2>&1 | tail -20
```
Expected: build succeeds (helper is unused for now — no call sites yet — but must compile). If `py_gen.cast<at::Generator>()` fails to resolve, fall back to `THPGenerator_Unpack(py_gen.ptr())` (from `torch/csrc/Generator.h`), which returns `at::Generator`.

- [ ] **Step 5: Commit**

```bash
git add csrc/aten/backends/flagos/python_op_caller.h csrc/aten/backends/flagos/python_op_caller.cc
git commit -m "feat(rng): add GetFlagosDefaultCudaGenerator C++ helper"
```

---

### Task 2: Codegen — inject shared generator into tensor-input RNG kernels

**Files:**
- Modify: `scripts/codegen/codegen_ops.py` — helper predicate + edits to `gen_functional_pure` (~1107), `gen_inplace` (~1151), `gen_out_variant` (~1192), `gen_tuple_return` (~1241)
- Modify (regenerate, do not hand-edit): `csrc/aten/generated/cuda_kernels.cc`
- Test: Task 4.

**Interfaces:**
- Consumes: `GetFlagosDefaultCudaGenerator(int64_t)` from Task 1.
- Produces: generated tensor-input RNG kernels emit, before the `at::<op>` call:
  `if (!generator.has_value()) generator = at::native::flagos::GetFlagosDefaultCudaGenerator(<idx>);`
  where `<idx>` is the first boxed input tensor's `.get_device()`.

- [ ] **Step 1: Add a shared injection-line helper near the body templates**

In `scripts/codegen/codegen_ops.py`, add a module-level helper above `gen_functional_pure` (~line 1105):

```python
def _generator_inject_line(args, device_expr):
    """If this op's schema carries a trailing `Generator?`, emit a line that
    fills an absent generator with the flagos shared CUDA generator (the one
    FlagGems reads), so torch.manual_seed unifies native + flaggems RNG.
    Returns '' for non-RNG ops. `device_expr` is a C++ expr yielding int64
    device index in the kernel body scope."""
    has_gen = any("Generator" in t for t, _ in args)
    if not has_gen:
        return ""
    # The generator parameter is always named `generator` in the faithful sig.
    return (
        f"  if (!generator.has_value()) generator = "
        f"at::native::flagos::GetFlagosDefaultCudaGenerator({device_expr});\n"
    )
```

- [ ] **Step 2: Wire it into `gen_functional_pure`**

In `gen_functional_pure` (~1128), the current return is:
```python
    return f"""{ret_type} {kn}({args_decl(args)}) {{
{holder_lines}  DeviceBoxingGuard guard({guard});
  auto result = {api}({call_args(args)});
  UnboxToFlagos(result);
  return result;
}}"""
```
Change it to insert the inject line after the guard. The device index comes from the first boxed tensor (`guard_names[0]`), which is on CUDA after boxing:
```python
    inject = _generator_inject_line(args, f"{guard_names[0]}.get_device()") if guard_names else ""
    return f"""{ret_type} {kn}({args_decl(args)}) {{
{holder_lines}  DeviceBoxingGuard guard({guard});
{inject}  auto result = {api}({call_args(args)});
  UnboxToFlagos(result);
  return result;
}}"""
```

- [ ] **Step 3: Wire it into `gen_inplace`**

In `gen_inplace` the body (~1180-1189) is built as `at::{base}(...)` or `self.method(...)` with a `DeviceBoxingGuard guard(self, ...)`. Insert the inject line after the guard line, using `self` as the device source. Locate the `f"""..."""` return block and add `{inject}` right after the `DeviceBoxingGuard guard(...)` line, with:
```python
    inject = _generator_inject_line(args, "self.get_device()")
```
(The first arg of an inplace RNG op is always `at::Tensor & self`.)

- [ ] **Step 4: Wire it into `gen_out_variant`**

In `gen_out_variant` (~1226/1234) the body uses `DeviceBoxingGuard guard(...)` over boxed inputs+out, then calls `at::{base}_outf(...)`. Insert `{inject}` after the guard line with the device source being the `out` tensor (always present and boxed) or the first guarded name:
```python
    inject = _generator_inject_line(args, f"{out_names[0]}.get_device()")
```
Use whatever variable name the function already computes for the out tensor(s); if it is a list, use its first element. If no out name variable exists, use the first entry of the guard names list.

- [ ] **Step 5: Wire it into `gen_tuple_return`**

`gen_tuple_return` (~1241, e.g. `PrivFusedDropout`) has `DeviceBoxingGuard guard(self, ...)`. Insert `{inject}` after the guard line with `self.get_device()` as device source (first tensor arg). Match the existing variable the function uses for guarded names; if it computes `guard = ", ".join(tensor_arg_names(args))`, use `tensor_arg_names(args)[0]`.

- [ ] **Step 6: Regenerate and eyeball the output**

Run:
```bash
python scripts/codegen/codegen_ops.py
grep -A4 "NormalInplaceKernelCuda\|MultinomialKernelCuda\|PrivFusedDropoutKernelCuda\|BernoulliTensorOutKernelCuda" csrc/aten/generated/cuda_kernels.cc | head -40
```
Expected: each shows `if (!generator.has_value()) generator = at::native::flagos::GetFlagosDefaultCudaGenerator(<tensor>.get_device());` between the guard and the `at::` call. Non-RNG kernels (e.g. `AddTensorKernelCuda`) unchanged.

- [ ] **Step 7: Build**

```bash
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation 2>&1 | tail -20
```
Expected: builds. If `ops.h`/`cuda_kernels.cc` cannot see `GetFlagosDefaultCudaGenerator`, ensure `cuda_kernels.cc` includes the caller header — add `#include "backends/flagos/python_op_caller.h"` to the codegen's include block for cuda_kernels.cc (search the `_CUDA_INCLUDES`/header-emit section near line 1538 and the cuda_kernels.cc writer near 1716) and regenerate.

- [ ] **Step 8: Commit**

```bash
git add scripts/codegen/codegen_ops.py csrc/aten/generated/cuda_kernels.cc
git commit -m "feat(rng): inject shared generator into tensor-input native RNG kernels"
```

---

### Task 3: Codegen — inject into factory RNG kernels (rand/randint/randperm/randn)

**Files:**
- Modify: `scripts/codegen/codegen_ops.py` — `gen_factory` (~1399), compute-factory branch (~1449-1469)
- Modify (regenerate): `csrc/aten/generated/cuda_kernels.cc`
- Test: Task 4.

**Interfaces:**
- Consumes: `GetFlagosDefaultCudaGenerator(int64_t)` (Task 1), `_generator_inject_line` (Task 2).
- Produces: factory RNG kernels (`Rand*Generator*`, `Randint*Generator*`, `RandpermGenerator*`, `Randn*Generator*`) inject the shared generator using `_cuda_dev.index()` before the `at::<base>(...)` call.

- [ ] **Step 1: Inject in the compute-factory branch**

In `gen_factory`, the compute branch (~1462) currently builds `make` as:
```python
            make = (
                f"  at::Device _req_dev = {device_arg}.has_value() ? *{device_arg} "
                f": at::Device(at::kPrivateUse1, 0);\n"
                "  at::Device _cuda_dev = _req_dev.type() == at::kPrivateUse1\n"
                "      ? at::Device(at::kCUDA, _req_dev.index()) : _req_dev;\n"
                f"  auto result = at::{base}({', '.join(call_names)});\n"
                "  if (result.device().type() == at::kCUDA) UnboxToFlagos(result);"
            )
```
Insert the generator injection after `_cuda_dev` is computed and before the `at::{base}` call. `_cuda_dev` is in scope; use its index:
```python
            inject = _generator_inject_line(args, "_cuda_dev.index()")
            make = (
                f"  at::Device _req_dev = {device_arg}.has_value() ? *{device_arg} "
                f": at::Device(at::kPrivateUse1, 0);\n"
                "  at::Device _cuda_dev = _req_dev.type() == at::kPrivateUse1\n"
                "      ? at::Device(at::kCUDA, _req_dev.index()) : _req_dev;\n"
                f"{inject}"
                f"  auto result = at::{base}({', '.join(call_names)});\n"
                "  if (result.device().type() == at::kCUDA) UnboxToFlagos(result);"
            )
```
The non-compute branches (`zeros`/`ones`/`full`/`scalar_tensor`) carry no `Generator?` arg, so `_generator_inject_line` returns `""` there — leave them unchanged.

- [ ] **Step 2: Regenerate and eyeball**

```bash
python scripts/codegen/codegen_ops.py
grep -A8 "RandintLowGeneratorKernelCuda\|RandpermGeneratorKernelCuda\|RandGeneratorKernelCuda" csrc/aten/generated/cuda_kernels.cc | head -40
```
Expected: each factory RNG kernel shows `if (!generator.has_value()) generator = at::native::flagos::GetFlagosDefaultCudaGenerator(_cuda_dev.index());` right before its `at::rand/randint/randperm(...)` call.

- [ ] **Step 3: Build**

```bash
FLAGGEMS_KERNEL=OFF FLAGGEMS_PYTHON=OFF CUDA_KERNEL=ON pip install -e . --no-build-isolation 2>&1 | tail -20
```
Expected: builds.

- [ ] **Step 4: Commit**

```bash
git add scripts/codegen/codegen_ops.py csrc/aten/generated/cuda_kernels.cc
git commit -m "feat(rng): inject shared generator into factory RNG kernels"
```

---

### Task 4: Reproducibility tests for the unified native RNG

**Files:**
- Create: `tests/integration/ops/test_rng_unified_dispatch.py`
- Test: itself.

**Interfaces:**
- Consumes: the injected native RNG kernels (Tasks 2-3). No new symbols.
- Produces: pytest coverage asserting reproducibility + seed-sensitivity for native RNG families and randperm.

- [ ] **Step 1: Write the failing test**

Create `tests/integration/ops/test_rng_unified_dispatch.py`:

```python
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

"""Unified RNG reproducibility for NATIVE (= cuda) RNG ops.

After the C++ generator-injection change, native RNG kernels fall back to the
shared per-device CUDA generator (torch.cuda.default_generators[device], the one
FlagGems reads) when the caller passes no generator. So torch.manual_seed now
makes native RNG reproducible too -- unifying the two RNG worlds. These ops route
to `= cuda` regardless of FLAGOS_USE_FLAGGEMS, so this test does not gate on it.
"""

import pytest

import torch_fl  # noqa: F401  (import before torch; installs preload + shim)
import torch

DEVICE = "flagos:0"


def _reproducible(op):
    torch.manual_seed(1234)
    a = op()
    torch.manual_seed(1234)
    b = op()
    return torch.equal(a.cpu(), b.cpu())


def _seed_sensitive(op):
    torch.manual_seed(1)
    a = op()
    torch.manual_seed(2)
    b = op()
    return not torch.equal(a.cpu(), b.cpu())


# Warm ATen_cuda + build the lazy shared generator before the parametrized runs.
@pytest.fixture(scope="module", autouse=True)
def _warmup():
    torch.manual_seed(0)
    _ = torch.rand(8, device=DEVICE)


_NATIVE_RNG_OPS = {
    "normal_": lambda: torch.empty(256, device=DEVICE).normal_(0.0, 1.0),
    "randint": lambda: torch.randint(0, 1000, (256,), device=DEVICE),
    "randint_low": lambda: torch.randint(5, 1000, (256,), device=DEVICE),
    "bernoulli_tensor": lambda: torch.bernoulli(
        torch.full((256,), 0.5, device=DEVICE)
    ),
    "uniform_func": lambda: torch.empty(256, device=DEVICE).uniform_(2.0, 5.0),
    "log_normal_": lambda: torch.empty(256, device=DEVICE).log_normal_(),
    "cauchy_": lambda: torch.empty(256, device=DEVICE).cauchy_(),
    "randperm": lambda: torch.randperm(512, device=DEVICE),
}


class TestNativeRngReproducible:
    @pytest.mark.parametrize("name", list(_NATIVE_RNG_OPS))
    def test_same_seed_same_draw(self, name):
        assert _reproducible(_NATIVE_RNG_OPS[name]), (
            f"{name} not reproducible under torch.manual_seed"
        )

    @pytest.mark.parametrize("name", list(_NATIVE_RNG_OPS))
    def test_different_seed_differs(self, name):
        assert _seed_sensitive(_NATIVE_RNG_OPS[name]), (
            f"{name} not sensitive to seed changes"
        )


class TestGeneratorPassthrough:
    """A user-supplied generator must still take effect (injection skipped)."""

    def test_explicit_generator_reproducible(self):
        g = torch.Generator(device="cuda")
        g.manual_seed(77)
        a = torch.empty(64, device=DEVICE).normal_(generator=g)
        g.manual_seed(77)
        b = torch.empty(64, device=DEVICE).normal_(generator=g)
        assert torch.equal(a.cpu(), b.cpu())
```

- [ ] **Step 2: Run to verify it fails on a pre-change build (informational)**

If run against the OLD build, `normal_`/`randint`/`randperm`/... would fail reproducibility. Since Tasks 1-3 already changed the build, run:
```bash
python -m pytest tests/integration/ops/test_rng_unified_dispatch.py -v 2>&1 | tail -30
```
Expected: PASS (proves the injection works). If any native op is still non-reproducible, its kernel body was missed by Tasks 2-3 — grep `cuda_kernels.cc` for that kernel and confirm the inject line is present; if absent, its category template needs the same treatment.

- [ ] **Step 3: Run the full RNG suites together (flaggems path unaffected)**

```bash
FLAGOS_USE_FLAGGEMS=1 HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python -m pytest tests/integration/ops/test_rng_dispatch.py tests/integration/ops/test_rng_unified_dispatch.py -v 2>&1 | tail -40
```
Expected: existing `test_rng_dispatch.py` (10) still PASS; new tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/ops/test_rng_unified_dispatch.py
git commit -m "test(rng): reproducibility for unified native RNG + generator passthrough"
```

---

### Task 5: Regression sweep + doc note

**Files:**
- Modify: `docs/superpowers/specs/2026-07-30-unified-rng-design.md` (append a short "Implemented" note with verified results) — optional but recommended.
- Test: existing native + flaggems_python suites.

**Interfaces:**
- Consumes: everything above. Produces: confidence that no existing behavior regressed.

- [ ] **Step 1: Run the native ops suite (no flaggems)**

```bash
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate torch-fl-210
export HF_HOME="$HF_HOME" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m pytest tests/integration/ops/ -m "not flaggems and not flaggems_python and not flaggems_cpp" \
  --deselect tests/integration/ops/test_conv1d_dispatch.py::test_conv1d_with_bias 2>&1 | tail -20
```
Expected: same pass/skip counts as baseline (~311 passed per env memory), no new failures.

- [ ] **Step 2: Run the flaggems_python suite**

```bash
FLAGOS_USE_FLAGGEMS=1 python -m pytest tests/integration/ops/ -m "flaggems_python" \
  --deselect tests/integration/ops/test_conv1d_dispatch.py::test_conv1d_with_bias 2>&1 | tail -20
```
Expected: no new failures vs baseline.

- [ ] **Step 3: Multi-device spot check (if host has >1 GPU)**

Create a throwaway check (delete after):
```bash
cat > /tmp/rng_multidev.py <<'PY'
import torch_fl
import torch
for idx in (0, 1):
    d = f"flagos:{idx}"
    try:
        torch.manual_seed(5); a = torch.empty(32, device=d).normal_()
        torch.manual_seed(5); b = torch.empty(32, device=d).normal_()
        print(d, "reproducible:", torch.equal(a.cpu(), b.cpu()))
    except Exception as e:
        print(d, "skip:", type(e).__name__, str(e)[:50])
PY
python /tmp/rng_multidev.py 2>&1 | grep -E "flagos:" ; rm -f /tmp/rng_multidev.py
```
Expected: `flagos:0 reproducible: True` and `flagos:1 reproducible: True` (or a clean skip if only 1 GPU is visible).

- [ ] **Step 4: Append implemented-note to the design doc and commit**

Add a brief section at the end of the design doc summarizing verified results (which native families are now reproducible, suite pass counts). Then:
```bash
git add docs/superpowers/specs/2026-07-30-unified-rng-design.md
git commit -m "docs(rng): record unified-RNG verification results"
```

---

## Self-Review

**Spec coverage:**
- Core mechanism (inject shared generator when absent) → Tasks 1-3. ✓
- C++ helper fetching `torch.cuda.default_generators[idx]` as `at::Generator` → Task 1. ✓
- Device-index derivation: tensor-input (`.get_device()`) → Task 2; factory (`_cuda_dev.index()`) → Task 3. ✓
- Exhaustive scope (all 80 generator kernels) → covered because Tasks 2-3 edit the shared body templates (`gen_functional_pure`/`gen_inplace`/`gen_out_variant`/`gen_tuple_return`/`gen_factory`) that emit ALL of them; `_generator_inject_line` self-selects on presence of a `Generator?` arg. ✓
- randperm-for-free → validated in Task 4 (`randperm` in `_NATIVE_RNG_OPS`). ✓
- Backward compat (explicit generator honored) → Task 4 `TestGeneratorPassthrough`. ✓
- Testing (reproducibility, cross-world, regression, multi-device) → Tasks 4-5. ✓
- Risk: per-call GIL cost → Task 1 caches per-index. ✓
- Risk: device-type assert → injected generator is CUDA, ops run post-boxing on CUDA; Task 4 exercises each family so a bad assert surfaces. ✓

**Placeholder scan:** No TBD/TODO; every code step has concrete code. The `gen_inplace`/`gen_out_variant`/`gen_tuple_return` steps reference "the existing variable the function uses" because those emitters compute guard names slightly differently — the implementer must read the 3-4 surrounding lines and use the actual local name. This is a read-then-insert instruction with the exact inject expression given, not a vague placeholder.

**Type consistency:** `GetFlagosDefaultCudaGenerator(int64_t) -> at::Generator` used identically in Tasks 1-3. `_generator_inject_line(args, device_expr) -> str` used identically in Tasks 2-3. The emitted C++ line is byte-identical across tasks (only `<device_expr>` differs). ✓
