---
name: amp-development
description: >
  Automatically develop and land measured automatic mixed precision support for
  torch_fl. Use this when an AMP operator, dtype, vendor runtime, autocast
  route, GradScaler path, or training workload is missing or failing. The
  workflow investigates the failure layer, chooses the existing CUDA/codegen or
  vendor path, implements the smallest complete fix, regenerates artifacts,
  runs CPU-reference and real-hardware tests, and reports unsupported scope
  explicitly. Do not stop at registration or a single operator smoke test.
---

# Automatic AMP development (torch_fl)

## Purpose

This skill turns an AMP request into a repeatable development workflow. It is
for **changing the repository**, not merely checking whether a known AMP path
works. Use [[amp-integration]] for a validation-only request; this skill uses
that contract after implementation.

The workflow is evidence-driven and fail-closed:

```text
request
  -> identify torch/vendor/operator scope
  -> reproduce and classify the failure
  -> choose the existing backend/codegen path
  -> implement the smallest complete change
  -> regenerate and prove idempotency
  -> add or extend contract tests
  -> run CPU-reference and hardware validation
  -> update measured support records
  -> run pre-PR checks and report exact boundaries
```

A successful import, a registered dispatch key, a process-group constructor,
or one matrix multiplication is not an AMP implementation. AMP work is complete
only for the scope that has passed the corresponding gates below.

## Inputs and defaults

Accept a request containing any subset of:

- target accelerator and hardware (`ACCELERATOR`, model, device index);
- PyTorch minor line and vendor torch path;
- target dtype (`float16`, `bfloat16`, or vendor-specific dtype);
- failing operator or AMP component;
- selected route (`cuda` boxing, vendor-native, FlagGems, or unknown);
- required workload (eager, GradScaler, training, DDP/FSDP2);
- whether the user wants a commit or PR.

Make routine defaults without blocking:

- device key: the torch_fl PrivateUse1 name, normally `flagos`;
- validation interpreter: the interpreter that imports the vendor torch used to
  build torch_fl;
- reference: CPU FP32 with tolerances justified per dtype and operator;
- target branch: the current stable PyTorch branch unless the user specifies
  another supported branch;
- implementation route: preserve the existing route unless measurement proves
  that it is the failing layer.

Ask a blocking question only when the target hardware, torch ABI, or requested
scope cannot be determined safely. Never silently substitute a CPU or NVIDIA
wheel for a vendor torch distribution.

## Non-negotiable boundaries

1. **No inferred support.** Do not infer AMP support from an API import, a route
   table, a compiled symbol, or a passing context manager.
2. **No hidden skips.** A skipped test is an evidence gap, not a pass. Record the
   interpreter, skip reason, and hardware availability.
3. **No dtype overclaims.** Do not claim BF16 from FP16, and do not claim a
   vendor-specific dtype merely because torch accepts it.
4. **No distributed overclaims.** Do not claim DDP, FSDP2, or multi-rank AMP from
   single-device eager training.
5. **No handwritten vendor operator kernels when codegen can express the path.**
   Extend the relevant registry, category/template, generated source, and
   routing configuration instead. If codegen cannot express the runtime
   behavior, document the concrete limitation and obtain explicit human review.
6. **No environment leakage.** Do not copy vendor drivers, SDKs, runtimes,
   XMLIR, or Python environments into a wheel or repository artifact.
7. **No destructive GitHub action without authorization.** Push ordinary
   development branches only to the contributor fork; never expose tokens.
8. **Keep claims scoped.** Every final report must distinguish autocast state,
   operator policy, GradScaler, training, distributed AMP, dtype, backend route,
   hardware, and performance.

## Phase 0 — establish the development ledger

Before editing, create an in-memory ledger (or a temporary evidence file outside
the repository) with these fields:

```text
request:
target_branch:
target_accelerator:
hardware_model:
device_key:
torch_version:
torch_path:
torch_fl_revision:
route_and_config:
requested_dtype:
requested_scope:
original_failure:
files_examined:
commands_run:
changes_made:
verification_results:
unsupported_scope:
```

Print the resolved environment using the vendor interpreter:

```bash
python - <<'PY'
import importlib.util
import os
import sys
import torch

# torch_fl must be imported before reading the device key: the PrivateUse1
# rename to "flagos" happens at import time. Without it the key still reads
# "privateuseone" and every autocast/device string in this workflow is wrong.
import torch_fl
from torch_fl._build_config import ACCELERATOR

print("python", sys.version.split()[0])
print("torch", torch.__version__)
print("torch_file", torch.__file__)
print("torch_fl_file", torch_fl.__file__)
print("build_accelerator", ACCELERATOR)
print("env_accelerator", os.environ.get("ACCELERATOR", "<unset>"))
print("device_key", torch._C._get_privateuse1_backend_name())
print("backend_config", os.environ.get("FLAGOS_BACKEND_CONFIG", "<unset>"))
print("flag_gems", importlib.util.find_spec("flag_gems") is not None)
print("flagcx", importlib.util.find_spec("flagcx") is not None)
PY
```

Compare `build_accelerator` from `torch_fl/_build_config.py` against the
requested `ACCELERATOR`. `_build_config.py` is generated at build time and is
what the test gates read, so a mismatch means the installed extension was built
for a different vendor and any AMP result from it is invalid.

Also record the vendor runtime, SDK, visible devices, compiler, relevant
`FLAGOS_*`, `GEMS_*`, `XPU_*`, `CUDA_*`, and `TORCH_*` variables, and the
selected `FLAGOS_BACKEND_CONFIG`. If the environment cannot import the target
vendor torch or discover the requested device, stop implementation claims and
report an environment blocker.

## Phase 1 — reproduce before designing

Reproduce the smallest failing case in a fresh process. Capture stdout,
stderr, exit status, and whether the process crashed. Classify the failure into
exactly one primary layer before editing:

| Symptom | Primary layer | First inspection |
|---|---|---|
| `Autocastflagos` dispatch error | registration | `csrc/aten/autocast.cc`, CMake definitions |
| output dtype is wrong | policy/routing | ATen policy macros, config, selected backend |
| `cudaErrorInvalidDeviceFunction` during cast | vendor cast/runtime | `_copy_from`, `_to_copy`, direct dtype cast |
| non-finite check crash | GradScaler kernel | generated foreach kernel, backend helper |
| scale changes but parameters update on overflow | scaler semantics | found-inf value, unscale, optimizer step |
| finite eager step but wrong training result | numerical/training | CPU reference, gradients, optimizer state |
| rank hang or unequal gradients | distributed AMP | process group, stream/order, scaler sync |
| only FlagGems route fails | compiler route | config, generated wrapper, compiler logs |
| one device index fails | device guard | current device, stream, pointer/view |

Run layer-isolating probes, not a large combined script. For a cast failure,
run direct FP32->target and target->FP32 conversions without autocast. For a
GradScaler failure, invoke `_amp_foreach_non_finite_check_and_unscale_` with
one finite tensor and one tensor containing `inf` or `nan` before testing a
model. Also invoke it twice with separate dtype groups and one shared
`found_inf` tensor: GradScaler groups parameters by `(device, dtype)`, and a
later finite group must not clear an overflow recorded by an earlier group. For
a training failure, first test a scalar parameter and SGD.

Do not change autocast policy to hide a vendor kernel failure. Do not route an
operator to FlagGems merely because its native path failed; prove the selected
route and compare both paths when relevant.

## Phase 2 — inspect the repository path

Read the complete relevant files before editing and trace the call chain:

- `csrc/aten/autocast.cc`: generic `AutocastPrivateUse1` policy registration;
- `csrc/aten/copy_ops.cc`: `_copy_from`, `_to_copy`, stream synchronization,
  and portable conversion paths;
- `csrc/aten/generated/cuda_kernels.cc`: generated CUDA-boxing wrappers;
- `scripts/codegen_ops.py`: source of truth for generated wrappers and configs;
- `csrc/CMakeLists.txt` and the top-level `CMakeLists.txt`: build definitions;
- `torch_fl/__init__.py` and `torch_fl/configs/`: route selection;
- `tests/integration/test_amp.py`: shared AMP contract tests;
- `docs/reference/operator-support.md`,
  `docs/reference/compatibility.md`, and vendor installation docs: measured
  support records.

For a native non-CUDA-compatible backend, also read [[native-op-backend]] and
follow its category/template rules. For a CUDA-compatible vendor, read
[[cuda-compat-vendor]] and preserve CUDA boxing unless measurement rules it out.
For a FlagGems route, read [[flaggems-integration]] and validate that route
separately rather than treating native/boxing evidence as FlagGems evidence.

Search for an existing implementation on another platform and reuse its helper,
macro, guard, test fixture, or codegen category where semantics match. Record
why a reused path is safe for the target vendor.

## Phase 3 — choose the implementation strategy

Use this decision table:

| Measured condition | Strategy |
|---|---|
| Generic policy missing but backend kernels work | enable the existing generic `AutocastPrivateUse1` policy table for the target build |
| CUDA-shaped vendor cast kernel works | keep generated CUDA boxing and add only policy/test wiring |
| CUDA-shaped vendor cast kernel fails | route conversion through an existing portable path; synchronize before blocking copies |
| Generated AMP kernel is ABI-compatible and works | keep generated dispatch and add contract coverage |
| Generated AMP kernel crashes or has a vendor ABI limitation | add a focused backend helper and route to it from the generator |
| FlagGems wrapper is the failing route | fix registry/schema/codegen/config; do not add a handwritten operator kernel |
| Native vendor API is required | add a category/template/mapping entry through native-op codegen |
| Requested dtype is unsupported | fail clearly or mark it not validated; do not silently fall back to a wrong dtype |

Prefer the smallest change that fixes the measured layer. Keep vendor-specific
preprocessor guards narrow (`USE_<VENDOR>`), use existing namespaces and stream
helpers, and make error messages name the AMP operation and expected device.

For a host-staged workaround, document all of the following in code and vendor
docs:

- the exact vendor runtime error or crash that necessitated it;
- the synchronization point before a blocking copy;
- whether strides, empty tensors, and dtype preservation are handled;
- that it is correctness-first and may be slower;
- the follow-up needed for a device-side implementation.

## Phase 4 — implement with generated artifacts

### Autocast registration

Use upstream policy macros (`AT_FORALL_LOWER_PRECISION_FP`, `AT_FORALL_FP32`,
`AT_FORALL_FP32_SET_OPT_DTYPE`, `AT_FORALL_PROMOTE`) and the existing dispatcher
helpers. Do not enumerate individual operators by hand when the generic table
already describes their policy. If safety restrictions such as banned BCE are
changed for another vendor, call out the cross-vendor semantic impact.

### GradScaler

The minimum implementation contract is:

```text
_amp_foreach_non_finite_check_and_unscale_
  -> detect inf/nan
  -> unscale every finite gradient exactly once
  -> preserve found_inf across all (device, dtype) gradient-group calls
  -> let scaler.step skip overflow updates
  -> let scaler.update grow/back off consistently
```

`found_inf` is an accumulator for one optimizer unscale operation. The PyTorch
GradScaler groups gradients by `(device, dtype)` and invokes the primitive once
per group with the same `found_inf` tensor. A finite later group must not reset
an overflow found by an earlier group; test this directly and with mixed
parameter dtypes.

If a generated wrapper is changed, modify `scripts/codegen_ops.py` first and
regenerate `csrc/aten/generated/cuda_kernels.cc`. Include all output variants
that the schema exposes. Do not hand-edit generated artifacts. Check that tensor
lists, `found_inf`, `inv_scale`, outputs, generators, and device views use the
exact generated signatures.

### Tests

Extend `tests/integration/test_amp.py` only when its contract applies to the
new measured accelerator. Otherwise add a focused platform-marked test. Tests
must:

- use the current logical device key, not a hard-coded CUDA API;
- compare values against CPU FP32 references;
- assert dtype, shape, device, finiteness, and parameter state;
- test finite and overflow paths separately;
- leave autocast state and global settings restored;
- skip only when the required hardware/runtime is absent, with a precise reason.

Unit tests must not require vendor hardware. Use source/config fakes for policy
and codegen logic. Hardware tests must use the vendor interpreter and explicit
environment variables.

## Phase 5 — regenerate and prove idempotency

Run the generator in the exact build environment. On Kunlun, the vendor
runtime may be initialized while torch imports, so preserve the runtime
variables required by the vendor interpreter:

```bash
unset XPU_CUPTI_ENABLE_DEVICE
export XPU_ENABLE_PROFILER_TRACING=1
FLAGOS_CODEGEN_ALL=1 python scripts/codegen_ops.py
```

Review every generated file. Restore unrelated generated drift instead of
bundling it. Check the generator exit status **before** comparing output, then
run it again and require identical patches (or an empty intended-scope diff):

```bash
set -eu
FLAGOS_CODEGEN_ALL=1 python scripts/codegen_ops.py
# Save the complete tracked generated patch. Inspect any untracked generated
# files separately before deciding whether they belong in the change.
git diff --binary > /tmp/amp-codegen-first.patch
FLAGOS_CODEGEN_ALL=1 python scripts/codegen_ops.py
git diff --binary > /tmp/amp-codegen-second.patch
cmp -s /tmp/amp-codegen-first.patch /tmp/amp-codegen-second.patch
printf '%s\n' "AMP codegen idempotent"
```

If the complete generator rewrites pre-existing unrelated artifacts, isolate
the intended generated delta and record the pre-existing drift. Never claim
idempotency by comparing only one hand-selected function while silently
committing other generated changes. A failed generator run is a failed gate,
even if two empty or unchanged patch files happen to compare equal.

## Phase 6 — build and run the contract in layers

Build through `setup.py`/`pip`, not by invoking `cmake` directly. `setup.py`
supplies build variables that the top-level `CMakeLists.txt` requires and that
are easy to omit by hand — notably `PYTHON_INCLUDE_DIR`, `PYTORCH_INSTALL_DIR`,
and `CMAKE_INSTALL_PREFIX`. A bare `cmake -S . -B <dir> -DACCELERATOR=<vendor>`
fails at configure time with `Cannot find Python directory`, and it also writes
`torch_fl/_build_config.py`, which the test gates read.

For Kunlun XPU, use the documented vendor invocation:

```bash
unset XPU_CUPTI_ENABLE_DEVICE
export XPU_ENABLE_PROFILER_TRACING=1
export LD_LIBRARY_PATH=/usr/local/xcudart/lib:/usr/local/xpu/lib:$LD_LIBRARY_PATH

ACCELERATOR=kunlun \
  XPU_ROOT=/usr/local/xpu \
  XCUDART_ROOT=/usr/local/xcudart \
  FLAGOS_BUILD_JOBS="$(nproc)" \
  /path/to/vendor/python setup.py build_ext --inplace
```

Use `pip install --no-build-isolation -e .` for an installed editable build, and
`build_ext --inplace` for an in-place source checkout. A plain `build_ext`
without `--inplace` leaves the extension in a temporary directory and
`import torch_fl._C` then fails. Consult the vendor installation document for
the authoritative flags rather than reconstructing them.

Install or select the newly built extension; never validate against a stale
library without printing its path and build identity:

```bash
python -c "import torch_fl, torch_fl._C; print(torch_fl.__file__); print(torch_fl._C.__file__)"
```

Run checks in this order:

1. **State:** enabled/disabled nesting, target dtype, exception cleanup, and
   restoration after a failed operator.
2. **Cast:** direct target casts and reverse casts, including float64 copy and
   empty tensors where applicable.
3. **Operator policy:** lower-precision (`mm`, `linear`, `conv2d`), FP32-
   preserving (`log`, `layer_norm`, loss/reduction), explicit dtype, promotion,
   and safety restrictions.
4. **GradScaler primitive:** one finite and one non-finite gradient; assert
   unscaled values and `found_inf`. Repeat across separate dtype groups with
   one shared `found_inf` and assert overflow accumulation.
5. **GradScaler semantics:** finite update/growth and overflow skip/backoff;
   assert parameters before and after the optimizer step, including an
   overflow in one dtype group with a finite second group.
6. **Single-device training:** deterministic model, fixed inputs, CPU FP32
   reference, finite outputs/gradients/parameters, and optimizer state.
7. **Distributed AMP, only if requested:** at least two ranks; use strict
   FlagCX selection when applicable; assert synchronized overflow decisions and
   optimizer steps. Otherwise report it not validated.

Use the shared suite where the platform gate is justified:

```bash
ACCELERATOR=<vendor> <vendor-env> python -m pytest tests/integration/test_amp.py -v
```

Also run focused config/codegen tests, `ruff check .`, `ruff format --check .`,
`python -m compileall -q torch_fl tests`, and relevant unit tests. Capture actual
output and exit status. If a broad unit suite fails for pre-existing vendor
reasons, rerun at the base or a clean worktree and report both results; never
hide the failure or call it an AMP pass.

## Phase 7 — automated completion gate

Before declaring the development task complete, evaluate this matrix. A box is
checked only with direct evidence:

| Gate | Required proof | Status label |
|---|---|---|
| Environment | intended vendor torch, extension, device, and runtime paths printed | environment resolved / blocked |
| Registration | target autocast dispatch key executes a policy operator | autocast registration validated |
| State | nesting and exception cleanup assertions pass | autocast state validated |
| Policy | all requested policy classes pass CPU-reference comparison | autocast validated for `<dtype>/<operators>` |
| GradScaler primitive | finite/non-finite and unscale assertions pass | non-finite handling validated |
| GradScaler semantics | finite update/growth and overflow skip/backoff pass | GradScaler validated for `<scope>` |
| Training | deterministic model update matches CPU reference | training validated for `<workload>` |
| Distributed | two or more ranks and synchronized scaler decisions pass | distributed AMP validated |
| Generation | second generator run produces no intended-scope diff | codegen idempotent |
| Quality | lint, format, compile, relevant tests pass | quality checks passed |
| Records | docs reflect measured hardware and unsupported scope | support records updated |

A missing gate must be reported as `not validated`, `blocked`, or `not
revalidated`, never as supported. The final response must include:

- files changed and why;
- root cause and the layer that was fixed;
- exact test commands and output summaries;
- hardware/interpreter/torch/runtime identity;
- generated artifact and idempotency result;
- known failures and whether they pre-date the change;
- explicit dtype, route, distributed, and performance boundaries.

## Phase 8 — commit and PR automation

Only when the user requests submission:

1. Read `.github/AI_AGENT_GUIDE.md`, `.github/CLAUDE_CODE_GUIDE.md`, and
   `.github/PULL_REQUEST_TEMPLATE/ai_agent_pr.md`.
2. Fetch the intended upstream branch before linting or rebasing.
3. Isolate the AMP change from unrelated commits and generated drift.
4. Use an English conventional commit with the required Claude co-author trailer.
5. Validate the AI PR body with:

   ```bash
   python scripts/validate_ai_pr.py --pr-body <body-file>
   ```

6. Push only to the contributor fork and create the PR against the requested
   upstream branch. Add `ai-generated` and request a human reviewer who is not
   the PR author.
7. Paste actual P800/XPU or other vendor test output into the PR. Do not paste
   a skip as a pass. Mention pre-existing failures and all unvalidated scope.

Do not force-push a shared branch, expose proxy credentials, or place tokens in
an evidence file, commit, PR body, or log.

## Failure recovery loop

When a gate fails, return to the smallest layer that can explain it:

```text
failure
  -> preserve complete log and exit status
  -> reduce to a minimal reproducer
  -> classify registration / policy / cast / runtime / scaler / training / comm
  -> inspect the existing path and comparable backend
  -> implement one focused change
  -> regenerate if applicable
  -> rerun the failed gate and all dependent gates
```

Never patch several unrelated layers at once. If the vendor runtime is unstable
or unavailable, finish static/codegen work that does not depend on it, then
report the runtime-dependent scope as not validated.

## Related skills

[[runtime-bringup]] · [[torch-version-port]] · [[cuda-compat-vendor]] ·
[[native-op-backend]] · [[flaggems-integration]] · [[pre-pr-checks]]
