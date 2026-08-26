---
name: amp-integration
description: >
  Integrate and validate automatic mixed precision for torch_fl across the
  flagos PrivateUse1 device, including autocast policy, dtype promotion,
  GradScaler, non-finite detection, model training, vendor capability gates,
  and CPU-reference evidence. Do not infer AMP support from import success or
  one lower-precision operator.
---

# AMP integration (torch_fl)

## Scope and prerequisites

Automatic mixed precision (AMP) is a coordinated runtime feature, not a dtype
alias. A complete integration covers autocast state, operator-specific dtype
policy, accumulation precision, gradient scaling, non-finite detection,
optimizer-step skipping, and the interaction with the selected torch_fl
operator route.

AMP support is ready only for the measured scope. A successful `torch.autocast`
context or one matrix multiplication does not prove training correctness.

Complete these first:

- [[runtime-bringup]] — device allocation, copies, streams, synchronization, and
  device guards pass on the target hardware.
- [[torch-version-port]] — the active torch minor line matches AMP APIs,
  dispatcher registrations, and generated bindings.
- [[cuda-compat-vendor]] or [[native-op-backend]] — identify whether lower
  precision is handled by CUDA boxing, a vendor-native kernel, or FlagGems.
- [[flaggems-integration]] — if autocast reaches FlagGems routes, validate those
  routes separately against CPU references.
- [[pre-pr-checks]] — fetch/rebase before linting or opening a PR.

## Non-negotiable correctness contract

A reported AMP integration must distinguish these claims:

| Claim | Required evidence |
|---|---|
| Autocast state | enabled/disabled nesting and target dtype are correct for `flagos` |
| Operator policy | lower-precision, FP32-preserving, explicit-dtype, and promotion cases match the documented policy |
| GradScaler | finite step, scale growth, overflow detection, skipped optimizer step, and scale backoff pass |
| Training | a model forward/backward/optimizer step remains finite and matches a CPU/reference baseline within a declared tolerance |
| Distributed AMP | DDP/FSDP or gradient synchronization with scaled gradients passes separately; do not infer it from eager AMP |
| Vendor scope | each dtype/backend combination is measured on actual hardware or labeled not validated |

If a gate is absent, use **AMP not validated for `<scope>`**, not generic
"AMP supported".

## Step 0 — record the runtime and policy

Capture the exact interpreter and runtime before changing code:

```bash
python - <<'PY'
import os
import sys
import torch
import torch_fl

print("python", sys.version)
print("torch", torch.__version__)
print("torch file", torch.__file__)
print("torch_fl", torch_fl.__file__)
print("accelerator", os.environ.get("ACCELERATOR", "<unset>"))
print("GEMS_VENDOR", os.environ.get("GEMS_VENDOR", "<unset>"))
print("FLAGOS_USE_FLAGGEMS", os.environ.get("FLAGOS_USE_FLAGGEMS", "<unset>"))
print("autocast dtypes", {
    device: torch.get_autocast_dtype(device)
    for device in ("flagos", "cuda", "cpu")
})
print("GradScaler", torch.amp.GradScaler)
PY
```

Record:

- torch and torch_fl revisions;
- Python ABI and vendor torch path;
- device model, driver, SDK, and visible device mapping;
- supported AMP dtypes (`float16`, `bfloat16`, or vendor-specific limits);
- whether autocast uses `flagos`, `cuda`, or another device key;
- selected operator route for tested operations;
- whether GradScaler uses device-side non-finite checks or a fallback;
- compiler/runtime flags and relevant `AMP_*`, `TORCH_*`, and vendor variables.

Do not replace a vendor torch distribution with a CPU or NVIDIA wheel merely to
make the AMP API import. Do not call `torch.cuda.amp.autocast` for a PrivateUse1
path unless the vendor explicitly aliases its runtime to CUDA and that alias has
been measured.

## Step 1 — validate autocast state and nesting

Use the current device key and test state restoration:

```python
assert not torch.is_autocast_enabled("flagos")
with torch.autocast("flagos", dtype=torch.float16):
    assert torch.is_autocast_enabled("flagos")
    assert torch.get_autocast_dtype("flagos") == torch.float16
    with torch.autocast("flagos", enabled=False):
        assert not torch.is_autocast_enabled("flagos")
    assert torch.is_autocast_enabled("flagos")
assert not torch.is_autocast_enabled("flagos")
```

Also test exception cleanup and nested contexts with different dtypes. A failed
operator must not leave autocast enabled or change the previous target dtype.
Test `torch.set_autocast_enabled`, `torch.set_autocast_dtype`, and thread/process
isolation when those APIs are used by the workload.

## Step 2 — validate operator dtype policy

Build a small matrix rather than sampling only `mm`:

1. **Lower precision:** `mm`, `linear`, and `conv2d` return the requested AMP
dtype when the backend supports it.
2. **FP32-preserving:** `log`, `layer_norm`, reductions/losses, and numerically
sensitive operations remain FP32 where the policy requires it.
3. **Explicit dtype:** `softmax(..., dtype=torch.float16)` honors an explicit
dtype while the default autocast policy may produce FP32.
4. **Promotion:** mixed `float16`/`float32` inputs such as `atan2` produce the
promoted or policy-required dtype.
5. **Safety restrictions:** unsafe operations such as binary cross entropy raise
the expected autocast error or follow the documented safe alternative.
6. **Unsupported dtype:** unsupported `bfloat16` or `float16` must fail clearly or
fall back according to policy; silently returning an incorrect dtype is failure.

For every case, compare output values against a CPU FP32 reference. Use
`torch.testing.assert_close` with tolerances justified by dtype and operation.
Check shape, device, dtype, finiteness, and alias/in-place behavior where
relevant. Do not infer AMP coverage from the route table alone.

## Step 3 — validate GradScaler semantics

The minimum GradScaler matrix is:

```python
scaler = torch.amp.GradScaler("flagos", init_scale=8.0)

# finite path: parameter updates and scale growth
# overflow path: non-finite check, optimizer step skipped, scale backs off
```

Explicitly test:

- `_amp_foreach_non_finite_check_and_unscale_` detects `inf` and `nan`;
- finite gradients are unscaled exactly once;
- `scaler.step(optimizer)` updates parameters on a finite step;
- an overflow skips the optimizer step;
- `scaler.update()` grows after the configured finite interval;
- `scaler.update()` backs off after overflow;
- scale state remains consistent across repeated iterations;
- sparse or unsupported gradients fail clearly if not supported.

Compare parameter values with a CPU calculation for a simple scalar model. Do
not accept a changed scale as proof that the optimizer step was correctly
skipped; assert both scale and parameter state.

## Step 4 — validate model training

Run a small deterministic model with both lower-precision dtypes supported by
the platform:

```python
model = torch.nn.Sequential(
    torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 4)
).to("flagos:0")
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
scaler = torch.amp.GradScaler("flagos", init_scale=8.0)

with torch.autocast("flagos", dtype=torch.float16):
    loss = torch.nn.functional.mse_loss(model(inputs), targets)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

Use fixed seeds and identical CPU/device inputs. Check:

- expected output and loss dtypes;
- finite activations, gradients, and parameters;
- loss trend or parameter update against a CPU FP32 reference;
- repeated steps with stable scale behavior;
- optimizer state and zeroing semantics;
- checkpoint/save-load of model, optimizer, and scaler state when required.

A device run that merely completes without checking parameter updates is not
training evidence.

## Step 5 — validate distributed AMP separately

AMP and distributed correctness are independent. When DDP is claimed, run at
least two ranks and assert:

- each rank selects the intended process-group backend;
- autocast state and target dtype are correct on every rank;
- scaled gradients are finite before unscale and equal after synchronization;
- overflow on one rank is handled consistently across the group;
- optimizer steps are either taken by all ranks or skipped by all ranks;
- model, optimizer, and scaler states remain synchronized.

For FlagCX-backed communication, use the strict harness from
[[flagcx-integration]] and make native fallback fatal. Do not report DDP AMP
support from a single-rank training loop.

## Step 6 — use the repository tests and add focused coverage

The shared AMP suite is:

```bash
pytest tests/integration/test_amp.py -v
```

It covers autocast nesting, lower-precision and FP32-preserving policies,
explicit dtype, promotion, unsafe BCE, non-finite detection, finite scaler
steps, scale growth, and overflow backoff. Run it with the vendor interpreter
and required runtime environment; the default test environment may skip it.

The suite is gated on the accelerators with measured evidence (DCU and Kunlun
XPU today). Extend that gate only after the vendor's own run passes on real
hardware. A full skip is an evidence gap, not a pass, and a pass on one
accelerator says nothing about another.

The source build must also include the AMP registration translation unit. Check
that `csrc/aten/autocast.cc` is compiled with the vendor's appropriate
`AutocastPrivateUse1` registrations. CUDA-compatible vendors use the generic
`USE_FLAGOS_AUTOCAST` build definition; other vendors need an explicit
compatible registration path before claiming operator autocast support.
`get_amp_supported_dtype()` only advertises candidate dtypes to torch; it does
not register autocast kernels.

For a vendor where the registration unit is absent, first validate the state
API, then report operator AMP as **not validated** rather than treating a
successful context manager as support. Do not add a handwritten per-operator
AMP implementation when the platform's generated registration/codegen path can
express the policy.

The expected failure signature for a missing registration is:

```text
NotImplementedError: Could not run 'aten::<op>' with arguments from the
'Autocastflagos' backend
```

Capture this separately from a vendor kernel failure or a GradScaler failure.
Before changing routing, verify the source/build condition and regenerate or
rebuild the extension in the intended vendor environment.

For vendor validation, use an explicit command such as:

```bash
unset XPU_CUPTI_ENABLE_DEVICE
XPU_ENABLE_PROFILER_TRACING=1 \
  ACCELERATOR=<vendor> GEMS_VENDOR=<vendor> \
  python -m pytest tests/integration/test_amp.py -v
```

A skip is an evidence gap; an `Autocastflagos` dispatch error is a failed AMP
operator integration, not a pass.

For a new platform, add tests under `tests/integration/` with a platform mark
and keep them skipped when the required device runtime is absent. Unit tests
must use fakes or CPU-compatible policy checks and must not require a vendor
runtime. Always include CPU-reference assertions.

Recommended focused commands:

```bash
ruff check .
ruff format --check .
pytest tests/integration/test_amp.py -v
python -m compileall -q torch_fl tests
```

A skipped platform test is an evidence gap, not a pass. Record the skip reason,
interpreter, and hardware availability.

## Step 7 — investigate failures by layer

| Symptom | Likely layer | Response |
|---|---|---|
| autocast context unavailable | torch device-key registration | inspect PrivateUse1 rename and torch AMP APIs |
| output dtype wrong | autocast policy or dispatcher route | inspect op policy and selected backend; compare CPU |
| FP16/BF16 kernel failure | vendor runtime/compiler | test dtype independently without autocast |
| non-finite check crashes | missing AMP kernel or wrong device view | inspect `_amp_foreach_non_finite_check_and_unscale_` registration |
| scaler updates but parameters do not | optimizer/scaler integration | assert before/after parameters and found-inf state |
| DDP hangs or gradients differ | collective ordering or scaler synchronization | run two-rank strict matrix and inspect backend selection |
| only FlagGems path fails | compiler route | disable FlagGems, compare CUDA/native route, then measure separately |
| only one device index fails | device guard/current-device bug | run device 0 and a nonzero index with explicit guards |

Do not fix a vendor kernel by changing AMP policy without proving the policy is
wrong. Keep routing, runtime, and numerical issues separate.

## Step 8 — record evidence and status

For each platform and dtype, record:

- torch-fl, torch, vendor runtime, and FlagGems revisions;
- hardware/device index and driver/SDK;
- autocast device key and default/explicit dtype;
- tested operator matrix and CPU tolerances;
- GradScaler initial/final scale and parameter before/after values;
- model training loss/gradient/parameter evidence;
- distributed world size, selected process group, and scaler synchronization;
- exact commands, output, exit status, and skip/failure reasons.

Use precise status labels:

```text
AMP autocast validated for <dtypes>/<operator scope>
AMP GradScaler validated for <finite/overflow scope>
AMP training validated for <model/workload scope>
AMP distributed training not validated
```

Do not claim BF16 because FP16 passed, or distributed AMP because eager AMP
passed. If hardware is unavailable, mark the relevant scope **not revalidated**.

## Step 9 — pre-PR gate

Before opening a PR:

```bash
git fetch flagos main
git rebase flagos/main
ruff check .
ruff format --check .
pytest tests/integration/test_amp.py -v
python -m compileall -q torch_fl tests
```

Then confirm:

- the active torch and torch_fl are the intended artifacts;
- autocast state is restored after nesting and exceptions;
- every claimed dtype has CPU-reference evidence;
- GradScaler finite and overflow paths both assert parameter state;
- DDP evidence is separate and multi-rank when claimed;
- skipped or unavailable hardware is explicitly reported;
- no vendor runtime, driver, or Python environment was copied into the package;
- all documentation and GitHub-facing text are in English.

## Related

[[runtime-bringup]] · [[torch-version-port]] · [[cuda-compat-vendor]] ·
[[native-op-backend]] · [[flaggems-integration]] · [[flagcx-integration]] ·
[[pre-pr-checks]]
