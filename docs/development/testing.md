# Testing Guide

This document describes the test structure, pytest markers, and commands for running torch_fl tests.

## Test Structure

Tests are organized by scope and purpose:

| Directory | Responsibility |
|-----------|----------------|
| `tests/unit/` | Fast, isolated unit tests for utilities, routing logic, and device registration |
| `tests/integration/` | End-to-end tests requiring GPU hardware: model inference, training, distributed |
| `tests/integration/ops/` | Individual operator correctness tests against PyTorch reference implementations |
| `tests/manual/` | Manual verification scripts for features that require specific hardware or interactive setup |
| `tests/perf/` | Performance benchmarks and profiling scripts |

## Pytest Markers

Operator tests (`tests/integration/ops/`) use markers to select backend-specific and platform-specific tests. All markers are registered in `tests/integration/ops/conftest.py`.

### Backend Markers

| Marker | Meaning | When to Use |
|--------|---------|-------------|
| `main_ops` | Representative operator in the CI smoke subset | Applied to frequently-used ops (add, matmul, conv, etc.) for fast feedback |
| `anyplatform` | Runs on any accelerator backend | Device-agnostic tests (operator registration, dispatch routing) |
| `cuda` | Requires CUDA boxing kernels or NVIDIA hardware | Tests asserting `-> cuda` backend routing |
| `metax` | Requires MetaX C++ (mxcc) backend | Tests asserting `-> metax` backend routing; skipped when `FLAGOS_METAX_BOXING=1` |
| `ascend` | Requires Ascend ACL backend | Tests asserting `-> ascend` backend routing |
| `musa` | Requires Moore Threads MUSA backend | Tests asserting `-> musa` backend routing |
| `flaggems` | Requires FlagGems runtime path on (`FLAGOS_USE_FLAGGEMS=1`) | Tests asserting `-> flagos_python` or vendor-fallback routing |
| `flaggems_python` | Requires FlagGems Python wrapper backend | Tests checking Python-layer integration (dispatch overhead, GIL behavior) |
| `flaggems_cpp` | Requires FlagGems C++ runtime (`FLAGOS_USE_FLAGGEMS_CPP=1` + wheel built with `FLAGGEMS_KERNEL=ON`) | Tests asserting `-> kFlagOs` (C++) dispatch |

### Cross-backend Contract Markers

Public PyTorch APIs whose behavior must be identical on every backend are tested once, by a single contract module, rather than copied per vendor. These markers are registered in `tests/integration/conftest.py` so they work for direct `tests/integration/` invocation:

| Marker | Meaning |
|--------|---------|
| `profiler` | Selects the whole `torch.profiler` contract (`tests/integration/test_profiler_contract.py`) |
| `profiler_device`, `profiler_kernel`, `profiler_runtime`, `profiler_memcpy`, `profiler_memset`, `profiler_flow`, `profiler_linkage`, `profiler_metadata` | Individual tracer capabilities |
| `amp` | Selects the whole `torch.amp` contract (`tests/integration/test_amp_contract.py`) |
| `amp_device`, `amp_grad_scaler` | AMP device compute and GradScaler route capabilities |

Each contract has a support module (`profiler_support.py`, `amp_support.py`) holding a frozen capability dataclass resolved from the active platform. A test that needs a capability the backend does not provide skips with a reason naming the platform, so an unimplemented vendor route reports as an intentional skip instead of a fabricated pass. Adding a backend means extending one capability table, not adding a test file.

Both support modules share `platform_support.detect_platform()` and must not import torch at module scope: they are loaded as pytest plugins before `torch_fl` preloads its device assets, and importing torch first breaks the required library initialization order.

### Platform Detection

Test filtering is automatic: `conftest.py` detects the active platform from `ACCELERATOR`, `lib/flagos_platform`, or `FLAGOS_BACKEND_CONFIG` and skips tests marked for unavailable backends.

## Running Tests

### Prerequisites

- **Unit tests**: No hardware dependencies; run on CPU-only environments.
- **Integration tests**: Require GPU hardware matching the build (CUDA, MetaX, Ascend, DCU, MUSA, or GCU).
- **Operator tests**: Require the SDK and runtime libraries for the target platform.

### Unit Tests

```bash
pytest tests/unit/ -v
```

### Platform-filtered Operator Tests

Run all operator tests for the current platform:

```bash
pytest tests/integration/ops/ -v
```

Run only main operators (CI smoke subset):

```bash
pytest tests/integration/ops/ -m main_ops -v
```

Run FlagGems-routed operators (requires `FLAGOS_USE_FLAGGEMS=1`):

```bash
FLAGOS_USE_FLAGGEMS=1 pytest tests/integration/ops/ -m "flaggems and main_ops" -v
```

Run FlagGems C++ operators (requires `FLAGGEMS_KERNEL=ON` build + `FLAGOS_USE_FLAGGEMS_CPP=1`):

```bash
FLAGOS_USE_FLAGGEMS_CPP=1 pytest tests/integration/ops/ -m "flaggems_cpp and main_ops" -v
```

### Manual FlagGems Overload Survey

`tests/manual/flaggems_overload_survey.py` measures active FlagGems routes as
exact ATen overloads on real accelerator hardware. It is a manual survey, not a
pytest test, and ordinary CI does not invoke it.

Run it with the generic FlagGems routing configuration:

```bash
python tests/manual/flaggems_overload_survey.py \
  --conf torch_fl/configs/backends_flaggems.conf \
  --out /tmp/flaggems-overloads.json
```

The JSON output contains per-profile evidence and an overload-level summary.
Running the same command resumes an interrupted survey; pass `--rerun` only when
existing results should be replaced. Use `--ops` for a comma-separated subset.
Record the hardware model, torch-fl and FlagGems revisions, configuration hash,
route-set hash, and harness version/hash with every published result. See
[Operator Support](../reference/operator-support.md) for the current baseline
and update procedure.

### Per-model Transformers Coverage Probe

`tests/manual/transformers_model_probe.py` measures one randomly initialized,
tiny HuggingFace transformers architecture at a time on a real accelerator. It
is deliberately separate from the model integration tests and from CI: a full
architecture sweep is an external loop over isolated single-model runs, not one
long process whose later results could be poisoned by an earlier device fault.

The probe defaults to the latest installed `transformers` and accepts an older
version explicitly. It does not install dependencies automatically; install the
requested version in the active torch-fl environment without replacing its
PyTorch:

```bash
python -m pip install --upgrade --no-deps transformers==5.16.1
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python tests/manual/transformers_model_probe.py \
  --model qwen3 --device flagos --out /tmp/qwen3.json
```

The probe runs transfer, forward, backward/optimizer, and short deterministic
generation layers in order, comparing the device result with a CPU result using
the same weights, seed, and dtype. A failure stops that model's later layers.
The JSON preserves layer status, CPU-fallback/native ATen observations, error
text, and the environment versions needed to interpret the result. A parameter
cap protects composite configurations whose top-level tiny overrides do not
shrink nested configs. Use `--list-models` to inspect the model types available
in the installed version. Issue filing is intentionally not part of this probe;
a future reporter consumes its JSON after fingerprint deduplication and a
baseline comparison.

### Official HuggingFace Transformers tests

`transformers-test <model>` (implemented by
`tests/manual/transformers_hf_tests.py`) runs HuggingFace's complete official test
files for one architecture at a time. It selects only
`transformers/tests/models/<module>/`, where the module name is resolved by
Transformers' `model_type_to_module_name()` (for example, `blip-2` maps to
`blip_2`). Each invocation uses a fresh subprocess so an accelerator fault does
not contaminate another model's result.

The installed Transformers wheel does not include its `tests/` tree. The runner
caches the complete, version-matched source tree under
`~/.cache/torch_fl/hf-tests/transformers-<version>/` (or `HF_COVERAGE_CACHE` /
`--cache-dir`) and verifies the source declaration against the installed wheel.
Use `--source-dir` for an already prepared checkout. It never installs or
upgrades torch or Transformers. Install the requested wheel without dependencies
so the PyTorch build used by torch-fl remains unchanged:

```bash
python -m pip install --no-deps transformers==5.16.1
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python tests/manual/transformers_hf_tests.py \
  --model qwen3 --transformers-version 5.16.1 --out /tmp/qwen3-official.json
```

The runner injects `flagos` through Transformers' official
`TRANSFORMERS_TEST_DEVICE_SPEC` contract using
`tests/manual/hf_device_spec.py`. The spec imports torch-fl before declaring the
PrivateUse1 device name; this is required because Transformers 5.16.1 validates
`TRANSFORMERS_TEST_DEVICE` with `torch.device()` before it loads a custom spec.
CUDA-only tests remain upstream skips and are recorded separately from ordinary
skips. Per-test JSON evidence includes the
pytest node ID, status, duration, failure detail, and captured output; collection
errors, missing optional dependencies, timeouts, and accelerator crashes are
kept distinct. `--offline` requires a previously cached source tree and also
sets the HuggingFace offline environment variables.

This runner is an execution and evidence tool only. It does not create GitHub
issues. Baseline promotion, failure fingerprinting, duplicate search, and issue
comments/creation consume these JSON results separately. To run every architecture
in the pinned Transformers registry, use the explicit all mode:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python tests/manual/transformers_hf_tests.py \
  --all --transformers-version 5.16.1 --out /tmp/transformers-all.json
```

All mode is a long hardware sweep, but each architecture still runs in its own
subprocess. "All" means the architecture test directories of the installed
Transformers registry; it does not enumerate Hub checkpoints or download
pretrained weights. Registry keys that share a test directory are visited once:
transformers 5.16.1 exposes 709 keys over 492 directories, because component
configs such as `blip_text_model` map to `tests/models/blip`. With `--out`, the
aggregate JSON is rewritten after every architecture, so an interrupted sweep
keeps what it already measured. The aggregate reports `attempted` separately
from `completed`, and an architecture absent from the pinned version is recorded
as `NOT_IN_VERSION` rather than as a failure.

### Model Tests

Model integration tests accept command-line options for model path and hyperparameters.


**Inference test**:

```bash
pytest tests/integration/test_inference.py --model <model-path> --max-new-tokens 128 -v
```

**Training test**:

```bash
pytest tests/integration/test_train.py --model <model-path> --steps 10 --batch-size 2 -v
```

Replace `<model-path>` with a Hugging Face model identifier (e.g., `Qwen/Qwen3-0.6B`) or a local directory containing model weights.

### Distributed Tests

Distributed tests require multi-GPU hardware and a collective communication backend (NCCL, FlagCX, or HCCL).

```bash
pytest tests/integration/test_distributed.py -v
```

### torch.compile Tests

Compile integration tests verify operator compatibility with PyTorch's inductor backend.

```bash
pytest tests/integration/test_compile.py -v
```

**Note**: Compile tests require a working torch.compile environment. On some platforms, additional environment setup may be needed (see [torch.compile Integration](../architecture/torch-compile-integration.md)).

### Profiler Tests

One contract validates the public `torch.profiler` API, Chrome trace export, and device timeline capture on every backend:

```bash
pytest tests/integration/test_profiler_contract.py -m profiler -v
```

Device-only cases (kernel, runtime, memcpy, memset, flow, linkage, metadata) skip on a backend whose tracer does not emit that category. Low-level PrivateUse1 dispatcher and CUPTI-bridge regressions stay in `tests/unit/test_profiler_privateuse1.py`.

### AMP Tests

One contract validates `torch.autocast("flagos")`, the shared `AutocastPrivateUse1` policy groups, and `torch.amp.GradScaler("flagos")` on every backend:

```bash
pytest tests/integration/test_amp_contract.py -m amp -v
```

The autocast API and policy-state cases are device-independent. Device compute, convolution, and GradScaler cases skip on a backend without those routes.

## Code Generation

torch_fl uses code generation to create operator bindings and backend-specific kernels. Generated files live under `csrc/aten/generated/` and should not be edited manually.

### Regenerating Operator Bindings

When PyTorch's operator schema changes or new operators are added, regenerate the CUDA boxing kernels:

```bash
python scripts/codegen_ops.py
```

This reads `torch._C._dispatch_tls_local_include()` and `native_functions.yaml` to generate `csrc/aten/generated/RegisterFlagOS.cpp` and related files.

### Platform-specific Codegen

| Script | Purpose | When to Run |
|--------|---------|-------------|
| `scripts/codegen_ops.py` | CUDA boxing kernels (kCUDA dispatch) | After PyTorch version bump or schema changes |
| `scripts/codegen_autograd.py` | Autograd backward operator wrappers | After adding autograd support for new operators |
| `scripts/codegen_ascend.py` | Ascend ACL kernel wrappers | After updating CANN version or adding Ascend ops |
| `scripts/codegen_gcu.py` | GCU topsaten kernel wrappers | After updating TopsRider SDK or adding GCU ops |
| `scripts/codegen_mudnn.py` | MUSA mudnn kernel wrappers | After updating MUSA toolkit or adding MUSA ops |

**Environment variables for codegen**:

- `FLAGOS_CODEGEN_ALL`: Set to `1` to regenerate all operator bindings (slow; usually unnecessary).
- Platform SDK paths (`ASCEND_HOME`, `TOPS_HOME`, `MUSA_HOME`, etc.): Required by platform-specific codegen scripts.

## Linting

torch_fl enforces code style with ruff. CI runs these exact commands:

```bash
ruff check .
ruff format --check .
git diff --check
```

**Required ruff version**: `0.15.12` (pinned in CI; install with `pip install ruff==0.15.12`).

Fix formatting issues automatically:

```bash
ruff format .
```

Fix linting issues automatically where possible:

```bash
ruff check --fix .
```

`git diff --check` detects trailing whitespace and conflicts. Run it before committing to catch issues early.

## CI Test Selection

GitHub Actions CI runs a subset of tests based on pytest marks:

- **Smoke tests**: `-m main_ops` (CUDA boxing kernels only)
- **FlagGems tests**: `-m "flaggems and main_ops"` (when `FLAGOS_USE_FLAGGEMS=1`)
- **Platform tests**: Vendor-specific runners filter by platform marker (e.g., `-m "ascend and main_ops"`)
- **Cross-backend contracts**: platform manifests under `.github/configs/` run the same `-m amp` and `-m profiler` commands rather than per-vendor test files. A manifest omits a contract only when the gap is recorded in the file (GCU currently omits the profiler contract: its CPU-only PyTorch/Kineto image supplies no PrivateUse1 resolver, so TOPSPTI activities never surface as device events).

To replicate CI behavior locally, use the same mark expressions shown above.

`.github/configs/<platform>.yml` is the only place that defines what a platform runs. Platforms
without dedicated hardware requirements go through `all-tests-common.yml`; ascend, cuda, dcu, and
metax keep their own pipelines (hardcoded runner labels and container images) but pull the test
list through `load-platform-tests.yml`, an `ubuntu-latest` job that parses the config with `yq`
and hands the result to the hardware job. Adding or renaming a step means editing only the
config: nothing lists tests inline. Keeping `yq` off the vendor images is deliberate, so no
accelerator container needs a YAML dependency.
