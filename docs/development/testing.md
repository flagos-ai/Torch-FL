# Testing Guide

This document describes the test structure, pytest markers, and commands for running torch_fl tests.

For platform selection, routing labels, the required CI gate, `/ci` comments and
bounded network retries, see [CI selection and maintainer controls](ci-controls.md).

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
| `cuda` | Requires a CUDA boxing build (`FLAGOS_BUILD_BOXING=1`); the default FlagGems-primary wheel has no `cuda` route, so the CUDA lane skips these | Tests asserting `-> cuda` backend routing |
| `metax` | Requires MetaX C++ (mxcc) backend | Tests asserting `-> metax` backend routing; skipped on MetaX (its lane builds the boxing wheel) |
| `ascend` | Requires Ascend ACL backend | Tests asserting `-> ascend` backend routing |
| `musa` | Requires Moore Threads MUSA backend | Tests asserting `-> musa` backend routing |
| `flaggems` | Asserts the FlagGems route from `backends_<platform>.conf` | Tests asserting `-> flagos_python` or vendor-fallback routing |
| `flaggems_python` | Requires FlagGems Python wrapper backend | Tests checking Python-layer integration (dispatch overhead, GIL behavior) |
| `flaggems_cpp` | Requires FlagGems C++ runtime (wheel built with `FLAGOS_BUILD_FLAGGEMS_CPP=ON`, which the build record then reports) | Tests asserting `-> kFlagOs` (C++) dispatch |

### Cross-backend Contract Markers

Public PyTorch APIs whose behavior must be identical on every backend are tested once, by a single contract module, rather than copied per vendor. These markers are registered in `tests/integration/conftest.py` so they work for direct `tests/integration/` invocation:

| Marker | Meaning |
|--------|---------|
| `profiler` | Selects the whole `torch.profiler` contract (`tests/integration/test_profiler_contract.py`) |
| `profiler_device`, `profiler_kernel`, `profiler_runtime`, `profiler_memcpy`, `profiler_memset`, `profiler_flow`, `profiler_linkage`, `profiler_metadata` | Individual tracer capabilities |
| `amp` | Selects the whole `torch.amp` contract (`tests/integration/test_amp_contract.py`) |
| `amp_device`, `amp_grad_scaler` | AMP device compute and GradScaler route capabilities |
| `backend_contract` | Gating correctness cases in `test_device_backend_contract.py` |
| `backend_smoke` | Non-raising public device API cases in the same file |
| `backend_capability` | Non-gating capability observations, reported in the test log and JUnit properties |

The profiler and AMP contracts have support modules (`profiler_support.py`,
`amp_support.py`) holding frozen capability dataclasses resolved from the active
platform. A test that needs a capability the backend does not provide skips with
a reason naming the platform, so an unimplemented vendor route reports as an
intentional skip instead of a fabricated pass. Adding a backend means extending
one capability table, not adding a test file. The device backend contract gates
only baseline behavior shared by all platforms; its separate complex-arithmetic
diagnostic reports the observed support state without granting a silent skip to
a baseline case. Mixed-index `add` rejection is part of the correctness gate and
is enforced before selecting a vendor or FlagGems route.

Both support modules share `platform_support.detect_platform()` and must not import torch at module scope: they are loaded as pytest plugins before `torch_fl` preloads its device assets, and importing torch first breaks the required library initialization order.

### Platform Detection

Test filtering is automatic: `conftest.py` detects the active platform from the wheel's build record (`_build_config.py`, plus the `PPU_SDK` environment or the `lib_ppu/` bundle for wheels that predate PPU's own record value), `lib/flagos_platform`, or the name of the conf `torch_fl.backend_config_path()` reports, and skips tests marked for unavailable backends.

## Running Tests

### Prerequisites

- **Unit tests**: No hardware dependencies; run on CPU-only environments.
- **Integration tests**: Require GPU hardware matching the build (CUDA, MetaX, Ascend, DCU, MUSA, or GCU).
- **Operator tests**: Require the SDK and runtime libraries for the target platform.

### Unit Tests

```bash
pytest tests/unit/ -v
```

### Shared device backend contract

Run the small cross-platform gate against an installed Torch-FL wheel and an
available accelerator:

```bash
python -m pytest tests/integration/test_device_backend_contract.py -v -s --tb=short
```

Every platform manifest runs this same command after its environment check.
`tests/integration/conftest.py` exits with `flagos device is not available`
before collecting contract failures when the runner or runtime is unavailable.
Core cases compare deterministic device results and metadata with CPU; the
two-device cases skip only when the hardware exposes fewer than two devices.
The mixed-device cases cover functional, in-place, and `out=` additions; a CPU
scalar tensor remains a legal operand.

The final `backend_capability` case prints the observed complex-arithmetic
status and records it as a JUnit property without failing a platform that does
not support that optional dtype path. Operator matrices, specialized dtype
routes, and storage lifetime belong to their dedicated suites.

### Platform-filtered Operator Tests

Run all operator tests for the current platform:

```bash
pytest tests/integration/ops/ -v
```

Run only main operators (CI smoke subset):

```bash
pytest tests/integration/ops/ -m main_ops -v
```

Run FlagGems-routed operators:

```bash
pytest tests/integration/ops/ -m "flaggems and main_ops" -v
```

Run FlagGems C++ operators (requires a wheel built with
`FLAGOS_BUILD_FLAGGEMS_CPP=ON`):

```bash
pytest tests/integration/ops/ -m "flaggems_cpp and main_ops" -v
```

Whether these tests are collected is decided by the build record, not by the
environment: `conftest.py` reads `KERNELS` from `torch_fl/_build_config.py` and
skips the mark when `flaggems_cpp` is not in it. A wheel with the C++ runtime
compiled in therefore runs them, and a wheel without it skips them, with no
variable to keep in step by hand. Actual routing still follows the conf's
`flaggems_cpp` keys.

### Manual FlagGems Overload Survey

`tests/manual/flaggems_overload_survey.py` measures active FlagGems routes as
exact ATen overloads on real accelerator hardware. It is a manual survey, not a
pytest test, and ordinary CI does not invoke it.

Run it with the platform conf under test (`backends_cuda.conf` on a CUDA host;
each vendor has its own `backends_<platform>.conf`):

```bash
python tests/manual/flaggems_overload_survey.py \
  --conf torch_fl/configs/backends_cuda.conf \
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
version explicitly. Prepare the environment with the `.claude/skills/test-dependencies/SKILL.md` guidance first:
install the requested Transformers wheel without dependencies, then install its
validated non-torch dependencies such as `accelerate` without replacing the
PyTorch build used by torch-fl:

```bash
python -m pip install --no-deps transformers==5.16.1
python -m pip install accelerate tokenizers safetensors huggingface_hub
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
upgrades packages. Prepare the environment with [[test-dependencies]] first;
install the requested wheel without dependencies and install required optional
packages such as `accelerate` explicitly, so the PyTorch build used by torch-fl
remains unchanged:

```bash
python -m pip install --no-deps transformers==5.16.1
python -m pip install accelerate tokenizers safetensors huggingface_hub
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
sets the HuggingFace offline environment variables. Before the first batch the
runner runs a preflight in a child process and records it under
`environment.preflight`; a failure there means nothing was measured, the run
exits `2`, and its report names the check that failed. Exit `0` means measured
and clean, `1` measured with failures to review, `2` not measured at all.

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
pytest tests/integration/test_qwen3_infer.py --model <model-path> --max-new-tokens 128 -v
```

**Training test**:

```bash
pytest tests/integration/test_qwen3_train.py --model <model-path> --steps 10 --batch-size 2 -v
```

Replace `<model-path>` with a Hugging Face model identifier (e.g., `Qwen/Qwen3-0.6B`) or a local directory containing model weights.

### Distributed Tests

Distributed tests require multi-GPU hardware and a collective communication
backend (NCCL, FlagCX, or HCCL). They are manual scripts, not pytest tests --
each spawns processes and binds multiple devices -- and live under
`tests/manual/`:

```bash
# ProcessGroupFlagOS all_reduce / broadcast / all_gather + DDP on N devices
LD_LIBRARY_PATH=... python tests/manual/test_flagos_dist_live.py --world-size 2

# Qwen3 DistributedDataParallel training steps on N devices
HF_HOME=<hf-cache> python tests/manual/test_qwen3_ddp_live.py
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
python scripts/codegen/codegen_ops.py
```

This reads `torch._C._dispatch_tls_local_include()` and `native_functions.yaml` to generate `csrc/aten/generated/RegisterFlagOS.cpp` and related files.

### Platform-specific Codegen

| Script | Purpose | When to Run |
|--------|---------|-------------|
| `scripts/codegen/codegen_ops.py` | CUDA boxing kernels (kCUDA dispatch) | After PyTorch version bump or schema changes |
| `scripts/codegen/codegen_boxing_flaggems_register.py` | Per-platform FlagGems register lists (`register_flaggems_<platform>.inc`) compiled by a FlagGems-primary CUDA-ABI build | After a conf route change on cuda/metax/dcu/ppu/tsingmicro, or a FlagGems op-table change |
| `scripts/codegen/codegen_autograd.py` | Autograd backward operator wrappers | After adding autograd support for new operators |
| `scripts/codegen/codegen_ascend.py` | Ascend ACL kernel wrappers | After updating CANN version or adding Ascend ops |
| `scripts/codegen/codegen_gcu.py` | GCU topsaten kernel wrappers | After updating TopsRider SDK or adding GCU ops |
| `scripts/codegen/codegen_mudnn.py` | MUSA mudnn kernel wrappers | After updating MUSA toolkit or adding MUSA ops |

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
`pyproject.toml` enforces this through ruff's `required-version`, so a different
version stops with `Required version ... does not match the running version ...`
and names both numbers, instead of reporting files the gate never reads.

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
- **FlagGems tests**: `-m "flaggems and main_ops"` (no env gate; the conf decides the route)
- **Platform tests**: Vendor-specific runners filter by platform marker (e.g., `-m "ascend and main_ops"`)
- **Cross-backend contracts**: platform manifests under `.github/configs/` run the same `-m amp` and `-m profiler` commands rather than per-vendor test files. A manifest omits a contract only when the gap is recorded in the file (GCU currently omits the profiler contract: its CPU-only PyTorch/Kineto image supplies no PrivateUse1 resolver, so TOPSPTI activities never surface as device events).

To replicate CI behavior locally, use the same mark expressions shown above.

`.github/configs/<platform>.yml` is the only place that defines what a platform runs. Ascend,
cuda, dcu, gcu, metax, musa, ppu, and kunlunxin all use one parameterized pipeline,
`integration-tests.yml`, dispatched as a matrix from `ci.yml` (one job per platform the discovery
step reports). Each platform's job reads its config through `load-platform-config.yml`, an
`ubuntu-latest` job that parses the config with `yq`, then builds, installs, and tests the wheel
in the same hardware job without transferring a wheel artifact. The runner label, image, mounts,
timeout, provisioning stage, and test manifest all come from the config, so adding or renaming a
step means editing only the config: nothing lists tests inline, and there is no per-platform
workflow to copy. Keeping `yq` off the vendor images is deliberate, so no accelerator container
needs a YAML dependency.

There is no generic integration fallback. A selected platform without a config fails discovery
instead of being silently omitted. `build-wheel.yml` -- the one parameterized wheel build, used by
`release.yml` and manual dispatches -- and `build-wheel-common.yml`, the single implementation it
calls, remain for release/manual wheel builds only; they do not run integration tests.
`wheel_artifact_name` is used by those build callers, while hardware CI builds and tests its wheel
in the same job.

## FlagCICD pytest reports

Each selected hardware job collects native `pytest-json-report` JSON from its
pytest processes and merges it into `pytest-<platform>.json` in the checkout.
The upload action's `report_path` names this generated file; `file_type` is
`pytest`. Both the service upload and artifact retention use a checkout-relative
report path so job containers resolve the generated file inside their mounted
workspace rather than using the runner host's checkout path from
`${{ github.workspace }}`. This reports test executions and outcomes, not Python
line/branch coverage or a percentage of supported operators. Repeated cases in
different groups remain separate executions, annotated with group/session metadata.

Collection is scoped to the hardware test step through `PYTEST_PLUGINS` and an
absolute report directory. Every pytest invocation gets a unique file, including
DCU's per-file unit runs. The collector removes its activation from each pytest
process's child environment so intentionally failing subprocess fixtures do not
pollute the platform result. It adds only its own helper directory to
`PYTHONPATH`, retaining tests against the installed wheel in the isolated cwd.

Completed reports survive a later group failure. Missing or truncated session
reports are recorded under `flagcicd.incomplete_sessions`; no passing cases are
invented for a killed process or an unexecuted group. The integration step's
outcome remains authoritative, including failures in non-pytest checks and the
existing opt-in teardown-crash policy. `flagcicd.complete` describes the started
pytest sessions only, not whether every configured group ran. Failure before any
pytest session produces no merged report. Cancelled jobs do not attempt uploads.

Raw session files and the merged report are retained as
`pytest-<platform>-<run_attempt>` workflow artifacts. Reporting and upload steps
have bounded timeouts and do not turn failed tests into successful CI. The
organization/repository secret `FLAGCICD_API_TOKEN` is passed explicitly from
`ci.yml` to the reusable integration workflow. An absent token skips the service
upload with a warning; it does not prevent report generation or artifact retention.
Fork PRs and the trusted fork diagnostic route retain artifacts but skip the
credentialed upload in the job executing fork code.

The uploader posts to
`https://flagcicd.flagos.net/flagcicd/api/v1/cicd/metrics/`, uses a
platform-specific `job_name`, and treats transport/service errors as warnings.
Job containers use their configured DNS and must be able to reach that HTTPS
endpoint with normal TLS certificate validation. No FlagCICD-specific hosts-file
mapping is injected into the containers.
Validate the upload's HTTP response and the corresponding FlagCICD record: a
green workflow alone does not prove ingestion. Backend parsing and dashboard
interpretation of the merged report require an actual authenticated upload.

## FlagCICD Python line coverage

Hardware jobs also collect real line coverage with coverage.py 7.10.7 and upload
`coverage-<platform>.json` with `file_type: coverage` and a distinct
`coverage-<platform>` job name. The pytest outcome report above remains a
separate upload. Test pass rate, workflow pass rate, Python line coverage and
operator support are different metrics.

The scope is **all Python source in the installed `torch_fl` wheel**, including
modules no test imports. Preparation resolves that directory through distribution
metadata without importing the accelerator. The collector starts before initial
conftest/test-module imports and writes a unique coverage.py data file for each
top-level pytest invocation, including DCU's per-file runs. Fixture-launched
sample subprocesses do not inherit activation. Python code executed only in
subprocesses spawned by a test, C/C++ extensions, external dependencies and
device kernels are outside this first implementation's scope. Source-tree unit
tests only contribute when they execute the installed package, not a checkout or
a fake package.

`python_coverage.py merge` combines executed-line sets with coverage.py, rather
than averaging session percentages. JSON filenames are portable `torch_fl/...`
paths; the denominator remains the installed wheel's Python files. No coverage
threshold is enforced. Failures retain the coverage saved before the failure;
missing/corrupt process data and the integration outcome are recorded in the
artifact's `metadata.json`. Its `complete` flag is false when the integration run failed or a
started coverage session did not finalize. No executed package lines means no
report, rather than a fabricated 0%/100% result. A killed process can lose its
unsaved coverage. Raw data, session markers and the merged JSON are retained in
`coverage-<platform>-<run_attempt>` artifacts, including on fork PRs. The uploaded
JSON itself is the unmodified native coverage.py report (`meta`, `files`,
`totals`); provenance/completeness metadata stays in the separate artifact file.

Both uploads keep the same optional-token/fork boundary and nonblocking error
policy. The MetaX job installs missing `curl` and CA certificates through its
Ubuntu image's package manager before credentialed uploads, inside the job
container. This requires root/package-mirror access; preparing these tools in a
future image removes that runtime dependency. DNS/routing problems on other
platforms still require runner network fixes. An HTTP 2xx plus the corresponding
parsed FlagCICD record is required to validate service ingestion for each report
type; green test/artifact steps alone do not establish it.
