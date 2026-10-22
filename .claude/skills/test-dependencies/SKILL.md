---
name: test-dependencies
description: >
  Prepare and validate isolated Python environments for torch_fl tests without
  replacing the PyTorch build or vendor runtime. Use this before model,
  integration, or manual tests when dependencies are missing or versions need
  to be pinned. Covers torch ABI protection, no-deps installs, optional test
  packages, package indexes, and an import/device smoke check.
---

# Test dependency environments (torch_fl)

## Scope

Use this skill whenever a test needs Python packages beyond the installed
`torch_fl` environment. The goal is a runnable, reproducible test environment,
not a package upgrade for its own sake.

The PyTorch build is part of the torch_fl ABI. Never let installing a test
package resolve a different torch version into the environment that contains
the compiled torch_fl extension. A successful import after an ABI-changing
upgrade is not evidence that the environment is valid.

For Transformers coverage, use this skill together with [[transformers-test]].
For platform bring-up or operator implementation, install only the dependencies
needed by that skill and keep this environment policy unchanged.

## Step 1 — inspect before installing

Record the active interpreter and the packages that define the test contract:

```bash
python - <<'PY'
import sys
print("python", sys.executable)
try:
    import torch
    print("torch", torch.__version__)
except ImportError:
    print("torch", "MISSING")
try:
    import torch_fl
    print("torch_fl", torch_fl.__file__)
except ImportError:
    print("torch_fl", "MISSING")
PY
python -m pip show torch torch_fl transformers accelerate 2>/dev/null || true
```

If the interpreter is not the one used to build `torch_fl`, stop and activate
that environment first. On a vendor box also record the accelerator, driver,
SDK, and vendor library versions before interpreting a test failure.

## Step 2 — choose isolation and pinning

Prefer one dedicated environment per `(torch_fl build, torch version,
transformers version)` tuple. Reuse an existing environment only after checking
that its torch version and torch_fl commit match the planned test.

When a test package declares `torch` as a dependency, install it with
`--no-deps` if torch is already present and validated. Install its non-torch
runtime dependencies explicitly. This prevents pip from silently replacing the
vendor-compatible torch wheel.

For a package that must be installed into a separate environment, create the
environment first and install the exact torch build required by torch_fl before
installing the test packages. Do not install the latest torch merely because a
test package accepts it.

## Step 3 — install the requested test dependencies

Install the smallest package set needed by the test. For Transformers official
model tests, the baseline set is:

```bash
python -m pip install --no-deps transformers==<version>
python -m pip install accelerate tokenizers safetensors huggingface_hub
```

`accelerate` is required by Transformers tests that use a `torch.device`
context manager, `device_map`, `tp_plan`, or `torch.set_default_device`. It is
not pulled in by a `--no-deps` Transformers install and its absence produces an
environment failure rather than a backend failure.

The exact Transformers version must match the cached official source tree. Do
not upgrade Transformers in place while retaining an older cached test source.
Use a new environment or a new versioned cache when changing versions.

For a normal package install, prefer an explicit index when the configured
mirror is unavailable, while preserving the proxy required by the environment:

```bash
source ../proxy.sh
python -m pip install --index-url https://pypi.org/simple <package>...
```

Never copy proxy credentials into source files, logs committed to the
repository, issue reports, or test evidence published outside the machine.

### Recovering Hugging Face connectivity

Official Transformers tests may download tiny fixtures from the Hugging Face
Hub even when the synthetic probe is fully offline. If `huggingface.co` is not
reachable, recover connectivity before classifying the test result:

1. Check whether the required files are already cached. If they are, rerun with
   `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` and do not treat cache misses as
   backend failures.
2. If a repository proxy is available, load it and retry the same command:

   ```bash
   source ../proxy.sh
   python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("hf-internal-testing/tiny-random-BertModel", "config.json"))'
   ```

3. If the origin remains unreachable, retry through the public Hub mirror:

   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   python -c 'from huggingface_hub import hf_hub_download; print(hf_hub_download("hf-internal-testing/tiny-random-BertModel", "config.json"))'
   ```

Use the proxy first when it is configured, then `HF_ENDPOINT` as the fallback.
Unset `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE` for either online attempt.
The endpoint variable is inherited by the official test subprocess. Record
whether the run used the default endpoint, a proxy, or `https://hf-mirror.com`,
but never record proxy URLs containing credentials.

`HF_ENDPOINT` only changes Hugging Face Hub downloads. The official runner's
version-matched Transformers source archive comes from GitHub; if that archive
cannot be downloaded, use `--source-dir`, prepare the versioned cache on a
machine with access, or rerun with `--offline` after the cache is complete.

## Step 4 — validate without changing the environment

Run an import and device smoke check immediately after installation:

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 python - <<'PY'
import accelerate
import torch
import torch_fl
import transformers

print("accelerate", accelerate.__version__)
print("torch", torch.__version__)
print("transformers", transformers.__version__)
print("torch_fl", torch_fl.__file__)
print("device_count", torch.flagos.device_count())
PY
```

Then verify that the torch version remains the one used by torch_fl:

```bash
python -m pip check
python -m pip show torch torch_fl accelerate transformers
```

A `pip check` warning caused solely by an intentionally installed
`--no-deps` package must be investigated; do not dismiss a torch conflict.
If torch changed, stop, restore the validated environment using the project's
pinned installation procedure, and rebuild/reinstall torch_fl before testing.

## Step 5 — run the test and record the environment

Record these values beside every result:

- Python executable and version
- torch version
- torch_fl commit or build identifier
- test package versions, especially Transformers and accelerate
- accelerator model, driver, SDK, and vendor library versions
- install command and any non-default package index

For Transformers coverage, set the custom device contract and run the official
runner only after the smoke check passes:

```bash
export TRANSFORMERS_TEST_DEVICE_SPEC=tests/manual/hf_device_spec.py
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
python tests/manual/transformers_hf_tests.py \
  --model <model> --transformers-version <version> \
  --out /tmp/<model>-official.json
```

Keep raw JSON and logs outside the repository. Do not commit package caches,
core dumps, credentials, or machine-local environment files.

## Failure classification

- Missing `accelerate`, tokenizer libraries, or optional Python packages is an
  **environment dependency failure**. Install the package and rerun the
  isolated test before filing a backend issue.
- A package resolver changing torch or producing an ABI/import error is an
  **environment setup failure**. Fix the environment before interpreting model
  results.
- A passing import/device smoke check followed by a reproducible accelerator
  error or wrong result is a **backend test finding**. Continue with the owning
  test skill's isolation, baseline, attribution, and deduplication steps.

This skill installs and validates dependencies only. It does not change routing,
patch backend code, create GitHub issues, or mark a failing test as supported.
